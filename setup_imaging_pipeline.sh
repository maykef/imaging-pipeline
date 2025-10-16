#!/bin/bash

################################################################################
# setup_imaging_pipeline.sh
#
# Lightsheet Microscopy Image Processing Pipeline – Full Installer
# - Creates/refreshes conda env: imaging_pipeline
# - Installs imaging stack (conda-forge for compiled libs)
# - Handles Blackwell (sm_120) GPUs by switching to PyTorch nightly cu124 wheels
#
# Usage: bash setup_imaging_pipeline.sh
################################################################################

set -euo pipefail
IFS=$'\n\t'

# Colors / logging
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# No root
if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  log_error "Do not run as root/sudo."
  exit 1
fi

################################################################################
# PRE-FLIGHT
################################################################################
log_info "Lightsheet Imaging Pipeline Setup (INDEPENDENT Installation)"
echo ""

# NVIDIA presence
if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Install NVIDIA drivers first."
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
log_info "GPU detected: ${GPU_NAME} (${GPU_MEMORY})"

# Disk space check (20 GB in $HOME)
AVAILABLE_SPACE_KB=$(df -k "$HOME" | tail -1 | awk '{print $4}')
REQUIRED_SPACE_KB=$((20 * 1024 * 1024))
if [ "$AVAILABLE_SPACE_KB" -lt "$REQUIRED_SPACE_KB" ]; then
  log_error "Insufficient disk space. Need >= 20 GB free in $HOME."
  exit 1
fi

log_success "Pre-flight checks passed!"
echo ""

read -p "Continue with installation? (y/N): " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] || { log_info "Installation cancelled."; exit 0; }

################################################################################
# PHASE 1: Miniforge
################################################################################
log_info "PHASE 1: Checking for Miniforge…"
MINIFORGE_HOME="$HOME/miniforge3"

if [ -d "$MINIFORGE_HOME" ]; then
  log_success "Miniforge found at $MINIFORGE_HOME"
elif command -v mamba &>/dev/null; then
  log_success "Mamba available in PATH"
else
  log_warning "Miniforge not found. Installing…"
  INST="/tmp/Miniforge3-Linux-x86_64.sh"
  URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  [ -f "$INST" ] || wget -O "$INST" "$URL" --progress=bar:force 2>&1
  bash "$INST" -b -p "$MINIFORGE_HOME"
  log_success "Miniforge installed."
fi
echo ""

################################################################################
# PHASE 2: Init mamba
################################################################################
log_info "PHASE 2: Initializing mamba…"
export MAMBA_ROOT_PREFIX="$MINIFORGE_HOME"
eval "$("$MINIFORGE_HOME/bin/mamba" shell hook --shell bash)"

# Persist (backward compatible; don’t fail if old mamba)
set +e
if "$MINIFORGE_HOME/bin/mamba" shell init -h 2>&1 | grep -q -- '--prefix'; then
  "$MINIFORGE_HOME/bin/mamba" shell init -s bash --prefix "$MINIFORGE_HOME" >/dev/null 2>&1 || true
else
  "$MINIFORGE_HOME/bin/mamba" shell init -s bash >/dev/null 2>&1 || true
fi
"$MINIFORGE_HOME/bin/conda" init bash >/dev/null 2>&1 || true
set -e
source ~/.bashrc 2>/dev/null || true

command -v mamba >/dev/null || { log_error "Mamba initialization failed."; exit 1; }
log_success "Mamba: $(mamba --version)"
echo ""

################################################################################
# PHASE 3: Create env
################################################################################
log_info "PHASE 3: Creating imaging_pipeline environment…"

if mamba env list | grep -qE '^\s*imaging_pipeline\s'; then
  log_warning "imaging_pipeline exists. Removing…"
  mamba env remove -n imaging_pipeline -y
fi

if [ -f "environment_imaging.yml" ]; then
  log_info "Creating from environment_imaging.yml…"
  mamba env create -f environment_imaging.yml --yes
else
  log_info "No YAML found. Creating minimal env…"
  mamba create -n imaging_pipeline python=3.11 pip numpy scipy scikit-image scikit-learn pandas matplotlib -y
fi

log_success "Base environment created."
echo ""

################################################################################
# PHASE 4: Install imaging stack (conda-forge/pytorch) + Blackwell fix
################################################################################
log_info "PHASE 4: Installing imaging packages…"
eval "$("$MINIFORGE_HOME/bin/mamba" shell hook --shell bash)"
mamba activate imaging_pipeline

# --- PyTorch via conda (try CUDA builds; fallback to CPU). We'll swap for Blackwell later. ---
set +e
CUDA_OK=0
for CUDA_VER in 12.4 12.3 12.1; do
  log_info "Trying CUDA PyTorch (pytorch-cuda=${CUDA_VER}) via conda…"
  mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=${CUDA_VER}
  if [ $? -eq 0 ]; then CUDA_OK=1; break; fi
done
set -e
if [ "$CUDA_OK" -eq 0 ]; then
  log_warning "CUDA PyTorch (conda) not available; installing CPU-only PyTorch."
  mamba install -y -c conda-forge pytorch torchvision torchaudio
fi

# --- Core compiled/image stack from conda-forge ---
mamba install -y -c conda-forge \
  aicspylibczi czifile tifffile imagecodecs \
  simpleitk \
  "zarr>=2.16.0,<3.0.0" ome-zarr \
  dask dask-image \
  napari pyqt \
  aicsimageio \
  opencv openpyxl seaborn numpy-stl

# --- Pip only for a few; avoid resolver backtracking (no deps; deps satisfied via conda) ---
python -m pip install --upgrade pip
python -m pip install --no-build-isolation --no-deps \
  "cellpose==3.0.8" \
  "napari-aicsimageio==0.7.2"

# --- Cellpose runtime dependencies from conda-forge ---
mamba install -y -c conda-forge fastremap numba natsort tqdm roifile

# --- Blackwell (sm_120) fix: swap to nightly cu124 wheels that include sm_120 support ---
if echo "$GPU_NAME" | grep -qi "Blackwell"; then
  log_info "Blackwell GPU detected. Switching to PyTorch nightly cu124 wheels for sm_120 support…"
  mamba remove -y pytorch torchvision torchaudio || true
  python -m pip install --upgrade --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124
fi

log_success "All imaging packages installed."
echo ""

################################################################################
# PHASE 5: Verify
################################################################################
log_info "PHASE 5: Verifying installation…"
eval "$("$MINIFORGE_HOME/bin/mamba" shell hook --shell bash)"
mamba activate imaging_pipeline

python << 'EOF'
import sys, importlib, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

checks = {
    'aicspylibczi': 'CZI I/O',
    'SimpleITK':    'Registration',
    'zarr':         'Storage',
    'cellpose':     'Segmentation',
    'napari':       'Visualization',
    'torch':        'Deep Learning',
    'pandas':       'Analysis',
    'fastremap':    'Cellpose dep',
}
failed=[]
for lib, desc in checks.items():
    try:
        importlib.import_module(lib)
        print(f"✓ {lib:20} ({desc})")
    except Exception as e:
        print(f"✗ {lib:20} ({desc}) - {e}")
        failed.append(lib)

print("\nGPU Status:")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM: {props.total_memory/1e9:.1f} GB")
        try:
            cap = torch.cuda.get_device_capability(0)
            print(f"  Capability: sm_{cap[0]}{cap[1]}")
        except Exception: pass
except Exception as e:
    print(f"  PyTorch check failed: {e}")

if failed:
    print(f"\n⚠ {len(failed)} package(s) failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All packages verified!")
EOF
echo ""

################################################################################
# PHASE 6: Workspace
################################################################################
log_info "PHASE 6: Setting up workspace…"
mkdir -p "$HOME/imaging-workspace"/{configs,scripts,notebooks,example_data,logs}
sudo mkdir -p /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis,cache} 2>/dev/null || true
sudo chown -R "$USER":"$USER" /scratch/imaging 2>/dev/null || true

cat > "$HOME/imaging-workspace/configs/pipeline_config.yaml" << 'CONFIG'
# Lightsheet Imaging Pipeline Configuration
imaging:
  input_dir: /tmp/dataset1
  output_base: /scratch/imaging
  czi:
    z_step_microns: 0.5
    tile_overlap_percent: 15
    background_subtraction: true

registration:
  method: "cross_correlation"
  max_iterations: 100

deconvolution:
  enabled: true
  iterations: 20

segmentation:
  cellpose_model: "nuclei"
  cellpose_diameter: 30
  gpu_enabled: true
  batch_size: 4

hardware:
  n_workers: 16
  max_memory_gb: 60
CONFIG

log_success "Workspace created at ~/imaging-workspace"
echo ""

################################################################################
# PHASE 7: Helper scripts
################################################################################
log_info "PHASE 7: Creating helper scripts…"

cat > "$HOME/start-imaging-pipeline.sh" << 'HELPER'
#!/bin/bash
echo "🔬 Lightsheet Imaging Pipeline"
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate imaging_pipeline
echo "✓ Environment: imaging_pipeline"
echo "✓ Workspace: ~/imaging-workspace"
echo "✓ Data: /scratch/imaging"
python - << 'PY'
import warnings, torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
print(f"✓ PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
PY
bash
HELPER
chmod +x "$HOME/start-imaging-pipeline.sh"

cat > "$HOME/switch-to-imaging.sh" << 'SWITCH'
#!/bin/bash
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate imaging_pipeline
bash
SWITCH
chmod +x "$HOME/switch-to-imaging.sh"

log_success "Helper scripts created"
echo ""

################################################################################
# PHASE 8: Final checks
################################################################################
log_info "PHASE 8: Final verification…"
log_success "Available environments:"
mamba env list | grep -E "^(base|imaging_pipeline)" || true

python - << 'EOF'
try:
    from aicspylibczi import CziFile
    print('✓ CZI I/O (aicspylibczi import ok)')
except Exception as e:
    print(f'⚠ aicspylibczi quick check failed: {e}')
try:
    from cellpose.core import use_gpu
    print(f'✓ Cellpose use_gpu(): {use_gpu()}')
except Exception as e:
    print(f'⚠ Cellpose quick check failed: {e}')
EOF

echo ""
log_success "=========================================="
log_success "  IMAGING PIPELINE READY!"
log_success "=========================================="
echo ""

log_info "Installation Summary:"
echo "  ✓ imaging_pipeline environment created"
echo "  ✓ Imaging libraries installed (conda-forge; PyTorch CUDA if available)"
echo "  ✓ Blackwell GPUs auto-switch to nightly cu124 wheels"
echo "  ✓ Workspace at ~/imaging-workspace"
echo "  ✓ GPU support checked"
echo ""

cat > "$HOME/imaging-pipeline-install.log" << LOG
Imaging Pipeline Installation Log
Date: $(date)
User: $USER
Environment: imaging_pipeline
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
LOG

log_success "Installation log: ~/imaging-pipeline-install.log"
log_success "READY FOR LIGHTSHEET IMAGE PROCESSING!"
