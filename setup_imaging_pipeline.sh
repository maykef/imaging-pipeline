#!/bin/bash

################################################################################
# Lightsheet Microscopy Image Processing Pipeline Setup Script
#
# COMPLETELY INDEPENDENT - Does NOT require llm-inference setup
#
# Creates separate conda environment: imaging_pipeline
# Installs Miniforge if needed
# Sets up all imaging tools
#
# Hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
# Usage: bash setup_imaging_pipeline.sh
################################################################################

set -e

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Don't run as root
if [ "$EUID" -eq 0 ]; then
  log_error "Please do not run this script as root or with sudo."
  log_info  "The script will ask for sudo password when needed."
  exit 1
fi

################################################################################
# PRE-FLIGHT CHECKS
################################################################################
log_info "Lightsheet Imaging Pipeline Setup (INDEPENDENT Installation)"
echo ""

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Please install NVIDIA drivers first."
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
log_info "GPU detected: $GPU_NAME ($GPU_MEMORY)"

# Check disk space
AVAILABLE_SPACE=$(df /home | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((20 * 1024 * 1024))  # 20GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
  log_error "Insufficient disk space. Need at least 20GB free in /home."
  exit 1
fi

log_success "Pre-flight checks passed!"
echo ""

################################################################################
# PHASE 1: INSTALL MINIFORGE (if needed)
################################################################################
log_info "PHASE 1: Checking for Miniforge..."

if [ -d "$HOME/miniforge3" ]; then
  log_success "Miniforge already installed at $HOME/miniforge3"
elif command -v mamba &>/dev/null; then
  log_success "Mamba already available in PATH"
else
  log_warning "Miniforge not found. Installing..."
  MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
  MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  wget -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL" --progress=bar:force 2>&1
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  log_success "Miniforge installed."
fi

################################################################################
# PHASE 2: INITIALIZE MAMBA
################################################################################
log_info "PHASE 2: Initializing Mamba..."
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"
eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"
$HOME/miniforge3/bin/mamba shell init -s bash --prefix "$HOME/miniforge3"
$HOME/miniforge3/bin/conda init bash || true
source ~/.bashrc || true
log_success "Mamba initialized: $(mamba --version)"
echo ""

################################################################################
# PHASE 3: CREATE ENVIRONMENT
################################################################################
log_info "PHASE 3: Creating imaging_pipeline environment..."

if mamba env list | grep -qE '^\s*imaging_pipeline\s'; then
  log_warning "imaging_pipeline exists. Removing..."
  mamba env remove -n imaging_pipeline -y
fi

if [ -f "environment_imaging.yml" ]; then
  mamba env create -f environment_imaging.yml --yes
else
  mamba create -n imaging_pipeline python=3.11 pip numpy scipy scikit-image scikit-learn pandas matplotlib -y
fi

log_success "Base environment created."
echo ""

################################################################################
# PHASE 4: INSTALL IMAGING PACKAGES
################################################################################
log_info "PHASE 4: Installing imaging packages..."

eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate imaging_pipeline

cat > /tmp/imaging_requirements.txt << 'REQS'
aicspylibczi>=3.3.0
czifile>=2019.7.2
tifffile>=2024.1.0
imagecodecs>=2024.1.0
SimpleITK>=2.2.0
pycpd>=0.2.1
zarr>=2.16.0
ome-zarr-py>=0.8.0
dask[array]>=2023.12.0
cellpose==3.0.8
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
napari>=0.4.18
napari-aicsimageio>=0.8.1
aicsimageio>=0.9.0
seaborn>=0.13.0
opencv-python>=4.8.0
openpyxl>=3.1.0
numpy-stl>=3.1.0
REQS

log_info "Installing packages via pip (this may take a while)..."
python -m pip install --upgrade pip
python -m pip install --upgrade -r /tmp/imaging_requirements.txt
rm /tmp/imaging_requirements.txt

log_success "All imaging packages installed."
echo ""

################################################################################
# PHASE 5: VERIFY INSTALLATION
################################################################################
log_info "PHASE 5: Verifying installation..."

eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate imaging_pipeline

python << 'EOF'
import sys
checks = {
    'aicspylibczi': 'CZI I/O',
    'SimpleITK': 'Registration',
    'zarr': 'Storage',
    'cellpose': 'Segmentation',
    'napari': 'Visualization',
    'torch': 'Deep Learning',
    'pandas': 'Analysis',
}
failed = []
for lib, desc in checks.items():
    try:
        __import__(lib)
        print(f"✓ {lib:20} ({desc})")
    except Exception as e:
        print(f"✗ {lib:20} ({desc}) - {e}")
        failed.append(lib)
if failed:
    print(f"\n⚠ {len(failed)} package(s) failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All packages verified!")

import torch
print("\nGPU Status:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
EOF

log_success "Verification complete."
echo ""

################################################################################
# PHASE 6: WORKSPACE + HELPERS
################################################################################
log_info "PHASE 6: Setting up workspace..."
mkdir -p ~/imaging-workspace/{configs,scripts,notebooks,example_data,logs}
mkdir -p /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis,cache}
sudo chown -R "$USER":"$USER" /scratch/imaging 2>/dev/null || true

cat > ~/imaging-workspace/configs/pipeline_config.yaml << 'CONFIG'
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

cat > ~/start-imaging-pipeline.sh << 'HELPER'
#!/bin/bash
echo "🔬 Lightsheet Imaging Pipeline"
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate imaging_pipeline
python -c "import torch; print(f'✓ GPU available: {torch.cuda.is_available()}')"
bash
HELPER
chmod +x ~/start-imaging-pipeline.sh

log_success "Helper script: ~/start-imaging-pipeline.sh"

################################################################################
# DONE
################################################################################
log_success "=========================================="
log_success "  IMAGING PIPELINE READY!"
log_success "=========================================="
