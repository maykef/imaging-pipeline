# Create requirements file
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
REQS# Create from YAML file if it exists, otherwise create minimal and pip install
if [ -f "environment_imaging.yml" ]; then
  log_info "Creating from environment_imaging.yml..."
  mamba env create -f environment_imaging.yml --yes
else
  log_info "environment_imaging.yml not found. Creating minimal environment..."
  mamba create -n imaging_pipeline python=3.11 pip numpy scipy scikit-image scikit-learn pandas matplotlib -y
fi

log_success "Base environment created."

# Always activate and run pip install
log_info "Activating imaging_pipeline..."
mamba activate imaging_pipeline

# Create requirements file
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

log_info "Installing imaging packages via pip (this may take 10-15 minutes)..."
pip install --upgrade -r /tmp/imaging_requirements.txt
rm /tmp/imaging_requirements.txt

log_success "All packages installed."#!/bin/bash

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

log_warning "This script will:"
echo "  1) Install Miniforge (if not present)"
echo "  2) Create imaging_pipeline conda environment"
echo "  3) Install all imaging libraries"
echo "  4) Set up workspace at ~/imaging-workspace"
echo ""
echo "Total time: ~25-35 min"
echo "Disk space: ~8-12 GB"
echo ""
log_warning "⚠️  COMPLETELY INDEPENDENT"
echo "  - Does NOT require llm-inference setup"
echo "  - Does NOT require existing Miniforge"
echo "  - Will install Miniforge if needed"
echo ""
read -p "Continue with installation? (y/N): " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] || { log_info "Installation cancelled."; exit 0; }

################################################################################
# PHASE 1: INSTALL MINIFORGE (if needed)
################################################################################
log_info "PHASE 1: Checking for Miniforge..."

if [ -d "$HOME/miniforge3" ]; then
  log_success "Miniforge already installed at $HOME/miniforge3"
  SKIP_MINIFORGE=true
elif command -v mamba &>/dev/null; then
  log_success "Mamba already available in PATH"
  SKIP_MINIFORGE=true
else
  log_warning "Miniforge not found. Installing..."
  MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
  MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  
  log_info "Downloading Miniforge (~500 MB)..."
  [ -f "$MINIFORGE_INSTALLER" ] || wget -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL" --progress=bar:force 2>&1
  
  log_info "Installing Miniforge..."
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  log_success "Miniforge installed."
fi

echo ""

################################################################################
# PHASE 2: INITIALIZE MAMBA
################################################################################
log_info "PHASE 2: Initializing Mamba..."

# Make sure mamba is available
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"

# Load mamba into THIS shell
eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"

# Persist for future shells
if $HOME/miniforge3/bin/mamba shell init -h | grep -q -- "--prefix"; then
  $HOME/miniforge3/bin/mamba shell init -s bash --prefix "$HOME/miniforge3"
else
  $HOME/miniforge3/bin/mamba shell init -s bash
fi

# Also initialize conda
$HOME/miniforge3/bin/conda init bash || true

# Ensure hook in ~/.bashrc
if ! grep -q 'mamba shell hook --shell bash' ~/.bashrc; then
  echo 'eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"' >> ~/.bashrc
fi

# Ensure login shells source ~/.bashrc
if ! grep -q 'source ~/.bashrc' ~/.profile 2>/dev/null; then
  echo '' >> ~/.profile
  echo '# Load interactive bash settings' >> ~/.profile
  echo 'if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi' >> ~/.profile
fi

# Reload
source ~/.bashrc || true

# Verify mamba works
if ! command -v mamba &> /dev/null; then
  log_error "Mamba initialization failed."
  log_info "Try: eval \"\$($HOME/miniforge3/bin/mamba shell hook --shell bash)\""
  exit 1
fi

log_success "Mamba initialized: $(mamba --version)"
echo ""

################################################################################
# PHASE 3: CREATE IMAGING_PIPELINE ENVIRONMENT
################################################################################
log_info "PHASE 3: Creating imaging_pipeline environment..."

# Remove if exists
if mamba env list | grep -qE '^\s*imaging_pipeline\s'; then
  log_warning "imaging_pipeline environment exists. Removing..."
  mamba env remove -n imaging_pipeline -y
fi

# Create from YAML file if it exists, otherwise create minimal and pip install
if [ -f "environment_imaging.yml" ]; then
  log_info "Creating from environment_imaging.yml..."
  mamba env create -f environment_imaging.yml --yes
else
  log_info "environment_imaging.yml not found. Creating minimal environment..."
  mamba create -n imaging_pipeline python=3.11 pip -y
  
  # Activate for pip installs
  mamba activate imaging_pipeline
  
  log_info "Installing imaging packages via pip..."
  
  # Create requirements file to avoid permission issues
  # NOTE: Verified working versions (2025-01-16)
  cat > /tmp/imaging_requirements.txt << 'REQS'
aicspylibczi>=3.3.0
czifile>=2019.7.2
SimpleITK>=2.2.0
pycpd>=0.2.1
zarr>=2.16.0
dask[array]>=2023.12.0
cellpose==3.0.8
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
napari>=0.4.18
napari-aicsimageio>=0.7.0
aicsimageio>=0.9.0
seaborn>=0.13.0
opencv-python>=4.8.0
openpyxl>=3.1.0
REQS
  
  log_info "Installing imaging packages via pip..."
  pip install --upgrade -r /tmp/imaging_requirements.txt
  rm /tmp/imaging_requirements.txt
fi

log_success "Environment created: imaging_pipeline"
echo ""

################################################################################
# PHASE 4: ACTIVATE AND VERIFY
################################################################################
log_info "PHASE 4: Verifying installation..."

# Activate
mamba activate imaging_pipeline

# Verify imports
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
        print(f"✗ {lib:20} ({desc}) - {str(e)[:40]}")
        failed.append(lib)

if failed:
    print(f"\n⚠ {len(failed)} package(s) failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All packages verified!")

# GPU check
import torch
print(f"\nGPU Status:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""

################################################################################
# PHASE 5: SETUP WORKSPACE
################################################################################
log_info "PHASE 5: Setting up workspace..."

mkdir -p ~/imaging-workspace/{configs,scripts,notebooks,example_data,logs}
mkdir -p /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis,cache}

# Set permissions
sudo chown -R "$USER":"$USER" /scratch/imaging 2>/dev/null || true

# Create config template
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
echo ""

################################################################################
# PHASE 6: CREATE HELPERS
################################################################################
log_info "PHASE 6: Creating helper scripts..."

cat > ~/start-imaging-pipeline.sh << 'HELPER'
#!/bin/bash
echo "🔬 Lightsheet Imaging Pipeline"
eval "$(mamba shell hook --shell bash)"
mamba activate imaging_pipeline
echo "✓ Environment: imaging_pipeline"
echo "✓ Workspace: ~/imaging-workspace"
echo "✓ Data: /scratch/imaging"
python -c "import torch; print(f'✓ GPU: {torch.cuda.is_available()}')"
bash
HELPER

chmod +x ~/start-imaging-pipeline.sh

cat > ~/switch-to-imaging.sh << 'SWITCH'
#!/bin/bash
eval "$(mamba shell hook --shell bash)"
mamba activate imaging_pipeline
bash
SWITCH

chmod +x ~/switch-to-imaging.sh

log_success "Helper scripts created"
echo ""

################################################################################
# FINAL VERIFICATION
################################################################################
log_info "PHASE 7: Final verification..."

# Environment list
log_success "Available environments:"
mamba env list | grep -E "^(base|imaging_pipeline)"

# Test CZI reading
python -c "from aicspylibczi import CziFile; print('✓ CZI I/O')"
python -c "from cellpose.core import use_gpu; print(f'✓ Cellpose GPU: {use_gpu()}')"

echo ""
log_success "=========================================="
log_success "  IMAGING PIPELINE READY!"
log_success "=========================================="
echo ""

log_info "Installation Summary:"
echo "  ✓ imaging_pipeline environment created"
echo "  ✓ All imaging libraries installed"
echo "  ✓ Workspace at ~/imaging-workspace"
echo "  ✓ GPU support confirmed"
echo ""

log_info "Quick Start:"
echo "  1) Close and reopen terminal"
echo "  2) Run: mamba activate imaging_pipeline"
echo "     OR: ~/start-imaging-pipeline.sh"
echo "  3) Edit config: ~/imaging-workspace/configs/pipeline_config.yaml"
echo ""

cat > ~/imaging-pipeline-install.log << LOG
Imaging Pipeline Installation Log
Date: $(date)
User: $USER
Environment: imaging_pipeline
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
LOG

log_success "Installation log: ~/imaging-pipeline-install.log"
log_success "READY FOR LIGHTSHEET IMAGE PROCESSING!"
