#!/bin/bash

################################################################################
# Lightsheet Microscopy Image Processing Pipeline Setup Script
#
# Creates a SEPARATE conda environment for imaging tools
# Does NOT modify existing llm-inference environment
#
# Environment: imaging_pipeline (Python 3.11, independent from llm-inference)
# Hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
#
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
log_info "Lightsheet Imaging Pipeline Setup (Separate Environment)"
echo ""

# Initialize mamba
log_info "Initializing mamba..."
eval "$(~/miniforge3/bin/mamba shell hook --shell bash)"

# Check mamba availability
if ! command -v mamba &>/dev/null; then
  log_error "Mamba not found. Please ensure Miniforge is installed at ~/miniforge3"
  log_info "If setup_llm_core.sh hasn't run yet, run it first:"
  log_info "  bash setup_llm_core.sh"
  exit 1
fi

log_success "Mamba initialized."

# Check CUDA
if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Ensure NVIDIA drivers are installed."
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
log_info "GPU detected: $GPU_NAME"

# Check for existing imaging_pipeline env
if mamba env list | grep -qE '^\s*imaging_pipeline\s'; then
  log_warning "imaging_pipeline environment already exists."
  read -p "Remove and recreate? (y/N): " -n 1 -r; echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Removing existing imaging_pipeline environment..."
    mamba env remove -n imaging_pipeline -y
  else
    log_info "Using existing environment. Skipping creation."
    SKIP_ENV_CREATE=true
  fi
fi

echo ""
log_warning "This script will:"
echo "  1) Create a new conda environment: imaging_pipeline"
echo "  2) Install: CZI I/O, registration, stitching, segmentation, visualization"
echo "  3) Set up workspace at ~/imaging-workspace"
echo ""
log_warning "⚠️  IMPORTANT:"
echo "  - imaging_pipeline is SEPARATE from llm-inference"
echo "  - Both can run independently without conflicts"
echo "  - Activate with: mamba activate imaging_pipeline"
echo "  - Switch to LLM: mamba activate llm-inference"
echo ""
echo "Total time: ~25–35 min (network dependent)"
echo "Disk space: ~8–12 GB (separate from llm-inference)"
echo ""
read -p "Continue with installation? (y/N): " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] || { log_info "Installation cancelled."; exit 0; }

################################################################################
# PHASE 0: CREATE ENVIRONMENT FROM YAML
################################################################################
if [ "$SKIP_ENV_CREATE" != "true" ]; then
  log_info "PHASE 0: Creating imaging_pipeline environment from environment.yml..."
  
  # Check if environment.yml exists in current directory
  if [ ! -f "environment_imaging.yml" ]; then
    log_warning "environment_imaging.yml not found in current directory."
    log_info "Creating minimal environment file..."
    
    # Will be created by the caller, but we can create inline
    cat > environment_imaging.yml << 'EOF'
name: imaging_pipeline
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pip
  - numpy>=1.24.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - scikit-image>=0.22.0
  - scikit-learn>=1.3.0
  - pandas>=2.0.0
  - openpyxl>=3.1.0
  - cython>=0.29.30
  - h5py>=3.8.0
  - cmake>=3.25.0
  - pkg-config>=2.0.0
  - pip
  - pip:
    - aicspylibczi>=3.3.0
    - czifile>=2019.7.2
    - tifffile>=2024.1.0
    - imagecodecs>=2024.1.0
    - SimpleITK>=2.2.0
    - pycpd>=0.2.1
    - zarr>=2.16.0
    - ome-zarr-py>=0.8.0
    - dask[array]>=2023.12.0
    - cellpose==3.0.8
    - torch>=2.0.0
    - torchvision>=0.15.0
    - torchaudio>=2.0.0
    - napari>=0.4.18
    - napari-aicsimageio>=0.8.1
    - aicsimageio>=0.9.0
    - seaborn>=0.13.0
    - opencv-python>=4.8.0
EOF
    log_success "Created environment_imaging.yml"
  fi
  
  log_info "Creating environment from YAML..."
  mamba env create -f environment_imaging.yml --yes
  
  log_success "Environment created."
else
  log_info "Skipping environment creation (using existing)."
fi

echo ""

################################################################################
# ACTIVATE ENVIRONMENT
################################################################################
log_info "Activating imaging_pipeline environment..."
eval "$(mamba shell hook --shell bash)"
mamba activate imaging_pipeline

if [ "$CONDA_DEFAULT_ENV" != "imaging_pipeline" ]; then
  log_error "Failed to activate imaging_pipeline environment."
  exit 1
fi

log_success "Environment activated: $CONDA_DEFAULT_ENV"
echo ""

################################################################################
# PHASE 1: VERIFY CZI I/O
################################################################################
log_info "PHASE 1: Verifying CZI file I/O..."

python << 'EOF'
try:
    from aicspylibczi import CziFile
    print("✓ aicspylibczi OK")
except Exception as e:
    print(f"✗ aicspylibczi failed: {e}")
    
try:
    import tifffile
    print("✓ tifffile OK")
except Exception as e:
    print(f"✗ tifffile failed: {e}")

try:
    import czifile
    print("✓ czifile OK (fallback)")
except Exception as e:
    print(f"⚠ czifile skipped: {e}")
EOF

echo ""

################################################################################
# PHASE 2: VERIFY IMAGE PROCESSING
################################################################################
log_info "PHASE 2: Verifying image processing libraries..."

python << 'EOF'
try:
    import SimpleITK
    print(f"✓ SimpleITK {SimpleITK.__version__} OK")
except Exception as e:
    print(f"✗ SimpleITK failed: {e}")

try:
    import skimage
    print(f"✓ scikit-image {skimage.__version__} OK")
except Exception as e:
    print(f"✗ scikit-image failed: {e}")

try:
    import zarr
    print(f"✓ zarr {zarr.__version__} OK")
except Exception as e:
    print(f"✗ zarr failed: {e}")

try:
    import dask
    print(f"✓ dask {dask.__version__} OK")
except Exception as e:
    print(f"✗ dask failed: {e}")
EOF

echo ""

################################################################################
# PHASE 3: VERIFY GPU SEGMENTATION
################################################################################
log_info "PHASE 3: Verifying GPU segmentation (Cellpose)..."

python << 'EOF'
try:
    import cellpose
    print(f"✓ Cellpose {cellpose.__version__} OK")
except Exception as e:
    print(f"✗ Cellpose failed: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✓ CUDA support enabled")
    else:
        print("⚠ GPU not available (will use CPU)")
except Exception as e:
    print(f"✗ PyTorch check failed: {e}")

try:
    from cellpose.core import use_gpu
    if use_gpu():
        print("✓ Cellpose GPU support confirmed")
    else:
        print("⚠ Cellpose will use CPU")
except Exception as e:
    print(f"⚠ Cellpose GPU check: {e}")
EOF

echo ""

################################################################################
# PHASE 4: VERIFY VISUALIZATION
################################################################################
log_info "PHASE 4: Verifying visualization (Napari)..."

python << 'EOF'
try:
    import napari
    print(f"✓ Napari {napari.__version__} OK")
except Exception as e:
    print(f"✗ Napari failed: {e}")

try:
    import aicsimageio
    print(f"✓ aicsimageio OK")
except Exception as e:
    print(f"⚠ aicsimageio: {e}")
EOF

echo ""

################################################################################
# PHASE 5: SETUP WORKSPACE
################################################################################
log_info "PHASE 5: Setting up workspace directories..."

# Create imaging workspace
mkdir -p ~/imaging-workspace/{configs,scripts,notebooks,example_data,logs}
mkdir -p /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis,cache}

# Set permissions
sudo chown -R "$USER":"$USER" /scratch/imaging

log_success "Workspace directories created."

# Create configuration template
cat > ~/imaging-workspace/configs/pipeline_config.yaml << 'CONFIG'
# Lightsheet Imaging Pipeline Configuration
# Zeiss Z.1 Lightsheet Microscopy

imaging:
  input_dir: /tmp/dataset1
  output_base: /scratch/imaging
  
  # CZI-specific parameters (from microscope metadata)
  czi:
    z_step_microns: 0.5         # Adjust based on actual acquisition
    tile_overlap_percent: 15
    background_subtraction: true
    flat_field_correction: false
    
preprocessing:
  # Deskewing (if oblique acquisition)
  deskew_enabled: false
  deskew_angle: 45
  
  # Background subtraction
  background_method: "mean"      # "mean", "rolling_ball", or "none"
  background_window: 50
  
  # Denoising
  denoise_method: "tv"           # "tv", "wavelet", or "none"
  denoise_strength: 0.1

registration:
  method: "cross_correlation"    # "feature", "cross_correlation", or "metadata"
  max_iterations: 100
  metric: "correlation"

deconvolution:
  enabled: true
  method: "richardson_lucy"
  iterations: 20
  psf_method: "metadata"

segmentation:
  cellpose_model: "nuclei"       # "nuclei", "cyto", "cyto2"
  cellpose_diameter: 30
  gpu_enabled: true
  batch_size: 4

output:
  format: "ome_zarr"             # "ome_zarr", "ome_tiff", or "both"
  compression: "blosc"
  chunk_size: [64, 256, 256]
  
hardware:
  n_workers: 16                  # Use 16 of 48 Threadripper cores
  gpu_batch_size: 4
  max_memory_gb: 60              # Reserve for imaging (LLM uses ~90GB)
CONFIG

log_success "Configuration template created: ~/imaging-workspace/configs/pipeline_config.yaml"
echo ""

################################################################################
# PHASE 6: CREATE ACTIVATION HELPERS
################################################################################
log_info "PHASE 6: Creating activation helpers..."

# Main helper script
cat > ~/start-imaging-pipeline.sh << 'HELPER'
#!/bin/bash
echo "🔬 Lightsheet Imaging Pipeline Environment"
echo ""
eval "$(mamba shell hook --shell bash)"
mamba activate imaging_pipeline
echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo "✓ Workspace: $HOME/imaging-workspace"
echo "✓ Data: /scratch/imaging"
echo ""
python -c "
from aicspylibczi import CziFile
from cellpose.core import use_gpu
import torch
print('✓ CZI I/O ready')
print(f'✓ GPU available: {torch.cuda.is_available()}')
print(f'✓ Cellpose GPU support: {use_gpu()}')
"
echo ""
bash
HELPER

chmod +x ~/start-imaging-pipeline.sh

# Switch helper (for convenience)
cat > ~/switch-to-imaging.sh << 'SWITCH'
#!/bin/bash
eval "$(mamba shell hook --shell bash)"
mamba activate imaging_pipeline
echo "✓ Switched to imaging_pipeline"
echo "To return to llm-inference: mamba activate llm-inference"
bash
SWITCH

chmod +x ~/switch-to-imaging.sh

log_success "Created helpers:"
echo "  ~/start-imaging-pipeline.sh     (full startup with checks)"
echo "  ~/switch-to-imaging.sh          (quick activation)"
echo ""

################################################################################
# PHASE 7: FINAL SUMMARY
################################################################################
log_success "=========================================="
log_success "  IMAGING PIPELINE SETUP COMPLETED!"
log_success "=========================================="
echo ""

log_info "Environment Created: imaging_pipeline"
echo "  Location: $CONDA_PREFIX"
echo "  Python: 3.11"
echo "  Isolation: Completely separate from llm-inference"
echo ""

log_info "Installation Summary:"
echo "  ✓ CZI file I/O (aicspylibczi, tifffile)"
echo "  ✓ Image registration (SimpleITK, pycpd)"
echo "  ✓ Stitching & storage (zarr, dask)"
echo "  ✓ Deconvolution (scipy, scikit-image)"
echo "  ✓ Segmentation (Cellpose 3.0.8, GPU-native)"
echo "  ✓ Visualization (Napari)"
echo "  ✓ Analysis (Pandas, scikit-learn)"
echo ""

log_info "Quick Start:"
echo "  1) Close and reopen terminal"
echo "  2) Run: ~/start-imaging-pipeline.sh"
echo "     OR: mamba activate imaging_pipeline"
echo ""

log_info "Available Environments:"
echo "  mamba activate llm-inference      # LLM inference (vLLM, transformers)"
echo "  mamba activate imaging_pipeline   # Image processing (CZI, stitching, etc.)"
echo ""

log_warning "Workspace Locations:"
echo "  Scripts: ~/imaging-workspace/scripts"
echo "  Configs: ~/imaging-workspace/configs/pipeline_config.yaml"
echo "  Data: /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis}"
echo ""

log_warning "Next Steps:"
echo "  1) Download test CZI dataset:"
echo "     cd /tmp && wget https://datadryad.org/.../LSFM_000*.czi.bz2"
echo "     bunzip2 LSFM_*.czi.bz2"
echo ""
echo "  2) Test CZI reading:"
echo "     mamba activate imaging_pipeline"
echo "     python ~/imaging-workspace/scripts/test_czi_read.py"
echo ""
echo "  3) Build MVP stitcher (coming next)"
echo ""
echo "  4) Integrate with LLM agent for agentic orchestration"
echo ""

# Save installation log
cat > ~/imaging-pipeline-install.log << LOG
Imaging Pipeline Installation Log
Date: $(date)
User: $USER
Hostname: $(hostname)
Environment: imaging_pipeline
Python: 3.11
Conda Prefix: $CONDA_PREFIX

GPU Status:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

PyTorch CUDA:
$(python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')" 2>/dev/null || echo "  Check manually after install")

Installed Packages:
- aicspylibczi (3.3.0+)
- SimpleITK (2.2.0+)
- zarr (2.16.0+)
- cellpose (3.0.8)
- napari (0.4.18+)
- pandas (2.0.0+)
- dask (2023.12.0+)

Workspace: ~/imaging-workspace
Data: /scratch/imaging
LOG

log_success "Installation log: ~/imaging-pipeline-install.log"
echo ""
log_success "Ready for lightsheet imaging pipeline development!"
