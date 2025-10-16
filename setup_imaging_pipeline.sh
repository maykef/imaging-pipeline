#!/bin/bash

################################################################################
# Lightsheet Microscopy Image Processing Pipeline Setup Script
#
# Installs imaging-specific libraries into the existing llm-inference env
# Optimized for: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
#
# CRITICAL: Run this AFTER setup_llm_core.sh has completed
# This script extends the llm-inference environment with imaging tools
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
log_info "Lightsheet Imaging Pipeline Setup (Blackwell-optimized)"
echo ""

# Check mamba/conda availability
if ! command -v mamba &>/dev/null; then
  log_error "Mamba not found. Ensure setup_llm_core.sh has completed first."
  exit 1
fi

# Check llm-inference env exists
if ! mamba env list | grep -qE '^\s*llm-inference\s'; then
  log_error "llm-inference environment not found."
  log_info "Run setup_llm_core.sh first to create the environment."
  exit 1
fi

log_success "llm-inference environment found."

# Ensure CUDA is available
if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Ensure NVIDIA drivers are installed."
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
log_info "GPU detected: $GPU_NAME"

echo ""
log_warning "This script will extend the llm-inference environment with:"
echo "  - CZI file I/O (aicspylibczi, czifile, tifffile)"
echo "  - Image registration (SimpleITK, pycpd, scikit-image)"
echo "  - Stitching & zarr (zarr, ome-zarr-py, dask)"
echo "  - Deconvolution (scipy.ndimage via scikit-image)"
echo "  - Segmentation (cellpose with GPU support)"
echo "  - Visualization (napari)"
echo "  - Analysis (pandas, scikit-learn, scipy)"
echo ""
log_warning "⚠️  COMPATIBILITY NOTES:"
echo "  - Cellpose 3.0.8+ is stable; 3.1.0+ may have torch tensor issues"
echo "  - All packages compatible with PyTorch 2.10 nightly (cu128)"
echo "  - GPU memory will be shared between LLM inference and image processing"
echo "  - Dask parallelization uses CPU cores; GPU for image ops where possible"
echo ""
echo "Total time: ~20–30 min (network dependent)"
echo "Additional disk space: ~5–10 GB"
echo ""
read -p "Continue with installation? (y/N): " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] || { log_info "Installation cancelled."; exit 0; }

################################################################################
# ACTIVATE ENVIRONMENT
################################################################################
log_info "Activating llm-inference environment..."
eval "$(mamba shell hook --shell bash)"
mamba activate llm-inference

if [ "$CONDA_DEFAULT_ENV" != "llm-inference" ]; then
  log_error "Failed to activate llm-inference environment."
  exit 1
fi

log_success "Environment activated: $CONDA_DEFAULT_ENV"
echo ""

################################################################################
# PHASE 1: CZI FILE I/O & CORE I/O
################################################################################
log_info "PHASE 1: Installing CZI file I/O libraries..."

pip install --upgrade "aicspylibczi>=3.3.0" \
                      "tifffile>=2024.1.0" \
                      "imagecodecs>=2024.1.0"

# Optional fallback: czifile (slower pure-Python impl, but good for edge cases)
pip install --upgrade "czifile>=2019.7.2" || log_warning "czifile install skipped (optional)"

log_success "CZI I/O installed."
python -c "from aicspylibczi import CziFile; print('✓ aicspylibczi OK')" || log_error "aicspylibczi import failed"
echo ""

################################################################################
# PHASE 2: IMAGE PROCESSING CORE
################################################################################
log_info "PHASE 2: Installing image processing libraries..."

# Core scientific stack (likely already present, but ensure compatible versions)
pip install --upgrade "numpy>=1.24.0" \
                      "scipy>=1.10.0" \
                      "scikit-image>=0.22.0" \
                      "scikit-learn>=1.3.0"

log_success "Image processing core installed."
python -c "import skimage; print(f'✓ scikit-image {skimage.__version__} OK')" || log_error "scikit-image import failed"
echo ""

################################################################################
# PHASE 3: REGISTRATION & STITCHING
################################################################################
log_info "PHASE 3: Installing registration & stitching libraries..."

# SimpleITK (robust registration, good CPU performance)
pip install --upgrade "SimpleITK>=2.2.0"

# Point Cloud Deformable Registration (Coherent Point Drift)
pip install --upgrade "pycpd>=0.2.1"

# Zarr for chunked, compressed I/O (critical for large datasets)
pip install --upgrade "zarr>=2.16.0" \
                      "ome-zarr>=0.8.0" \
                      "ome-zarr-py>=0.8.0"

# Dask for parallelized processing
pip install --upgrade "dask[array]>=2023.12.0" \
                      "dask-jobqueue>=0.9.0" || log_warning "dask-jobqueue skipped (optional)"

log_success "Registration & stitching libraries installed."
python -c "import SimpleITK; print(f'✓ SimpleITK {SimpleITK.__version__} OK')" || log_error "SimpleITK import failed"
python -c "import zarr; print(f'✓ zarr {zarr.__version__} OK')" || log_error "zarr import failed"
echo ""

################################################################################
# PHASE 4: CELLPOSE (GPU-ACCELERATED SEGMENTATION)
################################################################################
log_info "PHASE 4: Installing Cellpose (GPU segmentation)..."

# Cellpose 3.0.8 is stable and compatible with PyTorch 2.10 nightly
# 3.1.0+ may have torch tensor indexing issues—we pin to stable
pip install --upgrade "cellpose==3.0.8" --no-deps

# Ensure torch dependencies are satisfied (should already be present from llm-inference)
pip install --upgrade "torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0"

# Optional: Cellpose-SAM (more advanced, slower)
log_warning "cellpose-sam is optional; skipping for now. Install manually if needed."

log_success "Cellpose 3.0.8 installed."
python << 'EOF'
import cellpose
from cellpose.core import use_gpu
print(f'✓ Cellpose {cellpose.__version__} OK')
if use_gpu():
    print('✓ GPU support detected in Cellpose')
else:
    print('⚠ GPU not detected; Cellpose will run on CPU (slower)')
EOF
echo ""

################################################################################
# PHASE 5: VISUALIZATION (NAPARI)
################################################################################
log_info "PHASE 5: Installing Napari (interactive visualization)..."

# Napari is large (~200MB); deps include PyQt5, OpenGL
pip install --upgrade "napari>=0.4.18" \
                      "napari-aicsimageio>=0.8.1" \
                      "aicsimageio>=0.9.0"

log_success "Napari installed."
python -c "import napari; print(f'✓ Napari {napari.__version__} OK')" || log_error "Napari import failed"
echo ""

################################################################################
# PHASE 6: DATA ANALYSIS & REPORTING
################################################################################
log_info "PHASE 6: Installing data analysis libraries..."

pip install --upgrade "pandas>=2.0.0" \
                      "openpyxl>=3.1.0" \
                      "matplotlib>=3.7.0" \
                      "seaborn>=0.13.0" \
                      "numpy-stl>=3.1.0"

log_success "Data analysis libraries installed."
python -c "import pandas; print(f'✓ Pandas {pandas.__version__} OK')" || log_error "Pandas import failed"
echo ""

################################################################################
# PHASE 7: OPTIONAL ADVANCED TOOLS
################################################################################
log_info "PHASE 7: Installing optional advanced tools..."

# Stardist: 3D object detection (excellent for nuclei, but heavyweight)
pip install --upgrade "stardist>=0.8.4" || log_warning "Stardist install skipped (optional, GPU-heavy)"

# Numba for JIT compilation (improves registration speed)
pip install --upgrade "numba>=0.57.0" || log_warning "Numba install skipped (optional)"

# OpenCV (sometimes useful for classical image processing)
pip install --upgrade "opencv-python>=4.8.0" || log_warning "OpenCV install skipped (optional)"

log_success "Optional tools installed (or skipped)."
echo ""

################################################################################
# PHASE 8: STORAGE & WORKSPACE SETUP
################################################################################
log_info "PHASE 8: Setting up storage directories and workspace..."

# Create imaging-specific subdirectories under /scratch
mkdir -p /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis}
sudo chown -R "$USER":"$USER" /scratch/imaging

# Create workspace for pipeline code
mkdir -p ~/imaging-workspace/{configs,scripts,notebooks,example_data}

# Create config template
cat > ~/imaging-workspace/configs/pipeline_config.yaml << 'CONFIG'
# Lightsheet Imaging Pipeline Configuration
# Adapt these values for your specific datasets and hardware

imaging:
  input_dir: /tmp/dataset1
  output_base: /scratch/imaging
  
  # CZI-specific parameters
  czi:
    z_step_microns: 1.0  # From microscope metadata
    tile_overlap_percent: 15
    background_subtraction: true
    flat_field_correction: false
    
preprocessing:
  # Deskewing (if oblique acquisition)
  deskew_enabled: false
  deskew_angle: 45
  
  # Background subtraction
  background_method: "mean"  # or "rolling_ball"
  background_window: 50
  
  # Denoising
  denoise_method: "tv"  # "tv" or "wavelet" or "none"
  denoise_strength: 0.1

registration:
  method: "cross_correlation"  # "feature", "cross_correlation", or "metadata"
  max_iterations: 100
  metric: "correlation"

deconvolution:
  enabled: true
  method: "richardson_lucy"
  iterations: 20
  psf_method: "metadata"  # or "empirical"

segmentation:
  cellpose_model: "nuclei"  # "nuclei", "cyto", "cyto2"
  cellpose_diameter: 30
  gpu_enabled: true

output:
  format: "ome_zarr"  # "ome_zarr", "ome_tiff", or "both"
  compression: "blosc"
  chunk_size: [64, 256, 256]
  
hardware:
  n_workers: 16  # Threadripper has 32 cores; use ~16 for balance
  gpu_batch_size: 4
  max_memory_gb: 60  # Conservative; leave 30GB for OS + other tasks
CONFIG

log_success "Workspace created at ~/imaging-workspace"
ls -la ~/imaging-workspace/configs/
echo ""

################################################################################
# PHASE 9: FINAL VERIFICATION
################################################################################
log_info "PHASE 9: Running final verification..."
echo ""

python << 'EOF'
import sys

checks = {
    'aicspylibczi': 'CZI file I/O',
    'tifffile': 'TIFF I/O',
    'SimpleITK': 'Registration',
    'skimage': 'Image processing',
    'zarr': 'Chunked storage',
    'cellpose': 'GPU segmentation',
    'napari': 'Visualization',
    'pandas': 'Data analysis',
    'dask': 'Parallelization',
}

print("Library Verification:")
print("=" * 50)

failed = []
for lib, desc in checks.items():
    try:
        __import__(lib)
        print(f"✓ {lib:20} ({desc})")
    except Exception as e:
        print(f"✗ {lib:20} ({desc}) - {str(e)[:30]}")
        failed.append(lib)

print("=" * 50)

if failed:
    print(f"\n⚠ {len(failed)} library/libraries failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All libraries verified!")

# GPU Check for Cellpose
print("\nGPU Status:")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"  Error checking GPU: {e}")

EOF

echo ""
log_success "Verification complete!"
echo ""

################################################################################
# PHASE 10: ENVIRONMENT PERSISTENCE & SUMMARY
################################################################################
log_info "PHASE 10: Persisting environment updates..."

# Add imaging pipeline env var to bashrc
if ! grep -q "IMAGING_WORKSPACE=" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# Imaging Pipeline Configuration
export IMAGING_WORKSPACE="$HOME/imaging-workspace"
export IMAGING_DATA="/scratch/imaging"
export CELLPOSE_CACHE="/scratch/cache/cellpose"
EOF
fi

export IMAGING_WORKSPACE="$HOME/imaging-workspace"
export IMAGING_DATA="/scratch/imaging"
export CELLPOSE_CACHE="/scratch/cache/cellpose"

log_success "Environment variables set."
echo ""

# Create a helper script for easy activation
cat > ~/start-imaging-pipeline.sh << 'HELPER'
#!/bin/bash
echo "🔬 Starting Imaging Pipeline Environment"
echo ""
eval "$(mamba shell hook --shell bash)"
mamba activate llm-inference
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Workspace: $IMAGING_WORKSPACE"
echo "Data: $IMAGING_DATA"
echo ""
echo "Quick test:"
python -c "from aicspylibczi import CziFile; from cellpose.core import use_gpu; print(f'✓ CZI I/O ready'); print(f'✓ GPU segmentation: {use_gpu()}')"
echo ""
echo "Usage:"
echo "  cd $IMAGING_WORKSPACE"
echo "  python scripts/stitch_lightsheet.py --input /tmp/dataset1/scan.czi --output /scratch/imaging/stitched"
echo ""
bash
HELPER

chmod +x ~/start-imaging-pipeline.sh
log_success "Created activation helper: ~/start-imaging-pipeline.sh"
echo ""

################################################################################
# FINAL SUMMARY
################################################################################
log_success "=========================================="
log_success "  IMAGING PIPELINE SETUP COMPLETED!"
log_success "=========================================="
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
echo "  1) Close and reopen terminal (load env vars)"
echo "  2) Run: ~/start-imaging-pipeline.sh"
echo "  3) Or: mamba activate llm-inference && cd ~/imaging-workspace"
echo ""

log_info "Next Steps:"
echo "  - Review ~/imaging-workspace/configs/pipeline_config.yaml"
echo "  - Build imaging pipeline scripts in ~/imaging-workspace/scripts"
echo "  - Integrate with vLLM agent for agentic orchestration"
echo "  - Test with synthetic/sample CZI data before lab dataset"
echo ""

log_warning "Memory & GPU Notes:"
echo "  - PyTorch (LLM): Can use up to 90GB VRAM"
echo "  - Imaging pipeline: Will share GPU with LLM inference"
echo "  - Cellpose: Optimal for batch processing (GPU-accelerated)"
echo "  - Dask: Uses $(nproc) CPU cores for tile parallelization"
echo "  - Zarr: Efficient chunked I/O minimizes memory pressure"
echo ""

log_warning "Troubleshooting:"
echo "  - If GPU memory issues: Reduce cellpose batch size or use CPU"
echo "  - If import errors: Run 'pip install --upgrade --force-reinstall <package>'"
echo "  - For Cellpose 3.1.0+ torch issues: Pin to 3.0.8 (done here)"
echo ""

log_success "Ready for lightsheet imaging pipeline development!"

# Log installation
cat > ~/imaging-pipeline-install.log << LOG
Imaging Pipeline Installation Log (Blackwell-optimized)
Date: $(date)
User: $USER
Hostname: $(hostname)

Installed Packages:
- CZI I/O: aicspylibczi 3.3.0+, tifffile, czifile
- Registration: SimpleITK 2.2.0+, pycpd 0.2.1+
- Storage: zarr 2.16.0+, ome-zarr-py 0.8.0+
- Processing: scikit-image 0.22.0+, scipy 1.10.0+
- Segmentation: cellpose 3.0.8 (GPU-enabled)
- Visualization: napari 0.4.18+
- Analysis: pandas 2.0.0+, scikit-learn 1.3.0+

Hardware:
- CPU: AMD Threadripper 7970X (48 cores)
- GPU: RTX Pro 6000 Blackwell (96GB VRAM)
- VRAM allocated: Up to 60GB for imaging (shared with LLM)
- Workers: 16 (conservative to avoid contention)

Workspace: ~/imaging-workspace
Data: /scratch/imaging

GPU Status: $(nvidia-smi --query-gpu=name --format=csv,noheader)
PyTorch CUDA: $(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Check manually")
LOG

log_success "Installation log: ~/imaging-pipeline-install.log"
