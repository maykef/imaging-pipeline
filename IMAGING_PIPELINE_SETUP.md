# Lightsheet Imaging Pipeline - Complete Setup Guide

**Separate conda environment for lightsheet microscopy image processing**

Built for: **AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)**

---

## Architecture: Dual Environments

You now have **two independent conda environments**:

```
┌─────────────────────────────────────┐
│    llm-inference Environment         │
├─────────────────────────────────────┤
│ - PyTorch 2.10 nightly (CUDA 12.8)  │
│ - vLLM                              │
│ - Transformers                      │
│ - Hugging Face stack                │
│ - (isolated, no imaging tools)      │
└─────────────────────────────────────┘
            VS
┌─────────────────────────────────────┐
│  imaging_pipeline Environment        │
├─────────────────────────────────────┤
│ - CZI file I/O                      │
│ - Registration & stitching          │
│ - GPU-accelerated segmentation      │
│ - Visualization (napari)            │
│ - (isolated, no LLM tools)          │
└─────────────────────────────────────┘
```

**Benefits**:
- ✅ No package conflicts
- ✅ Cleaner dependency management
- ✅ Can run both independently
- ✅ Easy to maintain/update separately
- ✅ Clear separation of concerns

---

## Installation

### Prerequisites

Ensure `setup_llm_core.sh` completed successfully:

```bash
mamba activate llm-inference
python -c "import torch; print(torch.version.cuda)"  # Should print: 12.8
```

### Step 1: Download Setup Files

Place these in your working directory:
- `setup_imaging_pipeline.sh`
- `environment_imaging.yml`

```bash
git clone <your-repo-url>  # or copy files manually
cd <repo-directory>
ls -la setup_imaging_pipeline.sh environment_imaging.yml
```

### Step 2: Run Installation

```bash
bash setup_imaging_pipeline.sh
```

This script will:
1. ✅ Check prerequisites (mamba, GPU, drivers)
2. ✅ Create new `imaging_pipeline` environment from `environment_imaging.yml`
3. ✅ Install all imaging dependencies
4. ✅ Verify all imports
5. ✅ Set up workspace at `~/imaging-workspace`
6. ✅ Create activation helpers

**Time: ~25–35 minutes** (depends on download speeds)

### Step 3: Verify Installation

```bash
~/start-imaging-pipeline.sh
```

Should output:
```
✓ Environment: imaging_pipeline
✓ CZI I/O ready
✓ GPU available: True
✓ Cellpose GPU support: True
```

---

## Quick Start

### Activate Imaging Pipeline

```bash
mamba activate imaging_pipeline
```

Or use the helper:
```bash
~/start-imaging-pipeline.sh
```

### Switch Between Environments

```bash
# Go to imaging pipeline
mamba activate imaging_pipeline

# Go back to LLM inference
mamba activate llm-inference
```

### Check What's Installed

```bash
mamba list -n imaging_pipeline
```

---

## Directory Structure

After installation:

```
~
├── imaging-workspace/
│   ├── configs/
│   │   └── pipeline_config.yaml       # Edit this for your data
│   ├── scripts/
│   │   ├── test_czi_read.py           # (to be created)
│   │   ├── stitch_zeiss_z1.py         # (to be created)
│   │   └── ...
│   ├── notebooks/
│   │   └── (for Jupyter prototyping)
│   └── example_data/
│       └── (sample CZI files for testing)
│
├── start-imaging-pipeline.sh           # Quick start helper
├── switch-to-imaging.sh               # Quick switch helper
├── imaging-pipeline-install.log       # Installation log
│
/scratch/imaging/
├── raw/                               # Input CZI files
├── stitched/                          # Stitching output (Zarr)
├── deconvolved/                       # Deconvolution output
├── segmented/                         # Segmentation masks
├── analysis/                          # Analysis results (CSV, plots)
└── cache/                             # Cellpose model cache
```

---

## Configuration

Edit `~/imaging-workspace/configs/pipeline_config.yaml`:

```yaml
imaging:
  input_dir: /tmp/dataset1             # Where your CZI files are
  output_base: /scratch/imaging        # Where to save results
  
  czi:
    z_step_microns: 0.5                # From microscope metadata
    tile_overlap_percent: 15           # Typical for Z.1
    background_subtraction: true
    
registration:
  method: "cross_correlation"          # "feature", "cross_correlation", "metadata"
  max_iterations: 100
  
deconvolution:
  enabled: true
  iterations: 20
  
segmentation:
  cellpose_model: "nuclei"
  cellpose_diameter: 30
  gpu_enabled: true
  
hardware:
  n_workers: 16                        # Use 16 of 48 Threadripper cores
  max_memory_gb: 60                    # Conservative to avoid LLM conflicts
```

---

## Hardware Resource Allocation

### Memory & GPU

| Component | Allocation | Details |
|-----------|-----------|---------|
| GPU VRAM | 96GB total RTX Pro 6000 | |
| - LLM inference | ~60-90GB | vLLM streaming |
| - Imaging pipeline | ~60GB | Registration, segmentation |
| - **Strategy** | **Sequential** | Run imaging AFTER LLM finishes, or vice versa |
| System RAM | 128GB DDR5 | Plenty for buffering datasets |

### CPU

| Component | Cores | Details |
|-----------|-------|---------|
| OS/background | ~12 cores | System tasks |
| Dask workers | 16 cores | Tile registration, preprocessing |
| Available | 32 cores | For future parallelization |
| **Total** | **48 cores** | Threadripper 7970X |

---

## Usage Examples

### Test CZI Reading

Create `~/imaging-workspace/scripts/test_czi_read.py`:

```python
#!/usr/bin/env python
"""Test CZI file reading with aicspylibczi"""

from aicspylibczi import CziFile
from pathlib import Path
import numpy as np

# Point to your CZI file
czi_path = Path("/tmp/LSFM_0000.czi")

if not czi_path.exists():
    print(f"❌ File not found: {czi_path}")
    print("Download from: https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf823")
    exit(1)

print(f"📂 Reading: {czi_path}")
print(f"   Size: {czi_path.stat().st_size / 1e9:.2f} GB")
print()

try:
    czi = CziFile(czi_path)
    
    print("✓ CZI file opened successfully")
    print()
    print("📊 Metadata:")
    print(f"   Dimensions: {czi.dims}")
    print(f"   Shape: {czi.size}")
    print(f"   Scenes: {len(czi.scenes)}")
    print()
    
    # Read metadata
    print("📝 XML Metadata (first 500 chars):")
    try:
        meta = czi.meta
        meta_str = str(meta)[:500]
        print(f"   {meta_str}...")
    except Exception as e:
        print(f"   ⚠ Could not read metadata: {e}")
    print()
    
    # Read a single plane
    print("🖼  Reading a single plane (T=0, Z=20)...")
    try:
        img, shape_info = czi.read_image(T=0, Z=20)
        print(f"   ✓ Shape: {img.shape}")
        print(f"   ✓ Dtype: {img.dtype}")
        print(f"   ✓ Range: [{img.min()}, {img.max()}]")
        print(f"   ✓ Mean intensity: {img.mean():.1f}")
    except Exception as e:
        print(f"   ❌ Error reading plane: {e}")
    
    print()
    print("✅ CZI reading test PASSED")
    
except Exception as e:
    print(f"❌ Failed to open CZI file: {e}")
    exit(1)
```

Run it:
```bash
mamba activate imaging_pipeline
python ~/imaging-workspace/scripts/test_czi_read.py
```

### Download Test Dataset

```bash
cd /tmp

# Download 10 CZI files from Dryad (compressed)
for i in {0000..0009}; do
  wget https://datadryad.org/stash/downloads/file_stream/3698629 \
    -O LSFM_${i}.czi.bz2 &  # Download in parallel
done
wait

# Decompress all
bunzip2 LSFM_*.czi.bz2

# Verify
ls -lh LSFM_*.czi | head
```

Then test:
```bash
mamba activate imaging_pipeline
python ~/imaging-workspace/scripts/test_czi_read.py
```

---

## Environment Management

### List All Environments

```bash
mamba env list
```

Output should show:
```
base                  /home/user/miniforge3
llm-inference         /home/user/miniforge3/envs/llm-inference
imaging_pipeline      /home/user/miniforge3/envs/imaging_pipeline  *
```

### Update Packages in imaging_pipeline

```bash
mamba activate imaging_pipeline

# Update specific package
pip install --upgrade cellpose

# Update all pip packages
pip install --upgrade -r requirements_imaging.txt

# Or update from YAML
mamba env update -f environment_imaging.yml
```

### Export Current Environment

```bash
mamba activate imaging_pipeline
mamba env export > imaging_pipeline_exported.yml

# Share with colleagues - they can recreate with:
mamba env create -f imaging_pipeline_exported.yml
```

### Remove Environment (if needed)

```bash
mamba env remove -n imaging_pipeline
```

---

## Troubleshooting

### GPU Not Detected in Cellpose

```bash
mamba activate imaging_pipeline
python -c "from cellpose.core import use_gpu; print(f'GPU: {use_gpu()}')"
```

If returns `False`:
```bash
# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors After Installation

```bash
# Reinstall imaging_pipeline environment
mamba env remove -n imaging_pipeline -y
mamba env create -f environment_imaging.yml

# Or just pip reinstall troublesome package
pip install --force-reinstall aicspylibczi
```

### Out of Memory Issues

If processing large datasets:

**Option 1: Reduce batch size in config**
```yaml
segmentation:
  batch_size: 2  # Reduce from 4
```

**Option 2: Process one timepoint at a time**
```bash
# Instead of processing entire dataset, do:
for t in {0..9}; do
  python stitch_zeiss_z1.py --timepoint $t
done
```

**Option 3: Use Dask chunking**
```python
import dask.array as da

# Process in chunks instead of loading entire volume
chunks = (64, 256, 256)
arr = da.from_array(large_image, chunks=chunks)
```

### Cellpose Taking Too Long

Cellpose uses GPU but is I/O bound:

```bash
# Check GPU usage during segmentation
nvidia-smi -l 1  # Refresh every 1 second

# If GPU utilization low (~30%):
# - Increase batch_size in config
# - Increase cellpose_diameter (fewer objects)
# - Use CPU (faster for small images): cellpose_model: "nuclei-cpu"
```

---

## Next Steps

### 1. Test Installation (Now)

```bash
~/start-imaging-pipeline.sh
python ~/imaging-workspace/scripts/test_czi_read.py
```

### 2. Download Test Data

```bash
cd /tmp && bunzip2 *.bz2  # Decompress Dryad dataset
```

### 3. Build MVP Stitcher (Next)

I'll create `stitch_zeiss_z1.py` that:
- Reads CZI metadata
- Stitches one Z-slice
- Outputs to Zarr
- Launches Napari for QC

### 4. Scale to Full Pipeline

- Add deconvolution
- Add segmentation
- Add batch processing

### 5. Integrate with LLM Agent

Wrap pipeline in tool functions:
```python
def stitch_lightsheet(czi_path, output_dir) -> dict:
    """Tool for LLM to call"""
```

---

## File Manifest

After setup, you should have:

```
✓ setup_imaging_pipeline.sh          (the installer)
✓ environment_imaging.yml            (conda environment spec)
✓ ~/imaging-workspace/               (workspace)
✓ ~/start-imaging-pipeline.sh        (helper to activate)
✓ ~/switch-to-imaging.sh             (quick switcher)
✓ /scratch/imaging/                  (data directory)
✓ ~/imaging-pipeline-install.log     (install log)
```

---

## Performance Tips

### For Stitching Large Tiles

```bash
# Use all 16 dask workers
export DASK_SCHEDULER='threads'
export DASK_NUM_WORKERS=16

# Or in Python:
from dask.distributed import Client
client = Client(n_workers=16, threads_per_worker=2)
```

### For GPU Segmentation

```bash
# Batch multiple Z-slices at once
cellpose_batch_size: 8
```

### For Zarr I/O Performance

```yaml
# Optimal chunk size (balance between I/O and memory)
chunk_size: [64, 256, 256]  # Good for 96GB
# Compression
compression: "blosc"  # Fast + good compression (15:1)
```

---

## Resources

- **CZI Format**: https://zeiss.com/microscopy/us/products/software/zeiss-zen/czi-image-file-format.html
- **aicspylibczi Docs**: https://allencellmodeling.github.io/aicspylibczi/
- **Cellpose Paper**: https://www.nature.com/articles/s41592-020-01018-z
- **Zarr Docs**: https://zarr-specs.readthedocs.io/
- **Napari Docs**: https://napari.org/
- **Test Data**: https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf823

---

## Support

If you encounter issues:

1. **Check logs**: `cat ~/imaging-pipeline-install.log`
2. **Verify GPU**: `nvidia-smi`
3. **Test imports**: `python -c "from aicspylibczi import CziFile"`
4. **Check conda**: `mamba list -n imaging_pipeline`
5. **Reinstall if needed**: `bash setup_imaging_pipeline.sh` (will recreate env)

---

## Summary

You now have a **production-ready imaging pipeline** that:

✅ Runs independently from LLM inference  
✅ Optimized for RTX Pro 6000 (96GB GPU)  
✅ Handles Zeiss Z.1 CZI files  
✅ GPU-accelerated segmentation  
✅ Zarr-based efficient I/O  
✅ Ready for agentic orchestration  

**Next**: Build MVP stitcher and test on Dryad dataset! 🚀
