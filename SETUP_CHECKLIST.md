# Imaging Pipeline Setup Checklist

## Pre-Installation (5 minutes)

- [ ] Verify `llm-inference` environment works
  ```bash
  mamba activate llm-inference
  python -c "import torch; print(torch.version.cuda)"  # Should print 12.8
  ```

- [ ] Check GPU is available
  ```bash
  nvidia-smi  # Should show RTX Pro 6000 96GB
  ```

- [ ] Check disk space in /scratch
  ```bash
  df -h /scratch  # Need ~20GB free
  ```

- [ ] Have these files ready
  - [ ] `setup_imaging_pipeline.sh`
  - [ ] `environment_imaging.yml`
  - [ ] `requirements_imaging.txt` (optional, reference)

---

## Installation (30-35 minutes)

- [ ] Navigate to working directory with setup files
  ```bash
  ls -la setup_imaging_pipeline.sh environment_imaging.yml
  ```

- [ ] Run installer
  ```bash
  bash setup_imaging_pipeline.sh
  ```

- [ ] Answer prompts:
  - [ ] "Continue with installation?" → **y**
  - [ ] "Continue?" → **y**

- [ ] Wait for completion (~25-35 min)
  - [ ] Watch for phase completion messages
  - [ ] Look for green `[SUCCESS]` messages
  - [ ] No red `[ERROR]` messages

- [ ] Installer creates:
  - [ ] `imaging_pipeline` conda environment
  - [ ] `~/imaging-workspace/` directory
  - [ ] `~/start-imaging-pipeline.sh` helper
  - [ ] `~/switch-to-imaging.sh` helper
  - [ ] `/scratch/imaging/` directories
  - [ ] `~/imaging-pipeline-install.log`

---

## Post-Installation Verification (5 minutes)

### Step 1: Check Environment Created

```bash
mamba env list
```

- [ ] See `imaging_pipeline` in the list

### Step 2: Activate Environment

```bash
mamba activate imaging_pipeline
```

- [ ] Prompt changes to `(imaging_pipeline)`

### Step 3: Verify Key Packages

```bash
python -c "from aicspylibczi import CziFile; print('✓ CZI I/O')"
```

- [ ] Outputs: `✓ CZI I/O`

```bash
python -c "import SimpleITK; print('✓ Registration')"
```

- [ ] Outputs: `✓ Registration`

```bash
python -c "import cellpose; print('✓ Segmentation')"
```

- [ ] Outputs: `✓ Segmentation`

```bash
python -c "import napari; print('✓ Visualization')"
```

- [ ] Outputs: `✓ Visualization`

```bash
python -c "import zarr; print('✓ Storage')"
```

- [ ] Outputs: `✓ Storage`

### Step 4: Check GPU Support

```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

- [ ] Outputs: `GPU: True`

```bash
python -c "from cellpose.core import use_gpu; print(f'Cellpose GPU: {use_gpu()}')"
```

- [ ] Outputs: `Cellpose GPU: True`

```bash
nvidia-smi
```

- [ ] Shows RTX Pro 6000 with 96GB VRAM

### Step 5: Verify Workspace

```bash
ls -la ~/imaging-workspace/
```

- [ ] Directories exist:
  - [ ] `configs/`
  - [ ] `scripts/`
  - [ ] `notebooks/`
  - [ ] `example_data/`

```bash
ls -la /scratch/imaging/
```

- [ ] Directories exist:
  - [ ] `raw/`
  - [ ] `stitched/`
  - [ ] `deconvolved/`
  - [ ] `segmented/`
  - [ ] `analysis/`
  - [ ] `cache/`

### Step 6: Check Configuration

```bash
cat ~/imaging-workspace/configs/pipeline_config.yaml | head -20
```

- [ ] Config file exists and is readable

---

## Test with Real Data (Optional, 30 minutes)

### Download Test Dataset

```bash
cd /tmp

# Download one small CZI file to test
wget https://datadryad.org/stash/downloads/file_stream/3698629 \
  -O LSFM_0000.czi.bz2

# Decompress
bunzip2 LSFM_0000.czi.bz2
```

- [ ] File downloaded: `ls -lh /tmp/LSFM_0000.czi`

### Create Test Script

```bash
cat > ~/imaging-workspace/scripts/test_czi_read.py << 'EOF'
from aicspylibczi import CziFile
from pathlib import Path

czi_path = Path("/tmp/LSFM_0000.czi")
if not czi_path.exists():
    print("❌ File not found. Download from Dryad first.")
    exit(1)

print(f"Opening: {czi_path}")
czi = CziFile(czi_path)
print(f"✓ Dimensions: {czi.dims}")
print(f"✓ Shape: {czi.size}")

try:
    img, _ = czi.read_image(T=0, Z=20)
    print(f"✓ Read plane: {img.shape}")
    print("✅ CZI reading works!")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

- [ ] Script created

### Run Test

```bash
mamba activate imaging_pipeline
python ~/imaging-workspace/scripts/test_czi_read.py
```

- [ ] Outputs:
  ```
  Opening: /tmp/LSFM_0000.czi
  ✓ Dimensions: BSCZYX
  ✓ Shape: ...
  ✓ Read plane: (1, 1, 2, 1, 1024, 1024)
  ✅ CZI reading works!
  ```

---

## Troubleshooting Checklist

### If Environment Creation Failed

```bash
# Remove failed environment
mamba env remove -n imaging_pipeline -y

# Try again
bash setup_imaging_pipeline.sh
```

- [ ] Second attempt successful

### If Import Fails

```bash
# Check what's installed
mamba list -n imaging_pipeline | grep cellpose

# Reinstall specific package
pip install --force-reinstall cellpose==3.0.8
```

- [ ] Import works after reinstall

### If GPU Not Detected

```bash
# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall torch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- [ ] GPU detected after reinstall

### If Out of Disk Space

```bash
# Check /scratch
df -h /scratch

# Clean up if needed
rm -rf /scratch/imaging/cache/*
```

- [ ] Freed up space (need 20GB minimum)

---

## Environment Sanity Checks

### List Environments

```bash
mamba env list
```

Expected output:
```
base                     /home/user/miniforge3
llm-inference            /home/user/miniforge3/envs/llm-inference
imaging_pipeline         /home/user/miniforge3/envs/imaging_pipeline  *
```

- [ ] Both environments visible
- [ ] `imaging_pipeline` has asterisk (currently active)

### Switch Environments

```bash
mamba activate llm-inference
```

- [ ] Prompt changes to `(llm-inference)`
- [ ] Previous environment still exists

```bash
mamba activate imaging_pipeline
```

- [ ] Prompt changes back to `(imaging_pipeline)`

### Check Python Versions

```bash
mamba activate llm-inference
python --version  # Should be 3.11
```

- [ ] Shows Python 3.11

```bash
mamba activate imaging_pipeline
python --version  # Should be 3.11
```

- [ ] Shows Python 3.11 (same version, different packages)

---

## Helper Scripts Check

```bash
ls -lh ~/start-imaging-pipeline.sh ~/switch-to-imaging.sh
```

- [ ] Both files exist and are executable

### Test Main Helper

```bash
~/start-imaging-pipeline.sh
```

- [ ] Shows startup info
- [ ] Displays GPU status
- [ ] Activates environment
- [ ] Drops into bash shell

Press `exit` to continue:

```bash
exit
```

- [ ] Returns to previous shell

### Test Quick Switch

```bash
~/switch-to-imaging.sh
```

- [ ] Quickly activates imaging_pipeline
- [ ] Shows confirmation message

```bash
exit
```

- [ ] Returns to previous shell

---

## Configuration Check

```bash
cat ~/imaging-workspace/configs/pipeline_config.yaml
```

- [ ] File is valid YAML
- [ ] Contains expected sections:
  - [ ] `imaging:`
  - [ ] `czi:`
  - [ ] `registration:`
  - [ ] `deconvolution:`
  - [ ] `segmentation:`
  - [ ] `hardware:`

---

## Final Verification (Before Using)

### Run Complete Test

```bash
mamba activate imaging_pipeline

python -c "
from aicspylibczi import CziFile
import SimpleITK
import zarr
import cellpose
import napari
import torch
import pandas as pd

print('✓ CZI I/O')
print('✓ Registration')
print('✓ Storage')
print('✓ Segmentation')
print('✓ Visualization')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ GPU available: {torch.cuda.is_available()}')
print('✓ Data analysis')
print()
print('🎉 ALL SYSTEMS GO!')
"
```

- [ ] Outputs `🎉 ALL SYSTEMS GO!`

### Check Log File

```bash
tail -20 ~/imaging-pipeline-install.log
```

- [ ] Shows successful completion
- [ ] No error messages

---

## Deployment Checklist (Ready to Use)

- [ ] ✅ `imaging_pipeline` environment created
- [ ] ✅ All packages installed and verified
- [ ] ✅ Workspace directories created
- [ ] ✅ Configuration template ready
- [ ] ✅ GPU support confirmed
- [ ] ✅ Helper scripts working
- [ ] ✅ Both environments coexist peacefully

---

## Documentation Review

- [ ] Read `IMAGING_PIPELINE_SETUP.md` (full guide)
- [ ] Read `QUICK_REFERENCE.md` (cheat sheet)
- [ ] Saved `DEPLOYMENT_SUMMARY.md` for reference
- [ ] Bookmarked test data source: https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf823

---

## Success Indicators

✅ **You're done when:**

1. `imaging_pipeline` environment exists
2. All key packages import successfully
3. GPU is detected by PyTorch and Cellpose
4. Workspace directories created at `~/imaging-workspace` and `/scratch/imaging`
5. Helper scripts are executable
6. Configuration file is readable

**Expected time**: 30 min (install) + 5 min (verify) = **35 minutes total**

---

## Next: MVP Stitcher

After this checklist completes successfully:

1. Download test CZI from Dryad (optional)
2. Create `stitch_zeiss_z1.py` script (coming next)
3. Test stitching on single timepoint
4. Visualize output in Napari
5. Scale to full pipeline
6. Integrate with LLM agent

---

## Sign-Off

- [ ] **Date completed**: _______________
- [ ] **By**: _______________
- [ ] **System**: Threadripper 7970X + RTX Pro 6000
- [ ] **Status**: ✅ Ready for imaging pipeline development

---

## Quick Sanity Check (Anytime)

```bash
# 3-line verification
mamba activate imaging_pipeline && \
python -c "from aicspylibczi import CziFile; from cellpose.core import use_gpu; import torch; print(f'✓ Ready (GPU: {torch.cuda.is_available()}, Cellpose: {use_gpu()})')" && \
echo "✅ All good!"
```

Should output:
```
✓ Ready (GPU: True, Cellpose: True)
✅ All good!
```

---

**Checklist Version**: 1.0  
**Last Updated**: October 2025  
**For**: Zeiss Z.1 Lightsheet Imaging Pipeline  
**System**: Threadripper 7970X + RTX Pro 6000 Blackwell
