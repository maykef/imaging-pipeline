# Imaging Pipeline - Quick Reference Card

## TL;DR Setup

```bash
# 1. Copy files to working directory
cp setup_imaging_pipeline.sh environment_imaging.yml ~/

# 2. Run installer (30 min)
bash setup_imaging_pipeline.sh

# 3. Activate
mamba activate imaging_pipeline

# 4. Done!
```

---

## Two Environments

```
┌─ llm-inference ─────────────────┬─ imaging_pipeline ──────────────┐
│ mamba activate llm-inference    │ mamba activate imaging_pipeline │
│ • vLLM                          │ • CZI I/O                       │
│ • Transformers                  │ • Stitching                     │
│ • 90GB GPU                      │ • GPU segmentation              │
│ • LLM agent orchestration       │ • Napari visualization          │
└─────────────────────────────────┴─────────────────────────────────┘
```

---

## Quick Commands

### Activate Pipeline

```bash
# Method 1: Full startup with checks
~/start-imaging-pipeline.sh

# Method 2: Direct activation
mamba activate imaging_pipeline

# Method 3: Quick switch
~/switch-to-imaging.sh
```

### Check GPU

```bash
mamba activate imaging_pipeline
python -c "import torch; print(torch.cuda.is_available())"
python -c "from cellpose.core import use_gpu; print(use_gpu())"
nvidia-smi
```

### Test CZI Reading

```bash
mamba activate imaging_pipeline
python ~/imaging-workspace/scripts/test_czi_read.py
```

### List Installed Packages

```bash
mamba list -n imaging_pipeline
```

### Update Packages

```bash
mamba activate imaging_pipeline
pip install --upgrade cellpose  # or any package
```

---

## Directory Shortcuts

```bash
# Go to workspace
cd ~/imaging-workspace

# View config
cat ~/imaging-workspace/configs/pipeline_config.yaml

# Check logs
tail -f ~/imaging-workspace/logs/*.log

# Access data
ls /scratch/imaging/{raw,stitched,deconvolved,segmented,analysis}/
```

---

## File Locations

| What | Where |
|------|-------|
| Config | `~/imaging-workspace/configs/pipeline_config.yaml` |
| Scripts | `~/imaging-workspace/scripts/` |
| Notebooks | `~/imaging-workspace/notebooks/` |
| Input data | `/scratch/imaging/raw/` |
| Stitched output | `/scratch/imaging/stitched/` |
| Segmented output | `/scratch/imaging/segmented/` |
| Analysis | `/scratch/imaging/analysis/` |
| Logs | `~/imaging-workspace/logs/` |

---

## Resource Limits

```yaml
# Edit in pipeline_config.yaml
hardware:
  n_workers: 16           # Dask threads (16 of 48 cores)
  max_memory_gb: 60       # Max imaging RAM (leave 30GB for OS)
  gpu_batch_size: 4       # Cellpose batch
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not found | `pip install --force-reinstall torch` |
| Import error | `mamba env remove -n imaging_pipeline -y && bash setup_imaging_pipeline.sh` |
| Out of memory | Reduce batch_size in config |
| Slow processing | Check `nvidia-smi` for GPU utilization |
| Can't find CZI | Download from https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf823 |

---

## Install Files Needed

```
Working directory should contain:
✓ setup_imaging_pipeline.sh
✓ environment_imaging.yml
✓ requirements_imaging.txt  (optional, for reference)
```

Then run:
```bash
bash setup_imaging_pipeline.sh
```

---

## Test Dataset

**Download test CZI files:**

```bash
cd /tmp

# Download all 10 timepoints (bzip2 compressed)
for i in {0..9}; do
  wget https://datadryad.org/stash/downloads/file_stream/3698629 \
    -O LSFM_000${i}.czi.bz2
done

# Decompress
bunzip2 LSFM_*.czi.bz2

# Test
mamba activate imaging_pipeline
python ~/imaging-workspace/scripts/test_czi_read.py
```

---

## Next Steps Checklist

- [ ] Run `bash setup_imaging_pipeline.sh`
- [ ] Run `~/start-imaging-pipeline.sh` to verify
- [ ] Download test data from Dryad
- [ ] Run `python ~/imaging-workspace/scripts/test_czi_read.py`
- [ ] Edit `~/imaging-workspace/configs/pipeline_config.yaml`
- [ ] Create MVP stitcher script
- [ ] Test on single CZI timepoint
- [ ] Scale to full dataset
- [ ] Integrate with LLM agent

---

## Environment Info

**Created:** imaging_pipeline  
**Python:** 3.11  
**Location:** `~/miniforge3/envs/imaging_pipeline`  
**Isolation:** Complete (separate from llm-inference)  
**GPU:** RTX Pro 6000 (96GB VRAM shared with llm-inference)  
**CPU:** Threadripper 7970X (48 cores, 16 reserved for Dask)  

---

## One-Liner Reminders

```bash
# Activate imaging pipeline
mamba activate imaging_pipeline

# Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# List environments
mamba env list

# Switch to LLM
mamba activate llm-inference

# Remove imaging_pipeline
mamba env remove -n imaging_pipeline

# Update from YAML
mamba env update -f environment_imaging.yml

# Check conda version
mamba --version

# See what's installed
mamba list -n imaging_pipeline | grep -E "(cellpose|zarr|napari|SimpleITK)"
```

---

## Performance Monitoring

```bash
# Watch GPU in real-time during processing
nvidia-smi -l 1

# Check memory usage
free -h

# Monitor Dask workers
# (will show in output when processing)

# Check storage
df -h /scratch/imaging
du -sh /scratch/imaging/*
```

---

## Key Packages Versions

```
cellpose==3.0.8          (pinned - 3.1+ has issues)
aicspylibczi>=3.3.0      (latest, fast C++ backend)
SimpleITK>=2.2.0
zarr>=2.16.0
napari>=0.4.18
torch>=2.0.0
```

---

## Documentation Links

- Zeiss CZI: https://zeiss.com/microscopy/us/products/software/zeiss-zen/czi-image-file-format.html
- aicspylibczi: https://allencellmodeling.github.io/aicspylibczi/
- Cellpose: https://cellpose.readthedocs.io/
- SimpleITK: https://simpleitk.readthedocs.io/
- Zarr: https://zarr-specs.readthedocs.io/
- Napari: https://napari.org/
- Dask: https://dask.readthedocs.io/

---

## Emergency Commands

```bash
# If things break, start fresh:
mamba env remove -n imaging_pipeline -y
bash setup_imaging_pipeline.sh

# If you need to check what's wrong:
cat ~/imaging-pipeline-install.log

# See exactly what's installed:
mamba list -n imaging_pipeline > /tmp/imaging_packages.txt

# Create a backup of your environment:
mamba env export -n imaging_pipeline > imaging_pipeline_backup.yml
```

---

## Notes

- Both `llm-inference` and `imaging_pipeline` can coexist
- Switch between them with `mamba activate <env-name>`
- They don't interfere with each other
- GPU memory is shared (sequential use recommended)
- 96GB RTX Pro 6000 is sufficient for both workloads

**Ready?** → `bash setup_imaging_pipeline.sh` 🚀
