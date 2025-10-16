# Lightsheet Microscopy Image Processing Pipeline

**Agentic image processing workstation for Zeiss Z.1 lightsheet microscopy data**

Optimized for: **AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)**

---

## Overview

This imaging pipeline extends your `llm-inference` environment with GPU-accelerated tools for processing large lightsheet microscopy datasets locally, eliminating network bottlenecks that plague university labs.

**Target use case**: Cancer cell imaging with Zeiss Z.1 lightsheet microscope
- Input: Raw CZI files (50–1000+ GB)
- Processing: Stitching, deconvolution, segmentation
- Output: Publication-ready OME-Zarr or OME-TIFF + quantitative analysis

**Key value**: Run on-premise, on-demand, without uploading terabytes of data.

---

## Quick Start

### 1. Prerequisites

Ensure `setup_llm_core.sh` has completed successfully:

```bash
mamba activate llm-inference
python -c "import torch; print(torch.version.cuda)"  # Should print CUDA 12.8
```

### 2. Install Imaging Pipeline

```bash
bash setup_imaging_pipeline.sh
```

This will:
- Install CZI I/O, registration, stitching, segmentation, visualization libraries
- Set up `/scratch/imaging` workspace
- Create `~/imaging-workspace/` with configs and scripts
- Verify all dependencies

**Time: ~20–30 minutes**

### 3. Verify Installation

```bash
~/start-imaging-pipeline.sh
```

Should output:
```
✓ CZI I/O ready
✓ GPU segmentation: True
```

---

## Architecture

### Hardware Utilization

| Component | Allocation | Notes |
|-----------|-----------|-------|
| GPU VRAM | 90GB (LLM) + 60GB (imaging) | 96GB total; shared, not concurrent |
| CPU cores | 32 (OS) + 16 (dask workers) | Threadripper 7970X: 48 cores |
| Storage I/O | NVMe prioritized | Samsung 990 Pro + Crucial P2 = 3TB hot |
| Memory (system) | 128GB DDR5 | Sufficient for large dataset caching |

### Software Stack

```
User Interface (LLM Agent)
    ↓
vLLM (local inference)
    ↓
Tool Registry (imaging pipeline functions)
    ↓
┌──────────────────────────────────────┐
│   IMAGING PIPELINE (GPU/CPU hybrid)   │
├──────────────────────────────────────┤
│ Phase 1: CZI I/O      → aicspylibczi │
│ Phase 2: Preprocessing → scikit-image │
│ Phase 3: Stitching    → SimpleITK    │
│ Phase 4: Deconvolution → scipy       │
│ Phase 5: Visualization → napari      │
│ Phase 6: Segmentation  → cellpose    │
│ Phase 7: Analysis     → pandas       │
└──────────────────────────────────────┘
    ↓
Output (OME-Zarr, OME-TIFF, CSV, email)
```

---

## Installation Details

### What Gets Installed

**Core Libraries**:
- `aicspylibczi` (3.3.0+): Fast C++-backed CZI reader
- `SimpleITK`: Robust multi-modal registration
-
