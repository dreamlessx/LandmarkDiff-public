# Getting Started

## System Requirements

- Python 3.10 or later (3.11 recommended)
- PyTorch 2.1+ with CUDA support (for diffusion modes)
- 6 GB+ GPU VRAM for inference, 40 GB+ for training
- CPU-only mode available via TPS inference (no GPU needed)

Supported platforms: Linux (primary), macOS (MPS backend), Windows (WSL2 recommended).

## Installation

### From source (recommended)

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .
```

### With GPU extras

```bash
pip install -e ".[gpu]"
```

This adds xformers and triton for faster attention on NVIDIA GPUs.

### Optional dependency groups

```bash
# Training (wandb, deepspeed, webdataset)
pip install -e ".[train]"

# Evaluation metrics (torch-fidelity, LPIPS, scikit-image)
pip install -e ".[eval]"

# Gradio web demo
pip install -e ".[app]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Everything at once
pip install -e ".[train,eval,app,dev,gpu]"
```

## Quick Example

Three lines to run a prediction:

```python
from landmarkdiff.inference import LandmarkDiffPipeline

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
result = pipeline.generate("photo.jpg", procedure="rhinoplasty", intensity=60)
```

The `result` dictionary contains the final composited image (`result["output"]`), intermediate outputs (TPS warp, conditioning wireframe, surgical mask), original and deformed landmarks, and an ArcFace identity similarity score.

For CPU-only geometric warping without any GPU:

```python
pipeline = LandmarkDiffPipeline(mode="tps", device="cpu")
result = pipeline.generate("photo.jpg", procedure="rhinoplasty", intensity=60)
```

## Interactive Notebook

The [quickstart notebook](https://github.com/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb) walks through the full pipeline step by step, with inline visualizations of each stage. You can also open it directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dreamlessx/LandmarkDiff-public/blob/main/notebooks/quickstart.ipynb)

## Verify Installation

```bash
python -c "
import landmarkdiff
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
print('LandmarkDiff installed successfully')
print(f'Version: {landmarkdiff.__version__}')
"
```

## Next Steps

- [Installation details](install.md) for Docker, Apptainer, and troubleshooting
- [Quick Start tutorial](tutorials/quickstart.md) for a guided walkthrough
- [API Reference](api/landmarks.md) for the full module documentation
- [FAQ](faq.md) for common questions and issues
