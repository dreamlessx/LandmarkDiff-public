# Installation

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA support
- 6 GB+ GPU VRAM (inference) or 40 GB+ (training)

## Quick Install

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .
```

## Install Options

```bash
# Inference only (minimal)
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With Gradio demo
pip install -e ".[app]"

# With evaluation metrics
pip install -e ".[eval]"

# Everything
pip install -e ".[train,eval,app,dev]"

# Development (includes linting and testing)
pip install -e ".[dev]"
pre-commit install
```

## Docker

```bash
# Build
docker build -t landmarkdiff .

# Run Gradio demo
docker compose up landmarkdiff

# Run training
docker compose run train
```

## Apptainer/Singularity (HPC)

```bash
apptainer build landmarkdiff.sif containers/landmarkdiff.def
apptainer exec --nv landmarkdiff.sif python scripts/app.py
```

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

## Common Issues

### MediaPipe fails on headless server
```bash
# Install OpenGL dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### CUDA out of memory during inference
- Use `--mode tps` for CPU-only inference (no diffusion model needed)
- Reduce image resolution: `--resolution 256`
- Use CPU offloading: set `device="cpu"` in LandmarkDiffPipeline

### PyTorch CUDA version mismatch
```bash
# Check your CUDA version
nvidia-smi
# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
