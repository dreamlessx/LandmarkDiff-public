# Installation

## System Requirements

- **Python:** 3.10 or later (3.11 recommended)
- **OS:** Linux (primary), macOS (MPS backend for Apple Silicon), Windows (WSL2 recommended)
- **GPU:** NVIDIA GPU with 6+ GB VRAM for diffusion inference, 40+ GB for training
- **CPU-only:** TPS mode works without any GPU

## Quick Install

### From PyPI

```bash
pip install landmarkdiff
```

### From Source (recommended for development)

```bash
git clone https://github.com/dreamlessx/LandmarkDiff-public.git
cd LandmarkDiff-public
pip install -e .
```

## Install Options

LandmarkDiff uses optional dependency groups so you only install what you need.

### Core (inference only)

```bash
pip install -e .
```

Installs the base package with MediaPipe, PyTorch, diffusers, and transformers. Sufficient for running predictions in all four inference modes (TPS, img2img, ControlNet, ControlNet + IP-Adapter).

### Development

```bash
pip install -e ".[dev]"
pre-commit install
```

Includes testing (pytest), linting (ruff), type checking (mypy), and pre-commit hooks.

### Training

```bash
pip install -e ".[train]"
```

Adds training dependencies: wandb for experiment tracking, deepspeed for distributed training, and webdataset for large-scale data loading.

### Evaluation

```bash
pip install -e ".[eval]"
```

Adds evaluation metric libraries: torch-fidelity (FID), lpips, scikit-image (SSIM), and insightface (ArcFace identity similarity).

### Gradio Demo

```bash
pip install -e ".[app]"
```

Adds Gradio for the interactive web demo.

### GPU Acceleration

```bash
pip install -e ".[gpu]"
```

Adds xformers and triton for faster attention computation on NVIDIA GPUs.

### Everything

```bash
pip install -e ".[train,eval,app,dev,gpu]"
```

## PyTorch with CUDA

LandmarkDiff requires PyTorch with CUDA support for diffusion-based inference modes. If you have not installed PyTorch with CUDA yet:

```bash
# Check your CUDA version
nvidia-smi

# Install PyTorch matching your CUDA version
# CUDA 12.1 (most common on recent systems)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2/M3), PyTorch MPS backend is used automatically:

```bash
pip install torch torchvision
```

## Docker

### CPU-only Docker

For demos that only need TPS (geometric warping) mode:

```bash
docker build -f Dockerfile.cpu -t landmarkdiff:cpu .
docker run -p 7860:7860 landmarkdiff:cpu
```

### GPU Docker

For ControlNet and diffusion-based inference (requires NVIDIA GPU):

```bash
docker build -f Dockerfile.gpu -t landmarkdiff:gpu .
docker run --gpus all -p 7860:7860 landmarkdiff:gpu
```

GPU passthrough requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). See [Docker GPU Setup](docker-gpu.md) for detailed prerequisites, VRAM requirements by GPU tier, verification steps, and troubleshooting.

### Docker Compose

```bash
docker compose up app       # CPU demo on :7860
docker compose up gpu       # GPU demo on :7861
docker compose --profile training run train  # training (GPU)
```

## Apptainer / Singularity (HPC)

For HPC environments that do not allow Docker:

```bash
apptainer build landmarkdiff.sif containers/landmarkdiff.def
apptainer exec --nv landmarkdiff.sif python scripts/app.py
```

See [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md) for detailed HPC setup, multi-node training, and SLURM job scripts.

## Verify Installation

Run this after installing to confirm everything is working:

```bash
python -c "
import landmarkdiff
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
print('LandmarkDiff installed successfully')
print(f'Version: {landmarkdiff.__version__}')
"
```

For a more thorough check that includes PyTorch device detection:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('MPS backend available (Apple Silicon)')
else:
    print('CPU only (TPS mode will work, diffusion modes will be slow)')

from landmarkdiff.inference import LandmarkDiffPipeline, get_device
print(f'LandmarkDiff device: {get_device()}')
"
```

## Troubleshooting

### MediaPipe fails on headless server

MediaPipe requires OpenGL libraries. On headless Linux servers:

```bash
# Debian / Ubuntu
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# RHEL / CentOS / Rocky
sudo dnf install mesa-libGL glib2
```

### CUDA out of memory during inference

The full pipeline (SD 1.5 + ControlNet + post-processing) needs about 5.2 GB VRAM. If you run out of memory:

- Use `--mode tps` for CPU-only inference (no diffusion model, instant results)
- Reduce `num_inference_steps` (e.g., 20 instead of 30)
- Use CPU offloading: initialize with `device="cpu"` (slower but no VRAM limit)

### PyTorch CUDA version mismatch

If you see errors about CUDA version incompatibility:

```bash
# Check system CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### MediaPipe version compatibility

LandmarkDiff supports both the new Tasks API (MediaPipe >= 0.10.20) and the legacy Solutions API. If you encounter issues with one API, the code automatically falls back to the other. To force a specific MediaPipe version:

```bash
pip install mediapipe==0.10.14    # legacy Solutions API
pip install mediapipe>=0.10.20    # new Tasks API (recommended)
```

### ImportError for optional dependencies

Some features require optional packages:

```bash
# For LPIPS metric
pip install lpips

# For FID metric
pip install torch-fidelity

# For ArcFace identity similarity
pip install insightface onnxruntime

# For face restoration (CodeFormer/GFPGAN)
pip install codeformer-perceptor gfpgan

# For Real-ESRGAN background enhancement
pip install realesrgan
```

### Pre-commit hooks fail

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files    # fix all existing issues
```

## Next Steps

- [Getting Started](getting-started.md) for a quick example
- [Quickstart tutorial](tutorials/quickstart.md) for a guided walkthrough
- [API Reference](api/landmarks.md) for the full module documentation
- [FAQ](faq.md) for common questions
