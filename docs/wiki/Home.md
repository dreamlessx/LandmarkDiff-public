# LandmarkDiff

**Anatomically-conditioned latent diffusion for photorealistic facial surgery outcome prediction from standard clinical photography.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Live_Demo-orange)](https://huggingface.co/spaces/dreamlessx/LandmarkDiff)

LandmarkDiff predicts post-surgical facial appearance by combining MediaPipe 478-point face mesh extraction, Gaussian RBF landmark deformation, and ControlNet-conditioned Stable Diffusion 1.5 generation. It supports six surgical procedures and four inference modes ranging from instant CPU-only TPS warps to full GPU-accelerated diffusion.

---

## Quick Links

### Getting Started

| Page | Description |
|------|-------------|
| [Installation](Installation) | pip, conda, Docker, Apptainer/Singularity, GPU setup |
| [Quick Start](Quick-Start) | Tutorial with code examples for all 4 inference modes |

### Core Documentation

| Page | Description |
|------|-------------|
| [Architecture](Architecture) | Full pipeline diagram, from face mesh extraction to post-processing |
| [Procedures](Procedures) | All 6 supported surgical procedures with landmark indices and displacement patterns |
| [API Reference](API-Reference) | Key classes: `LandmarkDiffPipeline`, `FaceLandmarks`, `DeformationHandle`, etc. |
| [Configuration](Configuration) | YAML experiment config schema, CLI flags, environment variables |
| [Clinical Flags](Clinical-Flags) | Handling pathological conditions: vitiligo, Bell's palsy, keloid, Ehlers-Danlos |

### Training & Evaluation

| Page | Description |
|------|-------------|
| [Training](Training) | Model training, data preparation, DisplacementModel fitting |
| [Benchmarks](Benchmarks) | Inference speed, VRAM usage, training throughput, quality metrics |

### Usage & Deployment

| Page | Description |
|------|-------------|
| [Deployment](Deployment) | Docker, Gradio, REST API deployment options |
| [FAQ](FAQ) | VRAM requirements, inference modes, custom procedures, intensity scaling |
| [Troubleshooting](Troubleshooting) | Common errors and how to fix them |

### Development

| Page | Description |
|------|-------------|
| [Contributing](Contributing) | Development setup, PR process, testing requirements |

---

## Installation

```bash
# Core (inference only)
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With Gradio demo
pip install -e ".[app]"

# Full development
pip install -e ".[dev]"
```

See the [Installation](Installation) page for Docker, conda, HPC, and GPU setup instructions.

## Quickstart

```python
from landmarkdiff.inference import LandmarkDiffPipeline
import cv2

pipe = LandmarkDiffPipeline(mode="tps")
pipe.load()

image = cv2.imread("face.jpg")
result = pipe.generate(image, procedure="rhinoplasty", intensity=60)
cv2.imwrite("prediction.png", result["output"])
```

Or from the command line:

```bash
python -m landmarkdiff infer face.jpg --procedure rhinoplasty --intensity 60 --mode tps
```

See the [Quick Start](Quick-Start) guide for all 4 inference modes with detailed examples.

## Supported Procedures

| Procedure | Target Region | CPU Mode | GPU Mode | Community |
|-----------|--------------|----------|----------|-----------|
| rhinoplasty | Nose (tip, bridge, alar base) | Yes | Yes | |
| blepharoplasty | Eyelids (upper/lower) | Yes | Yes | |
| rhytidectomy | Face lift (jowl, midface, neck) | Yes | Yes | |
| orthognathic | Jaw (mandibular repositioning) | Yes | Yes | |
| brow_lift | Brow (lateral/medial elevation) | Yes | Yes | [Deepak8858](https://github.com/dreamlessx/LandmarkDiff-public/pull/35) |
| mentoplasty | Chin (advancement/reduction) | Yes | Yes | [P-r-e-m-i-u-m](https://github.com/dreamlessx/LandmarkDiff-public/pull/36) |

See the [Procedures](Procedures) page for landmark indices, displacement vectors, and clinical notes.

## Inference Modes

| Mode | Device | Speed | Quality | VRAM |
|------|--------|-------|---------|------|
| tps | CPU | ~50ms | Geometric only | 0 GB |
| img2img | GPU | ~5s | Good | 4 GB |
| controlnet | GPU | ~8s | Best | 6 GB |
| controlnet_ip | GPU | ~10s | Best + identity | 8 GB |

See the [Benchmarks](Benchmarks) page for detailed performance data across hardware.

## Live Demo

Try LandmarkDiff without installation at [huggingface.co/spaces/dreamlessx/LandmarkDiff](https://huggingface.co/spaces/dreamlessx/LandmarkDiff) (TPS mode, runs on CPU).

## License

MIT License. See [LICENSE](https://github.com/dreamlessx/LandmarkDiff-public/blob/main/LICENSE).
