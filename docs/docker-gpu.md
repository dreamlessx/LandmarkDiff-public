# Docker GPU Setup

This guide covers running LandmarkDiff with GPU acceleration inside Docker,
enabling `img2img`, `controlnet`, and `controlnet_ip` inference modes.

## Prerequisites

### NVIDIA driver

Your host machine needs a working NVIDIA driver. Verify with:

```bash
nvidia-smi
```

You should see your GPU model, driver version, and CUDA version. The driver
must support CUDA 12.1 or later (driver >= 530.xx).

### NVIDIA Container Toolkit

Docker does not pass GPUs to containers by default. Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to enable GPU passthrough.

**Ubuntu / Debian:**

```bash
# Add the NVIDIA container toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**RHEL / CentOS / Rocky / Fedora:**

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify the toolkit works:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If this prints your GPU info, the toolkit is installed correctly.

### Docker and Docker Compose

Docker Engine 19.03+ is required for `--gpus` support.
Docker Compose v2.x is required for the `deploy.resources.reservations.devices`
syntax used in the compose file.

```bash
docker --version    # 19.03+
docker compose version  # v2.x
```

## Dockerfile.gpu

The repository includes `Dockerfile.gpu`, a GPU-optimized container image
based on `nvidia/cuda:12.1.1-runtime-ubuntu22.04`. It uses the CUDA runtime
image (not the larger `devel` image) for a smaller footprint while still
supporting GPU-accelerated PyTorch inference.

Build the image:

```bash
docker build -t landmarkdiff:gpu -f Dockerfile.gpu .
```

The existing `Dockerfile` (no suffix) uses `nvidia/cuda:12.1.1-devel-ubuntu22.04`
and includes CUDA development headers. Use that one if you need to compile
custom CUDA extensions (e.g., xformers from source). For inference only,
`Dockerfile.gpu` is the better choice.

## Running with Docker

### Single container

```bash
# Basic GPU inference
docker run --gpus all -p 7860:7860 landmarkdiff:gpu

# Specify a single GPU
docker run --gpus '"device=0"' -p 7860:7860 landmarkdiff:gpu

# With persistent model cache (avoids re-downloading weights)
docker run --gpus all \
    -p 7860:7860 \
    -v model-cache:/root/.cache \
    -v ./models:/app/models \
    landmarkdiff:gpu

# Force a specific inference mode
docker run --gpus all \
    -p 7860:7860 \
    -e LANDMARKDIFF_MODE=controlnet \
    landmarkdiff:gpu
```

### Docker Compose

The `docker-compose.yml` includes a `gpu` service that uses `Dockerfile.gpu`:

```bash
# Start the GPU demo on port 7861
docker compose up gpu

# Or run in the background
docker compose up -d gpu
```

The `gpu` service exposes port 7861 by default so it does not conflict with
the CPU `app` service on port 7860. You can run both simultaneously:

```bash
docker compose up app gpu
```

There is also an `app-gpu` service that uses the larger `devel`-based
`Dockerfile` on port 7860, and a `train` service for GPU-accelerated training.
See the compose file comments for details.

## Verifying GPU access

After starting the container, verify that PyTorch can see the GPU:

```bash
# Shell into the running container
docker exec -it <container_id> bash

# Check NVIDIA driver visibility
nvidia-smi

# Check PyTorch CUDA access
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

You can also check from outside the container:

```bash
docker run --rm --gpus all landmarkdiff:gpu python -c \
    "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## GPU memory requirements

The table below shows approximate VRAM usage for each inference mode.
These numbers assume a single 512x512 image at default settings
(30 diffusion steps, batch size 1).

| Inference mode | VRAM usage | Minimum GPU |
|----------------|-----------|-------------|
| `tps` | ~0 (CPU only) | No GPU needed |
| `img2img` | ~4.0 GB | GTX 1070 (8 GB) |
| `controlnet` | ~5.2 GB | GTX 1070 (8 GB) |
| `controlnet_ip` | ~6.5 GB | RTX 2060 (8 GB) |

### Recommendations by GPU tier

**8 GB VRAM** (GTX 1070/1080, RTX 2060/2070, RTX 3060):
- All inference modes work.
- Close other GPU applications before running.
- Reduce `num_inference_steps` to 20 if you hit OOM errors.

**12 GB VRAM** (RTX 3060 12GB, RTX 4070):
- Comfortable for all inference modes.
- Can run the Gradio demo while other light GPU tasks are active.

**24+ GB VRAM** (RTX 3090, RTX 4090, A5000, A6000):
- No memory concerns for inference.
- Can handle multiple concurrent requests.
- Sufficient for training with small batch sizes.

### Reducing VRAM usage

If you run out of GPU memory:

```bash
# Use TPS mode (no GPU needed)
docker run -p 7860:7860 landmarkdiff:cpu

# Reduce diffusion steps (faster, slightly lower quality)
docker run --gpus all -p 7860:7860 \
    -e LANDMARKDIFF_NUM_STEPS=20 \
    landmarkdiff:gpu

# Use CPU offloading (slower but no VRAM limit)
docker run --gpus all -p 7860:7860 \
    -e LANDMARKDIFF_DEVICE=cpu \
    landmarkdiff:gpu
```

## Multi-GPU setups

To restrict the container to specific GPUs:

```bash
# Use only GPU 0
docker run --gpus '"device=0"' -p 7860:7860 landmarkdiff:gpu

# Use GPUs 0 and 1
docker run --gpus '"device=0,1"' -p 7860:7860 landmarkdiff:gpu
```

Or with environment variables:

```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 7860:7860 landmarkdiff:gpu
```

In Docker Compose, the `gpu` service is configured for a single GPU by default.
Edit `docker-compose.yml` to change `count: 1` to `count: all` if you want
all GPUs available to the container.

## Troubleshooting

### "could not select device driver" error

```
docker: Error response from daemon: could not select device driver
```

The NVIDIA Container Toolkit is not installed or not configured. Follow the
installation steps above, then restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### "no NVIDIA GPU device is present" inside container

Check that the host driver is working (`nvidia-smi` on the host) and that
you passed `--gpus all` or the compose `deploy.resources` section is present.

### CUDA version mismatch

If PyTorch reports a CUDA error, the driver on the host may be too old for
CUDA 12.1. Check the minimum driver version:

```bash
nvidia-smi  # look at "CUDA Version" in the top right
```

CUDA 12.1 requires driver >= 530.xx. If your driver is older, either update
the driver or use an older CUDA base image in the Dockerfile.

### OOM (out of memory) during inference

See the "Reducing VRAM usage" section above. The most common fix is switching
to `tps` mode or reducing `num_inference_steps`.

## Next steps

- [Deployment guide](tutorials/deployment.md) for REST API setup, HuggingFace Spaces, and production considerations
- [GPU training guide](GPU_TRAINING_GUIDE.md) for SLURM-based training on HPC clusters
- [Getting started](getting-started.md) for a quick overview
