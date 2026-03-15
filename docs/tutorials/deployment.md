# Deployment Guide

This guide covers deploying LandmarkDiff for production or demo use, including Docker, HuggingFace Spaces, REST API setup, and production considerations.

## Local Gradio Demo

The simplest deployment is the built-in Gradio interface:

```bash
pip install -e ".[app]"
python scripts/app.py
# Open http://localhost:7860
```

This launches a five-tab demo (single procedure, multi-comparison, intensity sweep, face analysis, multi-angle capture) on port 7860. The first run downloads model weights (~6 GB), which are cached for subsequent launches.

To use a different port:

```bash
python scripts/app.py --port 8080
```

## Docker Deployment

### CPU-only deployment

For demos that only need TPS (geometric warping) mode:

```bash
# Build
docker build -t landmarkdiff:cpu -f Dockerfile.cpu .

# Run
docker run -p 7860:7860 landmarkdiff:cpu
```

The CPU Dockerfile uses `python:3.11-slim`, installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`, and runs the Gradio demo in TPS mode. The resulting image is smaller and does not require any GPU drivers.

### GPU deployment

For ControlNet and diffusion-based inference:

```bash
# Build the GPU image (runtime CUDA, smaller footprint)
docker build -t landmarkdiff:gpu -f Dockerfile.gpu .

# Run with GPU passthrough
docker run --gpus all -p 7860:7860 landmarkdiff:gpu
```

`Dockerfile.gpu` uses `nvidia/cuda:12.1.1-runtime-ubuntu22.04` with Python 3.11. It requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

For detailed GPU prerequisites, VRAM requirements by GPU tier, verification steps, and troubleshooting, see the [Docker GPU Setup](../docker-gpu.md) guide.

### Docker Compose

The `docker-compose.yml` defines five services:

```bash
# CPU demo (default)
docker compose up app

# GPU demo (runtime image, recommended)
docker compose up gpu

# GPU demo (devel image, for compiling extensions)
docker compose up app-gpu

# Build Sphinx docs
docker compose run docs

# Training (requires GPU)
docker compose --profile training run train
```

**Service details:**

| Service | Dockerfile | GPU | Port | Description |
|---------|-----------|-----|------|-------------|
| `app` | Dockerfile.cpu | No | 7860 | TPS-mode Gradio demo |
| `gpu` | Dockerfile.gpu | Yes (1 GPU) | 7861 | GPU inference (runtime image) |
| `app-gpu` | Dockerfile | Yes (1 GPU) | 7860 | GPU inference (devel image) |
| `docs` | python:3.11-slim | No | -- | Sphinx documentation builder |
| `train` | Dockerfile | Yes (1 GPU) | -- | ControlNet training |

**Volumes:**

All services mount these volumes:

- `./data:/app/data` -- training data, test pairs
- `./checkpoints:/app/checkpoints` -- model checkpoints
- `model-cache:/root/.cache` -- shared HuggingFace model cache

Create the host directories before running:

```bash
mkdir -p data checkpoints
```

### Custom Docker configuration

To modify the default inference mode, set the `LANDMARKDIFF_MODE` environment variable:

```bash
docker run -e LANDMARKDIFF_MODE=controlnet --gpus all -p 7860:7860 landmarkdiff
```

To pre-download model weights during build (so the first inference is fast), add to the Dockerfile:

```dockerfile
RUN python -c "from diffusers import ControlNetModel; ControlNetModel.from_pretrained('CrucibleAI/ControlNetMediaPipeFace')"
```

## HuggingFace Spaces

### Setup

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Gradio** as the SDK.
3. Choose hardware:
   - **CPU Basic** (free) -- works for TPS mode only
   - **T4 Small** -- minimum for ControlNet inference
   - **A10G Small** -- recommended for faster inference
4. Push the repository contents to the Space.

The `scripts/app.py` Gradio demo is compatible with HuggingFace Spaces out of the box. It auto-detects the environment and sets `share=False` (Spaces already provides a public URL).

### Space configuration

Create an `app.py` at the repository root that imports and launches the demo, or point the Space to `scripts/app.py` in your `README.md` metadata:

```yaml
---
title: LandmarkDiff
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: scripts/app.py
pinned: false
---
```

### Environment variables

Set secrets in the Space settings (Settings > Repository Secrets):

- `HF_TOKEN` -- if using gated models
- `LANDMARKDIFF_MODE` -- set to `tps` for CPU Spaces

### Persistence

HuggingFace Spaces provides `/data` as a persistent volume. Use it for caching model weights:

```bash
export HF_HOME=/data/huggingface_cache
```

## REST API Setup

### FastAPI server

For programmatic access without the Gradio UI, wrap the pipeline in a FastAPI server:

```python
"""LandmarkDiff REST API server."""

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from landmarkdiff.inference import LandmarkDiffPipeline

logger = logging.getLogger(__name__)

# Pipeline singleton -- loaded once at startup
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline on startup, clean up on shutdown."""
    global _pipeline
    logger.info("Loading LandmarkDiff pipeline...")
    _pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
    _pipeline.load()
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="LandmarkDiff API",
    version="0.2.0",
    lifespan=lifespan,
)

VALID_PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    procedure: str = Query("rhinoplasty", enum=VALID_PROCEDURES),
    intensity: int = Query(60, ge=0, le=100),
    seed: int = Query(42, ge=0),
):
    """Generate a surgical prediction.

    Returns the predicted post-operative image as PNG.
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not loaded")

    # Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    # Run prediction
    try:
        result = _pipeline.generate(
            img,
            procedure=procedure,
            intensity=intensity,
            seed=seed,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    # Encode output as PNG
    output_img = result["output"]
    _, buffer = cv2.imencode(".png", output_img)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=prediction.png"},
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
    }
```

Save this as `scripts/api_server.py` and run with uvicorn:

```bash
pip install fastapi uvicorn python-multipart

uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Run prediction
curl -X POST http://localhost:8000/predict \
    -F "image=@face.jpg" \
    -F "procedure=rhinoplasty" \
    -F "intensity=60" \
    -o prediction.png
```

### Flask alternative

If you prefer Flask:

```python
"""Minimal Flask API for LandmarkDiff."""

import cv2
import numpy as np
from flask import Flask, request, send_file
import io

from landmarkdiff.inference import LandmarkDiffPipeline

app = Flask(__name__)

pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
pipeline.load()


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return {"error": "No image provided"}, 400

    procedure = request.form.get("procedure", "rhinoplasty")
    intensity = int(request.form.get("intensity", 60))

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}, 400

    result = pipeline.generate(img, procedure=procedure, intensity=intensity)

    _, buffer = cv2.imencode(".png", result["output"])
    return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

## Production Considerations

### Model caching

On first inference, the pipeline downloads ~6 GB of model weights from HuggingFace. For production deployments:

1. **Pre-download models** during the Docker build or deployment setup:
   ```bash
   export HF_HOME=/persistent/cache
   python -c "
   from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
   ControlNetModel.from_pretrained('CrucibleAI/ControlNetMediaPipeFace')
   "
   ```

2. **Use a persistent volume** for the cache so container restarts do not re-download:
   ```bash
   docker run -v model-cache:/root/.cache --gpus all -p 7860:7860 landmarkdiff
   ```

3. **Set `HF_HOME`** to a stable location:
   ```bash
   export HF_HOME=/data/huggingface_cache
   ```

### Batch processing

For processing many images (e.g., generating training data or running evaluations):

1. **Load the pipeline once** and reuse it:
   ```python
   pipeline = LandmarkDiffPipeline(mode="controlnet", device="cuda")
   pipeline.load()

   for image_path in image_paths:
       result = pipeline.generate(load_image(image_path), ...)
   ```

2. **Use the batch inference script** for directory-level processing:
   ```bash
   python examples/batch_inference.py /path/to/images/ \
       --procedure rhinoplasty \
       --intensity 50 \
       --output output/batch/
   ```

3. **For large batches on HPC**, use SLURM array jobs:
   ```bash
   #!/bin/bash
   #SBATCH --array=0-9
   #SBATCH --gres=gpu:1

   TOTAL=1000
   PER_JOB=$((TOTAL / 10))
   START=$((SLURM_ARRAY_TASK_ID * PER_JOB))

   python scripts/batch_process.py \
       --start $START --count $PER_JOB \
       --input data/images/ --output output/
   ```

### Resource limits

**Memory:** The full inference pipeline uses ~5.2 GB VRAM and ~4 GB CPU RAM. For a web server, budget at least 8 GB RAM per worker process.

**Concurrency:** The pipeline is not thread-safe. Use process-based concurrency:
- **uvicorn:** Run with `--workers N` where N is the number of GPUs.
- **gunicorn:** Use `--workers N --worker-class uvicorn.workers.UvicornWorker`.
- Each worker loads its own pipeline and GPU. Do not share pipeline objects across workers.

**Timeouts:** ControlNet inference takes 3-15 seconds depending on hardware. Set request timeouts accordingly:
```bash
uvicorn scripts.api_server:app --timeout-keep-alive 30
```

**Disk:** Model weights take ~6 GB on disk. Temporary files (intermediate images) are cleaned up automatically, but ensure enough temp space for concurrent requests.

### Rate limiting

For public-facing deployments, add rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    ...
```

### Authentication

Never expose the API without authentication in production:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != expected_token:
        raise HTTPException(401, "Invalid token")

@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict(...):
    ...
```

### Reverse proxy

Put the API behind nginx for SSL termination and additional security:

```nginx
server {
    listen 443 ssl;
    server_name landmarkdiff.example.com;

    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }
}
```

### Security checklist

- Use HTTPS for all connections
- Authenticate all API endpoints
- Rate limit requests to prevent abuse
- Validate and sanitize all inputs (file type, file size)
- Do not store patient photos without explicit consent
- Follow HIPAA guidelines if handling medical data
- Set `client_max_body_size` to a reasonable limit (10 MB)
- Log requests for audit trails but do not log image content
- Run the container as a non-root user in production

### Monitoring

Add basic metrics to track API health:

```python
import time
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("predict_total", "Total prediction requests", ["procedure"])
REQUEST_LATENCY = Histogram("predict_seconds", "Prediction latency in seconds")

@app.post("/predict")
async def predict(...):
    start = time.time()
    # ... run prediction ...
    REQUEST_LATENCY.observe(time.time() - start)
    REQUEST_COUNT.labels(procedure=procedure).inc()
```

## Next Steps

- [Evaluation Guide](evaluation.md) -- Measure model quality
- [Training Guide](training.md) -- Train your own checkpoint
- [FAQ](../faq.md) -- Common questions and troubleshooting
