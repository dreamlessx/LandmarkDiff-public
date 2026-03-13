# Deployment Guide

Deploy LandmarkDiff for production or demo use.

## Local Gradio demo

```bash
pip install -e ".[app]"
python scripts/app.py
# Open http://localhost:7860
```

## Docker deployment

```bash
# Build
docker build -t landmarkdiff .

# Run with GPU
docker run --gpus all -p 7860:7860 landmarkdiff

# Or use docker compose
docker compose up landmarkdiff
```

## Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space
2. Select "Gradio" as the SDK
3. Upload the repo contents
4. Set GPU hardware (T4 minimum)

The `scripts/app.py` Gradio demo is compatible with Hugging Face Spaces out of the box.

## Cloud deployment

### AWS / GCP / Azure

For production deployments, we recommend:

1. Use the Docker image with GPU instances (e.g., AWS `g5.xlarge`)
2. Put behind a reverse proxy (nginx) with authentication
3. Rate limit requests to prevent abuse
4. Store model weights on persistent storage (EBS/GCS)

### FastAPI server

For API-only deployment without the Gradio UI:

```python
from fastapi import FastAPI, UploadFile
from landmarkdiff.inference import LandmarkDiffPipeline

app = FastAPI()
pipeline = LandmarkDiffPipeline.from_pretrained("checkpoints/latest")

@app.post("/predict")
async def predict(image: UploadFile, procedure: str = "rhinoplasty", intensity: float = 0.6):
    img = load_image(await image.read())
    result = pipeline.generate(img, procedure=procedure, intensity=intensity)
    return {"image": encode_image(result)}
```

## Security considerations

- Never expose without authentication in production
- Rate limit API endpoints
- Validate and sanitize all inputs
- Do not store patient photos without consent
- Follow HIPAA guidelines for medical data
- Use HTTPS for all connections
