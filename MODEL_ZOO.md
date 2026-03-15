# Model Zoo

Pre-trained models and weights for LandmarkDiff.

## Base Models (Required)

These are third-party models that LandmarkDiff uses. They are downloaded automatically on first run.

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| Stable Diffusion 1.5 | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | ~4 GB | Base diffusion backbone |
| ControlNet MediaPipe Face | [CrucibleAI/ControlNetMediaPipeFace](https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace) | ~1.4 GB | Face mesh conditioning |
| MediaPipe Face Mesh | [google/mediapipe](https://google.github.io/mediapipe/) | ~5 MB | 478-point landmark detection |

## Post-Processing Models (Optional)

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| CodeFormer | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | ~400 MB | Face restoration (primary) |
| GFPGAN v1.4 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | ~350 MB | Face restoration (fallback) |
| Real-ESRGAN x4 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | ~64 MB | Background super-resolution |
| ArcFace | [insightface/buffalo_l](https://github.com/deepinsight/insightface) | ~250 MB | Identity verification |

## Fine-tuned Checkpoints

| Checkpoint | Dataset | Steps | FID | LPIPS | Status |
|-----------|---------|-------|-----|-------|--------|
| `phase_a_50k` | 50K synthetic pairs | 50K | TBD | TBD | Training |
| `phase_b_clinical` | Clinical + synthetic | TBD | TBD | TBD | Planned |

### Downloading checkpoints

```bash
# Phase A (50K steps) - coming soon
# Will be available via Hugging Face Hub
```

## Training Your Own

See [docs/GPU_TRAINING_GUIDE.md](docs/GPU_TRAINING_GUIDE.md) for instructions on training from scratch.

```bash
# Generate training data
python scripts/generate_synthetic_data.py --input data/ffhq_samples/ --output data/synthetic_pairs/ --num 50000

# Train Phase A
python scripts/train_controlnet.py --data_dir data/synthetic_pairs/ --output_dir checkpoints/ --num_train_steps 50000
```

## Hardware Requirements

| Task | Min VRAM | Recommended | Time |
|------|----------|-------------|------|
| Inference (single image) | 6 GB | 8 GB | ~5 sec |
| Inference (batch of 16) | 12 GB | 16 GB | ~30 sec |
| Training Phase A (10K steps) | 24 GB | 40 GB (A100) | ~1 hour |
| Training Phase A (50K steps) | 40 GB | 80 GB (A100) | ~6 hours |
| Training Phase B | 40 GB | 80 GB (A100) | ~30 hours |

## Planned Models

The following models are on the roadmap as LandmarkDiff moves toward a 3D-native pipeline (phone video scan to interactive 3D surgical preview). None of these are available yet; this section previews what is coming.

| Model | Approach | Purpose | Status |
|-------|----------|---------|--------|
| 3D Face Reconstruction | FLAME-based fitting or neural implicit (NeRF/3DGS) | Reconstruct a textured 3D face mesh from a short phone video scan | Research |
| 3D Deformation Model | Mesh-space surgical simulation | Apply procedure-specific displacements directly on 3D mesh vertices instead of 2D pixel warps | Research |
| Multi-View Consistency | View-conditioned diffusion or 3DGS rendering | Ensure deformed face renders consistently across arbitrary viewpoints | Research |
| Mobile-Optimized Inference | Distilled/quantized pipeline | Run landmark detection, reconstruction, and preview on-device with acceptable latency | Planned |

### 3D face reconstruction model

Replaces the current single-image 2D pipeline entry point. Given 10-30 frames from a phone video scan (patient rotating their head), reconstructs a FLAME mesh with per-vertex texture. Candidate approaches include DECA-style regression, optimization-based FLAME fitting from MediaPipe landmarks, and feed-forward 3DGS methods.

### 3D deformation model

Operates on the reconstructed mesh rather than on 2D pixel coordinates. Existing procedure presets (rhinoplasty, blepharoplasty, etc.) would be re-expressed as 3D vertex displacement fields, enabling anatomically grounded deformations that look correct from any viewing angle.

### Multi-view consistency model

Ensures that the deformed 3D representation renders without view-dependent artifacts. This may be handled implicitly by the 3D representation (mesh or 3DGS) or may require an additional consistency loss during training.

### Mobile-optimized inference model

A distilled or quantized version of the pipeline targeting on-device inference. The goal is real-time landmark tracking and capture guidance, with reconstruction offloaded to a server or run locally on modern phones.
