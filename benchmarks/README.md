# Benchmarks

Performance benchmarks for LandmarkDiff across different hardware.

## Inference Speed

| Hardware | Mode | Resolution | Time per image |
|----------|------|------------|----------------|
| A100 80GB | ControlNet (30 steps) | 512x512 | ~3 sec |
| A100 40GB | ControlNet (30 steps) | 512x512 | ~4 sec |
| RTX 4090 | ControlNet (30 steps) | 512x512 | ~5 sec |
| RTX 3090 | ControlNet (30 steps) | 512x512 | ~7 sec |
| T4 16GB | ControlNet (30 steps) | 512x512 | ~15 sec |
| M3 Pro (MPS) | ControlNet (30 steps) | 512x512 | ~45 sec |
| CPU (i9-13900K) | TPS only | 512x512 | ~0.5 sec |

## Landmark Extraction

| Hardware | Images/sec | Notes |
|----------|-----------|-------|
| Any modern CPU | ~30 fps | MediaPipe runs on CPU |

## Training Throughput

| Hardware | Batch size | Grad accum | Effective batch | Steps/hour |
|----------|-----------|------------|-----------------|------------|
| A100 80GB | 4 | 4 | 16 | ~600 |
| A100 40GB | 2 | 8 | 16 | ~400 |
| RTX 4090 | 2 | 8 | 16 | ~350 |
| RTX 3090 | 1 | 16 | 16 | ~200 |

## Memory Usage

| Component | VRAM |
|-----------|------|
| SD 1.5 (FP16) | ~2.5 GB |
| ControlNet (FP16) | ~1.5 GB |
| VAE (FP32) | ~0.5 GB |
| CodeFormer | ~0.4 GB |
| ArcFace | ~0.3 GB |
| **Total inference** | **~5.2 GB** |
| **Total training** | **~25 GB** |

## Running benchmarks

```bash
# Inference benchmark
python benchmarks/benchmark_inference.py --device cuda --num_images 100

# Landmark extraction benchmark
python benchmarks/benchmark_landmarks.py --num_images 1000

# Training throughput benchmark
python benchmarks/benchmark_training.py --device cuda --num_steps 100
```
