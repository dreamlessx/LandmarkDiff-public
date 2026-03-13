# Training Guide

Train LandmarkDiff from scratch on your own data.

## Overview

Training has two phases:

1. **Phase A** (synthetic data, diffusion loss only) - teaches the model to generate faces from landmark meshes
2. **Phase B** (clinical data, full loss) - fine-tunes on real surgical before/after pairs with identity and perceptual losses

## Phase A: Synthetic Training

### 1. Download face images

```bash
# FFHQ faces (5K for quick experiments, 50K for full training)
python scripts/download_ffhq.py --num 5000 --resolution 512 --output data/ffhq_samples/

# Or use multiple sources for diversity
python scripts/download_faces_multi.py --num 10000 --output data/faces_all/
```

### 2. Generate synthetic pairs

Each training pair consists of:
- Original face image
- Deformed landmark mesh (via TPS warp)
- TPS-warped face image (training target)

```bash
python scripts/generate_synthetic_data.py \
    --input data/ffhq_samples/ \
    --output data/synthetic_pairs/ \
    --num 50000

# Or use SLURM for parallel generation
sbatch scripts/gen_synthetic_slurm.sh
```

### 3. (Optional) Add clinical augmentations

```bash
python scripts/augment_pairs.py \
    --input data/synthetic_pairs/ \
    --output data/augmented_pairs/ \
    --augmentations lighting,color_temp,jpeg,noise
```

### 4. Train

```bash
# Single GPU
python scripts/train_controlnet.py \
    --data_dir data/synthetic_pairs/ \
    --output_dir checkpoints/ \
    --num_train_steps 50000 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 1e-5 \
    --mixed_precision bf16

# SLURM
sbatch scripts/train_slurm.sh
```

### 5. Monitor

```bash
# WandB (if online)
# Check wandb.ai for loss curves

# Offline
tail -f slurm-*.out
```

### Expected Phase A metrics (50K steps)

| Metric | Target |
|--------|--------|
| Training loss | < 0.15 |
| FID | < 120 |
| Generated samples | Faces follow landmark structure |

## Phase B: Clinical Fine-tuning

Phase B uses real before/after surgical photos with the full 4-term loss:

```
L_total = L_diffusion + w_1 * L_landmark + w_2 * L_identity + w_3 * L_perceptual
```

This phase is planned but not yet activated. See the [roadmap](../../README.md#roadmap) for status.

## Configuration

Edit `configs/training.yaml`:

```yaml
training:
  phase: "A"
  learning_rate: 1.0e-5
  train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_steps: 50000
  mixed_precision: "bf16"
  ema_decay: 0.9999
  checkpoint_every: 5000
  sample_every: 1000
```

## Critical training safeguards

| Setting | Value | Why |
|---------|-------|-----|
| Mixed precision | BF16 only | FP16 overflows on SD activations |
| VAE | Frozen | Gradient leak corrupts latent space |
| EMA | 0.9999 | Without it, checkpoints have high-frequency artifacts |
| Normalization | GroupNorm | BatchNorm unstable at small batch sizes |
| LR schedule | Cosine | Constant LR causes late-stage oscillation |

See [GPU Training Guide](../GPU_TRAINING_GUIDE.md) for HPC-specific instructions.
