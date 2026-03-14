# GPU Training Guide

Complete setup for training LandmarkDiff on SLURM-based HPC clusters with A100/A6000 GPUs.

## Prerequisites

- SLURM cluster with A100 GPU (80GB preferred, 40GB works with reduced batch)
- Apptainer/Singularity installed
- Fast scratch storage (Lustre preferred, e.g. `/scratch/$USER/`)
- Python 3.10+

## 1. Build Container

```bash
# On a build node (not login node)
apptainer build landmarkdiff.sif containers/landmarkdiff.def

# Or as sandbox for development
apptainer build --sandbox landmarkdiff_sandbox containers/landmarkdiff.def
```

### Verify container
```bash
apptainer exec --nv landmarkdiff.sif python -c "import diffusers; import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 2. Download Training Data

```bash
# On compute node (not login - uses bandwidth)
srun --partition=gpu --gres=gpu:0 --mem=16G --time=2:00:00 --pty bash

# Download 5000 FFHQ images (512x512 for SD1.5 native resolution)
python scripts/download_ffhq.py --num 5000 --resolution 512 --output /scratch/$USER/landmarkdiff/data/ffhq

# Generate synthetic training pairs
python scripts/generate_synthetic_data.py \
    --input /scratch/$USER/landmarkdiff/data/ffhq \
    --output /scratch/$USER/landmarkdiff/data/synthetic_pairs \
    --num 5000
```

**Data sizes:**
- 5000 FFHQ 512x512: ~2.5 GB
- 5000 synthetic pairs: ~12 GB (5 images per pair)
- 50K pairs (full training): ~120 GB

## 3. Configure Training

Edit `configs/training.yaml` or pass CLI args:

```yaml
# Phase A: 10K steps, diffusion loss only, synthetic TPS data
training:
  phase: "A"
  learning_rate: 1.0e-5
  train_batch_size: 4
  gradient_accumulation_steps: 4    # effective batch = 16
  num_train_steps: 10000
  mixed_precision: "bf16"           # NEVER fp16
  ema_decay: 0.9999
```

## 4. Submit Training Job

Edit `scripts/train_slurm.sh`:

```bash
# Set these paths:
CONTAINER="/path/to/landmarkdiff.sif"
DATA_DIR="/scratch/$USER/landmarkdiff/data/synthetic_pairs"
CKPT_DIR="/scratch/$USER/landmarkdiff/checkpoints"
WANDB_DIR="/scratch/$USER/landmarkdiff/wandb"
```

Submit:
```bash
sbatch scripts/train_slurm.sh
```

Monitor:
```bash
squeue -u $USER
tail -f slurm-*.out
```

## 5. Cluster-Specific Notes

Adapt these settings for your cluster:

```bash
# Example SLURM config - adjust for your cluster
#SBATCH --partition=gpu            # your GPU partition
#SBATCH --account=your_gpu_acc    # your account

# Modules (if not using container)
module load GCC/12.3.0
module load CUDA/12.1.1
module load Python/3.11.3

# Storage - use fast scratch storage
DATA_DIR="/scratch/$USER/landmarkdiff/data"
CKPT_DIR="/scratch/$USER/landmarkdiff/checkpoints"
```

### GPU recommendations
| GPU | VRAM | Notes |
|-----|------|-------|
| P100 | 16GB | Batch size 2, gradient accumulation 8 |
| V100 | 32GB | Batch size 4, gradient accumulation 4 |
| A6000 | 48GB | Batch size 4-8 |
| A100 | 40/80GB | Recommended, full batch size |

If your cluster uses Lustre, set striping for large datasets:
```bash
lfs setstripe -c -1 $DATA_DIR
```

## 6. Critical Safeguards

These are non-negotiable. Training will produce garbage without them:

| Safeguard | Why |
|-----------|-----|
| BF16 only | FP16 overflows on SD/FLUX activations |
| VAE frozen | Gradient leak corrupts entire latent space |
| EMA 0.9999 | Without it, checkpoints have HF artifacts |
| GroupNorm | Batch size 4 makes BatchNorm unstable |
| `--resume_from_checkpoint=latest` | Preemption restarts from step 0 without this |
| `--signal=B:USR1@300` | Saves checkpoint on preemption |
| Phase A: L_diffusion only | Perceptual loss against TPS warps penalizes realism |
| Pre-computed TPS warps | On-the-fly TPS CPU-bottlenecks GPU |

## 7. Expected Training Metrics

**Phase A (10K steps, ~6-8 hours on A100):**
- Loss should decrease monotonically
- Sample generations at 10K should show face structure following landmarks
- FID ~80-120 (will improve with more data + Phase B)

**Phase B (50K steps, ~30-40 hours on A100):**
- FID target: <50
- NME target: <0.05
- Identity sim target: >0.85
- SSIM target: >0.80

## 8. Sync WandB (Offline Mode)

After training completes:
```bash
wandb sync $WANDB_DIR/wandb/latest-run/
```

## 9. Local Development (Apple Silicon)

For inference and small experiments (not training):
```bash
cd /path/to/LandmarkDiff
source .venv/bin/activate
python -m landmarkdiff infer /path/to/face.jpg --procedure rhinoplasty
```

- SD1.5 inference: ~30-60 sec per image on MPS
- 36GB unified memory handles models easily
- Use for rapid iteration on conditioning/masking before cluster training
