# GPU Training Guide

Complete setup for training LandmarkDiff on HPC clusters (Vanderbilt ACCRE, UW Hyak, or any SLURM+A100 system).

## Prerequisites

- SLURM cluster with A100 GPU (80GB preferred, 40GB works with reduced batch)
- Apptainer/Singularity installed
- `/gscratch/` or equivalent fast storage (Lustre preferred)
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
# On compute node (not login — uses bandwidth)
srun --partition=gpu --gres=gpu:0 --mem=16G --time=2:00:00 --pty bash

# Download 5000 FFHQ images (512x512 for SD1.5 native resolution)
python scripts/download_ffhq.py --num 5000 --resolution 512 --output /gscratch/$GROUP/landmarkdiff/data/ffhq

# Generate synthetic training pairs
python scripts/generate_synthetic_data.py \
    --input /gscratch/$GROUP/landmarkdiff/data/ffhq \
    --output /gscratch/$GROUP/landmarkdiff/data/synthetic_pairs \
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
DATA_DIR="/gscratch/$GROUP/landmarkdiff/data/synthetic_pairs"
CKPT_DIR="/gscratch/$GROUP/landmarkdiff/checkpoints"
WANDB_DIR="/gscratch/$GROUP/landmarkdiff/wandb"
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

## 5. Vanderbilt ACCRE Specifics

```bash
# Partition
#SBATCH --partition=pascal         # or volta, ampere if available
#SBATCH --account=csb_gpu_acc     # your account

# Modules (if not using container)
module load GCC/12.3.0
module load CUDA/12.1.1
module load Python/3.11.3

# Storage
DATA_DIR="/scratch/$USER/landmarkdiff/data"
CKPT_DIR="/scratch/$USER/landmarkdiff/checkpoints"
```

### ACCRE GPU partitions
| Partition | GPU | VRAM | Max Time |
|-----------|-----|------|----------|
| pascal | P100 | 16GB | 14 days |
| volta | V100 | 32GB | 3 days |
| ampere (if available) | A100 | 40/80GB | varies |

**Note:** If only P100/V100 available, reduce batch size to 2 and increase gradient accumulation to 8.

## 6. UW Hyak Specifics

```bash
#SBATCH --partition=gpu-a100
#SBATCH --account=your_group

DATA_DIR="/gscratch/your_group/landmarkdiff/data"
CKPT_DIR="/gscratch/your_group/landmarkdiff/checkpoints"

# Lustre striping for large datasets
lfs setstripe -c -1 $DATA_DIR
```

## 7. Critical Safeguards

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

## 8. Expected Training Metrics

**Phase A (10K steps, ~6-8 hours on A100):**
- Loss should decrease monotonically
- Sample generations at 10K should show face structure following landmarks
- FID ~80-120 (will improve with more data + Phase B)

**Phase B (50K steps, ~30-40 hours on A100):**
- FID target: <50
- NME target: <0.05
- Identity sim target: >0.85
- SSIM target: >0.80

## 9. Sync WandB (Offline Mode)

After training completes:
```bash
wandb sync $WANDB_DIR/wandb/latest-run/
```

## 10. Mac M3 Pro (Local Development)

For inference and small experiments (not training):
```bash
cd ~/Projects/Surgery_Landmark
source .venv/bin/activate
python landmarkdiff/inference.py /path/to/face.jpg --procedure rhinoplasty
```

- SD1.5 inference: ~30-60 sec per image on MPS
- 36GB unified memory handles models easily
- Use for rapid iteration on conditioning/masking before cluster training
