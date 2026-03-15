# Training Guide

Train LandmarkDiff from scratch on your own data. This guide covers every step: data preparation, configuration, single-GPU and multi-GPU training, curriculum learning, monitoring, checkpointing, and SLURM submission.

## Overview

Training has two phases:

1. **Phase A** (synthetic data, diffusion loss only): teaches the model to generate faces conditioned on deformed landmark meshes
2. **Phase B** (clinical data, full loss): fine-tunes on real surgical before/after pairs with identity and perceptual losses

The multi-term loss for Phase B:

```
L_total = L_diffusion + w_1 * L_identity + w_2 * L_perceptual + w_3 * L_mask
```

Phase B resumes from a Phase A checkpoint.

## Prerequisites

```bash
# Install training dependencies
pip install -e ".[train]"

# Verify PyTorch + CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
```

The `[train]` extra includes wandb, deepspeed, webdataset, and accelerate.

## Data Preparation Pipeline

### Step 1: Download face images

```bash
# FFHQ faces (5K for quick experiments, 50K for full training)
python scripts/download_ffhq.py --num 5000 --resolution 512 --output data/ffhq_samples/

# Or use multiple sources for diversity
python scripts/download_faces_multi.py --num 10000 --output data/faces_all/
```

### Step 2: Generate synthetic training pairs

Each training pair consists of:
- **Input image**: original face (512x512)
- **Conditioning image**: deformed wireframe mesh overlay
- **Target image**: TPS-warped face (what the model should produce)
- **Procedure label**: which procedure was applied

```bash
python scripts/generate_synthetic_data.py \
    --input data/ffhq_samples/ \
    --output data/synthetic_pairs/ \
    --num 50000

# Check what was generated
python scripts/dataset_stats.py data/synthetic_pairs/
```

For large datasets, use the SLURM parallel generation script:

```bash
sbatch scripts/gen_synthetic_slurm.sh
```

The generator randomly samples from all registered procedures (rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow_lift, mentoplasty) unless you restrict it with `--procedure`.

### Step 3: (Optional) Add clinical augmentations

Clinical photos have different characteristics than FFHQ: variable lighting, JPEG compression, color temperature shifts, and noise. Adding these augmentations to synthetic data helps the model generalize:

```bash
python scripts/augment_pairs.py \
    --input data/synthetic_pairs/ \
    --output data/augmented_pairs/ \
    --augmentations lighting,color_temp,jpeg,noise
```

Clinical augmentation can also be applied online during training by setting `clinical_augment: true` in your config (Phase B only; disabled by default for Phase A).

### Step 4: Build the combined training dataset

```bash
python scripts/build_training_dataset.py \
    --input data/synthetic_pairs/ \
    --output data/training_combined/
```

This creates the directory structure expected by the training script: `*_input.png`, `*_conditioning.png`, `*_target.png` triplets plus a `metadata.json` for curriculum learning.

### Step 5: Create train/val splits

```bash
python scripts/create_test_split.py \
    --data_dir data/training_combined/ \
    --output_dir data/splits/ \
    --val_fraction 0.05
```

### Step 6: Run the preflight check

Before submitting a training job, verify everything is in order:

```bash
python scripts/preflight_training.py --config configs/phaseA_v3_curriculum.yaml
```

The preflight script checks dataset completeness, metadata presence, val/test splits, config validity, dependency installation, GPU availability, disk space, and existing checkpoints for resume.

## Configuration with YAML

All training parameters live in YAML config files under `configs/`. You can either edit a config file or pass CLI arguments to override individual settings.

### Available configs

| Config | Steps | Purpose |
|--------|-------|---------|
| `phaseA_quick.yaml` | 500 | Smoke test, debug loop |
| `phaseA_default.yaml` | 10,000 | Quick Phase A validation |
| `phaseA_production.yaml` | 50,000 | Full Phase A production run |
| `phaseA_v3_curriculum.yaml` | 100,000 | Phase A with curriculum learning |
| `phaseB.yaml` | 25,000 | Phase B fine-tuning |
| `phaseB_production.yaml` | 50,000 | Full Phase B production run |
| `phaseB_identity.yaml` | -- | Phase B with identity emphasis |

### Config structure

```yaml
# configs/phaseA_v3_curriculum.yaml
experiment_name: phaseA_v3_curriculum

model:
  base_model: runwayml/stable-diffusion-v1-5
  controlnet_conditioning_channels: 3
  controlnet_conditioning_scale: 1.0
  use_ema: true
  ema_decay: 0.9999
  gradient_checkpointing: true

training:
  phase: A
  learning_rate: 1.0e-5
  batch_size: 4
  gradient_accumulation_steps: 4    # effective batch = 16
  max_train_steps: 100000
  warmup_steps: 1000
  mixed_precision: bf16             # never fp16
  seed: 42
  optimizer: adamw
  adam_beta1: 0.9
  adam_beta2: 0.999
  weight_decay: 0.01
  max_grad_norm: 1.0
  lr_scheduler: cosine
  save_every_n_steps: 10000
  resume_from_checkpoint: auto
  validate_every_n_steps: 5000
  num_validation_samples: 8

data:
  train_dir: data/training_combined
  val_dir: data/validation
  image_size: 512
  num_workers: 8
  random_flip: true
  random_rotation: 5.0
  color_jitter: 0.1
  procedures:
    - rhinoplasty
    - blepharoplasty
    - rhytidectomy
    - orthognathic
  displacement_model_path: data/displacement_model.npz

wandb:
  enabled: true
  project: landmarkdiff
  tags: [phase-a, curriculum, v3-data]

output_dir: outputs/phaseA_v3_curriculum
```

### CLI overrides

Any config field can be overridden on the command line:

```bash
python scripts/train_controlnet.py \
    --config configs/phaseA_v3_curriculum.yaml \
    --learning_rate 5e-6 \
    --batch_size 2 \
    --num_train_steps 5000
```

## Single GPU Training

The simplest way to start:

```bash
python scripts/train_controlnet.py \
    --data_dir data/training_combined/ \
    --output_dir checkpoints/ \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_steps 50000 \
    --checkpoint_every 5000 \
    --sample_every 1000 \
    --resume_from_checkpoint latest \
    --phase A
```

Or with a config file:

```bash
python scripts/train_controlnet.py --config configs/phaseA_default.yaml
```

### Dry run

To test the training loop without actually running for many steps:

```bash
python scripts/dry_run_training.py --config configs/phaseA_quick.yaml
```

### GPU memory recommendations

| GPU | VRAM | Batch size | Gradient accumulation | Effective batch |
|-----|------|------------|----------------------|-----------------|
| P100 | 16 GB | 2 | 8 | 16 |
| V100 | 32 GB | 4 | 4 | 16 |
| A6000 | 48 GB | 4-8 | 2-4 | 16-32 |
| A100 (40 GB) | 40 GB | 4 | 4 | 16 |
| A100 (80 GB) | 80 GB | 8 | 4 | 32 |
| H100 | 80 GB | 8 | 4 | 32 |
| L40S | 48 GB | 4-8 | 2-4 | 16-32 |

If you run out of VRAM, enable gradient checkpointing (`gradient_checkpointing: true` in the model config) and reduce batch size.

## Multi-GPU Distributed Training (DDP)

The training script automatically detects PyTorch DDP environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) and activates distributed mode. Use `torchrun` to launch:

### 2 GPUs on a single node

```bash
torchrun --nproc_per_node=2 scripts/train_controlnet.py \
    --config configs/phaseA_v3_curriculum.yaml \
    --output_dir checkpoints/phaseA_ddp/
```

### 4 GPUs on a single node

```bash
torchrun --nproc_per_node=4 scripts/train_controlnet.py \
    --config configs/phaseA_v3_curriculum.yaml \
    --output_dir checkpoints/phaseA_ddp/ \
    --train_batch_size 4 \
    --gradient_accumulation_steps 2
```

With 4 GPUs, batch size 4, and gradient accumulation 2, the effective batch size is `4 * 4 * 2 = 32`.

### DDP behavior

- Only rank 0 saves checkpoints, logs to WandB, and generates sample images
- All ranks participate in gradient computation and synchronization
- The learning rate does not need to be scaled; effective batch size increases naturally through more GPUs
- Use `NCCL` backend for GPU-to-GPU communication (the default on Linux)

### Multi-node training (advanced)

For training across multiple nodes on a SLURM cluster:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train_controlnet.py \
    --config configs/phaseA_production.yaml
```

## Curriculum Learning (Phase A to Phase B)

### Phase A: Synthetic data, diffusion loss only

Phase A trains the ControlNet to follow landmark mesh conditioning using synthetic TPS-warped data. The model learns mesh-to-face generation without needing real surgical data.

Key Phase A settings:
- Loss: `L_diffusion` only
- Data: synthetic TPS pairs
- Learning rate: `1e-5`
- LR schedule: cosine decay with warmup
- Steps: 50,000-100,000 depending on dataset size

### Phase B: Clinical data, full multi-term loss

Phase B fine-tunes the Phase A checkpoint on paired clinical data with the full 4-term loss. This phase requires before/after surgical image pairs.

```bash
python scripts/train_controlnet.py \
    --config configs/phaseB.yaml
```

Key Phase B settings:
- Loss: `L_diffusion + 0.1 * L_identity + 0.05 * L_perceptual + 0.1 * L_mask`
- Data: clinical before/after pairs with augmentation
- Learning rate: `5e-6` (lower than Phase A)
- Resume from: `checkpoints/phaseA/latest`
- Steps: 25,000-50,000

The Phase B config automatically loads the Phase A checkpoint:

```yaml
training:
  phase: B
  resume_from: checkpoints/phaseA/latest
  loss_weights:
    diffusion: 1.0
    identity: 0.1
    perceptual: 0.05
    mask: 0.1
```

### Curriculum progression

The `phaseA_v3_curriculum.yaml` config supports progressive difficulty scheduling across waves of data. Each wave introduces more varied and challenging training examples. Metadata in `metadata.json` tracks which wave each pair belongs to, and the dataloader can sample accordingly.

## Monitoring with Weights & Biases

### Online mode (local machines)

```bash
wandb login
python scripts/train_controlnet.py --config configs/phaseA_default.yaml
```

Check https://wandb.ai for live loss curves, sample generations, and system metrics.

### Offline mode (HPC clusters)

Most HPC clusters have restricted internet access. Use offline mode:

```bash
export WANDB_MODE=offline
python scripts/train_controlnet.py --config configs/phaseA_default.yaml
```

After training completes, sync the offline run from a machine with internet access:

```bash
wandb sync outputs/phaseA_default/wandb/latest-run/
```

### Key metrics to watch

**Phase A (target at 50K steps):**

| Metric | Target | Notes |
|--------|--------|-------|
| Training loss | < 0.15 | Should decrease monotonically |
| FID | < 120 | Improves with more data and steps |
| Generated samples | -- | Faces should follow landmark structure |

**Phase B (target at 50K steps):**

| Metric | Target | Notes |
|--------|--------|-------|
| FID | < 50 | Significant improvement over Phase A |
| NME (landmark error) | < 0.05 | Landmarks match surgical plan |
| Identity similarity | > 0.85 | ArcFace cosine similarity |
| SSIM | > 0.80 | Structural similarity to target |

### Real-time monitoring

```bash
# Follow the SLURM log
tail -f slurm-*.out

# Use the training dashboard
python scripts/training_dashboard.py --output_dir checkpoints/

# Plot loss curves from existing runs
python scripts/plot_training_curves.py --run_dir outputs/phaseA_v3_curriculum/
```

## Checkpointing and Resume

### Automatic checkpointing

Checkpoints are saved every `save_every_n_steps` (default: 5,000 for Phase A, 1,000 for Phase B). Each checkpoint contains the ControlNet weights, optimizer state, EMA weights, and the training step count.

### Resume from interruption

Set `resume_from_checkpoint: auto` (or `latest`) in your config, or pass `--resume_from_checkpoint=latest` on the CLI. The training script will find the most recent checkpoint and continue from that step.

```bash
# Explicit resume from a specific checkpoint
python scripts/train_controlnet.py \
    --config configs/phaseA_default.yaml \
    --resume_from_checkpoint checkpoints/checkpoint-15000
```

### EMA weights

Exponential Moving Average (EMA) weights at decay rate 0.9999 are maintained throughout training and saved alongside each checkpoint. Use EMA weights for inference; they produce smoother, more stable outputs. The training script saves separate `checkpoint-*-ema` directories.

### Evaluate a checkpoint

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/checkpoint-50000 \
    --test_dir data/splits/test/ \
    --output eval_results/
```

## SLURM Submission

### Single-GPU SLURM job

The provided `scripts/train_slurm.sh` handles everything for a single-GPU run, including preemption handling:

```bash
#!/bin/bash
#SBATCH --job-name=surgery_controlnet
#SBATCH --partition=batch_gpu
#SBATCH --account=your_gpu_acc
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# === Skip-logic: don't rerun if already completed ===
CKPT_DIR="/path/to/LandmarkDiff/checkpoints"
FINAL_STEP=50000
if [ -d "$CKPT_DIR/checkpoint-${FINAL_STEP}" ]; then
    echo "Training already complete at step ${FINAL_STEP}. Exiting."
    exit 0
fi

# === Critical HPC safeguards ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline

# Trap preemption signal -> save checkpoint -> requeue
trap 'echo "Caught USR1 - saving checkpoint..."; kill -INT $TRAIN_PID; wait $TRAIN_PID; scontrol requeue $SLURM_JOB_ID' USR1

WORK_DIR="/path/to/LandmarkDiff"
DATA_DIR="${WORK_DIR}/data/training_combined"
WANDB_DIR="${WORK_DIR}/wandb"

mkdir -p "$CKPT_DIR" "$WANDB_DIR"
cd "$WORK_DIR"

# Activate environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

python scripts/train_controlnet.py \
    --data_dir=$DATA_DIR \
    --output_dir=$CKPT_DIR \
    --wandb_dir=$WANDB_DIR \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_steps=${FINAL_STEP} \
    --checkpoint_every=5000 \
    --sample_every=1000 \
    --resume_from_checkpoint=latest \
    --phase=A &

TRAIN_PID=$!
wait $TRAIN_PID
```

Key SLURM features:

- **`--signal=B:USR1@300`**: sends USR1 to the job 300 seconds before wall-time expiration
- **`--requeue`**: requeues the job after preemption
- **`trap ... USR1`**: catches the signal, sends SIGINT to the training process (triggering a checkpoint save), then requeues
- **`--resume_from_checkpoint=latest`**: picks up from the last saved checkpoint after requeue
- **Skip-logic**: checks if training is already complete before starting

Submit with:

```bash
sbatch scripts/train_slurm.sh
```

Monitor the job:

```bash
squeue -u $USER                                    # check queue status
tail -f slurm-*.out                                # follow training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS  # check resource usage
```

### Multi-GPU SLURM job

For multi-GPU DDP training on SLURM, replace the python command with torchrun:

```bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

torchrun --nproc_per_node=4 scripts/train_controlnet.py \
    --config configs/phaseA_production.yaml \
    --output_dir=$CKPT_DIR \
    --resume_from_checkpoint=latest &

TRAIN_PID=$!
wait $TRAIN_PID
```

## Critical Training Safeguards

These settings are non-negotiable. Training will produce garbage without them:

| Safeguard | Setting | Why |
|-----------|---------|-----|
| Mixed precision | BF16 only | FP16 overflows on SD1.5 activations |
| VAE | Frozen | Gradient leak corrupts the entire latent space |
| EMA | 0.9999 | Without it, checkpoints have high-frequency artifacts |
| Normalization | GroupNorm | BatchNorm is unstable at batch size 4 |
| LR schedule | Cosine | Constant LR causes late-stage oscillation |
| Grad clipping | max_norm 1.0 | Prevents gradient explosions |
| Resume | `--resume_from_checkpoint=latest` | Preemption restarts from step 0 without this |
| SLURM signal | `--signal=B:USR1@300` | Saves checkpoint before wall-time preemption |
| Phase A loss | L_diffusion only | Perceptual loss against TPS warps penalizes realism |
| TPS warps | Pre-computed | On-the-fly TPS CPU-bottlenecks the GPU |
| ControlNet scale | max 1.2 | Values above 1.2 cause activation saturation |

## Common Training Issues and Fixes

### Loss spikes or NaN

- Check that mixed precision is set to `bf16`, not `fp16`. FP16 overflow is the most common cause.
- Verify `max_grad_norm` is set (1.0 is a good default).
- If the gradient watchdog fires (logged as `GradientWatchdog: explosion detected`), the emergency save will trigger automatically. Resume from the last clean checkpoint.

### Generated images are blank or noisy

- Confirm the VAE is frozen (`vae_frozen: true`). Unfreezing the VAE is the #1 cause of latent space collapse.
- Check that EMA is enabled. Non-EMA checkpoints often have high-frequency noise.
- Inspect training samples at `sample_every` intervals. If early samples (step 1000) show no face structure at all, the conditioning images may be malformed.

### Training is slow

- Enable gradient checkpointing to trade compute for memory, allowing a larger batch size.
- Pre-compute TPS warps instead of generating them on-the-fly. The CPU-to-GPU data transfer is the bottleneck.
- Use `num_workers: 8` (or more, up to your CPU count) for the dataloader.
- On HPC with Lustre, set file striping: `lfs setstripe -c -1 $DATA_DIR`

### SLURM job keeps restarting from step 0

- Make sure `--resume_from_checkpoint=latest` is set.
- Verify checkpoint files exist in the output directory: `ls checkpoints/checkpoint-*`
- Check that the SLURM trap is correctly wired. The training process must receive SIGINT (not SIGKILL) to save a checkpoint.

### Out of disk space mid-training

- Each SD1.5 checkpoint is about 2 GB. A 100K-step run with checkpoints every 10K steps produces ~20 GB of checkpoints.
- Point `output_dir` to a scratch filesystem with enough space.
- Use `python scripts/clean_data.py` to prune old checkpoints, keeping only the latest N.

### WandB issues on HPC

- Set `WANDB_MODE=offline` before training.
- Sync after the job completes: `wandb sync outputs/*/wandb/latest-run/`
- If WandB is not installed, the training script falls back to console logging automatically.

### Phase B identity loss not decreasing

- The identity loss uses ArcFace embeddings. Make sure `insightface` and `onnxruntime` are installed: `pip install insightface onnxruntime`.
- If identity similarity starts very low (<0.3), the Phase A checkpoint may not be generating recognizable faces yet. Train Phase A longer.

## Next Steps

- [Evaluation](evaluation.md): evaluate your trained checkpoints
- [Custom Procedures](custom_procedures.md): add new surgical procedures
- [GPU Training Guide](../GPU_TRAINING_GUIDE.md): HPC-specific setup details
- [Deployment](deployment.md): deploy your trained model
