#!/bin/bash
#SBATCH --job-name=surgery_controlnet_v2
#SBATCH --partition=batch_gpu
#SBATCH --account=your_gpu_acc
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# === Phase A v2: Extended training with scaled-up dataset ===
# 50K steps, 10K+ synthetic pairs, resume from v1 checkpoint

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline

# Trap preemption signal -> save checkpoint -> requeue
trap 'echo "Caught USR1 - saving checkpoint..."; kill -INT $TRAIN_PID; wait $TRAIN_PID; scontrol requeue $SLURM_JOB_ID' USR1

WORK_DIR="/path/to/LandmarkDiff"
DATA_DIR="${WORK_DIR}/data/synthetic_pairs_v2"
CKPT_DIR="${WORK_DIR}/checkpoints_v2"
WANDB_DIR="${WORK_DIR}/wandb"

mkdir -p "$CKPT_DIR" "$WANDB_DIR"

cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Phase A v2 Training ==="
echo "Data: $DATA_DIR ($(ls $DATA_DIR/*_input.png 2>/dev/null | wc -l) pairs)"
echo "Checkpoints: $CKPT_DIR"
echo "Steps: 50000"
echo "Started: $(date)"

python scripts/train_controlnet.py \
    --data_dir=$DATA_DIR \
    --output_dir=$CKPT_DIR \
    --wandb_dir=$WANDB_DIR \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_steps=50000 \
    --checkpoint_every=10000 \
    --log_every=100 \
    --sample_every=2000 \
    --resume_from_checkpoint=latest \
    --phase=A &

TRAIN_PID=$!
wait $TRAIN_PID

echo "=== Training complete: $(date) ==="
