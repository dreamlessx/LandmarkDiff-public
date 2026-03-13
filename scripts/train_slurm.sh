#!/bin/bash
#SBATCH --job-name=surgery_controlnet
#SBATCH --partition=batch_gpu
#SBATCH --account=csb_gpu_acc
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# === Critical HPC safeguards from spec ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline

# Trap preemption signal -> save checkpoint -> requeue
trap 'echo "Caught USR1 — saving checkpoint..."; kill -INT $TRAIN_PID; wait $TRAIN_PID; scontrol requeue $SLURM_JOB_ID' USR1

# === Paths ===
WORK_DIR="/data/p_csb_meiler/agarwm5/landmarkdiff_work/LandmarkDiff"
DATA_DIR="${WORK_DIR}/data/synthetic_pairs"
CKPT_DIR="${WORK_DIR}/checkpoints"
WANDB_DIR="${WORK_DIR}/wandb"

mkdir -p "$CKPT_DIR" "$WANDB_DIR"

cd "$WORK_DIR"

# Activate conda environment
source /home/agarwm5/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

python scripts/train_controlnet.py \
    --data_dir=$DATA_DIR \
    --output_dir=$CKPT_DIR \
    --wandb_dir=$WANDB_DIR \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_steps=10000 \
    --checkpoint_every=5000 \
    --log_every=50 \
    --sample_every=1000 \
    --resume_from_checkpoint=latest \
    --phase=A &

TRAIN_PID=$!
wait $TRAIN_PID
