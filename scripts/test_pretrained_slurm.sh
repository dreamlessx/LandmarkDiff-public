#!/bin/bash
#SBATCH --job-name=tesla_test
#SBATCH --partition=batch_gpu
#SBATCH --account=your_gpu_acc
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-test-%j.out

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

python scripts/test_pretrained.py
