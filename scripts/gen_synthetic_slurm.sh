#!/bin/bash
#SBATCH --job-name=tesla_synth
#SBATCH --partition=batch
#SBATCH --account=YOUR_GROUP
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-synth-%A_%a.out
#SBATCH --array=0-3

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

PROCEDURES=("rhinoplasty" "blepharoplasty" "rhytidectomy" "orthognathic" "brow_lift" "mentoplasty")
PROC=${PROCEDURES[$SLURM_ARRAY_TASK_ID]}

echo "Generating synthetic pairs for $PROC at $(date)"
echo "SLURM Job: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID"

# Use the combined faces_all directory (14K images)
# Generate ~13K pairs per procedure (with varying intensities)
python scripts/generate_synthetic_pairs.py \
    --source_dir data/faces_all \
    --output data/synthetic_surgery_pairs \
    --procedure "$PROC" \
    --target 50000 \
    --size 512

echo "Completed $PROC at $(date)"
count=$(ls "data/synthetic_surgery_pairs/$PROC/" 2>/dev/null | wc -l)
echo "Final file count for $PROC: $count"
