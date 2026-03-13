#!/bin/bash
#SBATCH --job-name=tesla_mega
#SBATCH --partition=batch
#SBATCH --account=p_csb_meiler
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-mega-%A_%a.out
#SBATCH --array=0-3

WORK_DIR="/data/p_csb_meiler/agarwm5/landmarkdiff_work/LandmarkDiff"
cd "$WORK_DIR"

source /home/agarwm5/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

# Map array index to procedure
PROCEDURES=("rhinoplasty" "blepharoplasty" "rhytidectomy" "orthognathic")
PROC=${PROCEDURES[$SLURM_ARRAY_TASK_ID]}

# Chrome shared libs for compute nodes
export LD_LIBRARY_PATH="$WORK_DIR/tools/chrome/chrome-bin:${LD_LIBRARY_PATH:-}"

echo "Starting mega scrape for $PROC at $(date)"
echo "SLURM Job: $SLURM_JOB_ID, Array: $SLURM_ARRAY_TASK_ID"

python scripts/mega_scrape.py \
    --output data/real_surgery_pairs/raw \
    --target 50000 \
    --procedure "$PROC"

echo "Completed $PROC at $(date)"

# Count results
count=$(ls "data/real_surgery_pairs/raw/$PROC/" 2>/dev/null | wc -l)
echo "Final count for $PROC: $count images"
