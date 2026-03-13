#!/bin/bash
#SBATCH --job-name=tesla_process
#SBATCH --partition=batch
#SBATCH --account=YOUR_GROUP
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-process-%j.out

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "Processing all scraped images at $(date)"

python scripts/process_real_surgery.py \
    --raw_dir data/real_surgery_pairs/raw \
    --output data/real_surgery_pairs \
    --size 512

echo "Completed processing at $(date)"

# Count results
echo "=== Final Counts ==="
for d in data/real_surgery_pairs/pairs/*/; do
    proc=$(basename "$d")
    count=$(find "$d" -name "*_target.*" 2>/dev/null | wc -l)
    echo "  $proc: $count pairs"
done
