#!/bin/bash
#SBATCH --job-name=tesla_scrape
#SBATCH --partition=batch
#SBATCH --account=YOUR_GROUP
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-scrape-%j.out

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "Starting mass scrape at $(date)"
echo "Target: 50000 images per procedure"

python scripts/mass_scrape.py \
    --output data/real_surgery_pairs/raw \
    --target 50000 \
    --procedures rhinoplasty blepharoplasty rhytidectomy orthognathic

echo "Scrape completed at $(date)"

# Count results
echo "=== Final Counts ==="
for d in data/real_surgery_pairs/raw/*/; do
    proc=$(basename "$d")
    count=$(ls "$d" | wc -l)
    echo "  $proc: $count"
done
