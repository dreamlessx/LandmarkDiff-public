#!/bin/bash
#SBATCH --job-name=surgery_gen_pairs
#SBATCH --partition=batch
#SBATCH --account=p_csb_meiler
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-gen-%j.out

# Generate synthetic training pairs (CPU only, no GPU needed)

WORK_DIR="/data/p_csb_meiler/agarwm5/landmarkdiff_work/LandmarkDiff"
cd "$WORK_DIR"

source /home/agarwm5/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "Starting pair generation at $(date)"
echo "Source images:"
ls -d data/faces_multi/*/  2>/dev/null | while read d; do echo "  $d: $(ls "$d"/*.png 2>/dev/null | wc -l) images"; done
ls -d data/ffhq/ 2>/dev/null && echo "  data/ffhq: $(ls data/ffhq/*.png 2>/dev/null | wc -l) images"

python scripts/generate_pairs_scaled.py \
    --num 10000 \
    --data_root=data \
    --output=data/synthetic_pairs_v2 \
    --workers=1 \
    --seed=42

echo "Pair generation complete at $(date)"
echo "Total pairs: $(ls data/synthetic_pairs_v2/*_input.png 2>/dev/null | wc -l)"
