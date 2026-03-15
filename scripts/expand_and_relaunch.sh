#!/bin/bash
#SBATCH --job-name=tesla_expand
#SBATCH --partition=batch
#SBATCH --account=YOUR_GROUP
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-expand-%j.out

# Wait for CelebA extraction to finish, then merge all faces
# and relaunch synthetic generation with the full pool

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "=== Expanding face pool at $(date) ==="

# Create mega face directory with symlinks
MEGA_DIR="data/faces_mega"
mkdir -p "$MEGA_DIR"

# Link all existing faces
echo "Linking faces_all..."
for f in data/faces_all/*.{jpg,jpeg,png}; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/$(basename $f)" 2>/dev/null
done

echo "Linking celeba_hq_extracted..."
for f in data/celeba_hq_extracted/*.{jpg,jpeg,png}; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/celeba_hq_$(basename $f)" 2>/dev/null
done

echo "Linking ffhq..."
for f in data/ffhq/*.{jpg,jpeg,png}; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/ffhq_$(basename $f)" 2>/dev/null
done

echo "Linking celeba_hq..."
for f in data/faces_multi/celeba_hq/*.{jpg,jpeg,png}; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/multihq_$(basename $f)" 2>/dev/null
done

echo "Linking celeba..."
for f in data/faces_multi/celeba/*.{jpg,jpeg,png}; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/multi_$(basename $f)" 2>/dev/null
done

# Also link botox before/after images
echo "Linking botox..."
for f in data/hf_botox/data/extracted/before/*.png; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/botox_b_$(basename $f)" 2>/dev/null
done
for f in data/hf_botox/data/extracted/after/*.png; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$MEGA_DIR/botox_a_$(basename $f)" 2>/dev/null
done

TOTAL=$(ls "$MEGA_DIR" | wc -l)
echo "Total faces in mega pool: $TOTAL"

# Now submit the wave 2 synthetic generation jobs
echo "=== Submitting wave 2 synthetic generation at $(date) ==="

cat > ${WORK_DIR}/synth_wave2.sh << 'INNEREOF'
#!/bin/bash
#SBATCH --job-name=tesla_synth2
#SBATCH --partition=batch
#SBATCH --account=YOUR_GROUP
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=slurm-synth2-%A_%a.out
#SBATCH --array=0-3

WORK_DIR="/path/to/LandmarkDiff"
cd "$WORK_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

PROCEDURES=("rhinoplasty" "blepharoplasty" "rhytidectomy" "orthognathic" "brow_lift" "mentoplasty")
PROC=${PROCEDURES[$SLURM_ARRAY_TASK_ID]}

echo "Wave 2: Generating synthetic pairs for $PROC at $(date)"

# Use the mega face pool
python scripts/generate_synthetic_pairs.py \
    --source_dir data/faces_mega \
    --output data/synthetic_surgery_pairs_v2 \
    --procedure "$PROC" \
    --target 50000 \
    --size 512

echo "Completed $PROC at $(date)"
INNEREOF

sbatch ${WORK_DIR}/synth_wave2.sh
echo "Wave 2 submitted at $(date)"
