#!/bin/bash
# LandmarkDiff Local Demo — runs entirely on Mac M3 Pro
# Downloads a sample face, runs full pipeline, shows results
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== LandmarkDiff Local Demo ==="
echo ""

# Step 1: Download a few FFHQ sample images
echo "[1/4] Downloading sample face images..."
python scripts/download_ffhq.py --num 10 --resolution 512 --output data/ffhq_samples 2>&1 | tail -3

# Step 2: Generate synthetic training pairs
echo ""
echo "[2/4] Generating synthetic training pairs..."
python scripts/generate_synthetic_data.py --input data/ffhq_samples --output data/demo_pairs --num 5 2>&1 | tail -5

# Step 3: Run visual demo on first image
echo ""
echo "[3/4] Running pipeline demo..."
FIRST_IMAGE=$(ls data/ffhq_samples/*.png 2>/dev/null | head -1)
if [ -n "$FIRST_IMAGE" ]; then
    python scripts/demo.py "$FIRST_IMAGE" --procedure rhinoplasty --intensity 60
    python scripts/demo.py "$FIRST_IMAGE" --procedure blepharoplasty --intensity 50
else
    echo "No images found, running synthetic demo..."
    python scripts/demo.py
fi

# Step 4: Show results
echo ""
echo "[4/4] Results:"
echo ""
echo "  Pipeline outputs:    scripts/demo_output/"
echo "  Synthetic pairs:     data/demo_pairs/"
echo ""
echo "  Open in Finder:"
echo "    open scripts/demo_output/"
echo "    open data/demo_pairs/"
echo ""
echo "  To run ControlNet inference (generates actual before/after):"
echo "    python landmarkdiff/inference.py data/ffhq_samples/000000.png"
echo ""
echo "=== Done ==="
