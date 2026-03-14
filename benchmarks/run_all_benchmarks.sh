#!/usr/bin/env bash
# Run all LandmarkDiff benchmarks in sequence.
#
# Usage:
#   bash benchmarks/run_all_benchmarks.sh              # full run
#   bash benchmarks/run_all_benchmarks.sh --quick       # fast run with fewer iterations
#   bash benchmarks/run_all_benchmarks.sh --device cpu   # force CPU
#
# Results are saved to benchmarks/results/ as JSON files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Defaults
QUICK=0
DEVICE="cuda"
NUM_IMAGES=50
NUM_LANDMARKS=200
NUM_STEPS=100
BATCH_SIZE=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK=1
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--device cuda|cpu] [--output DIR]"
            echo ""
            echo "Options:"
            echo "  --quick       Run with reduced iterations for fast testing"
            echo "  --device      Device to use (default: cuda)"
            echo "  --output      Output directory for results (default: benchmarks/results/)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Adjust for quick mode
if [[ $QUICK -eq 1 ]]; then
    NUM_IMAGES=5
    NUM_LANDMARKS=20
    NUM_STEPS=10
    BATCH_SIZE=2
    echo "=== Quick mode: reduced iterations ==="
fi

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "LandmarkDiff Benchmark Suite"
echo "========================================"
echo "Device:     $DEVICE"
echo "Results:    $RESULTS_DIR"
echo "Quick mode: $( [[ $QUICK -eq 1 ]] && echo yes || echo no )"
echo "Started:    $(date -Iseconds)"
echo "========================================"
echo ""

PASSED=0
FAILED=0
SKIPPED=0

# --- Benchmark 1: Landmark Extraction ---
echo "--- [1/3] Landmark Extraction ---"
if python "${SCRIPT_DIR}/benchmark_landmarks.py" \
    --num_images "$NUM_LANDMARKS" \
    --output "$RESULTS_DIR" 2>&1; then
    echo "  PASSED"
    PASSED=$((PASSED + 1))
else
    echo "  FAILED (exit code $?)"
    FAILED=$((FAILED + 1))
fi
echo ""

# --- Benchmark 2: Inference Pipeline ---
echo "--- [2/3] Inference Pipeline ---"
# Skip if cuda requested but not available
if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  SKIPPED (CUDA not available)"
        SKIPPED=$((SKIPPED + 1))
    else
        if python "${SCRIPT_DIR}/benchmark_inference.py" \
            --num_images "$NUM_IMAGES" \
            --device "$DEVICE" \
            --output "$RESULTS_DIR" 2>&1; then
            echo "  PASSED"
            PASSED=$((PASSED + 1))
        else
            echo "  FAILED (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
    fi
elif [[ "$DEVICE" == "cpu" ]]; then
    # CPU mode only supports tps
    if python "${SCRIPT_DIR}/benchmark_inference.py" \
        --num_images "$NUM_IMAGES" \
        --device cpu \
        --mode tps \
        --output "$RESULTS_DIR" 2>&1; then
        echo "  PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
else
    if python "${SCRIPT_DIR}/benchmark_inference.py" \
        --num_images "$NUM_IMAGES" \
        --device "$DEVICE" \
        --output "$RESULTS_DIR" 2>&1; then
        echo "  PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
fi
echo ""

# --- Benchmark 3: Training Throughput ---
echo "--- [3/3] Training Throughput ---"
if ! python -c "import torch" 2>/dev/null; then
    echo "  SKIPPED (PyTorch not installed)"
    SKIPPED=$((SKIPPED + 1))
elif [[ "$DEVICE" == "cuda" ]] && ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  SKIPPED (CUDA not available)"
    SKIPPED=$((SKIPPED + 1))
else
    if python "${SCRIPT_DIR}/benchmark_training.py" \
        --device "$DEVICE" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --output "$RESULTS_DIR" 2>&1; then
        echo "  PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
fi
echo ""

# --- Summary ---
echo "========================================"
echo "Benchmark Summary"
echo "========================================"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "  Results: $RESULTS_DIR/"
echo "  Finished: $(date -Iseconds)"
echo "========================================"

# List result files
if ls "$RESULTS_DIR"/*.json 1>/dev/null 2>&1; then
    echo ""
    echo "Result files:"
    for f in "$RESULTS_DIR"/*.json; do
        echo "  $f"
    done
fi

exit $FAILED
