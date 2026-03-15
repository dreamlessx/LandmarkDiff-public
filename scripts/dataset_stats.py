"""Dataset statistics and quality analysis.

Analyzes training datasets for quality assurance before launching
training runs. Computes distribution statistics, checks for corruption,
and generates summary visualizations.

Usage:
    # Full analysis
    python scripts/dataset_stats.py --data_dir data/training_combined

    # Quick check (first 200 samples)
    python scripts/dataset_stats.py --data_dir data/training_combined --max_samples 200

    # Analyze specific wave
    python scripts/dataset_stats.py --data_dir data/synthetic_surgery_pairs_v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def analyze_dataset(
    data_dir: str,
    output_dir: str = "dataset_stats",
    max_samples: int = 0,
) -> dict:
    """Run comprehensive dataset analysis."""
    d = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not d.exists():
        print(f"ERROR: Directory not found: {d}")
        sys.exit(1)

    t0 = time.time()

    # Count files
    input_files = sorted(d.glob("*_input.png"))
    target_files = sorted(d.glob("*_target.png"))
    cond_files = sorted(d.glob("*_conditioning.png"))
    mask_files = sorted(d.glob("*_mask.png"))

    n_inputs = len(input_files)
    n_targets = len(target_files)
    n_conds = len(cond_files)
    n_masks = len(mask_files)

    print(f"Dataset: {d}")
    print(f"  Input files: {n_inputs:,}")
    print(f"  Target files: {n_targets:,}")
    print(f"  Conditioning files: {n_conds:,}")
    print(f"  Mask files: {n_masks:,}")

    # Completeness check
    input_prefixes = {f.stem.replace("_input", "") for f in input_files}
    target_prefixes = {f.stem.replace("_target", "") for f in target_files}
    cond_prefixes = {f.stem.replace("_conditioning", "") for f in cond_files}

    missing_targets = input_prefixes - target_prefixes
    missing_conds = input_prefixes - cond_prefixes

    if missing_targets:
        print(f"  WARNING: {len(missing_targets)} inputs missing targets")
    if missing_conds:
        print(f"  WARNING: {len(missing_conds)} inputs missing conditioning")

    # Load metadata if available
    meta_path = d / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"  Metadata: {len(metadata.get('pairs', {}))} entries")

    # Sample analysis
    samples_to_analyze = input_files
    if max_samples > 0:
        samples_to_analyze = input_files[:max_samples]

    print(f"\nAnalyzing {len(samples_to_analyze)} samples...")

    # Statistics collectors
    procedure_counts = Counter()
    wave_counts = Counter()
    intensity_values = []
    image_shapes = []
    corrupted = []
    pixel_stats = {"mean": [], "std": [], "min": [], "max": []}
    conditioning_stats = {"mean": [], "nonzero_ratio": []}
    mask_coverage = []
    fitzpatrick_counts = Counter()

    pairs_meta = metadata.get("pairs", {})

    for i, inp_path in enumerate(samples_to_analyze):
        prefix = inp_path.stem.replace("_input", "")

        # Read input image
        img = cv2.imread(str(inp_path))
        if img is None:
            corrupted.append(prefix)
            continue

        image_shapes.append(img.shape[:2])

        # Pixel statistics
        pixel_stats["mean"].append(float(img.mean()))
        pixel_stats["std"].append(float(img.std()))
        pixel_stats["min"].append(float(img.min()))
        pixel_stats["max"].append(float(img.max()))

        # Fitzpatrick classification (subsample for speed)
        if i % 10 == 0:
            try:
                from landmarkdiff.evaluation import classify_fitzpatrick_ita

                fitz = classify_fitzpatrick_ita(img)
                fitzpatrick_counts[fitz] += 1
            except Exception:
                pass

        # Conditioning analysis
        cond_path = d / f"{prefix}_conditioning.png"
        if cond_path.exists():
            cond = cv2.imread(str(cond_path))
            if cond is not None:
                conditioning_stats["mean"].append(float(cond.mean()))
                nonzero = (cond > 10).sum() / cond.size
                conditioning_stats["nonzero_ratio"].append(float(nonzero))

        # Mask analysis
        mask_path = d / f"{prefix}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                coverage = (mask > 128).sum() / mask.size
                mask_coverage.append(float(coverage))

        # Metadata-based stats
        if prefix in pairs_meta:
            meta = pairs_meta[prefix]
            procedure_counts[meta.get("procedure", "unknown")] += 1
            wave_counts[meta.get("wave", "unknown")] += 1
            if "intensity" in meta:
                intensity_values.append(float(meta["intensity"]))
        else:
            # Try to infer procedure from filename
            for proc in PROCEDURES:
                if proc in prefix:
                    procedure_counts[proc] += 1
                    break
            else:
                procedure_counts["unknown"] += 1

        if (i + 1) % 500 == 0:
            print(f"  Analyzed {i + 1}/{len(samples_to_analyze)}")

    # Compile results
    results = {
        "data_dir": str(d),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_counts": {
            "inputs": n_inputs,
            "targets": n_targets,
            "conditioning": n_conds,
            "masks": n_masks,
        },
        "completeness": {
            "missing_targets": len(missing_targets),
            "missing_conditioning": len(missing_conds),
            "corrupted": len(corrupted),
        },
        "samples_analyzed": len(samples_to_analyze) - len(corrupted),
        "procedure_distribution": dict(procedure_counts.most_common()),
        "wave_distribution": dict(wave_counts.most_common()),
    }

    # Image statistics
    if image_shapes:
        shapes = np.array(image_shapes)
        results["image_stats"] = {
            "unique_shapes": len(set(map(tuple, shapes))),
            "modal_shape": list(map(int, Counter(map(tuple, shapes)).most_common(1)[0][0])),
            "pixel_mean": round(float(np.mean(pixel_stats["mean"])), 2),
            "pixel_std": round(float(np.mean(pixel_stats["std"])), 2),
        }

    # Intensity statistics
    if intensity_values:
        results["intensity_stats"] = {
            "mean": round(float(np.mean(intensity_values)), 4),
            "std": round(float(np.std(intensity_values)), 4),
            "min": round(float(np.min(intensity_values)), 4),
            "max": round(float(np.max(intensity_values)), 4),
            "median": round(float(np.median(intensity_values)), 4),
        }

    # Conditioning statistics
    if conditioning_stats["mean"]:
        results["conditioning_stats"] = {
            "mean_pixel_value": round(float(np.mean(conditioning_stats["mean"])), 2),
            "mean_nonzero_ratio": round(float(np.mean(conditioning_stats["nonzero_ratio"])), 4),
        }

    # Mask statistics
    if mask_coverage:
        results["mask_stats"] = {
            "mean_coverage": round(float(np.mean(mask_coverage)), 4),
            "std_coverage": round(float(np.std(mask_coverage)), 4),
            "min_coverage": round(float(np.min(mask_coverage)), 4),
            "max_coverage": round(float(np.max(mask_coverage)), 4),
        }

    # Fitzpatrick distribution (scaled up from subsample)
    if fitzpatrick_counts:
        total_fitz = sum(fitzpatrick_counts.values())
        results["fitzpatrick_distribution"] = {
            k: {
                "count": v,
                "estimated_total": round(v * (len(samples_to_analyze) / max(total_fitz * 10, 1))),
                "proportion": round(v / total_fitz, 3),
            }
            for k, v in sorted(fitzpatrick_counts.items())
        }

    elapsed = time.time() - t0
    results["analysis_time_seconds"] = round(elapsed, 1)

    # Save results
    results_path = out / "dataset_stats.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  Dataset Statistics Summary")
    print(f"{'=' * 60}")
    print(f"  Total pairs: {n_inputs:,}")
    print(f"  Complete (all 4 files): {n_inputs - len(missing_targets) - len(missing_conds):,}")
    print(f"  Corrupted: {len(corrupted)}")
    print()

    print("  Procedure distribution:")
    total = sum(procedure_counts.values())
    for proc, count in procedure_counts.most_common():
        pct = count / max(total, 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {proc:<20} {count:>6,} ({pct:5.1f}%) {bar}")

    if wave_counts:
        print("\n  Wave distribution:")
        for wave, count in wave_counts.most_common():
            print(f"    Wave {wave}: {count:,}")

    if "intensity_stats" in results:
        s = results["intensity_stats"]
        print(
            f"\n  Intensity: mean={s['mean']:.3f}, std={s['std']:.3f}, "
            f"range=[{s['min']:.3f}, {s['max']:.3f}]"
        )

    if "mask_stats" in results:
        s = results["mask_stats"]
        print(
            f"  Mask coverage: mean={s['mean_coverage']:.3f}, "
            f"range=[{s['min_coverage']:.3f}, {s['max_coverage']:.3f}]"
        )

    if fitzpatrick_counts:
        print("\n  Fitzpatrick distribution (estimated from subsample):")
        for ftype, data in sorted(results.get("fitzpatrick_distribution", {}).items()):
            print(
                f"    Type {ftype}: ~{data['estimated_total']:,} ({data['proportion'] * 100:.1f}%)"
            )

    print(f"\n  Analysis time: {elapsed:.1f}s")
    print(f"  Results saved: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset statistics and analysis")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--output", default="dataset_stats", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to analyze (0=all)")
    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.output, args.max_samples)
