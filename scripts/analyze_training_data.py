#!/usr/bin/env python3
"""Analyze training dataset for quality, balance, and readiness.

Generates comprehensive statistics about the training data including:
1. Total pair count and file integrity
2. Procedure distribution (from metadata)
3. Fitzpatrick skin type distribution (via ITA classification)
4. Image quality statistics (resolution, brightness)
5. Conditioning signal quality check

Usage:
    python scripts/analyze_training_data.py --data_dir data/training_combined
    python scripts/analyze_training_data.py --data_dir data/training_combined --sample 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_ita(image: np.ndarray) -> float:
    """Compute Individual Typology Angle (ITA) for Fitzpatrick classification."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0]
    b = lab[:, :, 2]
    L_mean = np.mean(L)
    b_mean = np.mean(b)
    if abs(b_mean) < 1e-6:
        return 90.0
    ita = np.degrees(np.arctan2(L_mean - 50, b_mean))
    return float(ita)


def classify_fitzpatrick_from_ita(ita: float) -> str:
    """Map ITA angle to Fitzpatrick skin type."""
    if ita > 55:
        return "I"
    elif ita > 41:
        return "II"
    elif ita > 28:
        return "III"
    elif ita > 10:
        return "IV"
    elif ita > -30:
        return "V"
    else:
        return "VI"


def analyze_dataset(
    data_dir: str,
    sample_size: int = 0,
    output_path: str | None = None,
) -> dict:
    """Comprehensive training data analysis."""
    data_path = Path(data_dir)

    # Find all pairs
    input_files = sorted(data_path.glob("*_input.png"))
    total_pairs = len(input_files)

    if sample_size > 0 and sample_size < total_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_pairs, size=sample_size, replace=False)
        input_files = [input_files[i] for i in sorted(indices)]

    print(f"Dataset: {data_dir}")
    print(f"Total pairs: {total_pairs}")
    print(f"Analyzing: {len(input_files)} pairs")

    # Load metadata if available
    meta_path = data_path / "metadata.json"
    pairs_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        pairs_meta = meta.get("pairs", {})
        print(f"Metadata: {len(pairs_meta)} entries")

    # Analysis containers
    proc_counts = defaultdict(int)
    wave_counts = defaultdict(int)
    fitz_counts = defaultdict(int)
    resolutions = []
    brightness_values = []
    has_mask_count = 0
    has_cond_count = 0
    missing_target = 0
    cond_target_identical = 0
    quality_issues = []

    for i, inp_file in enumerate(input_files):
        prefix = inp_file.stem.replace("_input", "")

        # Check file existence
        target_file = data_path / f"{prefix}_target.png"
        cond_file = data_path / f"{prefix}_conditioning.png"
        mask_file = data_path / f"{prefix}_mask.png"

        if not target_file.exists():
            missing_target += 1
            continue

        if cond_file.exists():
            has_cond_count += 1
        if mask_file.exists():
            has_mask_count += 1

        # Procedure from metadata
        if prefix in pairs_meta:
            proc = pairs_meta[prefix].get("procedure", "unknown")
            wave = pairs_meta[prefix].get("wave", "unknown")
        else:
            proc = "unknown"
            for p in [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ]:
                if p in prefix:
                    proc = p
                    break
            wave = "unknown"

        proc_counts[proc] += 1
        wave_counts[wave] += 1

        # Read input image for quality analysis
        img = cv2.imread(str(inp_file))
        if img is None:
            quality_issues.append(f"{prefix}: unreadable input")
            continue

        h, w = img.shape[:2]
        resolutions.append((h, w))

        # Brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

        # Fitzpatrick via ITA
        ita = compute_ita(img)
        fitz = classify_fitzpatrick_from_ita(ita)
        fitz_counts[fitz] += 1

        # Check conditioning = target (often indicates fallback)
        if cond_file.exists():
            cond = cv2.imread(str(cond_file))
            target = cv2.imread(str(target_file))
            if cond is not None and target is not None:
                if cond.shape == target.shape and np.allclose(cond, target, atol=1):
                    cond_target_identical += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(input_files)}]...")

    # Compute statistics
    res_arr = np.array(resolutions)
    results = {
        "data_dir": str(data_dir),
        "total_pairs": total_pairs,
        "analyzed": len(input_files),
        "missing_target": missing_target,
        "has_conditioning": has_cond_count,
        "has_mask": has_mask_count,
        "cond_target_identical": cond_target_identical,
        "quality_issues": len(quality_issues),
        "procedure_distribution": dict(sorted(proc_counts.items())),
        "wave_distribution": dict(sorted(wave_counts.items())),
        "fitzpatrick_distribution": dict(sorted(fitz_counts.items())),
        "resolution": {
            "min_h": int(res_arr[:, 0].min()) if len(res_arr) else 0,
            "max_h": int(res_arr[:, 0].max()) if len(res_arr) else 0,
            "min_w": int(res_arr[:, 1].min()) if len(res_arr) else 0,
            "max_w": int(res_arr[:, 1].max()) if len(res_arr) else 0,
            "mode_h": int(np.median(res_arr[:, 0])) if len(res_arr) else 0,
            "mode_w": int(np.median(res_arr[:, 1])) if len(res_arr) else 0,
        },
        "brightness": {
            "mean": float(np.mean(brightness_values)) if brightness_values else 0,
            "std": float(np.std(brightness_values)) if brightness_values else 0,
            "min": float(np.min(brightness_values)) if brightness_values else 0,
            "max": float(np.max(brightness_values)) if brightness_values else 0,
        },
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING DATA ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Total pairs: {total_pairs}")
    print(f"Missing targets: {missing_target}")
    print(f"Has conditioning: {has_cond_count}/{len(input_files)}")
    print(f"Has mask: {has_mask_count}/{len(input_files)}")
    print(f"Cond=Target (fallback): {cond_target_identical}")

    print("\nProcedure Distribution:")
    for proc, count in sorted(proc_counts.items()):
        pct = count / len(input_files) * 100
        bar = "#" * int(pct / 2)
        print(f"  {proc:<20} {count:>6} ({pct:5.1f}%) {bar}")

    print("\nWave Distribution:")
    for wave, count in sorted(wave_counts.items()):
        print(f"  {wave:<20} {count:>6}")

    print("\nFitzpatrick Distribution:")
    total_fitz = sum(fitz_counts.values())
    for ftype in ["I", "II", "III", "IV", "V", "VI"]:
        count = fitz_counts.get(ftype, 0)
        pct = count / total_fitz * 100 if total_fitz else 0
        bar = "#" * int(pct / 2)
        print(f"  Type {ftype}: {count:>6} ({pct:5.1f}%) {bar}")

    print("\nImage Quality:")
    print(
        f"  Resolution: {results['resolution']['mode_h']}x{results['resolution']['mode_w']} (median)"
    )
    print(
        f"  Brightness: {results['brightness']['mean']:.1f} +/- {results['brightness']['std']:.1f}"
    )

    # Equity assessment
    if fitz_counts:
        max_fitz = max(fitz_counts.values())
        min_fitz = min(v for v in fitz_counts.values() if v > 0)
        ratio = max_fitz / max(min_fitz, 1)
        print("\nEquity Assessment:")
        print(f"  Max/Min Fitzpatrick ratio: {ratio:.1f}x")
        if ratio > 5:
            print("  WARNING: Significant skin tone imbalance (>5x)")
        elif ratio > 3:
            print("  CAUTION: Moderate skin tone imbalance (>3x)")
        else:
            print("  GOOD: Reasonable skin tone balance (<3x)")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training data quality")
    parser.add_argument("--data_dir", default="data/training_combined")
    parser.add_argument("--sample", type=int, default=0, help="Sample size (0=all)")
    parser.add_argument("--output", default="results/training_data_analysis.json")
    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.sample, args.output)
