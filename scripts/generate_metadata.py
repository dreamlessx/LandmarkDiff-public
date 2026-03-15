#!/usr/bin/env python3
"""Generate metadata.json for training datasets that lack it.

Infers procedure type, wave, and difficulty from filenames and image
properties. Essential for curriculum learning, stratified splitting,
and training data analysis.

Usage:
    python scripts/generate_metadata.py --data_dir data/training_combined
    python scripts/generate_metadata.py --data_dir data/training_combined --sample 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def infer_procedure(prefix: str) -> str:
    """Infer surgical procedure from filename."""
    prefix_lower = prefix.lower()
    for proc in PROCEDURES:
        if proc in prefix_lower:
            return proc
        # Handle abbreviations
        if proc[:5] in prefix_lower:
            return proc

    # Common patterns
    if "rhino" in prefix_lower or "nose" in prefix_lower:
        return "rhinoplasty"
    if "bleph" in prefix_lower or "eye" in prefix_lower:
        return "blepharoplasty"
    if "rhytid" in prefix_lower or "face" in prefix_lower or "lift" in prefix_lower:
        return "rhytidectomy"
    if "ortho" in prefix_lower or "jaw" in prefix_lower:
        return "orthognathic"

    return "unknown"


def infer_wave(prefix: str) -> str:
    """Infer data generation wave from filename patterns."""
    if "_v3_" in prefix or "_realistic" in prefix:
        return "wave3_realistic"
    if "_v2_" in prefix or "_displacement" in prefix:
        return "wave2_displacement"
    if "_v1_" in prefix:
        return "wave1_basic"
    # Check for numeric index patterns (e.g., pair_00123)
    match = re.search(r"pair_(\d+)", prefix)
    if match:
        idx = int(match.group(1))
        # Rough heuristic: earlier indices = earlier waves
        if idx < 5000:
            return "wave1_basic"
        elif idx < 15000:
            return "wave2_displacement"
        else:
            return "wave3_realistic"
    return "unknown"


def estimate_displacement_intensity(
    input_img: np.ndarray,
    cond_img: np.ndarray,
) -> float:
    """Estimate displacement intensity from input/conditioning difference.

    Uses structural difference between input and conditioning images
    to estimate how much displacement was applied.

    Returns intensity estimate in [0, 1].
    """
    if input_img.shape != cond_img.shape:
        cond_img = cv2.resize(cond_img, (input_img.shape[1], input_img.shape[0]))

    # Compute structural difference
    gray_in = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_cond = cv2.cvtColor(cond_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(gray_in - gray_cond)
    mean_diff = np.mean(diff) / 255.0

    # Map to [0, 1] — typical range is 0.01 to 0.15
    intensity = min(1.0, mean_diff / 0.15)
    return float(intensity)


def generate_metadata(
    data_dir: str,
    sample_size: int = 0,
    output_path: str | None = None,
) -> dict:
    """Generate metadata.json for a training dataset."""
    data_path = Path(data_dir)
    input_files = sorted(data_path.glob("*_input.png"))
    total = len(input_files)

    if sample_size > 0 and sample_size < total:
        rng = np.random.default_rng(42)
        indices = rng.choice(total, size=sample_size, replace=False)
        sampled_files = [input_files[i] for i in sorted(indices)]
    else:
        sampled_files = input_files

    print(f"Dataset: {data_dir}")
    print(f"Total pairs: {total}")
    print(f"Processing: {len(sampled_files)} pairs")

    pairs = {}
    proc_counts = defaultdict(int)
    wave_counts = defaultdict(int)
    intensity_vals = []

    for i, inp_file in enumerate(sampled_files):
        prefix = inp_file.stem.replace("_input", "")

        procedure = infer_procedure(prefix)
        wave = infer_wave(prefix)

        # Estimate intensity if conditioning exists
        intensity = 0.5  # default
        cond_path = data_path / f"{prefix}_conditioning.png"
        if cond_path.exists():
            input_img = cv2.imread(str(inp_file))
            cond_img = cv2.imread(str(cond_path))
            if input_img is not None and cond_img is not None:
                intensity = estimate_displacement_intensity(input_img, cond_img)

        pairs[prefix] = {
            "procedure": procedure,
            "wave": wave,
            "intensity": round(intensity, 4),
            "source": "synthetic",
        }

        proc_counts[procedure] += 1
        wave_counts[wave] += 1
        intensity_vals.append(intensity)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(sampled_files)}]...")

    # If we only sampled, extrapolate to full dataset
    # by assigning metadata based on sampling proportions
    if sample_size > 0 and sample_size < total:
        print(f"\nExtrapolating to full dataset ({total} pairs)...")
        # For non-sampled files, assign procedure randomly based on observed distribution
        total_sampled = sum(proc_counts.values())
        proc_probs = {p: c / total_sampled for p, c in proc_counts.items()}
        rng = np.random.default_rng(123)

        for inp_file in input_files:
            prefix = inp_file.stem.replace("_input", "")
            if prefix not in pairs:
                # Assign based on filename first, then random
                proc = infer_procedure(prefix)
                if proc == "unknown":
                    proc = rng.choice(
                        list(proc_probs.keys()),
                        p=list(proc_probs.values()),
                    )
                wave = infer_wave(prefix)
                pairs[prefix] = {
                    "procedure": proc,
                    "wave": wave,
                    "intensity": round(float(rng.normal(0.5, 0.15).clip(0.1, 1.0)), 4),
                    "source": "synthetic",
                }

    metadata = {
        "total_pairs": total,
        "generated_by": "generate_metadata.py",
        "procedure_distribution": dict(sorted(proc_counts.items())),
        "wave_distribution": dict(sorted(wave_counts.items())),
        "intensity_stats": {
            "mean": float(np.mean(intensity_vals)) if intensity_vals else 0,
            "std": float(np.std(intensity_vals)) if intensity_vals else 0,
            "min": float(np.min(intensity_vals)) if intensity_vals else 0,
            "max": float(np.max(intensity_vals)) if intensity_vals else 0,
        },
        "pairs": pairs,
    }

    # Print summary
    print(f"\n{'=' * 50}")
    print("METADATA SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total entries: {len(pairs)}")
    print("\nProcedure Distribution:")
    for proc, count in sorted(proc_counts.items()):
        pct = count / len(sampled_files) * 100
        print(f"  {proc:<20} {count:>6} ({pct:5.1f}%)")
    print("\nWave Distribution:")
    for wave, count in sorted(wave_counts.items()):
        print(f"  {wave:<25} {count:>6}")
    if intensity_vals:
        print(f"\nIntensity: {np.mean(intensity_vals):.3f} +/- {np.std(intensity_vals):.3f}")

    # Save
    if output_path is None:
        output_path = str(data_path / "metadata.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved to {output_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for training data")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--sample", type=int, default=0, help="Sample size for intensity estimation (0=all)"
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: data_dir/metadata.json)"
    )
    args = parser.parse_args()

    generate_metadata(args.data_dir, args.sample, args.output)
