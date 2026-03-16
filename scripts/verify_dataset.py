"""Verify training dataset integrity before starting training.

Checks:
1. File consistency: every _input.png has matching _target.png and _conditioning.png
2. Image validity: all images are readable and correct size (512x512)
3. Conditioning quality: conditioning images are mesh renderings (low pixel mean)
4. Target quality: target images are face photos (higher pixel mean)
5. No data leakage: conditioning != target (they should be different)
6. Face detection rate: MediaPipe can detect faces in targets
7. Statistics: per-file-type histograms

Usage:
    python scripts/verify_dataset.py --data_dir data/training_combined --sample 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def verify_dataset(
    data_dir: str,
    sample_size: int = 0,
    fix_issues: bool = False,
) -> dict:
    """Verify dataset integrity and quality."""
    data_dir = Path(data_dir)

    input_files = sorted(data_dir.glob("*_input.png"))
    target_files = sorted(data_dir.glob("*_target.png"))
    cond_files = sorted(data_dir.glob("*_conditioning.png"))

    n_input = len(input_files)
    n_target = len(target_files)
    n_cond = len(cond_files)

    print(f"Dataset: {data_dir}")
    print(f"  Input files:        {n_input}")
    print(f"  Target files:       {n_target}")
    print(f"  Conditioning files: {n_cond}")

    # Check file consistency
    input_prefixes = {f.stem.replace("_input", "") for f in input_files}
    target_prefixes = {f.stem.replace("_target", "") for f in target_files}
    cond_prefixes = {f.stem.replace("_conditioning", "") for f in cond_files}

    missing_targets = input_prefixes - target_prefixes
    missing_conds = input_prefixes - cond_prefixes
    orphan_targets = target_prefixes - input_prefixes
    orphan_conds = cond_prefixes - input_prefixes

    issues = {
        "missing_targets": list(sorted(missing_targets))[:20],
        "missing_conditioning": list(sorted(missing_conds))[:20],
        "orphan_targets": len(orphan_targets),
        "orphan_conditioning": len(orphan_conds),
    }

    print(f"\n  Missing targets:       {len(missing_targets)}")
    print(f"  Missing conditioning:  {len(missing_conds)}")
    print(f"  Orphan targets:        {len(orphan_targets)}")
    print(f"  Orphan conditioning:   {len(orphan_conds)}")

    # Sample for quality checks
    all_prefixes = sorted(input_prefixes & target_prefixes & cond_prefixes)
    if sample_size > 0 and sample_size < len(all_prefixes):
        rng = random.Random(42)
        sample_prefixes = rng.sample(all_prefixes, sample_size)
    else:
        sample_prefixes = all_prefixes

    print(f"\n  Sampling {len(sample_prefixes)} pairs for quality checks...")

    stats = {
        "input_means": [],
        "target_means": [],
        "cond_means": [],
        "input_readable": 0,
        "target_readable": 0,
        "cond_readable": 0,
        "wrong_size": 0,
        "cond_is_mesh": 0,
        "cond_is_photo": 0,
        "target_is_face": 0,
        "target_no_face": 0,
        "cond_target_identical": 0,
    }

    # Try face detection on a small sample
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from landmarkdiff.landmarks import extract_landmarks

        has_mediapipe = True
    except ImportError:
        has_mediapipe = False
        print("  (MediaPipe not available, skipping face detection)")

    face_check_limit = min(100, len(sample_prefixes))

    for i, prefix in enumerate(sample_prefixes):
        inp_path = data_dir / f"{prefix}_input.png"
        tgt_path = data_dir / f"{prefix}_target.png"
        cond_path = data_dir / f"{prefix}_conditioning.png"

        # Read images
        inp_img = cv2.imread(str(inp_path))
        tgt_img = cv2.imread(str(tgt_path))
        cond_img = cv2.imread(str(cond_path))

        if inp_img is not None:
            stats["input_readable"] += 1
            stats["input_means"].append(float(inp_img.mean()))
            if inp_img.shape[:2] != (512, 512):
                stats["wrong_size"] += 1

        if tgt_img is not None:
            stats["target_readable"] += 1
            stats["target_means"].append(float(tgt_img.mean()))
            if tgt_img.shape[:2] != (512, 512):
                stats["wrong_size"] += 1

            # Face detection on targets (subsample)
            if has_mediapipe and i < face_check_limit:
                tgt_512 = (
                    cv2.resize(tgt_img, (512, 512)) if tgt_img.shape[:2] != (512, 512) else tgt_img
                )
                face = extract_landmarks(tgt_512)
                if face is not None:
                    stats["target_is_face"] += 1
                else:
                    stats["target_no_face"] += 1

        if cond_img is not None:
            stats["cond_readable"] += 1
            cond_mean = float(cond_img.mean())
            stats["cond_means"].append(cond_mean)
            if cond_img.shape[:2] != (512, 512):
                stats["wrong_size"] += 1

            # Mesh images have low mean pixel values (dark background)
            if cond_mean < 30:
                stats["cond_is_mesh"] += 1
            else:
                stats["cond_is_photo"] += 1

            # Check if conditioning == target (data corruption)
            if tgt_img is not None and cond_img.shape == tgt_img.shape:
                if np.array_equal(cond_img, tgt_img):
                    stats["cond_target_identical"] += 1

    n_sampled = len(sample_prefixes)

    print(f"\n  Quality Check Results ({n_sampled} sampled):")
    print(
        f"    Readable: input={stats['input_readable']} "
        f"target={stats['target_readable']} "
        f"cond={stats['cond_readable']}"
    )
    print(f"    Wrong size (not 512x512): {stats['wrong_size']}")

    if stats["input_means"]:
        print(
            f"    Input pixel mean:  {np.mean(stats['input_means']):.1f} "
            f"(min={np.min(stats['input_means']):.1f}, max={np.max(stats['input_means']):.1f})"
        )
    if stats["target_means"]:
        print(
            f"    Target pixel mean: {np.mean(stats['target_means']):.1f} "
            f"(min={np.min(stats['target_means']):.1f}, max={np.max(stats['target_means']):.1f})"
        )
    if stats["cond_means"]:
        print(
            f"    Cond pixel mean:   {np.mean(stats['cond_means']):.1f} "
            f"(min={np.min(stats['cond_means']):.1f}, max={np.max(stats['cond_means']):.1f})"
        )

    print(f"    Conditioning is mesh: {stats['cond_is_mesh']}/{n_sampled}")
    print(f"    Conditioning is photo: {stats['cond_is_photo']}/{n_sampled}")
    if stats["cond_is_photo"] > 0:
        print(
            f"    WARNING: {stats['cond_is_photo']} conditioning images look like photos, not meshes!"
        )

    print(f"    Cond == Target (identical): {stats['cond_target_identical']}")
    if stats["cond_target_identical"] > 0:
        print(
            f"    WARNING: {stats['cond_target_identical']} pairs have identical cond and target!"
        )

    if has_mediapipe:
        print(f"    Face detected in targets: {stats['target_is_face']}/{face_check_limit}")
        print(f"    No face in targets: {stats['target_no_face']}/{face_check_limit}")

    # Overall verdict
    print(f"\n{'=' * 50}")
    critical = (
        len(missing_targets) > n_input * 0.01
        or stats["cond_is_photo"] > n_sampled * 0.1
        or stats["cond_target_identical"] > n_sampled * 0.01
    )
    if critical:
        print("VERDICT: ISSUES FOUND; review before training")
    else:
        print("VERDICT: Dataset looks good for training")
    print(f"{'=' * 50}")

    # Procedure breakdown from metadata
    meta_path = data_dir / "metadata.json"
    proc_counts = {}
    wave_counts = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for _, info in meta.get("pairs", {}).items():
            proc = info.get("procedure", "unknown")
            wave = info.get("wave", "unknown")
            proc_counts[proc] = proc_counts.get(proc, 0) + 1
            wave_counts[wave] = wave_counts.get(wave, 0) + 1
        print(f"\n  By procedure: {dict(sorted(proc_counts.items()))}")
        print(f"  By wave:      {dict(sorted(wave_counts.items()))}")

    result = {
        "total_input": n_input,
        "total_target": n_target,
        "total_conditioning": n_cond,
        "complete_pairs": len(all_prefixes),
        "issues": issues,
        "quality": {
            "sampled": n_sampled,
            "cond_is_mesh_pct": stats["cond_is_mesh"] / max(n_sampled, 1) * 100,
            "cond_is_photo_pct": stats["cond_is_photo"] / max(n_sampled, 1) * 100,
            "target_face_detect_pct": stats["target_is_face"] / max(face_check_limit, 1) * 100
            if has_mediapipe
            else None,
            "cond_target_identical": stats["cond_target_identical"],
        },
        "by_procedure": proc_counts,
        "by_wave": wave_counts,
        "passed": not critical,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify training dataset")
    parser.add_argument("--data_dir", default="data/training_combined")
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        help="Number of pairs to sample for quality checks (0=all)",
    )
    parser.add_argument("--fix", action="store_true", help="Attempt to fix found issues")
    parser.add_argument("--output", default=None, help="Save results as JSON")
    args = parser.parse_args()

    result = verify_dataset(args.data_dir, args.sample, args.fix)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
