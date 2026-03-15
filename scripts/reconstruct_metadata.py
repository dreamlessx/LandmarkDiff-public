#!/usr/bin/env python3
"""Reconstruct metadata.json for training_combined by retracing the build order.

The build_training_dataset.py script creates training_combined/ with numeric
indices (000000_input.png, etc.) by copying from source directories in a
specific order. This script re-traces that order to map each index back to
its original procedure, wave, and source.

Build order (from build_training_dataset.py):
  1. Synthetic wave 1: synthetic_surgery_pairs/{procedure}/ (sorted by procedure list)
  2. Synthetic wave 2: synthetic_surgery_pairs_v2/{procedure}/ (sorted by procedure list)
  3. Wave 3: synthetic_surgery_pairs_v3/ (flat directory, procedure from filename)
  4. Real pairs: real_surgery_pairs/pairs/ + augmented copies

Usage:
    python scripts/reconstruct_metadata.py
    python scripts/reconstruct_metadata.py --dry_run
    python scripts/reconstruct_metadata.py --verify 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent / "data"
COMBINED = BASE / "training_combined"
PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def count_valid_pairs(directory: Path) -> int:
    """Count pairs that have both input and target."""
    inputs = sorted(directory.glob("*_input.png"))
    count = 0
    for inp in inputs:
        prefix = inp.stem.replace("_input", "")
        if (directory / f"{prefix}_target.png").exists():
            count += 1
    return count


def count_augmented(real_count: int, augment_factor: int = 5) -> int:
    """Estimate augmented pairs (some may fail)."""
    return real_count * augment_factor


def reconstruct_metadata(
    verify_samples: int = 0,
    dry_run: bool = False,
    include_v2: bool = False,
) -> dict:
    """Reconstruct metadata by retracing the build order.

    Args:
        include_v2: Whether to include wave 2 (synthetic_surgery_pairs_v2/).
            Default False because the 30K combined dataset was built without v2.
    """
    # Count actual pairs in training_combined
    combined_inputs = sorted(COMBINED.glob("*_input.png"))
    # Build a set of existing indices for fast lookup
    existing_indices = {inp.stem.replace("_input", "") for inp in combined_inputs}
    total_combined = len(combined_inputs)
    print(f"Training combined: {total_combined} pairs")

    # Build the index mapping by tracing the exact same order as build_training_dataset.py
    metadata = {}
    idx = 0

    # Phase 1: Synthetic wave 1 (and optionally wave 2)
    syn_bases = [BASE / "synthetic_surgery_pairs"]
    if include_v2:
        syn_bases.append(BASE / "synthetic_surgery_pairs_v2")

    for syn_base in syn_bases:
        wave = "w2" if "_v2" in str(syn_base) else "w1"
        for proc in PROCEDURES:
            proc_dir = syn_base / proc
            if not proc_dir.exists():
                continue

            inputs = sorted(proc_dir.glob("*_input.png"))
            valid = 0
            for inp in inputs:
                prefix = inp.stem.replace("_input", "")
                if (proc_dir / f"{prefix}_target.png").exists():
                    out_prefix = f"{idx:06d}"
                    if out_prefix in existing_indices:
                        metadata[out_prefix] = {
                            "procedure": proc,
                            "source": "synthetic",
                            "wave": wave,
                            "original_prefix": prefix,
                            "original_dir": str(proc_dir.relative_to(BASE)),
                        }
                        valid += 1
                    idx += 1

            if valid > 0:
                print(f"  {proc} ({wave}): {valid} pairs → idx {idx - valid:06d}–{idx - 1:06d}")

    syn_total = idx
    print(f"Synthetic total: {syn_total}")

    # Phase 2: Wave 3 (flat directory, procedure in filename)
    v3_dir = BASE / "synthetic_surgery_pairs_v3"
    if v3_dir.exists():
        v3_inputs = sorted(v3_dir.glob("*_input.png"))
        v3_count = 0
        for inp in v3_inputs:
            prefix = inp.stem.replace("_input", "")
            if not (v3_dir / f"{prefix}_target.png").exists():
                continue

            proc = "unknown"
            for p in PROCEDURES:
                if prefix.startswith(p) or prefix.lower().startswith(p[:5]):
                    proc = p
                    break

            out_prefix = f"{idx:06d}"
            if out_prefix in existing_indices:
                metadata[out_prefix] = {
                    "procedure": proc,
                    "source": "synthetic_v3",
                    "wave": "w3",
                    "original_prefix": prefix,
                    "original_dir": str(v3_dir.relative_to(BASE)),
                }
                v3_count += 1
            idx += 1

        if v3_count > 0:
            print(f"  Wave 3: {v3_count} pairs → idx {idx - v3_count:06d}–{idx - 1:06d}")

    # Phase 3: Real pairs + augmented
    real_dir = BASE / "real_surgery_pairs" / "pairs"
    if real_dir.exists():
        real_inputs = sorted(real_dir.glob("*_input.png"))
        real_count = 0
        aug_count = 0

        for inp in real_inputs:
            prefix = inp.stem.replace("_input", "")
            if not (real_dir / f"{prefix}_target.png").exists():
                continue

            # Infer procedure from filename
            proc = "unknown"
            for p in PROCEDURES:
                if p in prefix.lower():
                    proc = p
                    break

            # Original real pair
            out_prefix = f"{idx:06d}"
            if out_prefix in existing_indices:
                metadata[out_prefix] = {
                    "procedure": proc,
                    "source": "real",
                    "wave": "real",
                    "original_prefix": prefix,
                    "original_dir": str(real_dir.relative_to(BASE)),
                }
                real_count += 1
            idx += 1

            # Augmented copies (5x default, some may fail)
            for aug_i in range(5):
                if idx > total_combined + syn_total:
                    break
                out_prefix = f"{idx:06d}"
                if out_prefix in existing_indices:
                    metadata[out_prefix] = {
                        "procedure": proc,
                        "source": "augmented",
                        "wave": "real",
                        "original_prefix": prefix,
                        "augmentation_idx": aug_i,
                        "original_dir": str(real_dir.relative_to(BASE)),
                    }
                    aug_count += 1
                    idx += 1
                else:
                    idx += 1

        print(f"  Real: {real_count} pairs")
        print(f"  Augmented: {aug_count} pairs")

    # Fill any remaining gaps (indices that exist in combined but we missed)
    missing = 0
    for inp in combined_inputs:
        prefix = inp.stem.replace("_input", "")
        if prefix not in metadata:
            metadata[prefix] = {
                "procedure": "unknown",
                "source": "unknown",
                "wave": "unknown",
            }
            missing += 1

    if missing > 0:
        print(f"\nUnmapped indices: {missing}")

    # Compute statistics
    from collections import defaultdict

    proc_counts = defaultdict(int)
    source_counts = defaultdict(int)
    wave_counts = defaultdict(int)

    for entry in metadata.values():
        proc_counts[entry["procedure"]] += 1
        source_counts[entry["source"]] += 1
        wave_counts[entry["wave"]] += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print("RECONSTRUCTED METADATA SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total entries: {len(metadata)}")
    print(f"Combined pairs: {total_combined}")
    print("\nProcedure Distribution:")
    for proc, count in sorted(proc_counts.items()):
        pct = count / len(metadata) * 100
        print(f"  {proc:<20} {count:>6} ({pct:5.1f}%)")
    print("\nSource Distribution:")
    for src, count in sorted(source_counts.items()):
        pct = count / len(metadata) * 100
        print(f"  {src:<20} {count:>6} ({pct:5.1f}%)")
    print("\nWave Distribution:")
    for wave, count in sorted(wave_counts.items()):
        pct = count / len(metadata) * 100
        print(f"  {wave:<25} {count:>6} ({pct:5.1f}%)")

    # Verification: spot-check a few samples by comparing pixels
    if verify_samples > 0:
        print(f"\n{'=' * 60}")
        print(f"VERIFICATION: checking {verify_samples} random samples")
        rng = np.random.default_rng(42)
        keys_with_originals = [
            k
            for k, v in metadata.items()
            if v.get("original_prefix") and v["source"] not in ("augmented", "unknown")
        ]
        sample_keys = rng.choice(
            keys_with_originals,
            size=min(verify_samples, len(keys_with_originals)),
            replace=False,
        )
        matches = 0
        for key in sample_keys:
            entry = metadata[key]
            combined_img = cv2.imread(str(COMBINED / f"{key}_input.png"))
            orig_dir = BASE / entry["original_dir"]
            orig_img = cv2.imread(str(orig_dir / f"{entry['original_prefix']}_input.png"))
            if combined_img is not None and orig_img is not None:
                # Resize if needed
                if combined_img.shape != orig_img.shape:
                    orig_img = cv2.resize(orig_img, (combined_img.shape[1], combined_img.shape[0]))
                diff = np.mean(np.abs(combined_img.astype(float) - orig_img.astype(float)))
                if diff < 1.0:  # should be nearly identical (copy)
                    matches += 1
                else:
                    print(
                        f"  MISMATCH: {key} (diff={diff:.1f}) → {entry['procedure']}/{entry['original_prefix']}"
                    )
            else:
                print(f"  MISSING: {key} or {entry['original_prefix']}")

        print(
            f"  Verified: {matches}/{len(sample_keys)} match ({matches / len(sample_keys) * 100:.0f}%)"
        )

    # Build final metadata object
    result = {
        "total_pairs": total_combined,
        "reconstructed_pairs": len(metadata),
        "generated_by": "reconstruct_metadata.py",
        "procedure_distribution": dict(sorted(proc_counts.items())),
        "source_distribution": dict(sorted(source_counts.items())),
        "wave_distribution": dict(sorted(wave_counts.items())),
        "pairs": {},
    }

    # Slim down pairs for JSON (drop original_dir/prefix to reduce size)
    for key, entry in metadata.items():
        result["pairs"][key] = {
            "procedure": entry["procedure"],
            "wave": entry["wave"],
            "source": entry["source"],
        }

    if not dry_run:
        out_path = COMBINED / "metadata.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out_path}")
        print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
    else:
        print("\n[DRY RUN] Would save metadata.json")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct metadata for training_combined")
    parser.add_argument(
        "--verify", type=int, default=0, help="Number of samples to verify by pixel comparison"
    )
    parser.add_argument("--dry_run", action="store_true", help="Print summary without saving")
    parser.add_argument(
        "--include_v2",
        action="store_true",
        help="Include wave 2 data (200K pairs, disabled by default)",
    )
    args = parser.parse_args()

    reconstruct_metadata(
        verify_samples=args.verify,
        dry_run=args.dry_run,
        include_v2=args.include_v2,
    )
