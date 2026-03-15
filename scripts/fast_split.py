#!/usr/bin/env python3
"""Fast symlink-based train/val/test split.

Creates splits using symbolic links instead of copying files.
Completes in seconds instead of hours for large datasets.

Usage:
    python scripts/fast_split.py
    python scripts/fast_split.py --val_frac 0.1 --test_frac 0.1
    python scripts/fast_split.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "training_combined"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def fast_split(
    data_dir: Path = DATA_DIR,
    output_dir: Path = SPLITS_DIR,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> dict:
    """Create stratified splits using symlinks (fast)."""
    # Find all pairs
    input_files = sorted(data_dir.glob("*_input.png"))
    total = len(input_files)
    print(f"Total pairs: {total:,}")

    if total == 0:
        print("No input files found")
        return {}

    # Load metadata for stratification
    meta = {}
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)
            meta = meta_data.get("pairs", {})
        print(f"Metadata: {len(meta)} entries")

    # Group by procedure (for stratified split)
    # Also group augmented pairs with their originals to prevent leakage
    proc_groups: dict[str, list[list[str]]] = defaultdict(list)

    # First pass: identify original→augmented relationships
    augmented_of: dict[str, str] = {}  # augmented prefix → original prefix
    originals: dict[str, list[str]] = defaultdict(list)  # original → [augmented prefixes]

    for inp in input_files:
        prefix = inp.stem.replace("_input", "")
        entry = meta.get(prefix, {})
        source = entry.get("source", "synthetic")

        if source == "augmented":
            # Find the nearest preceding non-augmented entry as its original
            orig_prefix = entry.get("original_prefix", prefix)
            # Map augmented → its procedure group key (original's prefix)
            augmented_of[prefix] = orig_prefix
            originals[orig_prefix].append(prefix)

    # Second pass: group into units (original + all its augmented copies)
    seen = set()
    for inp in input_files:
        prefix = inp.stem.replace("_input", "")
        if prefix in seen:
            continue

        entry = meta.get(prefix, {})
        proc = entry.get("procedure", "unknown")
        source = entry.get("source", "synthetic")

        if source == "augmented":
            continue  # will be grouped with its original

        # This is an original pair - collect it and all its augmented copies
        group = [prefix]
        for aug_prefix in originals.get(prefix, []):
            group.append(aug_prefix)
            seen.add(aug_prefix)
        seen.add(prefix)

        proc_groups[proc].append(group)

    # Print distribution
    print("\nProcedure distribution (source groups):")
    for proc in sorted(proc_groups.keys()):
        n_groups = len(proc_groups[proc])
        n_files = sum(len(g) for g in proc_groups[proc])
        print(f"  {proc:<20} {n_groups:>5} groups ({n_files:>6} pairs)")

    # Stratified split at the group level
    rng = np.random.default_rng(seed)
    train_prefixes, val_prefixes, test_prefixes = [], [], []
    split_stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for proc in sorted(proc_groups.keys()):
        groups = proc_groups[proc]
        n = len(groups)
        perm = rng.permutation(n)

        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_test - n_val

        for i in perm[:n_train]:
            for prefix in groups[i]:
                train_prefixes.append(prefix)
                split_stats["train"][proc] += 1

        for i in perm[n_train : n_train + n_val]:
            for prefix in groups[i]:
                val_prefixes.append(prefix)
                split_stats["val"][proc] += 1

        for i in perm[n_train + n_val :]:
            for prefix in groups[i]:
                test_prefixes.append(prefix)
                split_stats["test"][proc] += 1

    print("\nSplit allocation:")
    print(f"  Train: {len(train_prefixes):,} pairs")
    print(f"  Val:   {len(val_prefixes):,} pairs")
    print(f"  Test:  {len(test_prefixes):,} pairs")

    print("\nPer-procedure split:")
    for proc in sorted(set(p for s in split_stats.values() for p in s)):
        tr = split_stats["train"].get(proc, 0)
        va = split_stats["val"].get(proc, 0)
        te = split_stats["test"].get(proc, 0)
        print(f"  {proc:<20} train={tr:>5}  val={va:>4}  test={te:>4}")

    # Create symlink directories
    for split_name, prefixes in [
        ("train", train_prefixes),
        ("val", val_prefixes),
        ("test", test_prefixes),
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        for prefix in prefixes:
            for suffix in ["input", "target", "conditioning", "mask"]:
                src = data_dir / f"{prefix}_{suffix}.png"
                dst = split_dir / f"{prefix}_{suffix}.png"
                if src.exists() and not dst.exists():
                    os.symlink(str(src.resolve()), str(dst))
                    created += 1

        print(f"  {split_name}/: {created} symlinks created")

    # Save split info
    split_info = {
        "source": str(data_dir),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "method": "symlink",
        "leakage_prevention": True,
        "counts": {
            "train": len(train_prefixes),
            "val": len(val_prefixes),
            "test": len(test_prefixes),
        },
        "per_procedure": {split: dict(counts) for split, counts in split_stats.items()},
        "train_prefixes": sorted(train_prefixes),
        "val_prefixes": sorted(val_prefixes),
        "test_prefixes": sorted(test_prefixes),
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {output_dir / 'split_info.json'}")
    return split_info


def verify_splits(output_dir: Path = SPLITS_DIR) -> None:
    """Verify splits are valid (no overlap, symlinks intact)."""
    info_path = output_dir / "split_info.json"
    if not info_path.exists():
        print("No split_info.json found")
        return

    with open(info_path) as f:
        info = json.load(f)

    train_set = set(info["train_prefixes"])
    val_set = set(info["val_prefixes"])
    test_set = set(info["test_prefixes"])

    # Check overlap
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    print("Split verification:")
    print(f"  Train: {len(train_set):,} prefixes")
    print(f"  Val:   {len(val_set):,} prefixes")
    print(f"  Test:  {len(test_set):,} prefixes")
    print(f"  Train-Val overlap:  {len(tv)}")
    print(f"  Train-Test overlap: {len(tt)}")
    print(f"  Val-Test overlap:   {len(vt)}")

    # Check symlinks are valid
    broken = 0
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            print(f"  {split}/ MISSING!")
            continue
        for link in split_dir.iterdir():
            if link.is_symlink() and not link.exists():
                broken += 1

    print(f"  Broken symlinks: {broken}")

    if tv or tt or vt:
        print("\nWARNING: Data leakage detected!")
    elif broken > 0:
        print(f"\nWARNING: {broken} broken symlinks!")
    else:
        print("\nAll checks passed — no leakage, all symlinks valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast symlink-based data split")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(SPLITS_DIR))
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Verify existing splits")
    args = parser.parse_args()

    if args.verify:
        verify_splits(Path(args.output_dir))
    else:
        fast_split(
            Path(args.data_dir),
            Path(args.output_dir),
            args.val_frac,
            args.test_frac,
            args.seed,
        )
