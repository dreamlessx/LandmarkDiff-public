#!/usr/bin/env python3
"""Split HDA processed data into train/val/test sets with procedure stratification.

Creates stratified splits ensuring each procedure is represented proportionally
in train/val/test. Real surgery pairs go into val/test (higher value for evaluation),
augmented copies go into train.

Output:
    data/hda_splits/
    ├── train/    (augmented HDA pairs for Phase B training)
    ├── val/      (held-out HDA pairs for validation)
    ├── test/     (held-out HDA pairs for final evaluation)
    └── split_info.json

Usage:
    python scripts/split_hda_dataset.py
    python scripts/split_hda_dataset.py --val-frac 0.15 --test-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_metadata(hda_dir: Path) -> dict:
    """Load metadata from processed HDA directory."""
    meta_path = hda_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {hda_dir}")
    with open(meta_path) as f:
        return json.load(f)


def stratified_split(
    prefixes: list[str],
    procedures: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split prefixes into train/val/test with procedure stratification.

    Returns dict with keys 'train', 'val', 'test' → list of prefixes.
    """
    rng = np.random.default_rng(seed)

    # Group by procedure
    proc_to_prefixes: dict[str, list[str]] = defaultdict(list)
    for prefix, proc in zip(prefixes, procedures, strict=False):
        proc_to_prefixes[proc].append(prefix)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for _proc, proc_prefixes in proc_to_prefixes.items():
        shuffled = proc_prefixes.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Too few samples — put at least 1 in each
            n_train = max(1, n - 2)
            n_val = min(n_val, n - n_train - 1)
            n_test = n - n_train - n_val

        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])

    return splits


def copy_split(
    hda_dir: Path,
    output_dir: Path,
    prefixes: list[str],
    split_name: str,
) -> int:
    """Copy pairs for a split into the output directory."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for prefix in prefixes:
        for suffix in ["_input.png", "_target.png", "_conditioning.png", "_mask.png", "_canny.png"]:
            src = hda_dir / f"{prefix}{suffix}"
            if src.exists():
                shutil.copy2(src, split_dir / f"{prefix}{suffix}")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Split HDA data into train/val/test")
    parser.add_argument("--hda-dir", type=Path, default=ROOT / "data" / "hda_processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "hda_splits")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading metadata...")
    metadata = load_metadata(args.hda_dir)
    pairs_meta = metadata.get("pairs", {})

    if not pairs_meta:
        print("ERROR: No pairs found in metadata")
        sys.exit(1)

    prefixes = list(pairs_meta.keys())
    procedures = [pairs_meta[p].get("procedure", "unknown") for p in prefixes]

    print(f"Total pairs: {len(prefixes)}")
    proc_counts = defaultdict(int)
    for p in procedures:
        proc_counts[p] += 1
    for proc, count in sorted(proc_counts.items()):
        print(f"  {proc}: {count}")

    # Split
    splits = stratified_split(
        prefixes,
        procedures,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Copy files
    args.output.mkdir(parents=True, exist_ok=True)
    split_info = {"seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac}

    for split_name in ["train", "val", "test"]:
        n = copy_split(args.hda_dir, args.output, splits[split_name], split_name)
        print(f"  {split_name}: {n} pairs")
        split_info[split_name] = {
            "count": n,
            "prefixes": splits[split_name],
        }

        # Procedure breakdown per split
        split_procs = defaultdict(int)
        for prefix in splits[split_name]:
            proc = pairs_meta.get(prefix, {}).get("procedure", "unknown")
            split_procs[proc] += 1
        split_info[split_name]["procedures"] = dict(split_procs)
        for proc, count in sorted(split_procs.items()):
            print(f"    {proc}: {count}")

    # Copy metadata to each split
    for split_name in ["train", "val", "test"]:
        split_dir = args.output / split_name
        split_meta = {
            "source": metadata.get("source", "HDA"),
            "citation": metadata.get("citation", ""),
            "pairs": {p: pairs_meta[p] for p in splits[split_name] if p in pairs_meta},
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(split_meta, f, indent=2)

    # Save split info
    with open(args.output / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {args.output / 'split_info.json'}")


if __name__ == "__main__":
    main()
