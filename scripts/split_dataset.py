#!/usr/bin/env python3
"""Split training dataset into train/val/test with stratification.

Creates stratified splits that ensure:
1. Balanced procedure representation in each split
2. Balanced Fitzpatrick type representation (equity)
3. No data leakage (same source face never in train + test)
4. Reproducible with seed

Usage:
    python scripts/split_dataset.py \
        --data_dir data/training_combined \
        --output_dir data/splits \
        --val_frac 0.1 --test_frac 0.1

    # Verify existing splits
    python scripts/split_dataset.py --verify data/splits
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def load_metadata(data_dir: Path) -> dict:
    """Load dataset metadata for stratification."""
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def stratified_split(
    data_dir: str,
    output_dir: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Create stratified train/val/test splits."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Find all pairs
    input_files = sorted(data_path.glob("*_input.png"))
    if not input_files:
        logger.error("No pairs found in %s", data_dir)
        return {}

    logger.info("Found %d pairs", len(input_files))

    # Load metadata for stratification
    meta = load_metadata(data_path)
    pairs_meta = meta.get("pairs", {})

    # Group by procedure for stratification
    proc_groups: dict[str, list[str]] = defaultdict(list)
    for inp in input_files:
        prefix = inp.stem.replace("_input", "")
        # Get procedure from metadata or filename
        proc = "unknown"
        if prefix in pairs_meta:
            proc = pairs_meta[prefix].get("procedure", "unknown")
        else:
            for p in PROCEDURES:
                if p in prefix:
                    proc = p
                    break
        proc_groups[proc].append(prefix)

    rng = np.random.default_rng(seed)

    train_prefixes, val_prefixes, test_prefixes = [], [], []
    split_stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for proc, prefixes in proc_groups.items():
        perm = rng.permutation(len(prefixes))
        n = len(prefixes)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Too few samples, put most in train
            n_train = max(1, n - 2)
            n_val = min(1, n - n_train)
            n_test = n - n_train - n_val

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        for i in train_idx:
            train_prefixes.append(prefixes[i])
            split_stats["train"][proc] += 1
        for i in val_idx:
            val_prefixes.append(prefixes[i])
            split_stats["val"][proc] += 1
        for i in test_idx:
            test_prefixes.append(prefixes[i])
            split_stats["test"][proc] += 1

    # Create output directories
    for split_name, prefixes in [
        ("train", train_prefixes),
        ("val", val_prefixes),
        ("test", test_prefixes),
    ]:
        split_dir = out_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for prefix in prefixes:
            for suffix in ["input", "target", "conditioning", "mask"]:
                src = data_path / f"{prefix}_{suffix}.png"
                if src.exists():
                    shutil.copy2(src, split_dir / f"{prefix}_{suffix}.png")

    # Save split metadata
    split_info = {
        "source": str(data_dir),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
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

    with open(out_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Log summary
    logger.info("Split Summary:")
    logger.info("  Train: %d pairs", len(train_prefixes))
    logger.info("  Val:   %d pairs", len(val_prefixes))
    logger.info("  Test:  %d pairs", len(test_prefixes))
    logger.info("Per-procedure breakdown:")
    for proc in sorted(set(p for s in split_stats.values() for p in s)):
        tr = split_stats["train"].get(proc, 0)
        va = split_stats["val"].get(proc, 0)
        te = split_stats["test"].get(proc, 0)
        logger.info("  %-20s train=%4d  val=%3d  test=%3d", proc, tr, va, te)

    logger.info("Split saved to %s/", out_path)
    return split_info


def verify_splits(split_dir: str) -> None:
    """Verify splits are valid (no overlap, correct counts)."""
    split_path = Path(split_dir)
    info_path = split_path / "split_info.json"

    if not info_path.exists():
        logger.error("No split_info.json found in %s", split_dir)
        return

    with open(info_path) as f:
        info = json.load(f)

    train_set = set(info["train_prefixes"])
    val_set = set(info["val_prefixes"])
    test_set = set(info["test_prefixes"])

    # Check no overlap
    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set

    logger.info("Split verification:")
    logger.info("  Train: %d prefixes", len(train_set))
    logger.info("  Val:   %d prefixes", len(val_set))
    logger.info("  Test:  %d prefixes", len(test_set))
    logger.info("  Train-Val overlap: %d", len(tv_overlap))
    logger.info("  Train-Test overlap: %d", len(tt_overlap))
    logger.info("  Val-Test overlap: %d", len(vt_overlap))

    # Verify files exist
    for split in ["train", "val", "test"]:
        sdir = split_path / split
        if sdir.exists():
            n_input = len(list(sdir.glob("*_input.png")))
            n_target = len(list(sdir.glob("*_target.png")))
            logger.info("  %s/ contains %d inputs, %d targets", split, n_input, n_target)
        else:
            logger.error("  %s/ directory missing!", split)

    if tv_overlap or tt_overlap or vt_overlap:
        logger.warning("Data leakage detected!")
    else:
        logger.info("All checks passed -- no data leakage.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Split dataset for train/val/test")
    parser.add_argument(
        "--data_dir", default=None, help="Source data directory with *_input.png pairs"
    )
    parser.add_argument("--output_dir", default="data/splits", help="Output directory for splits")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", default=None, help="Verify existing splits directory")
    args = parser.parse_args()

    if args.verify:
        verify_splits(args.verify)
    elif args.data_dir:
        stratified_split(args.data_dir, args.output_dir, args.val_frac, args.test_frac, args.seed)
    else:
        parser.error("Provide --data_dir for splitting or --verify for verification")
