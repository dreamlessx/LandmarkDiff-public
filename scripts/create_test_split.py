"""Create train/val/test split from the combined training dataset.

Holds out percentages of pairs (stratified by procedure and source image)
for validation and evaluation. Ensures no source image appears in multiple
splits (prevents data leakage from augmented copies).

Must be run AFTER build_training_dataset.py.

Usage:
    # Basic 90/5/5 split
    python scripts/create_test_split.py \
        --data_dir data/training_combined \
        --test_dir data/test_pairs \
        --val_dir data/val_pairs \
        --test_fraction 0.05 \
        --val_fraction 0.05 \
        --seed 42

    # Test-only split (no validation)
    python scripts/create_test_split.py \
        --data_dir data/training_combined \
        --test_dir data/test_pairs \
        --test_fraction 0.05

    # Dry run (show what would be split without moving files)
    python scripts/create_test_split.py \
        --data_dir data/training_combined \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def extract_source_id(prefix: str, meta: dict) -> str:
    """Extract the source image ID to group augmented copies.

    Augmented pairs from the same source image must go to the same split
    to prevent data leakage.
    """
    if prefix in meta:
        info = meta[prefix]
        # If it's an augmented pair, use the original's source
        source = info.get("source", "synthetic")
        if source == "augmented" and "original_prefix" in info:
            return info["original_prefix"]
        # Use source_image if available
        if "source_image" in info:
            return info["source_image"]
    # Fall back to prefix without augmentation suffix
    # e.g., "rhinoplasty_000001_aug3" -> "rhinoplasty_000001"
    parts = prefix.rsplit("_aug", 1)
    return parts[0]


def create_split(
    data_dir: Path,
    test_dir: Path | None = None,
    val_dir: Path | None = None,
    test_fraction: float = 0.05,
    val_fraction: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """Create stratified train/val/test split.

    Groups by source image to prevent augmented copies leaking across splits.
    Stratifies by procedure for balanced evaluation.

    Returns split statistics dict.
    """
    data_dir = Path(data_dir)

    input_files = sorted(data_dir.glob("*_input.png"))
    if not input_files:
        print(f"No input files found in {data_dir}")
        return {}

    print(f"Total pairs: {len(input_files):,}")

    # Load metadata if available
    meta = {}
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_data = json.load(f)
            meta = meta_data.get("pairs", {})
        print(f"Metadata: {len(meta)} entries")

    # Group files by (procedure, source_id)
    # All augmented copies of the same source go to the same split
    proc_source_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for f in input_files:
        prefix = f.stem.replace("_input", "")

        # Determine procedure
        proc = "unknown"
        if prefix in meta:
            proc = meta[prefix].get("procedure", "unknown")
        else:
            for p in PROCEDURES:
                if p in prefix:
                    proc = p
                    break

        # Group by source image
        source_id = extract_source_id(prefix, meta)
        proc_source_groups[proc][source_id].append(f)

    # Print distribution
    print("\nProcedure distribution:")
    for proc in sorted(proc_source_groups.keys()):
        sources = proc_source_groups[proc]
        n_files = sum(len(v) for v in sources.values())
        print(f"  {proc}: {n_files:,} pairs ({len(sources)} source images)")

    # Stratified split by procedure at the source-image level
    rng = random.Random(seed)
    test_sources: dict[str, list[str]] = {}
    val_sources: dict[str, list[str]] = {}

    for proc in sorted(proc_source_groups.keys()):
        source_ids = sorted(proc_source_groups[proc].keys())
        rng.shuffle(source_ids)

        n_test = max(1, int(len(source_ids) * test_fraction))
        n_val = max(1, int(len(source_ids) * val_fraction)) if val_dir else 0

        test_sources[proc] = source_ids[:n_test]
        val_sources[proc] = source_ids[n_test : n_test + n_val]

    # Collect files for each split
    test_files = []
    val_files = []
    for proc in sorted(proc_source_groups.keys()):
        for source_id in test_sources.get(proc, []):
            test_files.extend(proc_source_groups[proc][source_id])
        for source_id in val_sources.get(proc, []):
            val_files.extend(proc_source_groups[proc][source_id])

    # Summary
    n_train = len(input_files) - len(test_files) - len(val_files)
    print("\nSplit allocation:")
    print(f"  Train: {n_train:,} pairs")
    if val_dir:
        print(f"  Val:   {len(val_files):,} pairs")
    print(f"  Test:  {len(test_files):,} pairs")

    # Per-procedure breakdown
    print("\nPer-procedure split:")
    for proc in sorted(proc_source_groups.keys()):
        n_test_proc = sum(len(proc_source_groups[proc][s]) for s in test_sources.get(proc, []))
        n_val_proc = sum(len(proc_source_groups[proc][s]) for s in val_sources.get(proc, []))
        n_total = sum(len(v) for v in proc_source_groups[proc].values())
        n_train_proc = n_total - n_test_proc - n_val_proc
        parts = [f"train={n_train_proc}"]
        if val_dir:
            parts.append(f"val={n_val_proc}")
        parts.append(f"test={n_test_proc}")
        print(f"  {proc}: {', '.join(parts)}")

    if dry_run:
        print("\n[DRY RUN] No files moved.")
        return {
            "total": len(input_files),
            "train": n_train,
            "val": len(val_files),
            "test": len(test_files),
        }

    # Move files
    def move_pairs(files: list[Path], dest: Path, label: str) -> int:
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for inp_file in files:
            prefix = inp_file.stem.replace("_input", "")
            for suffix in ["_input.png", "_target.png", "_conditioning.png", "_mask.png"]:
                src = data_dir / f"{prefix}{suffix}"
                if src.exists():
                    shutil.move(str(src), str(dest / src.name))
            moved += 1
        print(f"\n{label}: Moved {moved} pairs to {dest}")
        return moved

    moved_test = 0
    moved_val = 0

    if test_dir and test_files:
        moved_test = move_pairs(test_files, Path(test_dir), "Test")

    if val_dir and val_files:
        moved_val = move_pairs(val_files, Path(val_dir), "Val")

    remaining = len(list(data_dir.glob("*_input.png")))

    # Save split manifest for reproducibility
    manifest = {
        "seed": seed,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "total_pairs": len(input_files),
        "train_pairs": remaining,
        "val_pairs": moved_val,
        "test_pairs": moved_test,
        "test_prefixes": [f.stem.replace("_input", "") for f in test_files],
        "val_prefixes": [f.stem.replace("_input", "") for f in val_files],
    }

    manifest_path = data_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSplit manifest saved: {manifest_path}")

    print("\nFinal counts:")
    print(f"  Training: {remaining:,} pairs")
    if val_dir:
        print(f"  Validation: {moved_val:,} pairs")
    print(f"  Test: {moved_test:,} pairs")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified train/val/test split")
    parser.add_argument("--data_dir", default="data/training_combined")
    parser.add_argument("--test_dir", default="data/test_pairs")
    parser.add_argument("--val_dir", default=None, help="Validation set directory (optional)")
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show split plan without moving files"
    )
    args = parser.parse_args()

    create_split(
        Path(args.data_dir),
        Path(args.test_dir) if args.test_dir else None,
        Path(args.val_dir) if args.val_dir else None,
        args.test_fraction,
        args.val_fraction,
        args.seed,
        args.dry_run,
    )
