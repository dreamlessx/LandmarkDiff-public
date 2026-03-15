"""Build combined training dataset from all synthetic + real data sources.

Merges all data sources into a single flat directory with correct naming
for the ControlNet training script:
  {idx}_input.png      — input image (original face)
  {idx}_target.png     — target image (what model should generate)
  {idx}_conditioning.png — conditioning signal (MediaPipe tessellation mesh)
  {idx}_mask.png       — surgical mask (optional, float32)

Sources:
1. Synthetic pairs wave 1 (data/synthetic_surgery_pairs/{procedure}/)
2. Synthetic pairs wave 2 (data/synthetic_surgery_pairs_v2/{procedure}/)
3. Synthetic pairs wave 3 (data/synthetic_surgery_pairs_v3/) — realistic displacement model
4. Real surgery pairs (data/real_surgery_pairs/pairs/)
5. HDA Plastic Surgery Database (data/hda_processed/) — Rathgeb et al. CVPRW 2020

The conditioning image must be the tessellation mesh rendering, NOT the input photo.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def _copy_synthetic_pair(syn_dir: Path, output_dir: Path, start_idx: int) -> int:
    """Copy synthetic pairs preserving correct conditioning files.

    Returns number of pairs copied.
    """
    input_files = sorted(syn_dir.glob("*_input.png"))
    count = 0
    idx = start_idx

    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = syn_dir / f"{prefix}_target.png"
        mask_file = syn_dir / f"{prefix}_mask.png"

        if not target_file.exists():
            continue

        out_prefix = f"{idx:06d}"

        # Target = original face photo (what model learns to generate)
        shutil.copy2(target_file, output_dir / f"{out_prefix}_target.png")

        # Conditioning = original mesh rendering (ControlNet control signal)
        # _input.png IS the original mesh in our generation pipeline.
        # At training time: condition on original mesh → generate original face.
        # At inference time: swap in manipulated mesh → generate post-surgery face.
        #
        # Priority: _conditioning.png > _before_mesh.png > _input.png (all = original mesh)
        # NOTE: _manip_mesh.png is the manipulated mesh (inference only, NOT for training)
        cond_src = None
        for cond_suffix in ["_conditioning.png", "_before_mesh.png"]:
            candidate = syn_dir / f"{prefix}{cond_suffix}"
            if candidate.exists():
                cond_src = candidate
                break

        if cond_src is not None:
            shutil.copy2(cond_src, output_dir / f"{out_prefix}_conditioning.png")
        else:
            # _input.png is the original mesh — use it as conditioning
            shutil.copy2(inp_file, output_dir / f"{out_prefix}_conditioning.png")

        # Also copy _input.png (original mesh) for reference
        shutil.copy2(inp_file, output_dir / f"{out_prefix}_input.png")

        # Mask (optional)
        if mask_file.exists():
            shutil.copy2(mask_file, output_dir / f"{out_prefix}_mask.png")

        idx += 1
        count += 1

    return count


def _generate_conditioning_fallback(input_path: Path, output_path: Path) -> None:
    """Generate conditioning image from face photo when pre-rendered one is missing."""
    try:
        from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

        img = cv2.imread(str(input_path))
        if img is None:
            # Last resort: copy input as conditioning
            shutil.copy2(input_path, output_path)
            return
        h, w = img.shape[:2]
        face = extract_landmarks(img)
        if face is not None:
            cond = render_landmark_image(face, w, h)
            cv2.imwrite(str(output_path), cond)
        else:
            shutil.copy2(input_path, output_path)
    except Exception:
        shutil.copy2(input_path, output_path)


def _copy_v3_pairs(v3_dir: Path, output_dir: Path, start_idx: int) -> tuple[int, dict]:
    """Copy wave 3 (realistic displacement) pairs.

    V3 pairs use flat naming: {procedure}_{num}_input.png in a single directory.
    Returns (count, metadata_dict).
    """
    input_files = sorted(v3_dir.glob("*_input.png"))
    count = 0
    idx = start_idx
    meta = {}

    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = v3_dir / f"{prefix}_target.png"
        cond_file = v3_dir / f"{prefix}_conditioning.png"
        mask_file = v3_dir / f"{prefix}_mask.png"

        if not target_file.exists():
            continue

        out_prefix = f"{idx:06d}"
        shutil.copy2(inp_file, output_dir / f"{out_prefix}_input.png")
        shutil.copy2(target_file, output_dir / f"{out_prefix}_target.png")

        if cond_file.exists():
            shutil.copy2(cond_file, output_dir / f"{out_prefix}_conditioning.png")
        else:
            shutil.copy2(inp_file, output_dir / f"{out_prefix}_conditioning.png")

        if mask_file.exists():
            shutil.copy2(mask_file, output_dir / f"{out_prefix}_mask.png")

        # Infer procedure from prefix (e.g. "rhinoplasty_000042")
        proc = "unknown"
        for p in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            if prefix.startswith(p):
                proc = p
                break

        meta[out_prefix] = {"procedure": proc, "source": "synthetic_v3", "wave": "w3"}
        idx += 1
        count += 1

    return count, meta


def _copy_hda_pairs(
    hda_dir: Path,
    output_dir: Path,
    start_idx: int,
    augment: bool = True,
    augment_factor: int = 5,
) -> tuple[int, int, dict]:
    """Copy HDA Plastic Surgery Database pairs into the combined dataset.

    Returns (n_real, n_augmented, metadata_dict).
    """
    input_files = sorted(hda_dir.glob("*_input.png"))
    n_real = 0
    n_aug = 0
    idx = start_idx
    meta = {}

    # Load HDA metadata for procedure info
    hda_meta_path = hda_dir / "metadata.json"
    hda_meta = {}
    if hda_meta_path.exists():
        with open(hda_meta_path) as f:
            hda_meta = json.load(f).get("pairs", {})

    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = hda_dir / f"{prefix}_target.png"
        cond_file = hda_dir / f"{prefix}_conditioning.png"
        mask_file = hda_dir / f"{prefix}_mask.png"

        if not target_file.exists():
            continue

        # Get procedure from HDA metadata
        pair_info = hda_meta.get(prefix, {})
        proc = pair_info.get("procedure", "unknown")

        # Copy base pair
        out_prefix = f"{idx:06d}"
        shutil.copy2(inp_file, output_dir / f"{out_prefix}_input.png")
        shutil.copy2(target_file, output_dir / f"{out_prefix}_target.png")

        if cond_file.exists():
            shutil.copy2(cond_file, output_dir / f"{out_prefix}_conditioning.png")
        else:
            shutil.copy2(inp_file, output_dir / f"{out_prefix}_conditioning.png")

        if mask_file.exists():
            shutil.copy2(mask_file, output_dir / f"{out_prefix}_mask.png")

        meta[out_prefix] = {"procedure": proc, "source": "hda_real", "wave": "hda"}
        idx += 1
        n_real += 1

        # Augment real pairs — they're scarce and high-value
        if augment:
            cond_img = cv2.imread(str(cond_file)) if cond_file.exists() else None
            for aug_i in range(augment_factor):
                tgt_img = cv2.imread(str(target_file))
                aug_tgt = _augment_image(tgt_img, aug_i)
                if aug_tgt is None:
                    continue

                out_prefix = f"{idx:06d}"
                cv2.imwrite(str(output_dir / f"{out_prefix}_target.png"), aug_tgt)

                if cond_img is not None:
                    aug_cond = _augment_image(cond_img, aug_i)
                    if aug_cond is not None:
                        cv2.imwrite(str(output_dir / f"{out_prefix}_conditioning.png"), aug_cond)
                        cv2.imwrite(str(output_dir / f"{out_prefix}_input.png"), aug_cond)

                meta[out_prefix] = {"procedure": proc, "source": "hda_augmented", "wave": "hda"}
                idx += 1
                n_aug += 1

    return n_real, n_aug, meta


def build_dataset(
    synthetic_dirs: list[Path],
    real_pairs_dir: Path | None,
    output_dir: Path,
    v3_dir: Path | None = None,
    hda_dir: Path | None = None,
    augment_real: bool = True,
    augment_factor: int = 5,
    clear_existing: bool = True,
):
    """Build combined training dataset from all sources."""
    if clear_existing and output_dir.exists():
        # Count existing files to decide
        existing = list(output_dir.glob("*_input.png"))
        if existing:
            print(f"Clearing {len(existing)} existing pairs from {output_dir}")
            for f in output_dir.iterdir():
                f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    stats = {
        "synthetic_w1": 0,
        "synthetic_w2": 0,
        "synthetic_w3": 0,
        "real": 0,
        "augmented": 0,
        "hda_real": 0,
        "hda_augmented": 0,
    }
    metadata = {}  # idx -> {procedure, source, wave}

    # 1. Synthetic pairs (wave 1 + wave 2)
    for syn_dir in synthetic_dirs:
        if not syn_dir.exists():
            continue

        wave = "w2" if "_v2" in str(syn_dir) else "w1"
        proc = syn_dir.name

        n_before = idx
        count = _copy_synthetic_pair(syn_dir, output_dir, idx)
        for j in range(n_before, idx + count):
            metadata[f"{j:06d}"] = {"procedure": proc, "source": "synthetic", "wave": wave}
        idx += count

        key = f"synthetic_{wave}"
        stats[key] = stats.get(key, 0) + count
        print(f"  {proc} ({wave}): {count} pairs")

    print(f"Synthetic total: {stats['synthetic_w1']} (w1) + {stats['synthetic_w2']} (w2)")

    # 2. Wave 3: realistic displacement-based pairs (flat directory)
    if v3_dir and v3_dir.exists():
        print(f"\nProcessing wave 3 (realistic displacement) pairs from {v3_dir}...")
        v3_count, v3_meta = _copy_v3_pairs(v3_dir, output_dir, idx)
        metadata.update(v3_meta)
        idx += v3_count
        stats["synthetic_w3"] = v3_count
        print(f"  Wave 3: {v3_count} pairs")
    else:
        print("\nNo wave 3 directory found (skipping)")

    # 3. Real processed pairs
    if real_pairs_dir and real_pairs_dir.exists():
        input_files = sorted(real_pairs_dir.glob("*_input.png"))
        print(f"\nProcessing {len(input_files)} real pairs...")

        for inp_file in input_files:
            prefix = inp_file.stem.replace("_input", "")
            target_file = real_pairs_dir / f"{prefix}_target.png"

            if not target_file.exists():
                continue

            out_prefix = f"{idx:06d}"
            shutil.copy2(inp_file, output_dir / f"{out_prefix}_input.png")
            shutil.copy2(target_file, output_dir / f"{out_prefix}_target.png")

            # Conditioning: for both synthetic and real pairs, _input.png IS the
            # correct mesh (original mesh for synthetic, after mesh for real).
            # Use _conditioning.png if explicitly provided, otherwise _input.png.
            # NOTE: _before_mesh.png is the PRE-surgery mesh for real pairs —
            # do NOT use it as conditioning (it doesn't match the target face).
            cond_file = real_pairs_dir / f"{prefix}_conditioning.png"
            if cond_file.exists():
                shutil.copy2(cond_file, output_dir / f"{out_prefix}_conditioning.png")
            else:
                shutil.copy2(inp_file, output_dir / f"{out_prefix}_conditioning.png")

            # Infer procedure from real pair filename
            real_proc = "unknown"
            for p in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
                if p in prefix.lower():
                    real_proc = p
                    break
            metadata[out_prefix] = {"procedure": real_proc, "source": "real", "wave": "real"}

            idx += 1
            stats["real"] += 1

            # Augment real pairs (they're more valuable since they're real surgery)
            if augment_real:
                # Read the conditioning source (mesh) for augmentation
                cond_img = cv2.imread(str(output_dir / f"{out_prefix}_conditioning.png"))
                for aug_i in range(augment_factor):
                    tgt_img = cv2.imread(str(target_file))
                    aug_tgt = _augment_image(tgt_img, aug_i)

                    if aug_tgt is None:
                        continue

                    out_prefix = f"{idx:06d}"
                    cv2.imwrite(str(output_dir / f"{out_prefix}_target.png"), aug_tgt)

                    # Apply same augmentation to mesh — no MediaPipe needed
                    if cond_img is not None:
                        aug_cond = _augment_image(cond_img, aug_i)
                        if aug_cond is not None:
                            cv2.imwrite(
                                str(output_dir / f"{out_prefix}_conditioning.png"), aug_cond
                            )
                            cv2.imwrite(str(output_dir / f"{out_prefix}_input.png"), aug_cond)

                    metadata[out_prefix] = {
                        "procedure": real_proc,
                        "source": "augmented",
                        "wave": "real",
                    }

                    idx += 1
                    stats["augmented"] += 1

        print(f"  Added {stats['real']} real + {stats['augmented']} augmented pairs")

    # 4. HDA Plastic Surgery Database pairs
    if hda_dir and hda_dir.exists():
        input_files = sorted(hda_dir.glob("*_input.png"))
        print(f"\nProcessing {len(input_files)} HDA real surgery pairs...")
        n_hda, n_hda_aug, hda_meta = _copy_hda_pairs(
            hda_dir,
            output_dir,
            idx,
            augment=augment_real,
            augment_factor=augment_factor,
        )
        metadata.update(hda_meta)
        idx += n_hda + n_hda_aug
        stats["hda_real"] = n_hda
        stats["hda_augmented"] = n_hda_aug
        print(f"  Added {n_hda} HDA real + {n_hda_aug} HDA augmented pairs")
    else:
        print("\nNo HDA directory found (skipping)")

    total = sum(stats.values())
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {total} training pairs")
    print(f"  Synthetic Wave 1: {stats['synthetic_w1']}")
    print(f"  Synthetic Wave 2: {stats['synthetic_w2']}")
    print(f"  Synthetic Wave 3: {stats['synthetic_w3']}")
    print(f"  Real:             {stats['real']}")
    print(f"  Augmented:        {stats['augmented']}")
    print(f"  HDA Real:         {stats['hda_real']}")
    print(f"  HDA Augmented:    {stats['hda_augmented']}")
    print(f"{'=' * 50}")

    # Save metadata for test split stratification
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "pairs": metadata}, f)
    print(f"Metadata saved: {meta_path}")

    return total


def _augment_image(img: np.ndarray, aug_idx: int) -> np.ndarray | None:
    """Apply deterministic augmentation based on index."""
    if img is None:
        return None

    h, w = img.shape[:2]

    if aug_idx == 0:
        return cv2.flip(img, 1)
    elif aug_idx == 1:
        return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    elif aug_idx == 2:
        return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    elif aug_idx == 3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 3, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif aug_idx == 4:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -3, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    else:
        return img.copy()


def main():
    parser = argparse.ArgumentParser(description="Build combined training dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_combined",
        help="Output directory for combined dataset",
    )
    parser.add_argument(
        "--synthetic_dirs",
        nargs="+",
        default=["data/synthetic_surgery_pairs", "data/synthetic_surgery_pairs_v2"],
        help="Directories with synthetic pairs (each has procedure subdirs)",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="data/real_surgery_pairs/pairs",
        help="Directory with processed real pairs",
    )
    parser.add_argument(
        "--v3_dir",
        type=str,
        default="data/synthetic_surgery_pairs_v3",
        help="Directory with wave 3 (realistic displacement) pairs",
    )
    parser.add_argument(
        "--hda_dir",
        type=str,
        default="data/hda_processed",
        help="Directory with processed HDA Plastic Surgery pairs",
    )
    parser.add_argument(
        "--procedures",
        nargs="+",
        default=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"],
    )
    parser.add_argument("--no_augment", action="store_true", help="Skip augmenting real pairs")
    parser.add_argument(
        "--augment_factor", type=int, default=5, help="Number of augmented copies per real pair"
    )
    parser.add_argument(
        "--no_clear", action="store_true", help="Don't clear existing output directory"
    )
    args = parser.parse_args()

    # Build list of all synthetic subdirectories
    synthetic_dirs = []
    for base in args.synthetic_dirs:
        base_path = Path(base)
        for proc in args.procedures:
            proc_dir = base_path / proc
            if proc_dir.exists():
                synthetic_dirs.append(proc_dir)

    real_dir = Path(args.real_dir) if args.real_dir else None
    v3_dir = Path(args.v3_dir) if args.v3_dir else None
    hda_dir = Path(args.hda_dir) if args.hda_dir else None

    print(f"Synthetic sources: {len(synthetic_dirs)} procedure directories")
    print(f"Wave 3 dir: {v3_dir}")
    print(f"Real pairs dir: {real_dir}")
    print(f"HDA dir: {hda_dir}")
    print()

    build_dataset(
        synthetic_dirs=synthetic_dirs,
        real_pairs_dir=real_dir,
        output_dir=Path(args.output),
        v3_dir=v3_dir,
        hda_dir=hda_dir,
        augment_real=not args.no_augment,
        augment_factor=args.augment_factor,
        clear_existing=not args.no_clear,
    )


if __name__ == "__main__":
    main()
