"""Download face images from multiple free online datasets.

Sources (no auth required):
1. CelebA-HQ 256x256 (korexyz/celeba-hq-256x256) — 30K high-quality aligned faces
2. FFHQ 256x256 (merkol/ffhq-256) — 70K NVIDIA FFHQ faces
3. FairFace (HuggingFaceM4/FairFace) — 108K diverse demographics
4. LFW (bitmind/lfw) — 13K labeled faces in the wild

All resized to 512x512 for ControlNet training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image


DATASETS = {
    "celeba_hq": {
        "hf_id": "korexyz/celeba-hq-256x256",
        "split": "train",
        "image_key": "image",
        "max_default": 5000,
        "description": "CelebA-HQ 256x256 (30K aligned celebrity faces)",
    },
    "ffhq": {
        "hf_id": "merkol/ffhq-256",
        "split": "train",
        "image_key": "image",
        "max_default": 5000,
        "description": "FFHQ 256x256 (70K Flickr faces)",
    },
    "fairface": {
        "hf_id": "HuggingFaceM4/FairFace",
        "split": "train",
        "image_key": "image",
        "max_default": 3000,
        "description": "FairFace (108K diverse demographics, CC BY 4.0)",
    },
    "lfw": {
        "hf_id": "bitmind/lfw",
        "split": "train",
        "image_key": "image",
        "max_default": 2000,
        "description": "LFW (13K labeled faces in the wild)",
    },
    "celeba": {
        "hf_id": "nielsr/CelebA-faces",
        "split": "train",
        "image_key": "image",
        "max_default": 5000,
        "description": "CelebA (200K celebrity faces, lower res)",
    },
}


def download_dataset(
    dataset_name: str,
    num_images: int,
    output_dir: Path,
    resolution: int = 512,
    start_idx: int = 0,
) -> int:
    """Download images from a HuggingFace dataset.

    Returns number of images successfully downloaded.
    """
    from datasets import load_dataset

    config = DATASETS[dataset_name]
    ds_dir = output_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(ds_dir.glob("*.png")))
    if existing >= num_images:
        print(f"  {dataset_name}: {existing} images already exist, skipping")
        return existing

    print(f"  {dataset_name}: downloading {num_images} from {config['hf_id']}...")

    try:
        dataset = load_dataset(config["hf_id"], split=config["split"], streaming=True)
    except Exception as e:
        print(f"  {dataset_name}: FAILED to load — {e}")
        return 0

    count = existing
    img_key = config["image_key"]

    # Try alternative image keys if primary fails
    alt_keys = ["image", "img", "pixel_values", "face"]

    for i, sample in enumerate(dataset):
        if count >= num_images:
            break

        dest = ds_dir / f"{count:06d}.png"
        if dest.exists():
            count += 1
            continue

        img = None
        for key in [img_key] + alt_keys:
            img = sample.get(key)
            if img is not None:
                break

        if img is None:
            continue

        # Handle different image formats
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue

        # Resize to target resolution
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(dest)
        count += 1

        if count % 500 == 0:
            print(f"    {dataset_name}: {count}/{num_images}")

    print(f"  {dataset_name}: {count} images saved to {ds_dir}")
    return count


def consolidate(output_dir: Path, consolidated_dir: Path) -> int:
    """Copy all dataset images into a single flat directory for pair generation."""
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in consolidated_dir.glob("*.png"))

    idx = len(existing)
    total_new = 0

    for ds_dir in sorted(output_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for img_path in sorted(ds_dir.glob("*.png")):
            dest = consolidated_dir / f"{idx:06d}.png"
            if not dest.exists():
                import shutil
                shutil.copy2(img_path, dest)
                total_new += 1
            idx += 1

    print(f"\nConsolidated: {idx} total images in {consolidated_dir} ({total_new} new)")
    return idx


def main():
    parser = argparse.ArgumentParser(description="Download faces from multiple datasets")
    parser.add_argument("--output", default="data/faces_multi", help="Output directory")
    parser.add_argument("--consolidated", default="data/faces_all", help="Consolidated output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--datasets", nargs="+", default=["celeba_hq", "ffhq", "celeba", "fairface"],
                        choices=list(DATASETS.keys()))
    parser.add_argument("--num", type=int, default=None,
                        help="Override per-dataset count (default: use dataset-specific defaults)")
    parser.add_argument("--skip-consolidate", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for ds_name in args.datasets:
        num = args.num or DATASETS[ds_name]["max_default"]
        desc = DATASETS[ds_name]["description"]
        print(f"\n{'='*60}")
        print(f"Dataset: {desc}")
        print(f"Target: {num} images at {args.resolution}x{args.resolution}")
        print(f"{'='*60}")
        count = download_dataset(ds_name, num, out, args.resolution)
        total += count

    print(f"\n{'='*60}")
    print(f"Total downloaded: {total} images across {len(args.datasets)} datasets")
    print(f"{'='*60}")

    if not args.skip_consolidate:
        consolidate(out, Path(args.consolidated))


if __name__ == "__main__":
    main()
