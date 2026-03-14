"""Download FFHQ faces from HuggingFace (128 or 1024 res)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_from_huggingface(
    num_images: int = 1000,
    resolution: int = 128,
    output_dir: str = "data/ffhq",
    seed: int = 42,
) -> None:
    """Download FFHQ subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_images} FFHQ images at {resolution}x{resolution}...")
    print(f"Output: {out}")

    # Face dataset mirrors (ordered by reliability)
    dataset_ids = [
        "nielsr/CelebA-faces",
        "jlbaker361/CelebA-HQ-256",
        "merkol/ffhq-thumbnails-128",
    ]

    dataset = None
    for ds_id in dataset_ids:
        try:
            dataset = load_dataset(ds_id, split="train", streaming=True)
            print(f"  Using dataset: {ds_id}")
            break
        except Exception:
            continue

    if dataset is None:
        print("Could not access any face dataset. Trying wget fallback...")
        download_with_wget(num_images, output_dir)
        return

    count = 0
    for _i, sample in enumerate(dataset):
        if count >= num_images:
            break

        img = sample.get("image") or sample.get("img")
        if img is None:
            continue

        if resolution != img.size[0]:
            img = img.resize((resolution, resolution))

        img.save(out / f"{count:06d}.png")
        count += 1

        if count % 100 == 0:
            print(f"  Downloaded {count}/{num_images}")

    print(f"Done. {count} images saved to {out}")


def download_with_wget(
    num_images: int = 1000,
    output_dir: str = "data/ffhq",
) -> None:
    """Wget fallback if HuggingFace datasets isn't installed."""
    import urllib.request

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/merkol/ffhq-thumbnails-128/resolve/main/data"

    print(f"Downloading {num_images} FFHQ thumbnails...")
    for i in range(min(num_images, 70000)):
        url = f"{base_url}/{i:05d}.png"
        dest = out / f"{i:06d}.png"

        if dest.exists():
            continue

        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"  Downloaded {i + 1}/{num_images}")

    actual = len(list(out.glob("*.png")))
    print(f"Done. {actual} images in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FFHQ dataset")
    parser.add_argument("--num", type=int, default=1000, help="Number of images")
    parser.add_argument("--resolution", type=int, default=128, choices=[128, 256, 512, 1024])
    parser.add_argument("--output", default="data/ffhq", help="Output directory")
    parser.add_argument("--method", default="hf", choices=["hf", "wget"])
    args = parser.parse_args()

    if args.method == "hf":
        download_from_huggingface(args.num, args.resolution, args.output)
    else:
        download_with_wget(args.num, args.output)
