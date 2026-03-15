"""Generate synthetic pairs at scale - all procedures, varied intensities."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def find_all_face_images(data_root: Path) -> list[Path]:
    """Find all face images across all dataset directories."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = []

    # Check multiple possible locations
    search_dirs = [
        data_root / "ffhq",
        data_root / "faces_multi" / "celeba_hq",
        data_root / "faces_multi" / "ffhq",
        data_root / "faces_multi" / "celeba",
        data_root / "faces_multi" / "fairface",
        data_root / "faces_multi" / "lfw",
        data_root / "faces_all",
    ]

    for d in search_dirs:
        if d.exists():
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in extensions)
            image_files.extend(files)
            print(f"  Found {len(files)} images in {d}")

    # Deduplicate by absolute path
    seen = set()
    unique = []
    for f in image_files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def generate_single_pair(args: tuple) -> bool:
    """Generate a single pair (called in worker process)."""
    img_path, pair_idx, output_dir, procedure, intensity, seed = args

    try:
        # Import inside worker to avoid MediaPipe fork issues
        from landmarkdiff.synthetic.pair_generator import generate_pair, save_pair

        image = cv2.imread(str(img_path))
        if image is None:
            return False

        rng = np.random.default_rng(seed)
        pair = generate_pair(image, procedure=procedure, intensity=intensity, rng=rng)
        if pair is None:
            return False

        save_pair(pair, Path(output_dir), pair_idx)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate scaled synthetic training pairs")
    parser.add_argument("--num", type=int, default=10000, help="Total pairs to generate")
    parser.add_argument("--data_root", default="data", help="Root data directory")
    parser.add_argument("--output", default="data/synthetic_pairs_v2", help="Output directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (1=sequential, avoids MediaPipe issues)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for face images...")
    image_files = find_all_face_images(data_root)
    if not image_files:
        print("ERROR: No face images found. Run download scripts first.")
        return

    print(f"\nTotal source images: {len(image_files)}")
    print(f"Target pairs: {args.num}")
    print(f"Output: {output_dir}")

    # Check existing pairs
    existing = len(list(output_dir.glob("*_input.png")))
    if existing > 0:
        print(f"Found {existing} existing pairs, continuing from {existing}")

    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    rng = np.random.default_rng(args.seed)

    # Build work items: cycle through images × random procedures
    work_items = []
    for pair_idx in range(existing, args.num):
        img_path = image_files[pair_idx % len(image_files)]
        procedure = procedures[pair_idx % len(procedures)]
        intensity = float(rng.uniform(25, 95))
        seed = int(rng.integers(0, 2**31))
        work_items.append((img_path, pair_idx, str(output_dir), procedure, intensity, seed))

    print(f"\nGenerating {len(work_items)} pairs...")
    start = time.time()
    success = 0
    failed = 0

    if args.workers <= 1:
        # Sequential - safer with MediaPipe
        from landmarkdiff.synthetic.pair_generator import generate_pair, save_pair

        for _i, (img_path, pair_idx, out, procedure, intensity, seed) in enumerate(work_items):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    failed += 1
                    continue

                pair_rng = np.random.default_rng(seed)
                pair = generate_pair(image, procedure=procedure, intensity=intensity, rng=pair_rng)
                if pair is None:
                    failed += 1
                    continue

                save_pair(pair, Path(out), pair_idx)
                success += 1
            except Exception:
                failed += 1
                continue

            if (success + failed) % 100 == 0:
                elapsed = time.time() - start
                rate = success / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - success - failed) / rate if rate > 0 else 0
                print(
                    f"  Progress: {success}/{args.num} pairs | "
                    f"{failed} failed | "
                    f"{rate:.1f} pairs/sec | "
                    f"ETA: {eta / 60:.1f} min"
                )
    else:
        # Parallel - use ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_single_pair, item): item for item in work_items}
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1

                total_done = success + failed
                if total_done % 100 == 0:
                    elapsed = time.time() - start
                    rate = success / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {success}/{args.num} | {failed} failed | {rate:.1f}/sec")

    elapsed = time.time() - start
    print(f"\nDone! {success} pairs generated in {elapsed / 60:.1f} min ({failed} failed)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
