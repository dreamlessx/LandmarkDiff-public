"""Generate synthetic training pairs from FFHQ images.

Usage:
    # Generate 1000 pairs from downloaded FFHQ
    python scripts/generate_synthetic_data.py --input data/ffhq --num 1000

    # Generate with specific procedure focus
    python scripts/generate_synthetic_data.py --input data/ffhq --num 500 --procedure rhinoplasty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.synthetic.pair_generator import generate_pair, save_pair, PROCEDURES


def main(
    input_dir: str,
    output_dir: str = "data/synthetic_pairs",
    num_pairs: int = 1000,
    procedure: str | None = None,
    seed: int = 42,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in extensions)

    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} source images in {input_dir}")
    print(f"Generating {num_pairs} synthetic training pairs...")
    if procedure:
        print(f"Procedure: {procedure}")
    else:
        print(f"Procedures: {', '.join(PROCEDURES)} (random)")

    rng = np.random.default_rng(seed)
    generated = 0
    failed = 0

    for i in range(num_pairs * 2):  # oversample to account for face detection failures
        if generated >= num_pairs:
            break

        img_path = images[i % len(images)]
        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
            continue

        pair = generate_pair(image, procedure=procedure, rng=rng)
        if pair is None:
            failed += 1
            continue

        save_pair(pair, output_path, generated)
        generated += 1

        if generated % 50 == 0:
            print(f"  Generated {generated}/{num_pairs} (failed: {failed})")

    print(f"\nDone. {generated} pairs saved to {output_path}")
    print(f"Face detection failures: {failed}")

    # Save metadata
    meta_path = output_path / "metadata.txt"
    meta_path.write_text(
        f"source: {input_dir}\n"
        f"pairs: {generated}\n"
        f"procedure: {procedure or 'random'}\n"
        f"seed: {seed}\n"
        f"failures: {failed}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs")
    parser.add_argument("--input", required=True, help="Directory of face images")
    parser.add_argument("--output", default="data/synthetic_pairs")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--procedure", choices=PROCEDURES, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.output, args.num, args.procedure, args.seed)
