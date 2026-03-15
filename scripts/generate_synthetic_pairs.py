"""Generate synthetic before/after pairs at scale from CelebA/FFHQ faces."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset


def process_single_image(
    img_path: str,
    procedure: str,
    intensity: float,
    output_dir: str,
    pair_idx: int,
    target_size: int = 512,
) -> dict | None:
    """Process a single face image into a training pair."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Resize to target
        img = cv2.resize(img, (target_size, target_size))

        # Extract landmarks
        face = extract_landmarks(img)
        if face is None:
            return None

        # Apply surgical manipulation
        manipulated = apply_procedure_preset(face, procedure, intensity, target_size)

        # Render meshes
        original_mesh = render_landmark_image(face, target_size, target_size)
        manipulated_mesh = render_landmark_image(manipulated, target_size, target_size)

        out = Path(output_dir)

        # ControlNet training pair:
        # conditioning = original mesh (the face structure the model sees)
        # target = original realistic face (what the model learns to generate)
        #
        # At inference time, we feed the MANIPULATED mesh as conditioning,
        # and the model generates a realistic face matching that structure.
        # This way the model always generates realistic faces - never TPS artifacts.
        #
        # We also save the manipulated mesh for inference-time use.
        input_path = out / f"{pair_idx:06d}_{procedure}_input.png"
        target_path = out / f"{pair_idx:06d}_{procedure}_target.png"
        manip_mesh_path = out / f"{pair_idx:06d}_{procedure}_manip_mesh.png"

        cv2.imwrite(str(input_path), original_mesh)  # conditioning = mesh
        cv2.imwrite(str(target_path), img)  # target = real face
        cv2.imwrite(str(manip_mesh_path), manipulated_mesh)  # for inference

        # Compute displacement magnitude
        src_pts = face.pixel_coords
        dst_pts = manipulated.pixel_coords
        disp = dst_pts - src_pts
        mean_disp = float(np.mean(np.abs(disp)))

        return {
            "pair_id": pair_idx,
            "procedure": procedure,
            "source_image": img_path,
            "intensity": intensity,
            "input_path": str(input_path),
            "target_path": str(target_path),
            "manip_mesh_path": str(manip_mesh_path),
            "mean_displacement": mean_disp,
        }

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic surgery training pairs")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory with source face images (CelebA, FFHQ, etc.)",
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic_surgery_pairs", help="Output directory"
    )
    parser.add_argument(
        "--procedure",
        type=str,
        required=True,
        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"],
        help="Procedure to generate pairs for",
    )
    parser.add_argument("--target", type=int, default=50000, help="Target number of pairs")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--offset", type=int, default=0, help="Starting pair index offset")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output) / args.procedure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all source images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = sorted([str(f) for f in source_dir.rglob("*") if f.suffix.lower() in extensions])
    print(f"Found {len(all_images)} source images in {source_dir}")

    if not all_images:
        print("No images found!")
        return

    # Generate pairs with varying intensities
    # Each image can produce multiple pairs at different intensities
    pairs_per_image = max(1, args.target // len(all_images) + 1)
    print(f"Generating ~{pairs_per_image} pairs per image")
    print(f"Target: {args.target} pairs for {args.procedure}")

    intensity_range = (20, 100)  # mild to aggressive
    all_pairs = []
    pair_idx = args.offset
    processed = 0
    failed = 0

    for img_i, img_path in enumerate(all_images):
        if pair_idx - args.offset >= args.target:
            break

        # Generate multiple intensities for this image
        intensities = [random.uniform(*intensity_range) for _ in range(pairs_per_image)]

        for intensity in intensities:
            if pair_idx - args.offset >= args.target:
                break

            result = process_single_image(
                img_path, args.procedure, intensity, str(output_dir), pair_idx, args.size
            )

            if result is not None:
                all_pairs.append(result)
                pair_idx += 1
                processed += 1
            else:
                failed += 1

        if (img_i + 1) % 100 == 0:
            print(f"  [{img_i + 1}/{len(all_images)}] Generated {processed} pairs, {failed} failed")

    # Save metadata
    metadata_path = Path(args.output) / f"{args.procedure}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print("\n=== Summary ===")
    print(f"Procedure: {args.procedure}")
    print(f"Source images: {len(all_images)}")
    print(f"Generated pairs: {processed}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
