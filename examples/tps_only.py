"""CPU-only inference using thin-plate spline warping (no GPU needed)."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.synthetic.tps_warp import warp_image_tps


def main():
    parser = argparse.ArgumentParser(description="TPS-only inference (CPU)")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty")
    parser.add_argument(
        "--intensity", type=float, default=60.0, help="Deformation intensity (0-100)"
    )
    parser.add_argument("--output", type=str, default="output/tps_result.png")
    args = parser.parse_args()

    # load and resize
    img = Image.open(args.image).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    # extract and deform landmarks
    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected")
        return

    deformed = apply_procedure_preset(landmarks, args.procedure, intensity=args.intensity)

    # apply TPS warp
    src_pts = landmarks.pixel_coords[:, :2].copy()
    dst_pts = deformed.pixel_coords[:, :2].copy()

    # scale to 512x512
    src_pts[:, 0] *= 512 / landmarks.image_width
    src_pts[:, 1] *= 512 / landmarks.image_height
    dst_pts[:, 0] *= 512 / deformed.image_width
    dst_pts[:, 1] *= 512 / deformed.image_height

    warped = warp_image_tps(img_array, src_pts, dst_pts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(warped).save(str(output_path))
    print(f"Saved TPS result to {output_path}")
    print("Note: TPS mode produces geometric-only results without photorealistic generation.")
    print("For photorealistic output, use --mode controlnet (requires GPU).")


if __name__ == "__main__":
    main()
