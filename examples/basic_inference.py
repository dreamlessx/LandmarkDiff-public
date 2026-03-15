"""Basic inference example - predict surgical outcome for a single image."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset


def main():
    parser = argparse.ArgumentParser(description="Basic LandmarkDiff inference")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument(
        "--procedure",
        type=str,
        default="rhinoplasty",
        choices=[
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
    )
    parser.add_argument(
        "--intensity", type=float, default=60.0, help="Deformation intensity (0-100)"
    )
    parser.add_argument("--output", type=str, default="output/")
    parser.add_argument(
        "--mode", type=str, default="controlnet", choices=["controlnet", "img2img", "tps"]
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load image
    img = Image.open(args.image).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    # extract landmarks
    print(f"Extracting landmarks from {args.image}...")
    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected in image")
        return

    print(f"  Detected {len(landmarks.landmarks)} landmarks")

    # deform landmarks
    print(f"Applying {args.procedure} deformation (intensity={args.intensity})...")
    deformed = apply_procedure_preset(landmarks, args.procedure, intensity=args.intensity)

    # visualize mesh (always works, no GPU)
    original_mesh = render_wireframe(landmarks, (512, 512))
    deformed_mesh = render_wireframe(deformed, (512, 512))

    import cv2

    cv2.imwrite(str(output_dir / "mesh_original.png"), original_mesh)
    cv2.imwrite(str(output_dir / "mesh_deformed.png"), deformed_mesh)
    print(f"  Saved mesh visualizations to {output_dir}/")

    # full diffusion prediction (requires GPU)
    if args.mode in ("controlnet", "img2img"):
        try:
            from landmarkdiff.inference import LandmarkDiffPipeline

            print("Loading diffusion pipeline...")
            pipeline = LandmarkDiffPipeline(mode=args.mode, device="cuda")
            pipeline.load()

            print("Generating prediction...")
            result = pipeline.generate(
                img_array,
                procedure=args.procedure,
                intensity=args.intensity,
                mode=args.mode,
            )

            result["output"].save(str(output_dir / "prediction.png"))
            print(f"  Saved prediction to {output_dir}/")

        except Exception as e:
            print(f"  Diffusion pipeline not available: {e}")
            print("  Use --mode tps for CPU-only mode")

    elif args.mode == "tps":
        from landmarkdiff.synthetic.tps_warp import warp_image_tps

        src = landmarks.pixel_coords[:, :2].copy()
        dst = deformed.pixel_coords[:, :2].copy()
        src[:, 0] *= 512 / landmarks.image_width
        src[:, 1] *= 512 / landmarks.image_height
        dst[:, 0] *= 512 / deformed.image_width
        dst[:, 1] *= 512 / deformed.image_height

        warped = warp_image_tps(img_array, src, dst)
        Image.fromarray(warped).save(str(output_dir / "prediction_tps.png"))
        print(f"  Saved TPS prediction to {output_dir}/prediction_tps.png")

    print("Done!")


if __name__ == "__main__":
    main()
