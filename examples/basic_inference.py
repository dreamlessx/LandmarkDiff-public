"""Basic inference example - predict surgical outcome for a single image."""

import argparse
from pathlib import Path

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import draw_tessellation


def main():
    parser = argparse.ArgumentParser(description="Basic LandmarkDiff inference")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty",
                        choices=["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"])
    parser.add_argument("--intensity", type=float, default=0.6)
    parser.add_argument("--output", type=str, default="output/")
    parser.add_argument("--mode", type=str, default="controlnet",
                        choices=["controlnet", "img2img", "tps"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract landmarks
    print(f"Extracting landmarks from {args.image}...")
    landmarks = extract_landmarks(args.image)
    if landmarks is None:
        print("No face detected in image")
        return

    print(f"  Detected {len(landmarks.points)} landmarks")

    # Step 2: Deform landmarks
    print(f"Applying {args.procedure} deformation (intensity={args.intensity})...")
    deformed = apply_procedure_preset(landmarks, args.procedure, intensity=args.intensity)

    # Step 3: Visualize deformation (always works, no GPU needed)
    original_mesh = draw_tessellation(landmarks, (512, 512))
    deformed_mesh = draw_tessellation(deformed, (512, 512))

    import cv2
    cv2.imwrite(str(output_dir / "mesh_original.png"), original_mesh)
    cv2.imwrite(str(output_dir / "mesh_deformed.png"), deformed_mesh)
    print(f"  Saved mesh visualizations to {output_dir}/")

    # Step 4: Full diffusion prediction (requires GPU)
    if args.mode in ("controlnet", "img2img"):
        try:
            from landmarkdiff.inference import LandmarkDiffPipeline

            print("Loading diffusion pipeline...")
            pipeline = LandmarkDiffPipeline.from_pretrained("checkpoints/latest")

            print("Generating prediction...")
            result = pipeline.generate(
                args.image,
                procedure=args.procedure,
                intensity=args.intensity,
                mode=args.mode,
            )

            result["prediction"].save(str(output_dir / "prediction.png"))
            result["comparison"].save(str(output_dir / "comparison.png"))
            print(f"  Saved prediction to {output_dir}/")

        except Exception as e:
            print(f"  Diffusion pipeline not available: {e}")
            print("  Use --mode tps for CPU-only mode")

    elif args.mode == "tps":
        from landmarkdiff.synthetic.tps_warp import tps_warp
        from PIL import Image
        import numpy as np

        img = np.array(Image.open(args.image).resize((512, 512)))
        warped = tps_warp(img, landmarks.pixel_coords()[:, :2], deformed.pixel_coords()[:, :2])
        Image.fromarray(warped).save(str(output_dir / "prediction_tps.png"))
        print(f"  Saved TPS prediction to {output_dir}/prediction_tps.png")

    print("Done!")


if __name__ == "__main__":
    main()
