"""Visualize face mesh landmarks and deformation vectors."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset


def main():
    parser = argparse.ArgumentParser(description="Visualize landmarks and deformations")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty")
    parser.add_argument("--output", type=str, default="output/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load image and extract landmarks
    img = Image.open(args.image).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected")
        return

    # draw original mesh
    mesh_original = render_wireframe(landmarks, (512, 512))

    # deform at multiple intensities
    import cv2

    for intensity in [20, 40, 60, 80, 100]:
        deformed = apply_procedure_preset(landmarks, args.procedure, intensity=intensity)
        mesh_deformed = render_wireframe(deformed, (512, 512))

        # overlay: original in blue, deformed in red
        vis = np.zeros((512, 512, 3), dtype=np.uint8)
        vis[:, :, 0] = mesh_original  # original in blue channel
        vis[:, :, 2] = mesh_deformed  # deformed in red channel

        scale_orig = np.array([512 / landmarks.image_width, 512 / landmarks.image_height])
        scale_def = np.array([512 / deformed.image_width, 512 / deformed.image_height])
        original_px = landmarks.pixel_coords[:, :2] * scale_orig
        deformed_px = deformed.pixel_coords[:, :2] * scale_def

        # draw displacement arrows for moved landmarks
        for i in range(len(original_px)):
            dist = np.linalg.norm(deformed_px[i] - original_px[i])
            if dist > 1.0:
                pt1 = tuple(original_px[i].astype(int))
                pt2 = tuple(deformed_px[i].astype(int))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

        cv2.imwrite(str(output_dir / f"deformation_{args.procedure}_{intensity}.png"), vis)
        print(f"  Saved intensity {intensity}% visualization")

    print(f"Done! Check {output_dir}/")


if __name__ == "__main__":
    main()
