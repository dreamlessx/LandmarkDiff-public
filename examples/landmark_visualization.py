"""Visualize face mesh landmarks and deformation vectors."""

import argparse
import numpy as np
from pathlib import Path

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import draw_tessellation


def main():
    parser = argparse.ArgumentParser(description="Visualize landmarks and deformations")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--procedure", type=str, default="rhinoplasty")
    parser.add_argument("--output", type=str, default="output/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract landmarks
    landmarks = extract_landmarks(args.image)
    if landmarks is None:
        print("No face detected")
        return

    # Draw original mesh
    mesh_original = draw_tessellation(landmarks, (512, 512))

    # Deform at multiple intensities
    import cv2
    for intensity in [0.2, 0.4, 0.6, 0.8, 1.0]:
        deformed = apply_procedure_preset(landmarks, args.procedure, intensity=intensity)
        mesh_deformed = draw_tessellation(deformed, (512, 512))

        # Draw displacement vectors
        vis = np.zeros((512, 512, 3), dtype=np.uint8)
        vis[:, :, 0] = mesh_original  # original in blue
        vis[:, :, 2] = mesh_deformed  # deformed in red

        original_px = landmarks.pixel_coords[:, :2] * np.array([512 / landmarks.image_width, 512 / landmarks.image_height])
        deformed_px = deformed.pixel_coords[:, :2] * np.array([512 / deformed.image_width, 512 / deformed.image_height])

        # Draw displacement arrows for moved landmarks
        for i in range(len(original_px)):
            dist = np.linalg.norm(deformed_px[i] - original_px[i])
            if dist > 1.0:  # only draw significant displacements
                pt1 = tuple(original_px[i].astype(int))
                pt2 = tuple(deformed_px[i].astype(int))
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

        cv2.imwrite(str(output_dir / f"deformation_{args.procedure}_{int(intensity*100)}.png"), vis)
        print(f"  Saved intensity {intensity:.0%} visualization")

    print(f"Done! Check {output_dir}/")


if __name__ == "__main__":
    main()
