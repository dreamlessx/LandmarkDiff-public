"""Compare all procedures side-by-side on a single face."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS, apply_procedure_preset


def main():
    parser = argparse.ArgumentParser(description="Compare all procedures")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument(
        "--intensity", type=float, default=60.0, help="Deformation intensity (0-100)"
    )
    parser.add_argument("--output", type=str, default="output/comparison.png")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB").resize((512, 512))
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected")
        return

    procedures = list(PROCEDURE_LANDMARKS.keys())
    meshes = []

    # original
    original_mesh = render_wireframe(landmarks, (512, 512))
    meshes.append(("Original", Image.fromarray(original_mesh)))

    # each procedure
    for proc in procedures:
        deformed = apply_procedure_preset(landmarks, proc, intensity=args.intensity)
        mesh = render_wireframe(deformed, (512, 512))
        meshes.append((proc.capitalize(), Image.fromarray(mesh)))

    # create grid
    n = len(meshes)
    grid = Image.new("L", (512 * n, 512 + 40), 0)
    draw = ImageDraw.Draw(grid)

    for i, (name, mesh) in enumerate(meshes):
        grid.paste(mesh, (512 * i, 40))
        draw.text((512 * i + 200, 10), name, fill=255)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))
    print(f"Saved comparison grid to {output_path}")


if __name__ == "__main__":
    main()
