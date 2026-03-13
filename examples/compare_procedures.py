"""Compare all procedures side-by-side on a single face."""

import argparse
from pathlib import Path
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset, PROCEDURE_PRESETS
from landmarkdiff.conditioning import draw_tessellation


def main():
    parser = argparse.ArgumentParser(description="Compare all procedures")
    parser.add_argument("image", type=str, help="Path to input face image")
    parser.add_argument("--intensity", type=float, default=0.6)
    parser.add_argument("--output", type=str, default="output/comparison.png")
    args = parser.parse_args()

    landmarks = extract_landmarks(args.image)
    if landmarks is None:
        print("No face detected")
        return

    procedures = list(PROCEDURE_PRESETS.keys())
    meshes = []

    # Original
    original_mesh = draw_tessellation(landmarks, (512, 512))
    meshes.append(("Original", Image.fromarray(original_mesh)))

    # Each procedure
    for proc in procedures:
        deformed = apply_procedure_preset(landmarks, proc, intensity=args.intensity)
        mesh = draw_tessellation(deformed, (512, 512))
        meshes.append((proc.capitalize(), Image.fromarray(mesh)))

    # Create grid
    n = len(meshes)
    grid = Image.new("L", (512 * n, 512 + 40), 0)

    from PIL import ImageDraw
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
