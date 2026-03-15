"""Visualize real surgical displacement model for paper figures.

Generates:
1. Per-procedure displacement heatmaps overlaid on face mesh
2. Landmark displacement magnitude bar charts
3. Procedure comparison showing how different surgeries move different regions

Usage:
    python scripts/visualize_displacements.py \
        --model data/displacement_model.npz \
        --output paper/figures/displacements/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.displacement_model import DisplacementModel

# MediaPipe face mesh connections for drawing the mesh
# Subset of key connections for clean visualization
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263]
NOSE = [
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    19,
    94,
    2,
    164,
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    200,
    199,
    175,
    152,
]
LIPS_OUTER = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    61,
]


def create_mesh_canvas(size: int = 512, bg_color: int = 30) -> np.ndarray:
    """Create a dark canvas for mesh overlay."""
    return np.full((size, size, 3), bg_color, dtype=np.uint8)


def draw_mesh(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int] = (100, 100, 100),
    thickness: int = 1,
) -> None:
    """Draw face mesh outline on canvas."""
    h, w = canvas.shape[:2]

    for connections in [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS_OUTER]:
        for i in range(len(connections) - 1):
            idx1, idx2 = connections[i], connections[i + 1]
            if idx1 >= len(landmarks) or idx2 >= len(landmarks):
                continue
            pt1 = (int(landmarks[idx1, 0] * w), int(landmarks[idx1, 1] * h))
            pt2 = (int(landmarks[idx2, 0] * w), int(landmarks[idx2, 1] * h))
            cv2.line(canvas, pt1, pt2, color, thickness)


def draw_displacement_arrows(
    canvas: np.ndarray,
    base_landmarks: np.ndarray,
    displacements: np.ndarray,
    scale: float = 20.0,
    min_magnitude: float = 0.002,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """Draw displacement arrows with magnitude-based color coding."""
    h, w = canvas.shape[:2]
    magnitudes = np.linalg.norm(displacements, axis=1)
    max_mag = magnitudes.max() if magnitudes.max() > 0 else 1.0

    for i in range(len(base_landmarks)):
        mag = magnitudes[i]
        if mag < min_magnitude:
            continue

        bx = int(base_landmarks[i, 0] * w)
        by = int(base_landmarks[i, 1] * h)
        dx = int(displacements[i, 0] * w * scale)
        dy = int(displacements[i, 1] * h * scale)

        # Color by magnitude (normalized)
        norm_mag = int(mag / max_mag * 255)
        color_img = np.zeros((1, 1, 3), dtype=np.uint8)
        color_img[0, 0] = norm_mag
        colored = cv2.applyColorMap(color_img, colormap)
        color = tuple(int(c) for c in colored[0, 0])

        cv2.arrowedLine(canvas, (bx, by), (bx + dx, by + dy), color, 2, tipLength=0.3)
        cv2.circle(canvas, (bx, by), 2, color, -1)


def draw_heatmap(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    magnitudes: np.ndarray,
    radius: int = 20,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a heatmap overlay showing displacement magnitudes."""
    h, w = canvas.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    max_mag = magnitudes.max() if magnitudes.max() > 0 else 1.0

    for i in range(len(landmarks)):
        x = int(landmarks[i, 0] * w)
        y = int(landmarks[i, 1] * h)
        intensity = magnitudes[i] / max_mag

        # Gaussian splat
        cv2.circle(heat, (x, y), radius, float(intensity), -1)

    # Blur for smooth heatmap
    heat = cv2.GaussianBlur(heat, (31, 31), 0)
    heat = (heat / heat.max() * 255).astype(np.uint8) if heat.max() > 0 else heat.astype(np.uint8)

    colored = cv2.applyColorMap(heat, colormap)

    # Blend with canvas
    mask = (heat > 5).astype(np.float32)[:, :, np.newaxis]
    blended = (
        canvas.astype(np.float32) * (1 - mask * 0.7) + colored.astype(np.float32) * mask * 0.7
    ).astype(np.uint8)
    return blended


def create_procedure_visualization(
    model: DisplacementModel,
    procedure: str,
    size: int = 512,
    arrow_scale: float = 25.0,
) -> np.ndarray:
    """Create a comprehensive visualization for one procedure."""
    # Use a canonical face position (centered, normalized)
    # Generate landmarks at face center
    n = 478
    base_landmarks = np.zeros((n, 2), dtype=np.float32)
    # Use mean landmarks from the model's displacement mean as reference
    # Place them centered (displacements are relative, so use 0.5 center)
    base_landmarks[:, 0] = 0.5
    base_landmarks[:, 1] = 0.5

    # Get displacement field
    displacement = model.get_displacement_field(procedure, intensity=1.0)
    magnitudes = np.linalg.norm(displacement, axis=1)

    # Create visualization panels
    # Panel 1: Arrow visualization on mesh
    arrow_canvas = create_mesh_canvas(size, bg_color=20)

    # We need a real face image for the mesh. Use the displacement
    # field directly on a grid.
    # Just show the arrows at grid positions corresponding to face landmarks
    # Create a face-shaped arrangement using default MediaPipe proportions
    default_face = _get_default_face_positions(n)

    draw_mesh(arrow_canvas, default_face, color=(60, 60, 60))
    draw_displacement_arrows(
        arrow_canvas,
        default_face,
        displacement,
        scale=arrow_scale,
        min_magnitude=0.001,
    )

    # Panel 2: Heatmap
    heat_canvas = create_mesh_canvas(size, bg_color=20)
    draw_mesh(heat_canvas, default_face, color=(60, 60, 60))
    heat_canvas = draw_heatmap(heat_canvas, default_face, magnitudes, radius=15)

    # Add labels
    proc_label = procedure.capitalize()
    n_samples = model.n_samples.get(procedure, 0)
    mean_mag = magnitudes.mean()
    label = f"{proc_label} (n={n_samples}, mean={mean_mag:.4f})"

    cv2.putText(arrow_canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        heat_canvas,
        "Displacement Heatmap",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Combine horizontally
    return np.hstack([arrow_canvas, heat_canvas])


def _get_default_face_positions(n: int = 478) -> np.ndarray:
    """Generate default face landmark positions for visualization.

    Creates a canonical face layout matching approximate MediaPipe positions
    for a centered frontal face.
    """
    # Use a simple elliptical face model with key regions placed anatomically
    positions = np.zeros((n, 2), dtype=np.float32)

    # Distribute landmarks in face-like pattern
    # Face oval (outer)
    for i, idx in enumerate(FACE_OVAL):
        if idx < n:
            angle = 2 * np.pi * i / len(FACE_OVAL) - np.pi / 2
            positions[idx, 0] = 0.5 + 0.35 * np.cos(angle)
            positions[idx, 1] = 0.5 + 0.42 * np.sin(angle)

    # Eyes
    for i, idx in enumerate(LEFT_EYE):
        if idx < n:
            angle = 2 * np.pi * i / len(LEFT_EYE)
            positions[idx, 0] = 0.37 + 0.05 * np.cos(angle)
            positions[idx, 1] = 0.38 + 0.02 * np.sin(angle)

    for i, idx in enumerate(RIGHT_EYE):
        if idx < n:
            angle = 2 * np.pi * i / len(RIGHT_EYE)
            positions[idx, 0] = 0.63 + 0.05 * np.cos(angle)
            positions[idx, 1] = 0.38 + 0.02 * np.sin(angle)

    # Nose
    nose_bridge = [168, 6, 197, 195, 5, 4]
    for i, idx in enumerate(nose_bridge):
        if idx < n:
            positions[idx, 0] = 0.5
            positions[idx, 1] = 0.35 + i * 0.04

    # Nose tip
    nose_tip = [1, 2, 94, 19]
    for i, idx in enumerate(nose_tip):
        if idx < n:
            positions[idx, 0] = 0.5 + (i - 1.5) * 0.02
            positions[idx, 1] = 0.55

    # Lips
    for i, idx in enumerate(LIPS_OUTER):
        if idx < n:
            angle = 2 * np.pi * i / len(LIPS_OUTER)
            positions[idx, 0] = 0.5 + 0.07 * np.cos(angle)
            positions[idx, 1] = 0.65 + 0.02 * np.sin(angle)

    # Fill remaining landmarks with interpolated positions
    # Use a face-shaped distribution
    rng = np.random.default_rng(42)
    for i in range(n):
        if positions[i, 0] == 0.0 and positions[i, 1] == 0.0:
            # Random position within face region
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.05, 0.35)
            positions[i, 0] = 0.5 + r * np.cos(angle)
            positions[i, 1] = 0.5 + r * 0.9 * np.sin(angle)

    return positions


def create_comparison_figure(
    model: DisplacementModel,
    output_path: str,
    size: int = 400,
) -> None:
    """Create a 4-panel comparison figure for the paper."""
    panels = []
    for proc in [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]:
        if proc in model.procedures:
            panel = create_procedure_visualization(model, proc, size)
            panels.append(panel)

    if panels:
        figure = np.vstack(panels)
        cv2.imwrite(output_path, figure)
        print(f"Comparison figure saved: {output_path}")


def create_magnitude_chart(
    model: DisplacementModel,
    output_path: str,
) -> None:
    """Create a bar chart of per-region displacement magnitudes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for charts")
        return

    # Define anatomical regions
    regions = {
        "Nose bridge": [6, 197, 195, 5, 4],
        "Nose tip": [1, 2, 94, 19],
        "Left eye": [33, 246, 161, 160, 159, 158, 157, 173],
        "Right eye": [263, 466, 388, 387, 386, 385, 384, 398],
        "Upper lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "Lower lip": [17, 84, 181, 91, 146, 61, 375, 321, 405, 314],
        "Left jaw": [132, 58, 172, 136, 150, 149, 176, 148],
        "Right jaw": [361, 288, 397, 365, 379, 378, 400, 377],
        "Forehead": [10, 109, 67, 103, 54, 21, 338, 297, 332, 284, 251],
        "Chin": [152, 377, 400, 378, 379, 365, 397, 288, 361],
    }

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    fig, axes = plt.subplots(1, len(procedures), figsize=(16, 5), sharey=True)

    for ax, proc in zip(axes, procedures):
        if proc not in model.procedures:
            continue

        disp = model.get_displacement_field(proc, intensity=1.0)
        magnitudes = np.linalg.norm(disp, axis=1)

        region_means = []
        region_names = []
        for region_name, indices in regions.items():
            valid_idx = [i for i in indices if i < len(magnitudes)]
            if valid_idx:
                region_means.append(float(magnitudes[valid_idx].mean()))
                region_names.append(region_name)

        colors = plt.cm.viridis(np.array(region_means) / max(region_means))
        bars = ax.barh(range(len(region_names)), region_means, color=colors)
        ax.set_yticks(range(len(region_names)))
        ax.set_yticklabels(region_names, fontsize=8)
        ax.set_xlabel("Mean Displacement", fontsize=9)
        ax.set_title(proc.capitalize(), fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Per-Region Displacement Magnitudes by Procedure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Magnitude chart saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize displacement model")
    parser.add_argument(
        "--model", default="data/displacement_model.npz", help="Path to displacement model"
    )
    parser.add_argument("--output", default="paper/figures/displacements", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = DisplacementModel.load(args.model)
    print(f"Loaded model: {model.procedures}")

    # Individual procedure visualizations
    for proc in model.procedures:
        viz = create_procedure_visualization(model, proc, size=400)
        cv2.imwrite(str(out / f"{proc}_displacements.png"), viz)
        print(f"  {proc}: {out / f'{proc}_displacements.png'}")

    # Comparison figure
    create_comparison_figure(model, str(out / "displacement_comparison.png"), size=400)

    # Bar chart
    create_magnitude_chart(model, str(out / "displacement_magnitudes.png"))

    print(f"\nAll visualizations saved to {out}/")
