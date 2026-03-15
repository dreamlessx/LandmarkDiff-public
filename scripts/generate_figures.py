"""Generate publication-quality figures for the LandmarkDiff paper.

Produces:
1. Architecture diagram (Fig. 1) — pipeline overview
2. Qualitative results grid (Fig. 2) — procedure comparison
3. Conditioning signal visualization — mesh/canny/mask examples
4. Deformation visualization — before/after landmark overlays

All figures saved as 300 DPI PNGs suitable for LNCS at 12.2 cm width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Matplotlib setup for headless rendering
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Color palette (MICCAI-appropriate, colorblind-safe) ────────────
COLORS = {
    "input": "#4C72B0",  # Steel blue
    "landmarks": "#55A868",  # Sage green
    "deformation": "#C44E52",  # Muted red
    "conditioning": "#8172B2",  # Muted purple
    "diffusion": "#CCB974",  # Gold
    "output": "#64B5CD",  # Sky blue
    "arrow": "#444444",
    "bg": "#FAFAFA",
    "box_bg": "#FFFFFF",
}

LNCS_WIDTH_IN = 4.8  # 12.2 cm in inches


def generate_architecture_diagram(output_path: str = "paper/fig_architecture.png"):
    """Generate the main architecture diagram (Fig. 1).

    Shows the 5-stage pipeline with visual flow.
    """
    fig, ax = plt.subplots(1, 1, figsize=(LNCS_WIDTH_IN * 2, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Stage boxes
    stages = [
        {
            "x": 0.2,
            "label": "Input\nPhoto",
            "sublabel": "2D clinical\nphotograph",
            "color": COLORS["input"],
            "icon": "📷",
        },
        {
            "x": 2.0,
            "label": "Landmark\nExtraction",
            "sublabel": "MediaPipe\n478 points",
            "color": COLORS["landmarks"],
            "icon": "🔍",
        },
        {
            "x": 3.8,
            "label": "RBF\nDeformation",
            "sublabel": "Procedure-\nspecific",
            "color": COLORS["deformation"],
            "icon": "⬡",
        },
        {
            "x": 5.6,
            "label": "Conditioning\nSignal",
            "sublabel": "Mesh+Canny\n+Mask",
            "color": COLORS["conditioning"],
            "icon": "▦",
        },
        {
            "x": 7.4,
            "label": "ControlNet\n+ SD 1.5",
            "sublabel": "Latent\nDiffusion",
            "color": COLORS["diffusion"],
            "icon": "🎨",
        },
        {
            "x": 9.0,
            "label": "Output",
            "sublabel": "LAB color\ncomposite",
            "color": COLORS["output"],
            "icon": "✓",
        },
    ]

    box_w = 1.4
    box_h = 2.2

    for s in stages:
        # Box
        rect = FancyBboxPatch(
            (s["x"], 0.5),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=s["color"],
            edgecolor="white",
            alpha=0.15,
            linewidth=0,
        )
        ax.add_patch(rect)

        # Border
        rect2 = FancyBboxPatch(
            (s["x"], 0.5),
            box_w,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor="none",
            edgecolor=s["color"],
            linewidth=1.5,
        )
        ax.add_patch(rect2)

        cx = s["x"] + box_w / 2

        # Stage label (bold)
        ax.text(
            cx,
            2.1,
            s["label"],
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="#333333",
            linespacing=1.2,
        )

        # Sublabel
        ax.text(
            cx,
            1.1,
            s["sublabel"],
            ha="center",
            va="center",
            fontsize=5.5,
            color="#666666",
            linespacing=1.15,
            style="italic",
        )

    # Arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + box_w + 0.02
        x_end = stages[i + 1]["x"] - 0.02
        ax.annotate(
            "",
            xy=(x_end, 1.6),
            xytext=(x_start, 1.6),
            arrowprops=dict(
                arrowstyle="-|>",
                color=COLORS["arrow"],
                lw=1.5,
                mutation_scale=12,
            ),
        )

    # Stage numbers
    for i, s in enumerate(stages):
        if i == 0 or i == len(stages) - 1:
            continue
        cx = s["x"] + box_w / 2
        circle = plt.Circle((cx, 2.55), 0.15, color=s["color"], alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(
            cx,
            2.55,
            str(i),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    # Title
    ax.text(
        5.0,
        3.05,
        "LandmarkDiff Pipeline",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )

    plt.tight_layout(pad=0.2)
    out = PROJECT_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Architecture diagram saved: {out}")


def generate_conditioning_figure(
    image_path: str | None = None,
    output_path: str = "paper/fig_conditioning.png",
):
    """Generate conditioning signal visualization.

    Shows: Original | Landmarks | Tessellation Mesh | Canny Edges | Surgical Mask
    for each of the 4 procedures.
    """
    from landmarkdiff.conditioning import generate_conditioning
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask

    # Find a sample face image
    if image_path is None:
        candidates = [
            PROJECT_ROOT / "data" / "faces_mega",
            PROJECT_ROOT / "data" / "celeba_hq_extracted",
            PROJECT_ROOT / "data" / "ffhq",
            PROJECT_ROOT / "data" / "faces_all",
        ]
        img_file = None
        for cand in candidates:
            if cand.exists():
                files = sorted(cand.glob("*.jpg")) + sorted(cand.glob("*.png"))
                if files:
                    img_file = files[0]
                    break
        if img_file is None:
            print("No face images found, skipping conditioning figure")
            return
    else:
        img_file = Path(image_path)

    img = cv2.imread(str(img_file))
    if img is None:
        print(f"Could not load {img_file}")
        return

    size = 512
    img = cv2.resize(img, (size, size))
    face = extract_landmarks(img)
    if face is None:
        print("Face detection failed, trying next image...")
        return

    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    intensity = 65.0

    fig, axes = plt.subplots(4, 5, figsize=(LNCS_WIDTH_IN * 2, LNCS_WIDTH_IN * 1.6), dpi=300)
    fig.patch.set_facecolor("white")

    col_titles = ["Input", "Deformed Mesh", "Canny Edges", "Surgical Mask", "Overlay"]

    for row, proc in enumerate(procedures):
        # Manipulate landmarks
        manipulated = apply_procedure_preset(face, proc, intensity, size)
        mesh = render_landmark_image(manipulated, size, size)
        _, canny, _ = generate_conditioning(manipulated, size, size)
        mask = generate_surgical_mask(face, proc, size, size)

        # Input
        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(
            proc.capitalize(), fontsize=7, fontweight="bold", rotation=90, labelpad=10
        )

        # Mesh
        axes[row, 1].imshow(cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB))

        # Canny
        axes[row, 2].imshow(canny, cmap="gray")

        # Mask
        axes[row, 3].imshow(mask, cmap="hot", vmin=0, vmax=1)

        # Overlay: mesh on top of face
        overlay = img.copy()
        mesh_mask = np.any(mesh > 30, axis=2)
        overlay[mesh_mask] = (overlay[mesh_mask] * 0.4 + mesh[mesh_mask] * 0.6).astype(np.uint8)
        # Draw mask boundary
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            (mask_uint8 > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        axes[row, 4].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=7, fontweight="bold", pad=4)

    # Clean up axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    out = PROJECT_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Conditioning figure saved: {out}")


def generate_deformation_figure(
    output_path: str = "paper/fig_deformation.png",
):
    """Generate deformation visualization showing landmark displacement vectors.

    Shows original landmarks vs deformed landmarks with displacement arrows
    for each procedure.
    """
    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset

    # Find a face image
    candidates = [
        PROJECT_ROOT / "data" / "faces_mega",
        PROJECT_ROOT / "data" / "celeba_hq_extracted",
        PROJECT_ROOT / "data" / "ffhq",
    ]
    img_file = None
    for cand in candidates:
        if cand.exists():
            files = sorted(cand.glob("*.jpg")) + sorted(cand.glob("*.png"))
            if files:
                img_file = files[0]
                break
    if img_file is None:
        print("No face images found, skipping deformation figure")
        return

    img = cv2.imread(str(img_file))
    if img is None:
        return

    size = 512
    img = cv2.resize(img, (size, size))
    face = extract_landmarks(img)
    if face is None:
        return

    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    fig, axes = plt.subplots(1, 4, figsize=(LNCS_WIDTH_IN * 2, LNCS_WIDTH_IN * 0.55), dpi=300)
    fig.patch.set_facecolor("white")

    for col, proc in enumerate(procedures):
        ax = axes[col]
        manipulated = apply_procedure_preset(face, proc, 75.0, size)

        # Show face with landmarks
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.3)

        orig = face.pixel_coords
        deformed = manipulated.pixel_coords

        # Plot all landmarks (small, gray)
        ax.scatter(orig[:, 0], orig[:, 1], s=0.3, c="#CCCCCC", zorder=2)

        # Plot displacement arrows for landmarks that moved
        displacements = deformed - orig
        magnitudes = np.linalg.norm(displacements, axis=1)
        threshold = (
            np.percentile(magnitudes[magnitudes > 0.5], 50) if np.any(magnitudes > 0.5) else 1.0
        )

        moved = magnitudes > threshold
        if np.any(moved):
            # Color by magnitude
            moved_mags = magnitudes[moved]
            (moved_mags - moved_mags.min()) / (moved_mags.max() - moved_mags.min() + 1e-8)

            for i in np.where(moved)[0]:
                color_val = (magnitudes[i] - moved_mags.min()) / (
                    moved_mags.max() - moved_mags.min() + 1e-8
                )
                color = plt.cm.RdYlBu_r(color_val)
                ax.annotate(
                    "",
                    xy=(deformed[i, 0], deformed[i, 1]),
                    xytext=(orig[i, 0], orig[i, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6, mutation_scale=5),
                    zorder=3,
                )

            # Deformed landmarks (colored)
            ax.scatter(
                deformed[moved, 0],
                deformed[moved, 1],
                s=1.5,
                c=magnitudes[moved],
                cmap="RdYlBu_r",
                zorder=4,
                edgecolors="none",
            )

        ax.set_title(proc.capitalize(), fontsize=7, fontweight="bold", pad=3)
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.3)
    out = PROJECT_ROOT / output_path
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Deformation figure saved: {out}")


def generate_tps_baseline_figure(
    output_path: str = "paper/fig_tps_baseline.png",
    num_examples: int = 4,
):
    """Generate TPS-only baseline results for comparison.

    Shows: Input | Conditioning Mesh | TPS Warp | Mask | Composited
    This serves as the baseline that our diffusion model must beat.
    """
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    candidates = [
        PROJECT_ROOT / "data" / "faces_mega",
        PROJECT_ROOT / "data" / "celeba_hq_extracted",
        PROJECT_ROOT / "data" / "ffhq",
    ]
    img_files = []
    for cand in candidates:
        if cand.exists():
            files = sorted(cand.glob("*.jpg")) + sorted(cand.glob("*.png"))
            img_files.extend(files[: num_examples * 2])
            if len(img_files) >= num_examples * 2:
                break

    if len(img_files) < num_examples:
        print("Not enough face images, skipping TPS baseline figure")
        return

    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    size = 512

    fig, axes = plt.subplots(
        num_examples, 5, figsize=(LNCS_WIDTH_IN * 2, LNCS_WIDTH_IN * 1.6), dpi=300
    )
    fig.patch.set_facecolor("white")

    col_titles = ["Input", "Deformed Mesh", "TPS Warp", "Surgical Mask", "Composited"]

    generated = 0
    file_idx = 0

    while generated < num_examples and file_idx < len(img_files):
        img = cv2.imread(str(img_files[file_idx]))
        file_idx += 1
        if img is None:
            continue

        img = cv2.resize(img, (size, size))
        face = extract_landmarks(img)
        if face is None:
            continue

        proc = procedures[generated % len(procedures)]
        intensity = 65.0
        manipulated = apply_procedure_preset(face, proc, intensity, size)

        # Conditioning mesh
        mesh = render_landmark_image(manipulated, size, size)

        # TPS warp
        try:
            warped = warp_image_tps(img, face.pixel_coords, manipulated.pixel_coords)
        except Exception:
            continue

        # Mask
        mask = generate_surgical_mask(face, proc, size, size)

        # Composited: mask blend of warped into original
        mask_3ch = np.stack([mask] * 3, axis=-1)
        composited = (mask_3ch * warped + (1 - mask_3ch) * img).astype(np.uint8)

        row = generated

        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(
            proc.capitalize(), fontsize=7, fontweight="bold", rotation=90, labelpad=10
        )
        axes[row, 1].imshow(cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB))
        axes[row, 2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[row, 3].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))

        generated += 1

    if generated < num_examples:
        # Remove empty rows
        for r in range(generated, num_examples):
            for c in range(5):
                axes[r, c].set_visible(False)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=7, fontweight="bold", pad=4)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
    out = PROJECT_ROOT / output_path
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"TPS baseline figure saved: {out}")


def generate_intensity_sweep(
    output_path: str = "paper/fig_intensity_sweep.png",
):
    """Generate intensity sweep visualization.

    Shows a single face with rhinoplasty at intensities 20, 40, 60, 80, 100.
    """
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    candidates = [
        PROJECT_ROOT / "data" / "celeba_hq_extracted",
        PROJECT_ROOT / "data" / "faces_mega",
        PROJECT_ROOT / "data" / "ffhq",
    ]
    img_file = None
    for cand in candidates:
        if cand.exists():
            files = sorted(cand.glob("*.jpg")) + sorted(cand.glob("*.png"))
            for f in files:
                test_img = cv2.imread(str(f))
                if test_img is not None:
                    test_face = extract_landmarks(cv2.resize(test_img, (512, 512)))
                    if test_face is not None:
                        img_file = f
                        break
            if img_file:
                break

    if img_file is None:
        print("No suitable face found for intensity sweep")
        return

    img = cv2.resize(cv2.imread(str(img_file)), (512, 512))
    face = extract_landmarks(img)

    intensities = [0, 20, 40, 60, 80, 100]
    fig, axes = plt.subplots(
        2, len(intensities), figsize=(LNCS_WIDTH_IN * 2, LNCS_WIDTH_IN * 0.8), dpi=300
    )
    fig.patch.set_facecolor("white")

    for col, alpha in enumerate(intensities):
        if alpha == 0:
            mesh = render_landmark_image(face, 512, 512)
            warped = img.copy()
        else:
            manipulated = apply_procedure_preset(face, "rhinoplasty", float(alpha), 512)
            mesh = render_landmark_image(manipulated, 512, 512)
            try:
                warped = warp_image_tps(img, face.pixel_coords, manipulated.pixel_coords)
            except Exception:
                warped = img.copy()

        axes[0, col].imshow(cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"α={alpha}%", fontsize=6, fontweight="bold", pad=3)

        axes[1, col].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    axes[0, 0].set_ylabel("Mesh", fontsize=6, fontweight="bold", rotation=90, labelpad=8)
    axes[1, 0].set_ylabel("TPS Warp", fontsize=6, fontweight="bold", rotation=90, labelpad=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Rhinoplasty Intensity Sweep", fontsize=8, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.2)
    out = PROJECT_ROOT / output_path
    fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Intensity sweep figure saved: {out}")


def main(output_dir: str = "paper", output_format: str = "png"):
    """Generate all publication-quality figures.

    Args:
        output_dir: Directory to save figures to (default: paper)
        output_format: Output image format (default: png)
    """
    print("Generating LandmarkDiff paper figures...")
    print("=" * 50)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build output paths with format
    arch_out = f"{output_dir}/fig_architecture.{output_format}"
    cond_out = f"{output_dir}/fig_conditioning.{output_format}"
    deform_out = f"{output_dir}/fig_deformation.{output_format}"
    tps_out = f"{output_dir}/fig_tps_baseline.{output_format}"
    intensity_out = f"{output_dir}/fig_intensity_sweep.{output_format}"

    # 1. Architecture diagram (no face images needed)
    generate_architecture_diagram(arch_out)

    # 2. Conditioning signal visualization
    generate_conditioning_figure(output_path=cond_out)

    # 3. Deformation visualization
    generate_deformation_figure(output_path=deform_out)

    # 4. TPS baseline comparison
    generate_tps_baseline_figure(output_path=tps_out)

    # 5. Intensity sweep
    generate_intensity_sweep(output_path=intensity_out)

    print(f"\nAll figures generated in {output_dir}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the LandmarkDiff paper."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper",
        help="Output directory for figure images (default: paper)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="Output image format (default: png)",
    )

    args = parser.parse_args()
    main(output_dir=args.output, output_format=args.format)
