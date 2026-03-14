#!/usr/bin/env python3
"""Generate publication-quality figures for the LandmarkDiff paper.

Figures: architecture diagram, procedure grid (6x3), deformation overlay,
post-processing comparison, pipeline overview, conditioning ablation.

Usage:
    python scripts/generate_paper_figures.py \
        --input data/faces_all/000001.png --output paper/figures/ --figure all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from landmarkdiff.conditioning import generate_conditioning, render_wireframe
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]
PROC_SHORT = {
    "rhinoplasty": "Rhino",
    "blepharoplasty": "Bleph",
    "rhytidectomy": "Rhytid",
    "orthognathic": "Orthog",
    "brow_lift": "Brow Lift",
    "mentoplasty": "Mento",
}


def add_label(
    img: np.ndarray,
    text: str,
    position: str = "bottom",
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a text label to an image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    if position == "bottom":
        x, y = (img.shape[1] - tw) // 2, img.shape[0] - 10
    elif position == "top":
        x, y = (img.shape[1] - tw) // 2, th + 10
    else:
        x, y = 10, th + 10
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)
    return img


# -----------------------------------------------------------------------
# Figure 1: Architecture diagram
# -----------------------------------------------------------------------


def figure_architecture(output_dir: Path, dpi: int = 200):
    """Render pipeline architecture diagram using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping architecture diagram")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        (0.3, 1.2, 1.8, 1.6, "Input\nImage", "#4A90D9"),
        (2.5, 1.2, 1.8, 1.6, "MediaPipe\n478-pt Mesh", "#7B68EE"),
        (4.7, 1.2, 1.8, 1.6, "Gaussian RBF\nDeformation", "#E67E22"),
        (6.9, 1.2, 1.8, 1.6, "ControlNet\n(CrucibleAI)", "#27AE60"),
        (9.1, 1.2, 1.8, 1.6, "SD 1.5\nDiffusion", "#E74C3C"),
        (11.3, 1.2, 1.8, 1.6, "Post-\nProcessing", "#8E44AD"),
    ]

    for x, y, w, h, label, color in blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="white",
            alpha=0.85,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Arrows between blocks
    arrow_props = dict(
        arrowstyle="->",
        color="#333333",
        lw=1.5,
        connectionstyle="arc3,rad=0",
    )
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + blocks[i][2]
        x2 = blocks[i + 1][0]
        y_mid = blocks[i][1] + blocks[i][3] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    # Post-processing sub-labels
    pp_labels = ["CodeFormer", "Real-ESRGAN", "Hist Match", "Laplacian Blend"]
    for j, lbl in enumerate(pp_labels):
        ax.text(
            12.2,
            0.9 - j * 0.22,
            lbl,
            ha="center",
            va="center",
            fontsize=6,
            color="#555555",
        )

    ax.set_title(
        "LandmarkDiff Pipeline Architecture",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    out_path = output_dir / "fig_architecture.pdf"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    # Also save PNG for quick preview
    fig.savefig(str(output_dir / "fig_architecture.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Architecture diagram: %s", out_path)


# -----------------------------------------------------------------------
# Figure 2: 6 procedures x 3 intensities grid
# -----------------------------------------------------------------------


def figure_procedure_grid(image_path: str, output_dir: Path, size: int = 192):
    """6-procedure x 3-intensity comparison grid."""
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read %s", image_path)
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        logger.warning("No face detected in %s", image_path)
        return

    intensities = [33, 66, 100]
    rows = []

    # Header row
    header_panels = [_blank_panel(size, "Procedure")]
    for intensity in intensities:
        header_panels.append(_blank_panel(size, f"{intensity}%"))
    rows.append(np.hstack(header_panels))

    for proc in PROCEDURES:
        proc_label = PROC_SHORT.get(proc, proc)
        row_panels = [add_label(cv2.resize(img, (size, size)), proc_label, "bottom")]

        for intensity in intensities:
            try:
                manip = apply_procedure_preset(face, proc, float(intensity), image_size=512)
                mask = generate_surgical_mask(face, proc, 512, 512)
                warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
                composite = mask_composite(warped, img, mask)
                panel = cv2.resize(composite, (size, size))
            except Exception as e:
                logger.warning("  %s %d%%: %s", proc, intensity, e)
                panel = np.zeros((size, size, 3), dtype=np.uint8)
            row_panels.append(panel)

        rows.append(np.hstack(row_panels))

    grid = np.vstack(rows)
    out_path = output_dir / "fig_procedure_grid.png"
    cv2.imwrite(str(out_path), grid)
    logger.info("Procedure grid (6x3): %s (%dx%d)", out_path, grid.shape[1], grid.shape[0])


def _blank_panel(size: int, label: str) -> np.ndarray:
    """Create a black panel with centered text."""
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    return add_label(panel, label, "bottom")


# -----------------------------------------------------------------------
# Figure 3: Deformation visualization (landmark overlay before/after)
# -----------------------------------------------------------------------


def figure_deformation_overlay(
    image_path: str,
    output_dir: Path,
    size: int = 384,
    procedure: str = "rhinoplasty",
    intensity: float = 65.0,
):
    """Visualize landmark deformation: original (green) vs deformed (red) with arrows."""
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    manip = apply_procedure_preset(face, procedure, intensity, image_size=512)

    orig_pts = face.pixel_coords
    manip_pts = manip.pixel_coords

    # Panel 1: Original with landmarks
    panel_orig = img.copy()
    for pt in orig_pts:
        cv2.circle(panel_orig, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
    panel_orig = add_label(cv2.resize(panel_orig, (size, size)), "Original", "top")

    # Panel 2: Overlay showing displacement vectors
    panel_overlay = img.copy()
    # Draw displacement arrows for landmarks that moved more than 1px
    displacements = np.linalg.norm(manip_pts - orig_pts, axis=1)
    moved_mask = displacements > 1.0

    for i in range(len(orig_pts)):
        if moved_mask[i]:
            p1 = (int(orig_pts[i][0]), int(orig_pts[i][1]))
            p2 = (int(manip_pts[i][0]), int(manip_pts[i][1]))
            cv2.arrowedLine(panel_overlay, p1, p2, (0, 0, 255), 1, tipLength=0.3)
            cv2.circle(panel_overlay, p1, 2, (0, 255, 0), -1)
        else:
            pt = (int(orig_pts[i][0]), int(orig_pts[i][1]))
            cv2.circle(panel_overlay, pt, 1, (128, 128, 128), -1)

    panel_overlay = add_label(
        cv2.resize(panel_overlay, (size, size)),
        f"Deformation ({PROC_SHORT.get(procedure, procedure)})",
        "top",
    )

    # Panel 3: Warped result with manipulated landmarks
    warped = warp_image_tps(img, orig_pts, manip_pts)
    panel_warped = warped.copy()
    for pt in manip_pts:
        cv2.circle(panel_warped, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
    panel_warped = add_label(cv2.resize(panel_warped, (size, size)), "Warped", "top")

    row = np.hstack([panel_orig, panel_overlay, panel_warped])
    out_path = output_dir / f"fig_deformation_{procedure}.png"
    cv2.imwrite(str(out_path), row)
    logger.info("Deformation overlay (%s): %s", procedure, out_path)


# -----------------------------------------------------------------------
# Figure 4: Post-processing pipeline comparison
# -----------------------------------------------------------------------


def figure_postprocess_comparison(image_path: str, output_dir: Path, size: int = 256):
    """Compare output at each post-processing stage."""
    from landmarkdiff.postprocess import (
        frequency_aware_sharpen,
        histogram_match_skin,
        laplacian_pyramid_blend,
    )

    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    procedure = "rhinoplasty"
    manip = apply_procedure_preset(face, procedure, 65.0, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)

    # Stage 1: Raw TPS warp
    stage_raw = warped.copy()

    # Stage 2: + histogram matching
    stage_hist = histogram_match_skin(warped, img, mask)

    # Stage 3: + sharpening
    stage_sharp = frequency_aware_sharpen(stage_hist, strength=0.25)

    # Stage 4: + Laplacian blend
    stage_blend = laplacian_pyramid_blend(stage_sharp, img, mask)

    # Stage 5: Full composite (the final output via mask_composite)
    stage_final = mask_composite(warped, img, mask)

    panels = [
        (img, "Original"),
        (stage_raw, "TPS Warp"),
        (stage_hist, "+ Hist Match"),
        (stage_sharp, "+ Sharpen"),
        (stage_blend, "+ Lap Blend"),
        (stage_final, "Final Output"),
    ]

    grid_panels = []
    for panel_img, label in panels:
        if panel_img.ndim == 2:
            panel_img = cv2.cvtColor(panel_img, cv2.COLOR_GRAY2BGR)
        p = cv2.resize(panel_img, (size, size))
        p = add_label(p, label, "bottom")
        grid_panels.append(p)

    row = np.hstack(grid_panels)
    out_path = output_dir / "fig_postprocess.png"
    cv2.imwrite(str(out_path), row)
    logger.info("Post-processing comparison: %s", out_path)


# -----------------------------------------------------------------------
# Figure 5: Pipeline overview
# -----------------------------------------------------------------------


def figure_pipeline(image_path: str, output_dir: Path, size: int = 256):
    """Pipeline overview: Input -> Landmarks -> Wireframe -> Canny -> Mask -> TPS -> Output."""
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read %s", image_path)
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        logger.warning("No face in %s", image_path)
        return

    procedure = "rhinoplasty"
    landmark_img = render_landmark_image(face, 512, 512)
    wireframe = render_wireframe(face, 512, 512)
    _, canny, _ = generate_conditioning(face, 512, 512)

    manip = apply_procedure_preset(face, procedure, 65.0, image_size=512)
    mask = generate_surgical_mask(face, procedure, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    composite = mask_composite(warped, img, mask)

    panels = []
    labels = ["Input", "Landmarks", "Wireframe", "Canny", "Mask", "Warped", "Output"]
    images = [img, landmark_img, wireframe, canny, mask, warped, composite]

    for label, panel in zip(labels, images):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        elif panel.dtype == np.float32 or panel.dtype == np.float64:
            panel = (panel * 255).clip(0, 255).astype(np.uint8)
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (size, size))
        panel = add_label(panel, label, "bottom")
        panels.append(panel)

    grid = np.hstack(panels)
    out_path = output_dir / "fig_pipeline.png"
    cv2.imwrite(str(out_path), grid)
    logger.info("Pipeline figure: %s (%dx%d)", out_path, grid.shape[1], grid.shape[0])


# -----------------------------------------------------------------------
# Figure 6: Conditioning ablation
# -----------------------------------------------------------------------


def figure_conditioning(image_path: str, output_dir: Path, size: int = 256):
    """Generate conditioning signal comparison figure."""
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (512, 512))
    face = extract_landmarks(img)
    if face is None:
        return

    landmark_img, canny, wireframe = generate_conditioning(face, 512, 512)
    mesh_canny = cv2.add(wireframe, canny)
    full_cond = np.stack([wireframe, canny, wireframe], axis=-1)

    panels = [
        (img, "Input"),
        (cv2.cvtColor(wireframe, cv2.COLOR_GRAY2BGR), "Wireframe"),
        (cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR), "Canny"),
        (cv2.cvtColor(mesh_canny, cv2.COLOR_GRAY2BGR), "Mesh + Canny"),
        (landmark_img, "Tessellation"),
        (full_cond, "Full Cond."),
    ]

    grid_panels = []
    for panel_img, label in panels:
        p = cv2.resize(panel_img, (size, size))
        p = add_label(p, label, "bottom")
        grid_panels.append(p)

    row = np.hstack(grid_panels)
    out_path = output_dir / "fig_conditioning.png"
    cv2.imwrite(str(out_path), row)
    logger.info("Conditioning figure: %s", out_path)


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for paper figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the LandmarkDiff paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None, help="Single input face image path")
    parser.add_argument("--input_dir", default=None, help="Directory of face images")
    parser.add_argument("--output", default="paper/figures", help="Output directory")
    parser.add_argument("--max_images", type=int, default=4, help="Max images from input_dir")
    parser.add_argument("--size", type=int, default=256, help="Panel size in pixels")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for vector figures")
    parser.add_argument(
        "--figure",
        default="all",
        choices=[
            "all",
            "architecture",
            "procedure_grid",
            "deformation",
            "postprocess",
            "pipeline",
            "conditioning",
        ],
        help="Which figure(s) to generate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for paper figure generation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve an input image
    image_path = args.input
    if image_path is None:
        if args.input_dir:
            candidates = sorted(Path(args.input_dir).glob("*.png"))
            if candidates:
                image_path = str(candidates[0])
        if image_path is None:
            for candidate in [
                "data/faces_all/000001.png",
                "data/celeba_hq_extracted/000001.png",
            ]:
                if Path(candidate).exists():
                    image_path = candidate
                    break

    figures_generated = 0

    # Architecture diagram (no image needed)
    if args.figure in ("all", "architecture"):
        figure_architecture(output_dir, dpi=args.dpi)
        figures_generated += 1

    if image_path is None and args.figure not in ("architecture",):
        logger.warning("No input image found. Provide --input or --input_dir.")
        if figures_generated == 0:
            return 1
        logger.info("%d figures generated in %s/", figures_generated, output_dir)
        return 0

    if args.figure in ("all", "procedure_grid") and image_path:
        figure_procedure_grid(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "deformation") and image_path:
        for proc in PROCEDURES:
            figure_deformation_overlay(image_path, output_dir, args.size + 128, proc)
            figures_generated += 1

    if args.figure in ("all", "postprocess") and image_path:
        figure_postprocess_comparison(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "pipeline") and image_path:
        figure_pipeline(image_path, output_dir, args.size)
        figures_generated += 1

    if args.figure in ("all", "conditioning") and image_path:
        figure_conditioning(image_path, output_dir, args.size)
        figures_generated += 1

    logger.info("%d figures generated in %s/", figures_generated, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
