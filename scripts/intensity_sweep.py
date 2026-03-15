#!/usr/bin/env python3
"""Generate a deformation intensity sweep visualization for the paper.

This script demonstrates fine-grained controllability of the LandmarkDiff
pipeline by sweeping the landmark displacement intensity from 0.0 (no change)
to 2.0 (double the mean observed displacement) and beyond.  For each
intensity, the script:

1. Extracts MediaPipe 478-point face landmarks from a test input image.
2. Computes displacement vectors from the paired input/target images using
   ``extract_displacements`` (or, if no target is available, falls back to
   the hand-tuned presets via ``apply_procedure_preset``).
3. Scales the displacement vectors by the intensity factor.
4. Renders the modified landmark mesh as a 3-channel ControlNet conditioning
   image (CrucibleAI-compatible tessellation format).
5. Runs ControlNet inference (SD1.5 + fine-tuned checkpoint) to produce
   the predicted surgical outcome at that intensity.
6. Optionally applies mask-based compositing for seamless blending.

The final output is a grid figure (one row per procedure) where columns
represent increasing intensity.  Each column is labeled, and conditioning
images appear as a second row beneath each output row.

Intended for Fig. 5 (intensity sweep) in the paper.

Usage:
    python scripts/intensity_sweep.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/intensity_sweep/ \
        --num-steps 20 --guidance-scale 7.5

    # Custom intensities
    python scripts/intensity_sweep.py \
        --checkpoint checkpoints/phaseA/best \
        --test-dir data/hda_splits/test \
        --intensities 0.0,0.5,1.0,1.5,2.0,3.0

    # Specify exact image instead of auto-picking from test set
    python scripts/intensity_sweep.py \
        --checkpoint checkpoints/phaseB/best \
        --image data/hda_splits/test/rhinoplasty_Nose_01_input.png \
        --procedure rhinoplasty
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so ``import landmarkdiff`` always resolves,
# regardless of the working directory or PYTHONPATH state.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from landmarkdiff.displacement_model import extract_displacements
from landmarkdiff.inference import (
    PROCEDURE_PROMPTS,
    LandmarkDiffPipeline,
    mask_composite,
)

# ---- LandmarkDiff imports ------------------------------------------------
from landmarkdiff.landmarks import (
    FaceLandmarks,
    extract_landmarks,
    render_landmark_image,
)
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.postprocess import histogram_match_skin

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All six supported surgical procedures
# ---------------------------------------------------------------------------
PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]

# ---------------------------------------------------------------------------
# Checkpoint auto-discovery order (most preferred first)
# ---------------------------------------------------------------------------
CHECKPOINT_SEARCH_ORDER = [
    "checkpoints/phaseB/best",
    "checkpoints/phaseB/latest",
    "checkpoints/phaseB/final",
    "checkpoints_v2/final",
    "checkpoints/phaseA/best",
    "checkpoints/final",
]


# =========================================================================
# Helper utilities
# =========================================================================


def find_checkpoint(explicit_path: str | None) -> Path:
    """Resolve the ControlNet checkpoint directory.

    If an explicit path is provided, validate it.  Otherwise, walk through
    ``CHECKPOINT_SEARCH_ORDER`` relative to ``ROOT`` and return the first
    existing directory.

    Raises:
        FileNotFoundError: If no checkpoint can be located.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_dir():
            return p
        # Try relative to ROOT as well
        pr = ROOT / explicit_path
        if pr.is_dir():
            return pr
        raise FileNotFoundError(f"Checkpoint not found at '{explicit_path}' (also tried {pr})")

    for candidate in CHECKPOINT_SEARCH_ORDER:
        p = ROOT / candidate
        if p.is_dir():
            logger.info("Auto-detected checkpoint: %s", p)
            return p

    raise FileNotFoundError(
        "No checkpoint found.  Searched:\n  "
        + "\n  ".join(str(ROOT / c) for c in CHECKPOINT_SEARCH_ORDER)
    )


def find_test_image_for_procedure(
    test_dir: Path,
    procedure: str,
) -> tuple[Path, Path | None] | None:
    """Find one (input, target) pair for a given procedure in the test set.

    The HDA split naming convention is:
        ``{procedure}_{SubType}_{id}_input.png``
        ``{procedure}_{SubType}_{id}_target.png``

    We pick the first alphabetically-sorted pair that has both files.

    Returns:
        (input_path, target_path) if both exist, (input_path, None) if only
        input exists, or None if nothing matches the procedure.
    """
    # Glob for input images matching the procedure prefix
    candidates = sorted(test_dir.glob(f"{procedure}_*_input.png"))
    if not candidates:
        # Fallback: any image whose stem starts with the procedure name
        candidates = sorted(test_dir.glob(f"{procedure}_*.png"))
        candidates = [c for c in candidates if "_input" in c.stem or "_target" not in c.stem]

    for inp in candidates:
        # Derive the expected target path
        target = inp.parent / inp.name.replace("_input.png", "_target.png")
        if target.exists():
            return inp, target
        # Return input-only pair if no target
        return inp, None

    return None


def compute_scaled_landmarks(
    face_source: FaceLandmarks,
    displacements: np.ndarray,
    intensity: float,
    image_w: int = 512,
    image_h: int = 512,
) -> FaceLandmarks:
    """Create a new FaceLandmarks with displacement vectors scaled by intensity.

    The source landmarks are in normalized [0, 1] space.  Displacement vectors
    are also in normalized space.  We simply do:

        new_landmarks = source + displacements * intensity

    and clip to [0.01, 0.99] to avoid edge artifacts.

    Args:
        face_source: Original face landmarks.
        displacements: (478, 2) displacement vectors in normalized space
            (i.e., target - source landmarks).
        intensity: Scalar multiplier.  0.0 = identity, 1.0 = original
            displacement, 2.0 = doubled, etc.
        image_w: Width for the new FaceLandmarks metadata.
        image_h: Height for the new FaceLandmarks metadata.

    Returns:
        New FaceLandmarks with scaled displacements applied.
    """
    new_lm = face_source.landmarks.copy()  # (478, 3)

    # Scale and apply only x,y displacements (z stays unchanged)
    n = min(len(new_lm), len(displacements))
    new_lm[:n, 0] += displacements[:n, 0] * intensity
    new_lm[:n, 1] += displacements[:n, 1] * intensity

    # Clamp to valid normalized range to prevent off-canvas landmarks
    new_lm[:, 0] = np.clip(new_lm[:, 0], 0.01, 0.99)
    new_lm[:, 1] = np.clip(new_lm[:, 1], 0.01, 0.99)

    return FaceLandmarks(
        landmarks=new_lm,
        image_width=image_w,
        image_height=image_h,
        confidence=face_source.confidence,
    )


def add_text_label(
    image: np.ndarray,
    text: str,
    position: str = "top",
    font_scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Overlay a text label on a BGR image using OpenCV.

    Args:
        image: BGR numpy array to annotate (modified in-place and returned).
        text: Label text string.
        position: "top" or "bottom".
        font_scale: OpenCV font scale.
        color: Text color in BGR.
        bg_color: Background rectangle color.

    Returns:
        The annotated image (same object).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    h, w = image.shape[:2]
    if position == "top":
        # Draw background rectangle at top center
        x = (w - tw) // 2
        y = th + 6
        cv2.rectangle(image, (x - 4, 0), (x + tw + 4, y + baseline + 2), bg_color, -1)
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        x = (w - tw) // 2
        y = h - baseline - 4
        cv2.rectangle(image, (x - 4, y - th - 2), (x + tw + 4, h), bg_color, -1)
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image


def create_grid(
    images: list[np.ndarray],
    conditioning_images: list[np.ndarray] | None = None,
    labels: list[str] | None = None,
    cell_size: int = 256,
    label_height: int = 30,
    border: int = 2,
    title: str | None = None,
    title_height: int = 40,
) -> np.ndarray:
    """Arrange a list of images into a labeled grid.

    Layout (top to bottom):
    - Optional title bar
    - Labels bar (intensity values)
    - Row of output images
    - Row of conditioning images (if provided)

    Args:
        images: List of BGR output images (one per intensity).
        conditioning_images: Optional list of conditioning mesh images.
        labels: Optional text labels for each column.
        cell_size: Resize each cell to this square size.
        label_height: Height of the label bar in pixels.
        border: Border width between cells in pixels.
        title: Optional title text for the top of the grid.
        title_height: Height of the title bar.

    Returns:
        BGR grid image.
    """
    n_cols = len(images)
    # Number of rows: 1 for outputs, optionally 1 for conditioning
    n_rows = 2 if conditioning_images else 1

    total_w = n_cols * cell_size + (n_cols + 1) * border
    total_h = (
        n_rows * cell_size + (n_rows + 1) * border + label_height + (title_height if title else 0)
    )

    # Dark gray background
    grid = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    # Draw title if provided
    y_offset = 0
    if title:
        cv2.putText(
            grid,
            title,
            (border + 4, title_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset = title_height

    # Draw column labels
    for col_idx in range(n_cols):
        x_start = border + col_idx * (cell_size + border)
        label_text = labels[col_idx] if labels else f"Col {col_idx}"

        # Center the label text over the column
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        tx = x_start + (cell_size - tw) // 2
        ty = y_offset + label_height - 8

        cv2.putText(
            grid,
            label_text,
            (tx, ty),
            font,
            font_scale,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    y_offset += label_height

    # Draw output images (row 0)
    for col_idx, img in enumerate(images):
        x_start = border + col_idx * (cell_size + border)
        y_start = y_offset + border

        resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_LANCZOS4)
        grid[y_start : y_start + cell_size, x_start : x_start + cell_size] = resized

    y_offset += cell_size + border

    # Draw conditioning images (row 1) if provided
    if conditioning_images:
        for col_idx, cond in enumerate(conditioning_images):
            x_start = border + col_idx * (cell_size + border)
            y_start = y_offset + border

            resized = cv2.resize(cond, (cell_size, cell_size), interpolation=cv2.INTER_LANCZOS4)
            grid[y_start : y_start + cell_size, x_start : x_start + cell_size] = resized

    return grid


# =========================================================================
# Main sweep logic
# =========================================================================


def run_intensity_sweep(
    pipeline: LandmarkDiffPipeline,
    image_path: Path,
    target_path: Path | None,
    procedure: str,
    intensities: list[float],
    output_dir: Path,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.9,
    seed: int = 42,
    cell_size: int = 256,
    save_individuals: bool = True,
) -> np.ndarray:
    """Run the full intensity sweep for one (image, procedure) pair.

    For each intensity value, we:
    1. Scale the displacement vectors
    2. Render a conditioning image from the modified landmarks
    3. Run ControlNet inference
    4. Composite result into the original
    5. Collect into a labeled grid

    Args:
        pipeline: Loaded LandmarkDiffPipeline in 'controlnet' mode.
        image_path: Path to the input face image.
        target_path: Optional path to the ground-truth post-surgery target
            (used to extract real displacement vectors).
        procedure: Surgical procedure name.
        intensities: List of intensity scaling factors (e.g. [0.0, 0.5, 1.0]).
        output_dir: Directory to save outputs.
        num_steps: Number of diffusion inference steps.
        guidance_scale: Classifier-free guidance scale.
        controlnet_conditioning_scale: ControlNet conditioning strength.
        seed: Random seed for reproducibility.
        cell_size: Size of each cell in the output grid.
        save_individuals: If True, save each intensity output as a separate image.

    Returns:
        The assembled grid image (BGR numpy array).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and resize input image to 512x512 ----
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_512 = cv2.resize(img, (512, 512))

    # ---- Extract landmarks from input ----
    face_source = extract_landmarks(img_512)
    if face_source is None:
        raise ValueError(f"No face detected in {image_path}")

    # ---- Get displacement vectors ----
    # Prefer real displacements from paired target; fall back to presets
    if target_path and target_path.exists():
        logger.info("Extracting real displacements from input/target pair")
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            raise FileNotFoundError(f"Could not load target: {target_path}")
        target_512 = cv2.resize(target_img, (512, 512))

        disp_result = extract_displacements(img_512, target_512)
        if disp_result is not None:
            # Raw displacement vectors: (478, 2) in normalized space
            displacements = disp_result["displacements"]
            logger.info(
                "Displacement magnitude: mean=%.5f, max=%.5f",
                np.mean(np.linalg.norm(displacements, axis=1)),
                np.max(np.linalg.norm(displacements, axis=1)),
            )
        else:
            # Face detection failed on target — fall back to preset
            logger.warning(
                "Displacement extraction failed (face detection on target). Falling back to preset."
            )
            displacements = _get_preset_displacements(face_source, procedure)
    else:
        # No target image — use the hand-tuned preset manipulation
        logger.info("No target image available; using preset displacements")
        displacements = _get_preset_displacements(face_source, procedure)

    # ---- Generate surgical mask (from source landmarks, procedure-specific) ----
    mask = generate_surgical_mask(face_source, procedure, 512, 512)

    # ---- ControlNet prompt ----
    prompt = PROCEDURE_PROMPTS.get(procedure, "a photo of a person's face")

    # ---- Sweep across intensities ----
    output_images: list[np.ndarray] = []
    conditioning_images: list[np.ndarray] = []
    labels: list[str] = []

    for intensity in intensities:
        t0 = time.time()
        logger.info(
            "  Generating %s @ intensity=%.2f ...",
            procedure,
            intensity,
        )

        # Scale displacement vectors by intensity factor
        modified_face = compute_scaled_landmarks(
            face_source,
            displacements,
            intensity,
            image_w=512,
            image_h=512,
        )

        # Render the conditioning image (CrucibleAI mesh format)
        cond_img = render_landmark_image(modified_face, 512, 512)
        conditioning_images.append(cond_img)

        # Run ControlNet inference
        if intensity == 0.0:
            # At intensity 0, the conditioning is identical to the source —
            # the output should closely resemble the input.  We still run
            # inference to show the pipeline baseline.
            pass

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Use the pipeline's internal _generate_controlnet method for
        # direct control over the conditioning image.
        raw_output = pipeline._generate_controlnet(
            img_512,
            cond_img,
            prompt,
            num_steps,
            guidance_scale,
            controlnet_conditioning_scale,
            generator,
        )

        # Composite into original image using mask blending and skin matching
        composited = mask_composite(raw_output, img_512, mask)

        # Optional histogram matching for consistent skin tones across
        # the sweep — each intensity should differ in shape, not color
        composited = histogram_match_skin(composited, img_512, mask)

        output_images.append(composited)
        labels.append(f"x{intensity:.2f}")

        elapsed = time.time() - t0
        logger.info("    Done (%.1fs)", elapsed)

        # Save individual outputs if requested
        if save_individuals:
            fname = f"{procedure}_intensity_{intensity:.2f}.png"
            cv2.imwrite(str(output_dir / fname), composited)
            cond_fname = f"{procedure}_cond_{intensity:.2f}.png"
            cv2.imwrite(str(output_dir / cond_fname), cond_img)

    # ---- Assemble into a grid figure ----
    title = f"{procedure.capitalize()} — Intensity Sweep"
    grid = create_grid(
        images=output_images,
        conditioning_images=conditioning_images,
        labels=labels,
        cell_size=cell_size,
        title=title,
    )

    # Save the grid
    grid_path = output_dir / f"{procedure}_sweep_grid.png"
    cv2.imwrite(str(grid_path), grid)
    logger.info("Grid saved: %s", grid_path)

    # Also save a version without conditioning (just outputs) for the main
    # paper figure where space is tight
    grid_no_cond = create_grid(
        images=output_images,
        conditioning_images=None,
        labels=labels,
        cell_size=cell_size,
        title=title,
    )
    grid_no_cond_path = output_dir / f"{procedure}_sweep_grid_compact.png"
    cv2.imwrite(str(grid_no_cond_path), grid_no_cond)

    # Save the input image for reference
    cv2.imwrite(str(output_dir / f"{procedure}_input.png"), img_512)

    return grid


def _get_preset_displacements(
    face: FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
    image_size: int = 512,
) -> np.ndarray:
    """Compute displacement vectors from the preset manipulation engine.

    Applies the preset at a reference intensity, then computes the difference
    from the source landmarks in normalized coordinate space.

    Args:
        face: Source face landmarks.
        procedure: Surgical procedure name.
        intensity: Reference intensity for the preset (UI scale 0-100).
        image_size: Image size for pixel-coordinate calculation.

    Returns:
        (478, 2) displacement vectors in normalized coordinate space.
    """
    from landmarkdiff.manipulation import apply_procedure_preset

    manipulated = apply_procedure_preset(face, procedure, intensity, image_size=image_size)

    # Compute displacement in normalized space
    displacements = manipulated.landmarks[:, :2] - face.landmarks[:, :2]
    return displacements


# =========================================================================
# Entry point
# =========================================================================


def main() -> None:
    """Parse CLI arguments and run the intensity sweep pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate deformation intensity sweep figures for the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full sweep for all procedures:\n"
            "  python scripts/intensity_sweep.py --test-dir data/hda_splits/test\n\n"
            "  # Single image, custom intensities:\n"
            "  python scripts/intensity_sweep.py \\\n"
            "    --image path/to/face.png --procedure rhinoplasty \\\n"
            "    --intensities 0.0,0.5,1.0,2.0,4.0"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to fine-tuned ControlNet checkpoint directory.  "
            "If omitted, auto-discovers from standard locations: "
            + ", ".join(CHECKPOINT_SEARCH_ORDER)
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/hda_splits/test",
        help="Path to the HDA test split directory (default: data/hda_splits/test)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Path to a specific input face image.  If provided, overrides "
            "--test-dir and processes only this image."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to the paired target image (optional, for real displacements).",
    )
    parser.add_argument(
        "--procedure",
        type=str,
        default=None,
        choices=PROCEDURES,
        help=(
            "Which procedure to run.  If omitted and --test-dir is used, all procedures are run."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/intensity_sweep",
        help="Output directory for sweep results (default: paper/intensity_sweep/)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of diffusion inference steps (default: 20)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.9,
        help="ControlNet conditioning scale (default: 0.9)",
    )
    parser.add_argument(
        "--intensities",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated list of intensity factors (default: 0.0,0.25,...,2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=256,
        help="Size of each cell in the grid figure in pixels (default: 256)",
    )
    parser.add_argument(
        "--no-save-individuals",
        action="store_true",
        help="Skip saving individual intensity output images",
    )

    args = parser.parse_args()

    # ---- Parse intensities ----
    intensities = [float(x.strip()) for x in args.intensities.split(",")]
    logger.info("Intensities: %s", intensities)

    # ---- Resolve paths ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find checkpoint ----
    ckpt_path = find_checkpoint(args.checkpoint)
    logger.info("Using checkpoint: %s", ckpt_path)

    # ---- Initialize pipeline ----
    logger.info("Initializing LandmarkDiff pipeline (controlnet mode)...")
    pipeline = LandmarkDiffPipeline(
        mode="controlnet",
        controlnet_checkpoint=str(ckpt_path),
    )
    pipeline.load()
    logger.info("Pipeline loaded on %s", pipeline.device)

    # ---- Determine which images/procedures to process ----
    sweep_tasks: list[tuple[Path, Path | None, str]] = []

    if args.image:
        # Single image mode — process one (image, procedure) pair
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error("Image not found: %s", img_path)
            sys.exit(1)

        procedure = args.procedure or "rhinoplasty"
        target = Path(args.target) if args.target else None
        sweep_tasks.append((img_path, target, procedure))

    else:
        # Test directory mode — find one image per procedure
        test_dir = Path(args.test_dir)
        if not test_dir.is_dir():
            # Try relative to ROOT
            test_dir = ROOT / args.test_dir
        if not test_dir.is_dir():
            logger.error("Test directory not found: %s", args.test_dir)
            sys.exit(1)

        procedures = [args.procedure] if args.procedure else PROCEDURES
        for proc in procedures:
            pair = find_test_image_for_procedure(test_dir, proc)
            if pair is None:
                logger.warning(
                    "No test image found for procedure '%s' in %s — skipping",
                    proc,
                    test_dir,
                )
                continue
            inp_path, tgt_path = pair
            sweep_tasks.append((inp_path, tgt_path, proc))
            logger.info("Found test pair for %s: %s", proc, inp_path.name)

    if not sweep_tasks:
        logger.error("No images to process.  Check --test-dir or --image.")
        sys.exit(1)

    # ---- Run sweeps ----
    all_grids: list[np.ndarray] = []
    t_total = time.time()

    for img_path, tgt_path, procedure in sweep_tasks:
        logger.info(
            "=== Running intensity sweep: %s (%s) ===",
            procedure,
            img_path.name,
        )
        grid = run_intensity_sweep(
            pipeline=pipeline,
            image_path=img_path,
            target_path=tgt_path,
            procedure=procedure,
            intensities=intensities,
            output_dir=output_dir / procedure,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            seed=args.seed,
            cell_size=args.cell_size,
            save_individuals=not args.no_save_individuals,
        )
        all_grids.append(grid)

    # ---- Create combined multi-procedure grid ----
    # Stack all procedure grids vertically for a single paper figure
    if len(all_grids) > 1:
        # Pad all grids to the same width (they should already match, but
        # handle edge cases where different procedures have different column
        # counts due to missing data)
        max_w = max(g.shape[1] for g in all_grids)
        padded = []
        for g in all_grids:
            if g.shape[1] < max_w:
                pad = np.full((g.shape[0], max_w - g.shape[1], 3), 30, dtype=np.uint8)
                g = np.hstack([g, pad])
            padded.append(g)

        combined = np.vstack(padded)
        combined_path = output_dir / "all_procedures_sweep.png"
        cv2.imwrite(str(combined_path), combined)
        logger.info("Combined multi-procedure grid: %s", combined_path)

        # Also save a copy directly into the paper/ directory for easy
        # inclusion in the LaTeX document
        paper_fig = ROOT / "paper" / "fig_intensity_sweep.png"
        cv2.imwrite(str(paper_fig), combined)
        logger.info("Paper figure updated: %s", paper_fig)

    elif len(all_grids) == 1:
        # Single procedure — copy to paper figure too
        paper_fig = ROOT / "paper" / "fig_intensity_sweep.png"
        cv2.imwrite(str(paper_fig), all_grids[0])
        logger.info("Paper figure updated: %s", paper_fig)

    elapsed_total = time.time() - t_total
    logger.info(
        "=== Intensity sweep complete (%d procedures, %.1fs total) ===",
        len(sweep_tasks),
        elapsed_total,
    )
    logger.info("Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
