"""LandmarkDiff -- Facial surgery outcome prediction demo (TPS on CPU)."""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS, apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_URL = "https://github.com/dreamlessx/LandmarkDiff-public"
PROCEDURES = list(PROCEDURE_LANDMARKS.keys())
EXAMPLE_DIR = Path(__file__).parent / "examples"
EXAMPLE_IMAGES = sorted(EXAMPLE_DIR.glob("*.png")) if EXAMPLE_DIR.exists() else []

PROCEDURE_INFO = {
    "rhinoplasty": "Nose reshaping (bridge, tip, alar width)",
    "blepharoplasty": "Eyelid surgery (lid position, canthal tilt)",
    "rhytidectomy": "Facelift (midface, jawline tightening)",
    "orthognathic": "Jaw surgery (maxilla/mandible repositioning)",
    "brow_lift": "Brow elevation, forehead ptosis reduction",
    "mentoplasty": "Chin surgery (projection, vertical height)",
}

# ---------------------------------------------------------------------------
# Bilateral symmetry landmark pairs (MediaPipe face mesh indices)
# ---------------------------------------------------------------------------
SYMMETRY_PAIRS: dict[str, list[tuple[int, int]]] = {
    "eyes": [
        (33, 263),
        (133, 362),
        (159, 386),
        (145, 374),
    ],
    "brows": [
        (70, 300),
        (63, 293),
        (105, 334),
        (66, 296),
        (107, 336),
    ],
    "cheeks": [
        (116, 345),
        (123, 352),
        (147, 376),
        (187, 411),
        (205, 425),
    ],
    "mouth": [
        (61, 291),
        (78, 308),
        (95, 324),
    ],
    "jaw": [
        (172, 397),
        (136, 365),
        (150, 379),
        (149, 378),
        (176, 400),
    ],
}

# Midline landmarks: forehead top and chin bottom
MIDLINE_TOP = 10
MIDLINE_BOTTOM = 152


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def _normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert any input image format (RGBA, grayscale, etc.) to BGR uint8."""
    if image is None:
        raise ValueError("No image provided")

    img = np.asarray(image)

    # Handle float images (0-1 range)
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    # Ensure uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        # Grayscale -> BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            # RGBA -> BGR (drop alpha)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif channels == 3:
            # RGB -> BGR (Gradio sends RGB)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif channels == 1:
            return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _auto_adjust_brightness(image_bgr: np.ndarray) -> np.ndarray:
    """Auto-adjust brightness/contrast if the image is too dark or washed out.

    Uses CLAHE on the L channel of LAB color space for adaptive histogram
    equalization that preserves color balance.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l = float(np.mean(l_channel))

    # Only adjust if clearly too dark (<60) or washed out (>200)
    if mean_l < 60 or mean_l > 200:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image_bgr


def _prepare_image(image_rgb: np.ndarray, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: normalize, resize, auto-adjust.

    Returns (image_bgr_512, image_rgb_512).
    """
    image_bgr = _normalize_to_bgr(image_rgb)
    image_bgr = resize_preserve_aspect(image_bgr, size)
    image_bgr = _auto_adjust_brightness(image_bgr)
    image_rgb_out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb_out


def _detect_face_with_hints(image_bgr: np.ndarray) -> tuple[FaceLandmarks | None, str]:
    """Extract landmarks with better error messages for common failure modes.

    Returns (face_or_None, error_hint_string).
    """
    try:
        face = extract_landmarks(image_bgr)
    except Exception as exc:
        logger.error("Landmark extraction failed: %s\n%s", exc, traceback.format_exc())
        return None, f"Landmark extraction error: {exc}"

    if face is not None:
        return face, ""

    # Try to give a more useful hint about why detection failed.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Check if image is mostly black (too dark even after auto-adjust)
    if float(np.mean(gray)) < 30:
        return None, (
            "No face detected -- the image appears very dark. Try a photo with better lighting."
        )

    # Check for very low contrast (washed out)
    if float(np.std(gray)) < 15:
        return None, (
            "No face detected -- the image has very low contrast. "
            "Try a photo with more natural lighting."
        )

    # Check aspect ratio -- extremely tall/wide may indicate a side profile crop
    aspect = w / max(h, 1)
    if aspect > 2.5 or aspect < 0.4:
        return None, (
            "No face detected -- unusual aspect ratio. Use a standard portrait or headshot photo."
        )

    return None, (
        "No face detected. Make sure the photo shows a clear, "
        "well-lit frontal face. Side profiles and heavily occluded "
        "faces are not supported."
    )


# ---------------------------------------------------------------------------
# Symmetry analysis
# ---------------------------------------------------------------------------


def compute_symmetry_score(
    face: FaceLandmarks,
) -> tuple[float, dict[str, float]]:
    """Compute bilateral facial symmetry from a FaceLandmarks object.

    Reflects left-side landmarks across the facial midline and measures the
    Euclidean distance to their right-side counterparts. Distances are
    normalized by the inter-pupil distance to make the score scale-invariant.

    Args:
        face: FaceLandmarks with .pixel_coords property returning (478, 2).

    Returns:
        (overall_score, region_scores) where scores are 0-100
        (100 = perfectly symmetric).
    """
    coords = face.pixel_coords  # (478, 2) -- property, not method

    # Compute facial midline from forehead top (10) and chin bottom (152)
    mid_top = coords[MIDLINE_TOP]  # (2,)
    mid_bot = coords[MIDLINE_BOTTOM]  # (2,)

    # Midline direction vector and unit normal
    midline_dir = mid_bot - mid_top
    midline_len = np.linalg.norm(midline_dir)
    if midline_len < 1e-6:
        # Degenerate case -- landmarks are stacked
        return 0.0, {region: 0.0 for region in SYMMETRY_PAIRS}

    midline_unit = midline_dir / midline_len
    # Normal to midline (pointing right)
    midline_normal = np.array([midline_unit[1], -midline_unit[0]])

    # Normalization factor: use inter-eye distance (outer corners 33 <-> 263)
    # for scale-invariant scoring
    inter_eye = float(np.linalg.norm(coords[33] - coords[263]))
    if inter_eye < 1e-6:
        inter_eye = midline_len * 0.4  # fallback

    region_scores: dict[str, float] = {}
    all_distances: list[float] = []

    for region, pairs in SYMMETRY_PAIRS.items():
        region_dists: list[float] = []
        for left_idx, right_idx in pairs:
            left_pt = coords[left_idx]
            right_pt = coords[right_idx]

            # Reflect left point across the midline:
            # 1. Vector from midline top to the point
            v = left_pt - mid_top
            # 2. Component along the midline normal
            normal_component = np.dot(v, midline_normal)
            # 3. Reflected point: subtract twice the normal component
            reflected = left_pt - 2.0 * normal_component * midline_normal

            # Distance between reflected-left and actual-right
            dist = float(np.linalg.norm(reflected - right_pt))
            region_dists.append(dist)

        # Normalize by inter-eye distance and convert to 0-100 score
        if region_dists:
            mean_dist = float(np.mean(region_dists))
            # Normalized distance as fraction of inter-eye distance
            norm_dist = mean_dist / inter_eye
            # Convert to score: 0 distance = 100, large distance = 0
            # Use exponential decay so small asymmetries are penalized gently
            score = 100.0 * np.exp(-3.0 * norm_dist)
            region_scores[region] = round(max(0.0, min(100.0, score)), 1)
            all_distances.extend(region_dists)
        else:
            region_scores[region] = 0.0

    # Overall score: weighted mean of all pair distances
    if all_distances:
        overall_norm = float(np.mean(all_distances)) / inter_eye
        overall = 100.0 * np.exp(-3.0 * overall_norm)
        overall = round(max(0.0, min(100.0, overall)), 1)
    else:
        overall = 0.0

    return overall, region_scores


def render_symmetry_overlay(
    image_bgr: np.ndarray,
    face: FaceLandmarks,
    region_scores: dict[str, float],
) -> np.ndarray:
    """Draw a symmetry visualization overlay on the image.

    Draws the facial midline and color-codes bilateral landmark pairs by
    their region symmetry score: green (>80), yellow (50-80), red (<50).
    """
    canvas = image_bgr.copy()
    coords = face.pixel_coords

    # Draw midline
    mid_top = coords[MIDLINE_TOP].astype(int)
    mid_bot = coords[MIDLINE_BOTTOM].astype(int)
    cv2.line(canvas, tuple(mid_top), tuple(mid_bot), (255, 200, 0), 2, cv2.LINE_AA)

    # Small label at midline top
    cv2.putText(
        canvas,
        "midline",
        (int(mid_top[0]) + 5, int(mid_top[1]) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )

    def _score_color(score: float) -> tuple[int, int, int]:
        """BGR color based on symmetry score."""
        if score >= 80:
            return (0, 200, 0)  # green
        elif score >= 50:
            return (0, 200, 220)  # yellow (BGR)
        else:
            return (0, 0, 220)  # red

    for region, pairs in SYMMETRY_PAIRS.items():
        score = region_scores.get(region, 0.0)
        color = _score_color(score)

        for left_idx, right_idx in pairs:
            lp = coords[left_idx].astype(int)
            rp = coords[right_idx].astype(int)

            # Draw landmark dots
            cv2.circle(canvas, tuple(lp), 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(rp), 3, color, -1, cv2.LINE_AA)

            # Draw thin connecting line across midline
            cv2.line(canvas, tuple(lp), tuple(rp), color, 1, cv2.LINE_AA)

    # Draw region labels with scores
    # Position labels near each region's centroid
    region_label_offsets: dict[str, tuple[int, int]] = {
        "eyes": (0, -15),
        "brows": (0, -10),
        "cheeks": (15, 0),
        "mouth": (0, 10),
        "jaw": (0, 15),
    }

    for region, pairs in SYMMETRY_PAIRS.items():
        score = region_scores.get(region, 0.0)
        color = _score_color(score)

        # Compute centroid of the region landmarks
        region_pts = []
        for left_idx, right_idx in pairs:
            region_pts.append(coords[left_idx])
            region_pts.append(coords[right_idx])
        centroid = np.mean(region_pts, axis=0).astype(int)

        ox, oy = region_label_offsets.get(region, (0, 0))
        label_pos = (int(centroid[0]) + ox, int(centroid[1]) + oy)

        label = f"{region}: {score:.0f}"
        cv2.putText(
            canvas,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def _format_symmetry_text(
    overall: float,
    region_scores: dict[str, float],
    prefix: str = "",
) -> str:
    """Format symmetry scores into a readable text block."""
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append(f"Overall symmetry: {overall:.1f}/100")
    for region, score in region_scores.items():
        bar_len = int(score / 5)
        bar = "|" * bar_len + "." * (20 - bar_len)
        lines.append(f"  {region:>6s}: {score:5.1f}  [{bar}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symmetry tab callbacks
# ---------------------------------------------------------------------------


def analyze_symmetry(image_rgb: np.ndarray):
    """Analyze facial symmetry from an uploaded photo."""
    if image_rgb is None:
        b = _blank()
        return b, "Upload a face photo to analyze symmetry."

    try:
        image_bgr, image_rgb_512 = _prepare_image(image_rgb, 512)
    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        b = _blank()
        return b, f"Image conversion failed: {exc}"

    face, hint = _detect_face_with_hints(image_bgr)
    if face is None:
        return image_rgb_512, hint

    overall, region_scores = compute_symmetry_score(face)
    overlay_bgr = render_symmetry_overlay(image_bgr, face, region_scores)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    text = _format_symmetry_text(overall, region_scores)
    return overlay_rgb, text


def analyze_symmetry_comparison(
    pre_image_rgb: np.ndarray,
    post_image_rgb: np.ndarray,
):
    """Compare symmetry between pre- and post-procedure photos."""
    b = _blank()
    empty = (b, b, "Upload both pre and post photos to compare.")

    if pre_image_rgb is None or post_image_rgb is None:
        return empty

    try:
        pre_bgr, _ = _prepare_image(pre_image_rgb, 512)
        post_bgr, _ = _prepare_image(post_image_rgb, 512)
    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        return b, b, f"Image conversion failed: {exc}"

    pre_face, pre_hint = _detect_face_with_hints(pre_bgr)
    if pre_face is None:
        return b, b, f"Pre-procedure: {pre_hint}"

    post_face, post_hint = _detect_face_with_hints(post_bgr)
    if post_face is None:
        return b, b, f"Post-procedure: {post_hint}"

    pre_overall, pre_regions = compute_symmetry_score(pre_face)
    post_overall, post_regions = compute_symmetry_score(post_face)

    pre_overlay = render_symmetry_overlay(pre_bgr, pre_face, pre_regions)
    post_overlay = render_symmetry_overlay(post_bgr, post_face, post_regions)

    pre_rgb = cv2.cvtColor(pre_overlay, cv2.COLOR_BGR2RGB)
    post_rgb = cv2.cvtColor(post_overlay, cv2.COLOR_BGR2RGB)

    lines = []
    lines.append(_format_symmetry_text(pre_overall, pre_regions, prefix="-- Pre-procedure --"))
    lines.append("")
    lines.append(_format_symmetry_text(post_overall, post_regions, prefix="-- Post-procedure --"))
    lines.append("")

    delta = post_overall - pre_overall
    direction = "improved" if delta > 0 else "decreased"
    lines.append(f"Change: {delta:+.1f} ({direction})")

    # Per-region deltas
    for region in pre_regions:
        d = post_regions.get(region, 0.0) - pre_regions[region]
        lines.append(f"  {region:>6s}: {d:+.1f}")

    return pre_rgb, post_rgb, "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def warp_image_tps(image, src_pts, dst_pts):
    """Thin-plate spline warp (CPU only)."""
    from landmarkdiff.synthetic.tps_warp import warp_image_tps as _warp

    return _warp(image, src_pts, dst_pts)


def resize_preserve_aspect(image, size=512):
    """Resize to square canvas, padding to preserve aspect ratio."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def mask_composite(warped, original, mask):
    """Alpha-blend warped region into original using mask."""
    mask_3 = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
    return (warped * mask_3 + original * (1.0 - mask_3)).astype(np.uint8)


def _blank():
    return np.zeros((512, 512, 3), dtype=np.uint8)


def process_image(image_rgb, procedure, intensity):
    """Run the TPS pipeline on a single image, including symmetry scores."""
    if image_rgb is None:
        b = _blank()
        return b, b, b, b, "Upload a face photo to begin."

    t0 = time.monotonic()

    try:
        image_bgr, image_rgb_512 = _prepare_image(image_rgb, 512)
    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        b = _blank()
        return b, b, b, b, f"Image conversion failed: {exc}"

    face, hint = _detect_face_with_hints(image_bgr)
    if face is None:
        if hint:
            return image_rgb_512, image_rgb_512, image_rgb_512, image_rgb_512, hint
        return (
            image_rgb_512,
            image_rgb_512,
            image_rgb_512,
            image_rgb_512,
            "No face detected. Try a clearer, well-lit frontal photo.",
        )

    try:
        manipulated = apply_procedure_preset(face, procedure, float(intensity), image_size=512)
        wireframe = render_wireframe(manipulated, width=512, height=512)
        wireframe_rgb = cv2.cvtColor(wireframe, cv2.COLOR_GRAY2RGB)

        mask = generate_surgical_mask(face, procedure, 512, 512)
        mask_vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        warped = warp_image_tps(image_bgr, face.pixel_coords, manipulated.pixel_coords)
        composited = mask_composite(warped, image_bgr, mask)
        composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)

        displacement = np.mean(np.linalg.norm(manipulated.pixel_coords - face.pixel_coords, axis=1))
        elapsed = time.monotonic() - t0

        # Compute symmetry for original and predicted result
        pre_overall, pre_regions = compute_symmetry_score(face)
        post_overall, post_regions = compute_symmetry_score(manipulated)
        sym_delta = post_overall - pre_overall

        info_lines = [
            f"Procedure: {procedure.replace('_', ' ').title()}",
            f"Intensity: {intensity:.0f}%",
            f"Landmarks: {len(face.landmarks)}",
            f"Avg displacement: {displacement:.1f} px",
            f"Confidence: {face.confidence:.2f}",
            f"Time: {elapsed:.2f}s | Mode: TPS (CPU)",
            "",
            f"Symmetry (before): {pre_overall:.1f}/100",
            f"Symmetry (after):  {post_overall:.1f}/100",
            f"Symmetry change:   {sym_delta:+.1f}",
        ]
        info = "\n".join(info_lines)
        return wireframe_rgb, mask_vis, composited_rgb, image_rgb_512, info

    except Exception as exc:
        logger.error("Processing failed: %s\n%s", exc, traceback.format_exc())
        b = _blank()
        return b, b, b, b, f"Processing error: {exc}"


def compare_procedures(image_rgb, intensity):
    """Compare all six procedures at the same intensity."""
    if image_rgb is None:
        return [_blank()] * len(PROCEDURES)

    try:
        image_bgr, _ = _prepare_image(image_rgb, 512)
        face, _ = _detect_face_with_hints(image_bgr)
        if face is None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return [rgb] * len(PROCEDURES)

        results = []
        for proc in PROCEDURES:
            manip = apply_procedure_preset(face, proc, float(intensity), image_size=512)
            mask = generate_surgical_mask(face, proc, 512, 512)
            warped = warp_image_tps(image_bgr, face.pixel_coords, manip.pixel_coords)
            comp = mask_composite(warped, image_bgr, mask)
            results.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        return results
    except Exception as exc:
        logger.error("Compare failed: %s\n%s", exc, traceback.format_exc())
        return [_blank()] * len(PROCEDURES)


def intensity_sweep(image_rgb, procedure):
    """Generate results at 0%, 20%, 40%, 60%, 80%, 100% intensity."""
    if image_rgb is None:
        return []

    try:
        image_bgr, _ = _prepare_image(image_rgb, 512)
        face, _ = _detect_face_with_hints(image_bgr)
        if face is None:
            return []

        results = []
        for val in [0, 20, 40, 60, 80, 100]:
            if val == 0:
                results.append((cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "0%"))
                continue
            manip = apply_procedure_preset(face, procedure, float(val), image_size=512)
            mask = generate_surgical_mask(face, procedure, 512, 512)
            warped = warp_image_tps(image_bgr, face.pixel_coords, manip.pixel_coords)
            comp = mask_composite(warped, image_bgr, mask)
            results.append((cv2.cvtColor(comp, cv2.COLOR_BGR2RGB), f"{val}%"))
        return results
    except Exception as exc:
        logger.error("Sweep failed: %s\n%s", exc, traceback.format_exc())
        return []


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

_proc_table = "\n".join(
    f"| {name.replace('_', ' ').title()} | {desc} |" for name, desc in PROCEDURE_INFO.items()
)

with gr.Blocks(
    title="LandmarkDiff",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        f"# LandmarkDiff\n\n"
        f"Facial surgery outcome prediction from clinical photography. "
        f"Upload a face photo, pick a procedure, adjust intensity.\n\n"
        f"| Procedure | Effect |\n|---|---|\n{_proc_table}\n\n"
        f"[GitHub]({GITHUB_URL}) | "
        f"[Docs]({GITHUB_URL}/tree/main/docs) | "
        f"[Wiki]({GITHUB_URL}/wiki)"
    )

    # -- Tab 1: Single Procedure --
    with gr.Tab("Single Procedure"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Face Photo", type="numpy", height=350)
                procedure = gr.Radio(
                    choices=PROCEDURES,
                    value="rhinoplasty",
                    label="Procedure",
                )
                intensity = gr.Slider(
                    0,
                    100,
                    50,
                    step=1,
                    label="Intensity (%)",
                    info="0 = no change, 100 = maximum",
                )
                run_btn = gr.Button("Generate", variant="primary", size="lg")
                info_box = gr.Textbox(label="Info", lines=10, interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    out_wireframe = gr.Image(label="Deformed Wireframe", height=256)
                    out_mask = gr.Image(label="Surgical Mask", height=256)
                with gr.Row():
                    out_result = gr.Image(label="Predicted Result", height=256)
                    out_original = gr.Image(label="Original", height=256)

        if EXAMPLE_IMAGES:
            gr.Examples(
                examples=[[str(p)] for p in EXAMPLE_IMAGES],
                inputs=[input_image],
                label="Example faces (click to load)",
            )

        outputs = [out_wireframe, out_mask, out_result, out_original, info_box]
        _inputs = [input_image, procedure, intensity]
        run_btn.click(fn=process_image, inputs=_inputs, outputs=outputs)
        # Auto-trigger on image upload and procedure change, but not on every
        # slider tick during drag (each tick would re-run TPS on free CPU,
        # causing severe lag). Use .release so it fires once on mouse-up.
        input_image.change(fn=process_image, inputs=_inputs, outputs=outputs)
        procedure.change(fn=process_image, inputs=_inputs, outputs=outputs)
        intensity.release(fn=process_image, inputs=_inputs, outputs=outputs)

    # -- Tab 2: Compare Procedures --
    with gr.Tab("Compare All"):
        gr.Markdown("All six procedures at the same intensity, side by side.")
        with gr.Row():
            with gr.Column(scale=1):
                cmp_image = gr.Image(label="Face Photo", type="numpy", height=300)
                cmp_intensity = gr.Slider(0, 100, 50, step=1, label="Intensity (%)")
                cmp_btn = gr.Button("Compare", variant="primary", size="lg")
            with gr.Column(scale=2):
                cmp_outputs = []
                for row_idx in range(2):
                    with gr.Row():
                        for col_idx in range(3):
                            idx = row_idx * 3 + col_idx
                            if idx < len(PROCEDURES):
                                cmp_outputs.append(
                                    gr.Image(
                                        label=PROCEDURES[idx].replace("_", " ").title(),
                                        height=200,
                                    )
                                )

        if EXAMPLE_IMAGES:
            gr.Examples(
                examples=[[str(p)] for p in EXAMPLE_IMAGES],
                inputs=[cmp_image],
                label="Examples",
            )

        cmp_btn.click(fn=compare_procedures, inputs=[cmp_image, cmp_intensity], outputs=cmp_outputs)

    # -- Tab 3: Intensity Sweep --
    with gr.Tab("Intensity Sweep"):
        gr.Markdown("See a procedure at 0% through 100% in six steps.")
        with gr.Row():
            with gr.Column(scale=1):
                sweep_image = gr.Image(label="Face Photo", type="numpy", height=300)
                sweep_proc = gr.Radio(choices=PROCEDURES, value="rhinoplasty", label="Procedure")
                sweep_btn = gr.Button("Sweep", variant="primary", size="lg")
            with gr.Column(scale=2):
                sweep_gallery = gr.Gallery(label="0% to 100%", columns=3, height=400)

        if EXAMPLE_IMAGES:
            gr.Examples(
                examples=[[str(p)] for p in EXAMPLE_IMAGES],
                inputs=[sweep_image],
                label="Examples",
            )

        sweep_btn.click(
            fn=intensity_sweep,
            inputs=[sweep_image, sweep_proc],
            outputs=[sweep_gallery],
        )

    # -- Tab 4: Symmetry Analysis --
    with gr.Tab("Symmetry Analysis"):
        gr.Markdown(
            "### Bilateral Facial Symmetry\n\n"
            "Analyzes left-right symmetry across five facial regions "
            "(eyes, brows, cheeks, mouth, jaw) using MediaPipe 478-point "
            "face mesh landmark pairs. The midline is computed from the "
            "forehead apex to the chin, and left landmarks are reflected "
            "across it to measure deviation from the right side.\n\n"
            "**Score interpretation:** 90-100 = highly symmetric, "
            "70-89 = mild asymmetry, <70 = notable asymmetry."
        )

        with gr.Tabs():
            # Sub-tab: Single image analysis
            with gr.Tab("Single Photo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sym_image = gr.Image(
                            label="Face Photo",
                            type="numpy",
                            height=350,
                        )
                        sym_btn = gr.Button(
                            "Analyze Symmetry",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column(scale=1):
                        sym_overlay = gr.Image(label="Symmetry Overlay", height=350)

                sym_scores_box = gr.Textbox(
                    label="Symmetry Scores",
                    lines=8,
                    interactive=False,
                )

                if EXAMPLE_IMAGES:
                    gr.Examples(
                        examples=[[str(p)] for p in EXAMPLE_IMAGES],
                        inputs=[sym_image],
                        label="Examples",
                    )

                sym_btn.click(
                    fn=analyze_symmetry,
                    inputs=[sym_image],
                    outputs=[sym_overlay, sym_scores_box],
                )

            # Sub-tab: Pre vs post comparison
            with gr.Tab("Pre vs Post Comparison"):
                gr.Markdown(
                    "Upload a pre-procedure and post-procedure photo to compare "
                    "how symmetry changed."
                )
                with gr.Row():
                    sym_pre_image = gr.Image(
                        label="Pre-Procedure",
                        type="numpy",
                        height=300,
                    )
                    sym_post_image = gr.Image(
                        label="Post-Procedure",
                        type="numpy",
                        height=300,
                    )
                sym_cmp_btn = gr.Button(
                    "Compare Symmetry",
                    variant="primary",
                    size="lg",
                )
                with gr.Row():
                    sym_pre_overlay = gr.Image(
                        label="Pre Symmetry Overlay",
                        height=300,
                    )
                    sym_post_overlay = gr.Image(
                        label="Post Symmetry Overlay",
                        height=300,
                    )
                sym_cmp_box = gr.Textbox(
                    label="Comparison",
                    lines=14,
                    interactive=False,
                )

                sym_cmp_btn.click(
                    fn=analyze_symmetry_comparison,
                    inputs=[sym_pre_image, sym_post_image],
                    outputs=[sym_pre_overlay, sym_post_overlay, sym_cmp_box],
                )

    gr.Markdown(
        f"<div style='text-align:center;color:#999;font-size:0.8em;padding:8px'>"
        f"LandmarkDiff v0.2.2 | TPS on CPU | MediaPipe 478-point mesh | "
        f"<a href='{GITHUB_URL}'>GitHub</a> | MIT License</div>"
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
