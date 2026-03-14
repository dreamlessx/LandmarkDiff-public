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
from landmarkdiff.landmarks import extract_landmarks
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
    """Run the TPS pipeline on a single image."""
    if image_rgb is None:
        b = _blank()
        return b, b, b, b, "Upload a face photo to begin."

    t0 = time.monotonic()

    try:
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = resize_preserve_aspect(image_bgr, 512)
        image_rgb_512 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        b = _blank()
        return b, b, b, b, f"Image conversion failed: {exc}"

    try:
        face = extract_landmarks(image_bgr)
    except Exception as exc:
        logger.error("Landmark extraction failed: %s\n%s", exc, traceback.format_exc())
        b = _blank()
        return b, b, b, b, f"Landmark extraction error: {exc}"

    if face is None:
        return (
            image_rgb_512, image_rgb_512, image_rgb_512, image_rgb_512,
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

        displacement = np.mean(
            np.linalg.norm(manipulated.pixel_coords - face.pixel_coords, axis=1)
        )
        elapsed = time.monotonic() - t0

        info = (
            f"Procedure: {procedure.replace('_', ' ').title()}\n"
            f"Intensity: {intensity:.0f}%\n"
            f"Landmarks: {len(face.landmarks)}\n"
            f"Avg displacement: {displacement:.1f} px\n"
            f"Confidence: {face.confidence:.2f}\n"
            f"Time: {elapsed:.2f}s | Mode: TPS (CPU)"
        )
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
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = resize_preserve_aspect(image_bgr, 512)
        face = extract_landmarks(image_bgr)
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
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = resize_preserve_aspect(image_bgr, 512)
        face = extract_landmarks(image_bgr)
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
    f"| {name.replace('_', ' ').title()} | {desc} |"
    for name, desc in PROCEDURE_INFO.items()
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
                    choices=PROCEDURES, value="rhinoplasty", label="Procedure",
                )
                intensity = gr.Slider(
                    0, 100, 50, step=1, label="Intensity (%)",
                    info="0 = no change, 100 = maximum",
                )
                run_btn = gr.Button("Generate", variant="primary", size="lg")
                info_box = gr.Textbox(label="Info", lines=6, interactive=False)

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
        run_btn.click(fn=process_image, inputs=[input_image, procedure, intensity], outputs=outputs)
        for trigger in [input_image, procedure, intensity]:
            trigger.change(fn=process_image, inputs=[input_image, procedure, intensity], outputs=outputs)

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
                inputs=[cmp_image], label="Examples",
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
                inputs=[sweep_image], label="Examples",
            )

        sweep_btn.click(fn=intensity_sweep, inputs=[sweep_image, sweep_proc], outputs=[sweep_gallery])

    gr.Markdown(
        f"<div style='text-align:center;color:#999;font-size:0.8em;padding:8px'>"
        f"LandmarkDiff v0.2.2 | TPS on CPU | MediaPipe 478-point mesh | "
        f"<a href='{GITHUB_URL}'>GitHub</a> | MIT License</div>"
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
