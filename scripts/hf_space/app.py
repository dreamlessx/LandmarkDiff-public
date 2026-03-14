"""LandmarkDiff Hugging Face Spaces Demo - TPS-only (CPU)."""

from __future__ import annotations

import logging
import traceback

import cv2
import gradio as gr
import numpy as np

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS, apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v0.2.1"

GITHUB_URL = "https://github.com/dreamlessx/LandmarkDiff-public"
DOCS_URL = f"{GITHUB_URL}/tree/main/docs"
WIKI_URL = f"{GITHUB_URL}/wiki"
DISCUSSIONS_URL = f"{GITHUB_URL}/discussions"

PROCEDURE_DESCRIPTIONS = {
    "rhinoplasty": "Nose reshaping -- adjusts nasal bridge, tip projection, and alar width",
    "blepharoplasty": "Eyelid surgery -- modifies upper/lower lid position and canthal tilt",
    "rhytidectomy": "Facelift -- tightens midface and jawline contours",
    "orthognathic": "Jaw surgery -- repositions maxilla and mandible for skeletal alignment",
    "brow_lift": "Brow lift -- elevates brow position and reduces forehead ptosis",
    "mentoplasty": "Chin surgery -- adjusts chin projection and vertical height",
}


def warp_image_tps(image, src_pts, dst_pts):
    """Thin-plate spline warp (CPU only)."""
    from landmarkdiff.synthetic.tps_warp import warp_image_tps as _warp

    return _warp(image, src_pts, dst_pts)


def mask_composite(warped, original, mask):
    """Alpha blend warped into original using mask."""
    mask_3 = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
    return (warped * mask_3 + original * (1.0 - mask_3)).astype(np.uint8)


PROCEDURES = list(PROCEDURE_LANDMARKS.keys())


def _error_result(msg):
    """Return a 5-tuple of blanks + error message for the UI."""
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    return blank, blank, blank, blank, msg


def process_image(image_rgb, procedure, intensity):
    """Process a single image through the TPS pipeline."""
    if image_rgb is None:
        return _error_result("Upload a face photo to begin.")

    try:
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (512, 512))
        image_rgb_512 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        return _error_result(f"Image conversion failed: {exc}")

    try:
        face = extract_landmarks(image_bgr)
    except Exception as exc:
        logger.error("Landmark extraction failed: %s\n%s", exc, traceback.format_exc())
        return _error_result(f"Landmark extraction error: {exc}")

    if face is None:
        return (
            image_rgb_512,
            image_rgb_512,
            image_rgb_512,
            image_rgb_512,
            "No face detected. Try a clearer photo with good lighting.",
        )

    try:
        manipulated = apply_procedure_preset(face, procedure, float(intensity), image_size=512)

        wireframe = render_wireframe(manipulated, width=512, height=512)
        wireframe_rgb = cv2.cvtColor(wireframe, cv2.COLOR_GRAY2RGB)

        mask = generate_surgical_mask(face, procedure, 512, 512)
        mask_vis = (mask * 255).astype(np.uint8)

        warped = warp_image_tps(image_bgr, face.pixel_coords, manipulated.pixel_coords)
        composited = mask_composite(warped, image_bgr, mask)
        composited_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)

        side_by_side = np.hstack([image_rgb_512, composited_rgb])

        displacement = np.mean(np.linalg.norm(manipulated.pixel_coords - face.pixel_coords, axis=1))

        info = (
            f"Procedure: {procedure}\n"
            f"Intensity: {intensity:.0f}%\n"
            f"Landmarks: {len(face.landmarks)}\n"
            f"Avg displacement: {displacement:.1f} px\n"
            f"Confidence: {face.confidence:.2f}\n"
            f"Mode: TPS (CPU)"
        )
        return wireframe_rgb, mask_vis, composited_rgb, side_by_side, info

    except Exception as exc:
        logger.error("Processing failed: %s\n%s", exc, traceback.format_exc())
        return _error_result(f"Processing error: {exc}")


def compare_procedures(image_rgb, intensity):
    """Compare all procedures at the same intensity."""
    if image_rgb is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return [blank] * len(PROCEDURES)

    try:
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (512, 512))

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
        logger.error("Compare procedures failed: %s\n%s", exc, traceback.format_exc())
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return [blank] * len(PROCEDURES)


def intensity_sweep(image_rgb, procedure):
    """Generate intensity sweep from 0 to 100."""
    if image_rgb is None:
        return []

    try:
        image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (512, 512))

        face = extract_landmarks(image_bgr)
        if face is None:
            return []

        steps = [0, 20, 40, 60, 80, 100]
        results = []
        for val in steps:
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
        logger.error("Intensity sweep failed: %s\n%s", exc, traceback.format_exc())
        return []


# -- Build the procedure table for the description --
_proc_rows = "\n".join(
    f"| **{name.replace('_', ' ').title()}** | {desc} |"
    for name, desc in PROCEDURE_DESCRIPTIONS.items()
)

HEADER_MD = f"""
# LandmarkDiff

**Anatomically-conditioned facial surgery outcome prediction from standard clinical photography**

Upload a face photo, select a procedure, and adjust intensity to see a predicted
surgical outcome in real time.
This demo runs TPS (thin-plate spline) warping on CPU. The full package also supports
GPU-accelerated ControlNet and img2img inference modes.

---

### Supported Procedures

| Procedure | Description |
|-----------|-------------|
{_proc_rows}

---

### How It Works

1. **Landmark detection** -- MediaPipe extracts a 478-point facial mesh from the input photo.
2. **Anatomical displacement** -- Procedure-specific presets shift landmark subsets by calibrated
   vectors (intensity 0-100 controls magnitude).
3. **TPS deformation** -- A thin-plate spline maps source landmarks to displaced targets, warping
   the image smoothly while preserving non-surgical regions.
4. **Masked compositing** -- A procedure-aware mask blends the warped region back into the
   original, keeping hair, background, and uninvolved anatomy intact.

In GPU modes the deformed wireframe is passed to a ControlNet-conditioned Stable Diffusion
pipeline for photorealistic rendering, followed by CodeFormer + Real-ESRGAN post-processing.

---

[GitHub]({GITHUB_URL}) | \
[Documentation]({DOCS_URL}) | \
[Wiki]({WIKI_URL}) | \
[Discussions]({DISCUSSIONS_URL})
"""

FOOTER_MD = f"""
---
<p style="text-align:center; color:#888; font-size:0.85em;">
LandmarkDiff {VERSION} &middot;
<a href="{GITHUB_URL}">GitHub</a> &middot;
<a href="{WIKI_URL}">Wiki</a> &middot;
<a href="{DISCUSSIONS_URL}">Discussions</a> &middot;
MIT License
</p>
"""


with gr.Blocks(
    title="LandmarkDiff - Surgical Outcome Prediction",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(HEADER_MD)

    with gr.Tab("Single Procedure"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Face Photo", type="numpy", height=350)
                procedure = gr.Radio(
                    choices=PROCEDURES,
                    value="rhinoplasty",
                    label="Surgical Procedure",
                )
                intensity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Intensity (%)",
                    info="0 = no change, 100 = maximum effect",
                )
                run_btn = gr.Button("Generate Preview", variant="primary", size="lg")
                info_box = gr.Textbox(label="Info", lines=6, interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    out_wireframe = gr.Image(label="Deformed Wireframe", height=256)
                    out_mask = gr.Image(label="Surgical Mask", height=256)
                with gr.Row():
                    out_result = gr.Image(label="Predicted Result", height=256)
                    out_sidebyside = gr.Image(label="Before / After", height=256)

        run_btn.click(
            fn=process_image,
            inputs=[input_image, procedure, intensity],
            outputs=[out_wireframe, out_mask, out_result, out_sidebyside, info_box],
        )
        for trigger in [input_image, procedure, intensity]:
            trigger.change(
                fn=process_image,
                inputs=[input_image, procedure, intensity],
                outputs=[out_wireframe, out_mask, out_result, out_sidebyside, info_box],
            )

    with gr.Tab("Compare Procedures"):
        gr.Markdown("Compare all six procedures side by side at the same intensity.")
        with gr.Row():
            with gr.Column(scale=1):
                cmp_image = gr.Image(label="Upload Face Photo", type="numpy", height=300)
                cmp_intensity = gr.Slider(0, 100, 50, step=1, label="Intensity (%)")
                cmp_btn = gr.Button("Compare All", variant="primary", size="lg")
            with gr.Column(scale=2):
                cmp_outputs = []
                rows_needed = (len(PROCEDURES) + 2) // 3
                for row_idx in range(rows_needed):
                    with gr.Row():
                        for col_idx in range(3):
                            proc_idx = row_idx * 3 + col_idx
                            if proc_idx < len(PROCEDURES):
                                cmp_outputs.append(
                                    gr.Image(
                                        label=PROCEDURES[proc_idx].replace("_", " ").title(),
                                        height=200,
                                    )
                                )

        cmp_btn.click(
            fn=compare_procedures,
            inputs=[cmp_image, cmp_intensity],
            outputs=cmp_outputs,
        )

    with gr.Tab("Intensity Sweep"):
        gr.Markdown(
            "See how a procedure looks across intensity levels (0% through 100% in 20% steps)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                sweep_image = gr.Image(label="Upload Face Photo", type="numpy", height=300)
                sweep_procedure = gr.Radio(
                    choices=PROCEDURES,
                    value="rhinoplasty",
                    label="Procedure",
                )
                sweep_btn = gr.Button("Generate Sweep", variant="primary", size="lg")
            with gr.Column(scale=2):
                sweep_gallery = gr.Gallery(
                    label="Intensity Sweep (0% - 100%)", columns=3, height=400
                )

        sweep_btn.click(
            fn=intensity_sweep,
            inputs=[sweep_image, sweep_procedure],
            outputs=[sweep_gallery],
        )

    gr.Markdown(FOOTER_MD)

if __name__ == "__main__":
    demo.launch(show_error=True)
