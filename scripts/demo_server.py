"""Interactive clinical demo server for LandmarkDiff.

Provides a Gradio-based web interface for surgeons and patients to:
1. Upload a face photograph
2. Select a surgical procedure (rhinoplasty, blepharoplasty, etc.)
3. Adjust the deformation intensity
4. View the predicted post-surgical outcome in real-time

Designed for clinical deployment: runs on a single GPU, requires only
a smartphone photo, and produces results in ~10 seconds.

Usage:
    python scripts/demo_server.py --checkpoint checkpoints/phaseA/best
    python scripts/demo_server.py --share  # Create a public Gradio link

Requirements:
    pip install gradio>=4.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Global pipeline state (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────

_PIPE = None
_DEVICE = None
_LOADED_CKPT = None


def load_pipeline(checkpoint_path: str | Path) -> None:
    """Load the ControlNet pipeline into global state."""
    global _PIPE, _DEVICE, _LOADED_CKPT

    if _PIPE is not None and str(checkpoint_path) == _LOADED_CKPT:
        return

    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    checkpoint_path = Path(checkpoint_path)
    print(f"Loading pipeline from {checkpoint_path}...")

    controlnet_subdir = (
        "controlnet_ema" if (checkpoint_path / "controlnet_ema").exists() else "controlnet"
    )
    controlnet = ControlNetModel.from_pretrained(
        str(checkpoint_path / controlnet_subdir),
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable memory-efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  xformers enabled")
    except Exception:
        pass

    pipe.to("cuda")
    _PIPE = pipe
    _DEVICE = pipe.device
    _LOADED_CKPT = str(checkpoint_path)
    print(f"  Pipeline loaded on {_DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference logic
# ──────────────────────────────────────────────────────────────────────────────

PROCEDURE_PROMPTS = {
    "rhinoplasty": "high quality photo of a face after rhinoplasty nose reshaping surgery, realistic skin texture, natural appearance",
    "blepharoplasty": "high quality photo of a face after blepharoplasty eyelid surgery, realistic skin texture, natural appearance",
    "rhytidectomy": "high quality photo of a face after facelift surgery, realistic skin texture, younger appearance, natural",
    "orthognathic": "high quality photo of a face after jaw surgery, realistic skin texture, corrected bite, natural appearance",
}

NEGATIVE_PROMPT = "blurry, distorted, low quality, deformed, scarring, bruising, swelling, unnatural, painting, drawing"


def predict(
    input_image: np.ndarray,
    procedure: str,
    intensity: float = 1.0,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> dict:
    """Run the full LandmarkDiff prediction pipeline.

    Args:
        input_image: Input face photo (H, W, 3) RGB uint8.
        procedure: Surgical procedure name.
        intensity: Deformation intensity multiplier (0.0-2.0).
        num_steps: Number of diffusion inference steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed for reproducibility.

    Returns:
        dict with: prediction, conditioning, landmarks_before, landmarks_after,
                   processing_time, metrics
    """
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask
    from landmarkdiff.postprocess import histogram_match_skin

    t0 = time.time()

    # Convert from RGB (Gradio) to BGR (OpenCV)
    input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Resize to 512x512
    input_resized = cv2.resize(input_bgr, (512, 512))

    # 1. Extract landmarks
    landmarks = extract_landmarks(input_resized)
    if landmarks is None:
        return {
            "error": "Could not detect face landmarks. Please ensure the photo "
            "shows a clear, front-facing view of the face.",
            "processing_time": time.time() - t0,
        }

    # 2. Apply procedure-specific deformation
    manipulated = apply_procedure_preset(
        landmarks,
        procedure,
        intensity=intensity * 100.0,
        image_size=512,
    )

    # 3. Render conditioning image (MediaPipe mesh wireframe)
    conditioning = render_landmark_image(manipulated, 512, 512)

    # 4. Also render the "before" mesh for comparison
    before_mesh = render_landmark_image(landmarks, 512, 512)

    # 5. Run ControlNet inference
    cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
    cond_pil = Image.fromarray(cond_rgb)

    prompt = PROCEDURE_PROMPTS.get(procedure, PROCEDURE_PROMPTS["rhinoplasty"])
    generator = torch.Generator("cuda").manual_seed(seed)

    with torch.no_grad():
        result = _PIPE(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=cond_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    pred_pil = result.images[0]
    pred_bgr = cv2.cvtColor(np.array(pred_pil), cv2.COLOR_RGB2BGR)

    # 6. Post-processing: mask compositing + color matching
    try:
        mask = generate_surgical_mask(manipulated, procedure, 512, 512)
        mask_f = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)

        try:
            matched = histogram_match_skin(pred_bgr, input_resized, mask_f)
        except Exception:
            matched = pred_bgr

        if mask_f.ndim == 2:
            mask_f = mask_f[:, :, np.newaxis]

        composited = (mask_f * matched + (1.0 - mask_f) * input_resized).astype(np.uint8)
    except Exception:
        composited = pred_bgr

    processing_time = time.time() - t0

    # Convert results to RGB for Gradio
    return {
        "prediction": cv2.cvtColor(composited, cv2.COLOR_BGR2RGB),
        "raw_prediction": cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB),
        "conditioning": cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB),
        "before_mesh": cv2.cvtColor(before_mesh, cv2.COLOR_BGR2RGB),
        "input_resized": cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB),
        "processing_time": processing_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Gradio interface
# ──────────────────────────────────────────────────────────────────────────────


def create_demo(checkpoint_path: str | Path):
    """Create the Gradio demo interface."""
    import gradio as gr

    load_pipeline(checkpoint_path)

    def run_prediction(
        input_img,
        procedure,
        intensity,
        num_steps,
        guidance_scale,
        seed,
    ):
        """Gradio callback for prediction."""
        if input_img is None:
            return None, None, None, "Please upload a face photo."

        result = predict(
            input_image=input_img,
            procedure=procedure.lower(),
            intensity=intensity,
            num_steps=int(num_steps),
            guidance_scale=guidance_scale,
            seed=int(seed),
        )

        if "error" in result:
            return None, None, None, result["error"]

        status = f"Done in {result['processing_time']:.1f}s"

        return (
            result["prediction"],
            result["conditioning"],
            result["raw_prediction"],
            status,
        )

    with gr.Blocks(
        title="LandmarkDiff — Facial Surgery Outcome Prediction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # LandmarkDiff: Facial Surgery Outcome Prediction

            Upload a front-facing face photo, select a surgical procedure,
            and adjust the intensity to preview the predicted post-surgical outcome.

            **For research and educational purposes only. Not a substitute for
            professional medical consultation.**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Face Photo",
                    type="numpy",
                    sources=["upload", "webcam"],
                )

                procedure = gr.Dropdown(
                    choices=["Rhinoplasty", "Blepharoplasty", "Rhytidectomy", "Orthognathic"],
                    value="Rhinoplasty",
                    label="Procedure",
                )

                intensity = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Deformation Intensity",
                    info="0 = no change, 1 = standard, 2 = aggressive",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Diffusion Steps",
                        info="More steps = higher quality but slower",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        info="Change for different variations",
                    )

                predict_btn = gr.Button("Predict Outcome", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    output_image = gr.Image(label="Predicted Outcome", type="numpy")
                with gr.Row():
                    conditioning_image = gr.Image(
                        label="Landmark Conditioning",
                        type="numpy",
                    )
                    raw_output = gr.Image(
                        label="Raw Diffusion Output",
                        type="numpy",
                    )

        # Wire up the prediction
        predict_btn.click(
            fn=run_prediction,
            inputs=[input_image, procedure, intensity, num_steps, guidance_scale, seed],
            outputs=[output_image, conditioning_image, raw_output, status_text],
        )

        # Example inputs
        gr.Examples(
            examples=[
                ["Rhinoplasty", 1.0],
                ["Blepharoplasty", 0.8],
                ["Rhytidectomy", 1.2],
                ["Orthognathic", 1.0],
            ],
            inputs=[procedure, intensity],
        )

        gr.Markdown(
            """
            ---
            **LandmarkDiff** | Anatomically-Conditioned Latent Diffusion for Facial Surgery Prediction

            Pipeline: MediaPipe 478 landmarks → RBF displacement → 3-channel conditioning → ControlNet/SD1.5

            *This demo is part of a research project. Results are for illustration only.*
            """
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LandmarkDiff interactive demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "phaseA" / "best",
        help="Path to model checkpoint",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server host address")
    args = parser.parse_args()

    print("=" * 60)
    print("LandmarkDiff Interactive Demo")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print()

    demo = create_demo(args.checkpoint)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
