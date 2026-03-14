"""Latent space interpolation for LandmarkDiff.

Generates smooth transitions between two different surgical outcomes by
interpolating in the diffusion latent space. Demonstrates the model's
disentangled representation — geometry from conditioning, texture from latent.

Usage:
    python scripts/latent_interpolation.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/interpolation/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors.

    SLERP produces more natural interpolation than linear interpolation
    for high-dimensional latent spaces because it follows a geodesic on
    the unit hypersphere.
    """
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    v0_norm = v0_flat / v0_flat.norm()
    v1_norm = v1_flat / v1_flat.norm()

    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        # Nearly parallel — fall back to linear
        result = (1 - t) * v0_flat + t * v1_flat
    else:
        sin_omega = torch.sin(omega)
        result = (torch.sin((1 - t) * omega) / sin_omega) * v0_flat + (
            torch.sin(t * omega) / sin_omega
        ) * v1_flat

    # Rescale to preserve original magnitude
    target_norm = (1 - t) * v0_flat.norm() + t * v1_flat.norm()
    result = result / result.norm() * target_norm

    return result.reshape(v0.shape).to(v0.dtype)


def generate_with_fixed_noise(
    pipe,
    conditioning_pil: Image.Image,
    noise: torch.Tensor,
    prompt: str,
    neg_prompt: str,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
) -> np.ndarray:
    """Generate from a fixed initial noise tensor (no seed randomness).

    This allows us to control the latent space starting point precisely
    for interpolation experiments.
    """
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=pipe.device)

    latents = noise.clone() * scheduler.init_noise_sigma

    # Encode prompt
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    text_emb = text_emb.to(dtype=pipe.unet.dtype)

    # Negative prompt
    neg_input = pipe.tokenizer(
        neg_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    neg_emb = pipe.text_encoder(neg_input.input_ids.to(pipe.device))[0]
    neg_emb = neg_emb.to(dtype=pipe.unet.dtype)

    # Conditioning
    cond_tensor = pipe.prepare_image(
        conditioning_pil,
        width=512,
        height=512,
        batch_size=1,
        num_images_per_prompt=1,
        device=pipe.device,
        dtype=pipe.controlnet.dtype,
    )

    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(latents, t)

        # ControlNet
        down_samples, mid_sample = pipe.controlnet(
            scaled,
            t,
            encoder_hidden_states=text_emb,
            controlnet_cond=cond_tensor,
            return_dict=False,
        )

        # CFG: unconditional + conditional
        noise_uncond = pipe.unet(
            scaled,
            t,
            encoder_hidden_states=neg_emb,
        ).sample
        noise_cond = pipe.unet(
            scaled,
            t,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample

        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    decoded = ((decoded + 1) / 2).clamp(0, 1)
    img = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Latent space interpolation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/interpolation")
    parser.add_argument("--n-interp", type=int, default=8, help="Number of interpolation steps")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--max-pairs", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("Loading pipeline...")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    ckpt = Path(args.checkpoint)
    if (ckpt / "controlnet_ema").exists():
        ckpt = ckpt / "controlnet_ema"

    controlnet = ControlNetModel.from_pretrained(str(ckpt))
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load test pairs
    from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))[: args.max_pairs]

    prompt = "high quality photo of a face after cosmetic surgery, realistic skin texture"
    neg_prompt = "blurry, distorted, low quality, deformed"

    for idx, inp_file in enumerate(input_files):
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not target_file.exists():
            continue

        print(f"\n[{idx + 1}/{len(input_files)}] {prefix}")

        input_img = cv2.resize(cv2.imread(str(inp_file)), (512, 512))
        target_img = cv2.resize(cv2.imread(str(target_file)), (512, 512))

        # Use existing conditioning or generate it
        if cond_file.exists():
            conditioning = cv2.imread(str(cond_file))
            conditioning = cv2.resize(conditioning, (512, 512))
        else:
            face_lm = extract_landmarks(target_img)
            if face_lm is None:
                continue
            conditioning = render_landmark_image(face_lm, 512, 512)

        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        # Generate two different noise vectors (different "styles")
        gen_a = torch.Generator(device="cpu").manual_seed(42)
        noise_a = torch.randn(1, 4, 64, 64, generator=gen_a, device=device, dtype=dtype)

        gen_b = torch.Generator(device="cpu").manual_seed(123)
        noise_b = torch.randn(1, 4, 64, 64, generator=gen_b, device=device, dtype=dtype)

        # Generate endpoints
        print("  Generating endpoint A (seed=42)...")
        generate_with_fixed_noise(
            pipe,
            cond_pil,
            noise_a,
            prompt,
            neg_prompt,
            args.num_steps,
            args.guidance_scale,
        )
        print("  Generating endpoint B (seed=123)...")
        generate_with_fixed_noise(
            pipe,
            cond_pil,
            noise_b,
            prompt,
            neg_prompt,
            args.num_steps,
            args.guidance_scale,
        )

        # Interpolate
        t_values = np.linspace(0, 1, args.n_interp)
        interp_imgs = []

        for i, t in enumerate(t_values):
            print(f"  Interpolation t={t:.2f} ({i + 1}/{args.n_interp})...")
            noise_interp = slerp(noise_a, noise_b, t)
            img_interp = generate_with_fixed_noise(
                pipe,
                cond_pil,
                noise_interp,
                prompt,
                neg_prompt,
                args.num_steps,
                args.guidance_scale,
            )
            interp_imgs.append(img_interp)

        # Build grid: row of interpolated images with labels
        label_h = 30
        cell_size = 256
        n_cols = len(interp_imgs) + 2  # input + interp + target
        grid_w = n_cols * cell_size
        grid_h = cell_size + label_h

        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Labels
        labels = ["Input"] + [f"t={t:.2f}" for t in t_values] + ["Target"]
        for j, label in enumerate(labels):
            x = j * cell_size + 5
            cv2.putText(grid, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Images
        all_imgs = [input_img] + interp_imgs + [target_img]
        for j, img in enumerate(all_imgs):
            small = cv2.resize(img, (cell_size, cell_size))
            grid[label_h : label_h + cell_size, j * cell_size : (j + 1) * cell_size] = small

        fname = f"interpolation_{prefix}.png"
        cv2.imwrite(str(out_dir / fname), grid)
        print(f"  Saved: {fname}")

    print(f"\nAll interpolation grids saved to {out_dir}")


if __name__ == "__main__":
    main()
