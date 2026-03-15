"""Progressive denoising visualization.

Shows the diffusion process at multiple timesteps, from pure noise to
the final generated image. This is a common and informative figure for
diffusion model papers — it demonstrates:
  1. How structure emerges from noise
  2. That the model converges to anatomically plausible faces
  3. The ControlNet's influence at different noise levels

Generates a grid: [conditioning | t=T | t=0.8T | ... | t=0 | target]

Usage:
    python scripts/progressive_denoising.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/fig_progressive_denoising.png
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


def load_pipeline(checkpoint: str, device: torch.device):
    """Load ControlNet pipeline."""
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    ckpt = Path(checkpoint)
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
    return pipe


def run_with_intermediates(
    pipe,
    conditioning_pil: Image.Image,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    cn_scale: float = 1.0,
    seed: int = 42,
    capture_steps: list[int] | None = None,
) -> list[tuple[int, Image.Image]]:
    """Run inference and capture intermediate latent states.

    Returns list of (step_index, decoded_image) tuples.
    capture_steps specifies which step indices to decode (0-indexed).
    If None, captures at evenly spaced intervals.
    """
    from diffusers import UniPCMultistepScheduler

    device = pipe.device
    dtype = pipe.unet.dtype

    # Set up scheduler
    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    if capture_steps is None:
        # Capture ~8 evenly spaced steps + first and last
        n_captures = min(8, num_steps)
        capture_steps = sorted(
            set(
                [0]
                + [int(i * (num_steps - 1) / (n_captures - 1)) for i in range(n_captures)]
                + [num_steps - 1]
            )
        )

    # Encode prompt
    gen = torch.Generator(device="cpu").manual_seed(seed)
    prompt = "high quality photo of a face after cosmetic surgery"
    neg_prompt = "blurry, distorted, low quality"

    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)

    neg_input = pipe.tokenizer(
        neg_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    neg_emb = pipe.text_encoder(neg_input.input_ids.to(device))[0].to(dtype=dtype)

    # Classifier-free guidance: concat negative + positive
    text_emb_cfg = torch.cat([neg_emb, text_emb], dim=0)

    # Prepare conditioning image
    cond_tensor = pipe.prepare_image(
        conditioning_pil,
        width=512,
        height=512,
        batch_size=1,
        num_images_per_prompt=1,
        device=device,
        dtype=pipe.controlnet.dtype,
    )
    # Duplicate for CFG
    cond_tensor_cfg = torch.cat([cond_tensor] * 2, dim=0)

    # Initial noise
    latents = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    intermediates = []

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # ControlNet
            down_samples, mid_sample = pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_emb_cfg,
                controlnet_cond=cond_tensor_cfg,
                conditioning_scale=cn_scale,
                return_dict=False,
            )

            # UNet
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_emb_cfg,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Capture intermediate if requested
            if i in capture_steps:
                decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded_np = decoded[0].permute(1, 2, 0).float().cpu().numpy()
                decoded_np = (decoded_np * 255).astype(np.uint8)
                intermediates.append((i, Image.fromarray(decoded_np)))

    return intermediates


def make_progressive_grid(
    rows: list[dict],
    n_display_steps: int = 8,
) -> np.ndarray:
    """Create a grid showing progressive denoising for multiple samples.

    rows: list of dicts with keys:
        - "conditioning": PIL image
        - "target": PIL image
        - "intermediates": list of (step_idx, PIL image)
        - "label": str label for the row
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    # Columns: conditioning | step_0 | step_1 | ... | step_N | target
    # Select evenly spaced intermediates
    first_row_intermediates = rows[0]["intermediates"]
    total_steps = len(first_row_intermediates)
    if total_steps > n_display_steps:
        indices = [
            int(i * (total_steps - 1) / (n_display_steps - 1)) for i in range(n_display_steps)
        ]
    else:
        indices = list(range(total_steps))
        n_display_steps = total_steps

    n_cols = 2 + n_display_steps  # conditioning + steps + target

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, row_data in enumerate(rows):
        intermediates = row_data["intermediates"]
        total = intermediates[-1][0] + 1  # total denoising steps

        # Column 0: conditioning
        ax = axes[row_idx, 0]
        ax.imshow(row_data["conditioning"])
        if row_idx == 0:
            ax.set_title("Conditioning", fontsize=9, fontweight="bold")
        ax.set_ylabel(row_data.get("label", ""), fontsize=8, rotation=0, labelpad=60, va="center")
        ax.set_xticks([])
        ax.set_yticks([])

        # Intermediate columns
        for col_offset, idx in enumerate(indices):
            ax = axes[row_idx, 1 + col_offset]
            step_idx, img = intermediates[idx]
            ax.imshow(img)
            if row_idx == 0:
                t_frac = 1.0 - step_idx / max(total - 1, 1)
                ax.set_title(f"t={t_frac:.1f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Last column: target
        ax = axes[row_idx, -1]
        ax.imshow(row_data["target"])
        if row_idx == 0:
            ax.set_title("Target", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Progressive Denoising: From Noise to Surgical Prediction", fontsize=13, y=1.01)
    plt.tight_layout()

    # Render to numpy
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def main():
    parser = argparse.ArgumentParser(description="Progressive denoising visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/fig_progressive_denoising.png")
    parser.add_argument("--num-steps", type=int, default=20, help="Total inference steps")
    parser.add_argument(
        "--num-display", type=int, default=8, help="Number of intermediate steps to display"
    )
    parser.add_argument("--max-rows", type=int, default=4, help="Max number of sample rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.checkpoint, device)

    # Load test pairs — one per procedure for diversity
    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))

    # Try to get one per procedure
    procedures_seen = set()
    selected_files = []
    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        proc = "unknown"
        for p in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            if p in prefix:
                proc = p
                break
        if proc not in procedures_seen:
            procedures_seen.add(proc)
            selected_files.append((inp_file, proc))
            if len(selected_files) >= args.max_rows:
                break

    # If not enough diversity, fill with remaining
    if len(selected_files) < args.max_rows:
        for inp_file in input_files:
            if inp_file not in [s[0] for s in selected_files]:
                prefix = inp_file.stem.replace("_input", "")
                proc = "unknown"
                for p in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
                    if p in prefix:
                        proc = p
                        break
                selected_files.append((inp_file, proc))
                if len(selected_files) >= args.max_rows:
                    break

    print(f"Processing {len(selected_files)} samples with {args.num_steps} steps each")

    # Determine which steps to capture (all of them for flexibility)
    capture_steps = list(range(args.num_steps))

    rows = []
    for inp_file, proc in selected_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not target_file.exists() or not cond_file.exists():
            continue

        print(f"  [{len(rows) + 1}/{len(selected_files)}] {prefix} ({proc})")

        # Load images
        target_img = cv2.resize(cv2.imread(str(target_file)), (512, 512))
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_pil = Image.fromarray(target_rgb)

        conditioning = cv2.resize(cv2.imread(str(cond_file)), (512, 512))
        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        intermediates = run_with_intermediates(
            pipe,
            cond_pil,
            num_steps=args.num_steps,
            seed=args.seed,
            capture_steps=capture_steps,
        )

        rows.append(
            {
                "conditioning": cond_pil,
                "target": target_pil,
                "intermediates": intermediates,
                "label": proc.capitalize(),
            }
        )

    if not rows:
        print("No valid samples found")
        return

    # Generate grid
    grid = make_progressive_grid(rows, n_display_steps=args.num_display)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(str(out_path), quality=95)
    print(f"\nSaved: {out_path}")
    print(f"Grid size: {grid.shape[1]}x{grid.shape[0]} px")


if __name__ == "__main__":
    main()
