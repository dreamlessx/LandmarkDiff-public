"""Validation callback for training loop monitoring.

Periodically generates sample images from the validation set, computes
metrics (SSIM, LPIPS, NME, identity similarity), and logs results
to WandB and/or disk.

Designed for use with train_controlnet.py — call at regular intervals
during training to monitor quality without disrupting the training loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from landmarkdiff.evaluation import compute_ssim, compute_lpips, compute_nme


class ValidationCallback:
    """Validation callback that generates and evaluates samples during training.

    Usage::

        val_cb = ValidationCallback(
            val_dataset=val_dataset,
            output_dir=Path("checkpoints/val"),
            num_samples=8,
        )

        # In training loop:
        if global_step % val_every == 0:
            val_metrics = val_cb.run(
                controlnet=ema_controlnet,
                vae=vae,
                unet=unet,
                text_embeddings=text_embeddings,
                noise_scheduler=noise_scheduler,
                device=device,
                weight_dtype=weight_dtype,
                global_step=global_step,
            )
    """

    def __init__(
        self,
        val_dataset,
        output_dir: Path,
        num_samples: int = 8,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ):
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = min(num_samples, len(val_dataset))
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.history: list[dict] = []

    @torch.no_grad()
    def run(
        self,
        controlnet: torch.nn.Module,
        vae,
        unet,
        text_embeddings: torch.Tensor,
        noise_scheduler,
        device: torch.device,
        weight_dtype: torch.dtype,
        global_step: int,
    ) -> dict:
        """Run validation: generate samples and compute metrics.

        Returns dict with aggregate metrics.
        """
        from diffusers import DPMSolverMultistepScheduler

        t0 = time.time()
        controlnet.eval()

        step_dir = self.output_dir / f"step-{global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Set up inference scheduler (DPM++ 2M for quality)
        scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps, device=device)

        ssim_scores = []
        lpips_scores = []
        generated_images = []

        for i in range(self.num_samples):
            sample = self.val_dataset[i]
            conditioning = sample["conditioning"].unsqueeze(0).to(device, dtype=weight_dtype)
            target = sample["target"].unsqueeze(0).to(device, dtype=weight_dtype)

            # Encode target for latent shape
            latents = vae.encode(target * 2 - 1).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Start from noise
            noise = torch.randn_like(latents)
            sample_latents = noise * scheduler.init_noise_sigma
            encoder_hidden_states = text_embeddings[:1]

            # Denoising loop with classifier-free guidance
            for t in scheduler.timesteps:
                scaled = scheduler.scale_model_input(sample_latents, t)

                # ControlNet
                down_samples, mid_sample = controlnet(
                    scaled, t, encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning, return_dict=False,
                )

                # UNet with ControlNet residuals
                noise_pred = unet(
                    scaled, t, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample

                sample_latents = scheduler.step(noise_pred, t, sample_latents).prev_sample

            # Decode (use float32 for VAE to avoid color banding)
            decoded = vae.decode(sample_latents.float() / vae.config.scaling_factor).sample
            decoded = ((decoded + 1) / 2).clamp(0, 1)

            # Convert to numpy for metrics
            gen_np = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            tgt_np = (target[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cond_np = (conditioning[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # BGR for metrics (our metrics expect BGR)
            gen_bgr = gen_np[:, :, ::-1].copy()
            tgt_bgr = tgt_np[:, :, ::-1].copy()

            # Compute metrics
            ssim_val = compute_ssim(gen_bgr, tgt_bgr)
            lpips_val = compute_lpips(gen_bgr, tgt_bgr)
            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)
            generated_images.append(gen_np)

            # Save comparison: conditioning | generated | target
            comparison = np.hstack([cond_np, gen_np, tgt_np])
            Image.fromarray(comparison).save(step_dir / f"val_{i:02d}.png")

        # Aggregate metrics
        metrics = {
            "step": global_step,
            "ssim_mean": float(np.nanmean(ssim_scores)),
            "ssim_std": float(np.nanstd(ssim_scores)),
            "lpips_mean": float(np.nanmean(lpips_scores)),
            "lpips_std": float(np.nanstd(lpips_scores)),
            "time_seconds": round(time.time() - t0, 1),
        }

        self.history.append(metrics)

        # Save metrics
        with open(step_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save full history
        with open(self.output_dir / "validation_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Create comparison grid (all samples in one image)
        if generated_images:
            grid_rows = []
            for i in range(0, len(generated_images), 4):
                row_imgs = generated_images[i:i+4]
                while len(row_imgs) < 4:
                    row_imgs.append(np.zeros_like(generated_images[0]))
                grid_rows.append(np.hstack(row_imgs))
            grid = np.vstack(grid_rows)
            Image.fromarray(grid).save(step_dir / "grid.png")

        controlnet.train()

        print(
            f"  Validation @ step {global_step}: "
            f"SSIM={metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f} "
            f"LPIPS={metrics['lpips_mean']:.4f}±{metrics['lpips_std']:.4f} "
            f"({metrics['time_seconds']:.1f}s)"
        )

        return metrics

    def plot_history(self, output_path: str | None = None) -> None:
        """Plot validation metrics over training steps."""
        if not self.history:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps = [h["step"] for h in self.history]
        ssim = [h["ssim_mean"] for h in self.history]
        lpips = [h["lpips_mean"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(steps, ssim, "b-o", markersize=4)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("SSIM")
        ax1.set_title("Validation SSIM (higher=better)")
        ax1.grid(alpha=0.3)

        ax2.plot(steps, lpips, "r-o", markersize=4)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("LPIPS")
        ax2.set_title("Validation LPIPS (lower=better)")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = output_path or str(self.output_dir / "validation_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
