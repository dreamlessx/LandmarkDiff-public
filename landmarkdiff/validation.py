"""Validation callback for training loop monitoring.

Periodically generates sample images from the validation set, computes
metrics (SSIM, LPIPS, NME, identity similarity), and logs results
to WandB and/or disk.

Designed for use with train_controlnet.py -- call at regular intervals
during training to monitor quality without disrupting the training loop.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from landmarkdiff.evaluation import compute_lpips, compute_ssim

logger = logging.getLogger(__name__)


class ValidationCallback:
    """Validation callback that generates and evaluates samples during training.

    Usage::

        val_cb = ValidationCallback(
            val_dataset=val_dataset,
            output_dir=Path("checkpoints/val"),
            num_samples=8,
            samples_per_procedure=2,
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
        samples_per_procedure: int = 2,
    ):
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = min(num_samples, len(val_dataset))
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.samples_per_procedure = samples_per_procedure
        self.history: list[dict] = []

        # Pre-build per-procedure index map for stratified sampling
        self._procedure_indices = self._build_procedure_map()

    def _build_procedure_map(self) -> dict[str, list[int]]:
        """Build a mapping of procedure name to dataset indices."""
        from collections import defaultdict

        proc_indices: dict[str, list[int]] = defaultdict(list)
        ds = self.val_dataset

        if hasattr(ds, "_sample_procedures") and ds._sample_procedures:
            for idx, pair_path in enumerate(ds.pairs):
                prefix = pair_path.stem.replace("_input", "")
                proc = ds._sample_procedures.get(prefix, "unknown")
                proc_indices[proc].append(idx)
        elif hasattr(ds, "get_procedure"):
            for idx in range(len(ds)):
                proc = ds.get_procedure(idx)
                proc_indices[proc].append(idx)

        # Drop "unknown" if we have labeled procedures
        known = {k: v for k, v in proc_indices.items() if k != "unknown"}
        return dict(known) if known else dict(proc_indices)

    def _select_per_procedure_indices(self) -> list[tuple[int, str]]:
        """Select sample indices ensuring each procedure is represented.

        Returns list of (dataset_index, procedure_name) tuples.
        Falls back to first N sequential indices when no procedure metadata
        is available.
        """
        if not self._procedure_indices:
            return [(i, "unknown") for i in range(self.num_samples)]

        selected: list[tuple[int, str]] = []
        for proc, indices in sorted(self._procedure_indices.items()):
            for idx in indices[: self.samples_per_procedure]:
                selected.append((idx, proc))
        return selected

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

        Returns dict with aggregate and per-procedure metrics.
        """
        from diffusers import DDIMScheduler

        t0 = time.time()
        controlnet.eval()

        step_dir = self.output_dir / f"step-{global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Set up inference scheduler (DDIM for robustness during validation)
        scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps, device=device)

        ssim_scores = []
        lpips_scores = []
        generated_images = []

        # Per-procedure metric accumulators
        proc_ssim: dict[str, list[float]] = {}
        proc_lpips: dict[str, list[float]] = {}

        # Use per-procedure selection instead of sequential indices
        per_proc = self._select_per_procedure_indices()

        for sample_num, (idx, proc) in enumerate(per_proc):
            sample = self.val_dataset[idx]
            conditioning = sample["conditioning"].unsqueeze(0).to(device, dtype=weight_dtype)
            target = sample["target"].unsqueeze(0).to(device, dtype=weight_dtype)

            # Encode target for latent shape (VAE needs float32)
            latents = vae.encode((target * 2 - 1).float()).latent_dist.sample()
            latents = (latents * vae.config.scaling_factor).to(weight_dtype)

            # Start from noise
            noise = torch.randn_like(latents)
            sample_latents = noise * scheduler.init_noise_sigma
            encoder_hidden_states = text_embeddings[:1]

            # Denoising loop with autocast to handle BF16/FP32 dtype
            # mismatches in timestep embeddings
            with torch.autocast("cuda", dtype=weight_dtype):
                for t in scheduler.timesteps:
                    scaled = scheduler.scale_model_input(sample_latents, t)

                    # ControlNet
                    down_samples, mid_sample = controlnet(
                        scaled,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=conditioning,
                        return_dict=False,
                    )

                    # UNet with ControlNet residuals
                    noise_pred = unet(
                        scaled,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_samples,
                        mid_block_additional_residual=mid_sample,
                    ).sample

                    sample_latents = scheduler.step(
                        noise_pred,
                        t,
                        sample_latents,
                    ).prev_sample

            # Decode -- cast VAE to float32 temporarily to avoid color banding
            # and prevent dtype mismatch (latents float32 vs VAE weights bf16)
            vae_dtype = next(vae.parameters()).dtype
            vae.to(torch.float32)
            decoded = vae.decode(sample_latents.float() / vae.config.scaling_factor).sample
            vae.to(vae_dtype)
            decoded = ((decoded + 1) / 2).clamp(0, 1)

            # Convert to numpy for metrics
            gen_np = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            tgt_np = (target[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cond_np = (conditioning[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )

            # BGR for metrics (our metrics expect BGR)
            gen_bgr = gen_np[:, :, ::-1].copy()
            tgt_bgr = tgt_np[:, :, ::-1].copy()

            # Compute metrics
            ssim_val = compute_ssim(gen_bgr, tgt_bgr)
            lpips_val = compute_lpips(gen_bgr, tgt_bgr)
            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)
            generated_images.append(gen_np)

            # Accumulate per-procedure metrics
            proc_ssim.setdefault(proc, []).append(ssim_val)
            proc_lpips.setdefault(proc, []).append(lpips_val)

            # Save comparison: conditioning | generated | target
            proc_tag = proc.replace(" ", "_")
            comparison = np.hstack([cond_np, gen_np, tgt_np])
            Image.fromarray(comparison).save(step_dir / f"val_{sample_num:02d}_{proc_tag}.png")

        # Aggregate metrics
        metrics: dict = {
            "step": global_step,
            "ssim_mean": float(np.nanmean(ssim_scores)),
            "ssim_std": float(np.nanstd(ssim_scores)),
            "lpips_mean": float(np.nanmean(lpips_scores)),
            "lpips_std": float(np.nanstd(lpips_scores)),
            "time_seconds": round(time.time() - t0, 1),
        }

        # Per-procedure breakdown
        per_procedure: dict[str, dict] = {}
        for proc in sorted(proc_ssim.keys()):
            per_procedure[proc] = {
                "ssim_mean": float(np.nanmean(proc_ssim[proc])),
                "lpips_mean": float(np.nanmean(proc_lpips[proc])),
                "n_samples": len(proc_ssim[proc]),
            }
        metrics["per_procedure"] = per_procedure

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
                row_imgs = generated_images[i : i + 4]
                while len(row_imgs) < 4:
                    row_imgs.append(np.zeros_like(generated_images[0]))
                grid_rows.append(np.hstack(row_imgs))
            grid = np.vstack(grid_rows)
            Image.fromarray(grid).save(step_dir / "grid.png")

        controlnet.train()

        # Log summary with per-procedure breakdown
        proc_summary = " | ".join(
            f"{p}: SSIM={v['ssim_mean']:.3f}" for p, v in sorted(per_procedure.items())
        )
        logger.info(
            "  Validation @ step %d: SSIM=%.4f+/-%.4f LPIPS=%.4f+/-%.4f (%.1fs)",
            global_step,
            metrics["ssim_mean"],
            metrics["ssim_std"],
            metrics["lpips_mean"],
            metrics["lpips_std"],
            metrics["time_seconds"],
        )
        if proc_summary:
            logger.info("    Per-procedure: %s", proc_summary)

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


# ---------------------------------------------------------------------------
# Landmark file format validation
# ---------------------------------------------------------------------------

MEDIAPIPE_LANDMARK_COUNT = 478
DLIB_LANDMARK_COUNT = 68
VALID_LANDMARK_COUNTS = {MEDIAPIPE_LANDMARK_COUNT, DLIB_LANDMARK_COUNT}


@dataclass
class LandmarkValidationResult:
    """Structured result from validate_landmarks()."""

    valid: bool
    landmark_count: int | None = None
    dimensions: int | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"[{status}] Landmark file validation"]
        if self.landmark_count is not None:
            lines.append(f"  Count: {self.landmark_count}")
        if self.dimensions is not None:
            lines.append(f"  Dimensions: {self.dimensions}D")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_landmarks(
    path,
    expected_count: int = MEDIAPIPE_LANDMARK_COUNT,
    min_confidence: float = 0.0,
    pixel_coords: bool = False,
    image_size: int = 512,
) -> LandmarkValidationResult:
    """Validate a landmark file (JSON or CSV) before processing."""
    path = Path(path)
    errors, warnings = [], []

    if not path.exists():
        return LandmarkValidationResult(valid=False, errors=[f"File not found: {path}"])
    if not path.is_file():
        return LandmarkValidationResult(valid=False, errors=[f"Not a file: {path}"])

    suffix = path.suffix.lower()
    if suffix not in (".json", ".csv"):
        return LandmarkValidationResult(
            valid=False,
            errors=[f"Unsupported format '{suffix}'. Expected .json or .csv"],
        )

    try:
        if suffix == ".json":
            coords, confidences = _parse_landmark_json(path)
        else:
            coords, confidences = _parse_landmark_csv(path)
    except ValueError as exc:
        return LandmarkValidationResult(valid=False, errors=[str(exc)])

    landmark_count = len(coords)
    dimensions = len(coords[0]) if coords else 0

    if expected_count is not None and landmark_count != expected_count:
        errors.append(f"Expected {expected_count} landmarks, got {landmark_count}.")
    if dimensions not in (2, 3):
        errors.append(
            f"Each landmark must have 2 or 3 coordinates, got {dimensions}."
        )

    upper = float(image_size) if pixel_coords else 1.0
    nan_idx, inf_idx, oob_idx = [], [], []
    for i, lm in enumerate(coords):
        for v in lm:
            if math.isnan(v):
                nan_idx.append(i)
                break
            if math.isinf(v):
                inf_idx.append(i)
                break
        else:
            if any(v < 0.0 or v > upper for v in lm):
                oob_idx.append(i)

    if nan_idx:
        errors.append(f"{len(nan_idx)} landmark(s) contain NaN: indices {nan_idx[:5]}")
    if inf_idx:
        errors.append(f"{len(inf_idx)} landmark(s) contain Inf: indices {inf_idx[:5]}")
    if oob_idx:
        warnings.append(
            f"{len(oob_idx)} landmark(s) out of bounds: indices {oob_idx[:5]}"
        )

    if confidences and min_confidence > 0:
        low = [i for i, c in enumerate(confidences) if c < min_confidence]
        if low:
            warnings.append(
                f"{len(low)} landmark(s) below confidence {min_confidence}"
            )

    return LandmarkValidationResult(
        valid=len(errors) == 0,
        landmark_count=landmark_count,
        dimensions=dimensions,
        errors=errors,
        warnings=warnings,
    )


def _parse_landmark_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if isinstance(data, list):
        coords, confidences = data, []
    elif isinstance(data, dict):
        if "landmarks" not in data:
            raise ValueError("JSON must have a 'landmarks' key or be a bare list")
        coords = data["landmarks"]
        confidences = data.get("confidence", [])
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data).__name__}")
    parsed = []
    for i, lm in enumerate(coords):
        if not isinstance(lm, (list, tuple)) or len(lm) < 2:
            raise ValueError(f"Landmark {i} must be a list of 2+ numbers")
        try:
            parsed.append([float(v) for v in lm])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Landmark {i} non-numeric: {exc}") from exc
    return parsed, [float(c) for c in confidences]


def _parse_landmark_csv(path):
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if not rows:
        raise ValueError("CSV file is empty")
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    parsed = []
    for i, row in enumerate(rows[start:], start=start):
        if not row:
            continue
        if len(row) < 2:
            raise ValueError(f"Row {i} has fewer than 2 columns")
        try:
            parsed.append([float(v) for v in row[:3]])
        except ValueError as exc:
            raise ValueError(f"Row {i} non-numeric: {exc}") from exc
    if not parsed:
        raise ValueError("CSV contains no data rows")
    return parsed, []
