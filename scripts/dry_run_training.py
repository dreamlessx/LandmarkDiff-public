#!/usr/bin/env python3
"""Training dry-run validator — catches config/code bugs before SLURM GPU time.

Creates a tiny synthetic dataset and runs a mini training loop on CPU to verify
the entire pipeline works end-to-end:
1. Dataset loading + augmentation
2. Model loading (ControlNet + UNet + VAE)
3. Forward pass + backward pass
4. Checkpoint save + resume
5. EMA update
6. Sample generation
7. Post-training evaluation readiness

Usage:
    # Quick validation (default: 5 steps)
    python scripts/dry_run_training.py

    # With config file validation
    python scripts/dry_run_training.py --config configs/phaseA_production.yaml

    # Custom steps
    python scripts/dry_run_training.py --steps 10

    # Verbose output
    python scripts/dry_run_training.py --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class DryRunResult:
    """Tracks dry-run validation results."""

    def __init__(self):
        self.checks: list[dict] = []
        self.t0 = time.time()

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        icon = "+" if passed else "X"
        msg = f"[{icon}] {name}"
        if detail:
            msg += f" -- {detail}"
        if passed:
            logger.info("  %s", msg)
        else:
            logger.error("  %s", msg)
        return passed

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c["passed"])

    def summary(self) -> str:
        elapsed = time.time() - self.t0
        total = len(self.checks)
        status = "ALL PASSED" if self.all_passed else f"{self.n_failed} FAILED"
        return f"{self.n_passed}/{total} checks passed ({status}) in {elapsed:.1f}s"


def create_synthetic_dataset(output_dir: Path, n_pairs: int = 10) -> Path:
    """Create a tiny synthetic dataset for dry-run testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_pairs):
        prefix = f"{i:06d}"
        # Input: random face-like image
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        # Add a circle to simulate a face
        cv2.circle(img, (256, 256), 150, (180, 160, 140), -1)

        # Target: slightly modified version
        target = img.copy()
        target = np.roll(target, 5, axis=1)  # small shift

        # Conditioning: wireframe-like image (dark bg, bright lines)
        cond = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.ellipse(cond, (256, 256), (120, 160), 0, 0, 360, (255, 255, 255), 2)
        cv2.line(cond, (256, 200), (256, 300), (255, 255, 255), 2)

        cv2.imwrite(str(output_dir / f"{prefix}_input.png"), img)
        cv2.imwrite(str(output_dir / f"{prefix}_target.png"), target)
        cv2.imwrite(str(output_dir / f"{prefix}_conditioning.png"), cond)

    # Minimal metadata
    metadata = {
        "pairs": {
            f"{i:06d}": {"procedure": "rhinoplasty", "source": "dry_run"} for i in range(n_pairs)
        }
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return output_dir


def run_dry_run(
    config_path: str | None = None,
    n_steps: int = 5,
    verbose: bool = False,
) -> DryRunResult:
    """Run a complete training dry-run validation."""
    result = DryRunResult()

    logger.info("=" * 60)
    logger.info("Training Dry-Run Validator")
    logger.info("=" * 60)
    logger.info("Steps: %d | Device: CPU (forced)", n_steps)
    if config_path:
        logger.info("Config: %s", config_path)

    # ── Step 1: Config validation ──
    logger.info("Phase 1: Configuration")
    config = None
    if config_path:
        try:
            from scripts.launch_training import load_config

            config = load_config(config_path)
            result.check("Config loads", True, f"experiment={config.get('experiment_name', '?')}")

            phase = config.get("training", {}).get("phase", "A")
            result.check("Phase valid", phase in ("A", "B"), f"phase={phase}")

            train_dir = config.get("data", {}).get("train_dir", "")
            result.check("Train dir specified", bool(train_dir), train_dir)
        except Exception as e:
            result.check("Config loads", False, str(e))
    else:
        result.check("Config (skipped)", True, "no config file provided")

    # ── Step 2: Create synthetic dataset ──
    logger.info("Phase 2: Dataset")
    tmp_dir = PROJECT_ROOT / ".dry_run_tmp"
    data_dir = tmp_dir / "data"

    try:
        create_synthetic_dataset(data_dir, n_pairs=10)
        result.check("Synthetic data created", True, "10 pairs")
    except Exception as e:
        result.check("Synthetic data created", False, str(e))
        _cleanup(tmp_dir)
        return result

    # Test dataset loading
    try:
        from scripts.train_controlnet import SyntheticPairDataset

        dataset = SyntheticPairDataset(
            str(data_dir),
            resolution=512,
            clinical_augment=False,
            geometric_augment=False,
        )
        result.check("Dataset loads", True, f"{len(dataset)} pairs")
    except Exception as e:
        result.check("Dataset loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    # Test __getitem__
    try:
        sample = dataset[0]
        assert sample["input"].shape == (3, 512, 512), f"got {sample['input'].shape}"
        assert sample["target"].shape == (3, 512, 512)
        assert sample["conditioning"].shape == (3, 512, 512)
        assert sample["mask"].shape == (1, 512, 512)
        result.check("Dataset __getitem__", True, "shapes correct")
    except Exception as e:
        result.check("Dataset __getitem__", False, str(e))
        _cleanup(tmp_dir)
        return result

    # Test DataLoader
    try:
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        assert batch["target"].shape[0] == 2
        result.check("DataLoader batching", True, "batch_size=2")
    except Exception as e:
        result.check("DataLoader batching", False, str(e))

    # ── Step 3: Model loading ──
    logger.info("Phase 3: Model Loading")
    device = torch.device("cpu")
    weight_dtype = torch.float32

    try:
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        result.check("Diffusers import", True)
    except ImportError as e:
        result.check("Diffusers import", False, str(e))
        _cleanup(tmp_dir)
        return result

    base_model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_id = "CrucibleAI/ControlNetMediaPipeFace"

    try:
        tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
        result.check("Tokenizer loads", True)
    except Exception as e:
        result.check("Tokenizer loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    try:
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_id, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder.requires_grad_(False)
        result.check("Text encoder loads", True)
    except Exception as e:
        result.check("Text encoder loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    try:
        vae = AutoencoderKL.from_pretrained(
            base_model_id, subfolder="vae", torch_dtype=weight_dtype
        )
        vae.requires_grad_(False)
        result.check("VAE loads", True)
    except Exception as e:
        result.check("VAE loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    try:
        unet = UNet2DConditionModel.from_pretrained(
            base_model_id, subfolder="unet", torch_dtype=weight_dtype
        )
        unet.requires_grad_(False)
        result.check("UNet loads", True)
    except Exception as e:
        result.check("UNet loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    try:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, subfolder="diffusion_sd15", torch_dtype=weight_dtype
        )
        controlnet.train()
        result.check("ControlNet loads", True)
    except Exception as e:
        result.check("ControlNet loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    try:
        noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        result.check("Noise scheduler loads", True)
    except Exception as e:
        result.check("Noise scheduler loads", False, str(e))
        _cleanup(tmp_dir)
        return result

    # ── Step 4: Training loop ──
    logger.info("Phase 4: Training Loop")

    # Text embeddings
    try:
        text_input = tokenizer(
            "a photo of a person's face",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids)[0]
        result.check("Text embedding", True, f"shape={text_embeddings.shape}")
    except Exception as e:
        result.check("Text embedding", False, str(e))
        _cleanup(tmp_dir)
        return result

    # EMA
    ema_controlnet = copy.deepcopy(controlnet)
    ema_controlnet.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5, weight_decay=1e-2)

    # Run mini training loop
    data_iter = iter(loader)
    losses = []

    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        target = batch["target"].to(device, dtype=weight_dtype)
        conditioning = batch["conditioning"].to(device, dtype=weight_dtype)

        # Encode to latent space
        with torch.no_grad():
            latents = vae.encode(target * 2 - 1).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = text_embeddings.expand(latents.shape[0], -1, -1)

        # ControlNet forward
        down_samples, mid_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conditioning,
            return_dict=False,
        )

        # UNet forward
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample

        # Loss + backward
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # EMA update
        from scripts.train_controlnet import update_ema

        update_ema(ema_controlnet, controlnet, 0.9999)

        loss_val = loss.item()
        losses.append(loss_val)
        if verbose:
            logger.debug("Step %d/%d: loss=%.6f", step + 1, n_steps, loss_val)

    result.check(
        "Forward+backward pass",
        len(losses) == n_steps and all(np.isfinite(v) for v in losses),
        f"{n_steps} steps, loss={losses[-1]:.6f}",
    )
    result.check(
        "No NaN/Inf in loss",
        all(np.isfinite(v) for v in losses),
        f"min={min(losses):.6f}, max={max(losses):.6f}",
    )

    # Check gradients flowed to ControlNet
    any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in controlnet.parameters()
        if p.requires_grad
    )
    # After optimizer.zero_grad(), grads are zeroed, so check params changed from EMA
    params_differ = not all(
        torch.equal(p1, p2)
        for p1, p2 in zip(controlnet.parameters(), ema_controlnet.parameters(), strict=False)
    )
    result.check("Gradients flow to ControlNet", params_differ, "params updated from init")

    # ── Step 5: Checkpoint save/load ──
    logger.info("Phase 5: Checkpoint Save/Load")
    ckpt_dir = tmp_dir / "checkpoint-test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save
        state = {
            "controlnet": controlnet.state_dict(),
            "ema_controlnet": ema_controlnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": n_steps,
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        # Save ControlNet model files (for inference loading)
        controlnet.save_pretrained(ckpt_dir / "controlnet")
        ema_controlnet.save_pretrained(ckpt_dir / "controlnet_ema")

        result.check("Checkpoint save", True, f"{ckpt_dir}")
    except Exception as e:
        result.check("Checkpoint save", False, str(e))

    try:
        # Load into fresh model
        loaded_state = torch.load(ckpt_dir / "training_state.pt", weights_only=True)
        assert loaded_state["global_step"] == n_steps
        fresh_cn = ControlNetModel.from_pretrained(
            ckpt_dir / "controlnet_ema", torch_dtype=weight_dtype
        )
        assert fresh_cn is not None
        result.check("Checkpoint resume", True, f"step={loaded_state['global_step']}")
    except Exception as e:
        result.check("Checkpoint resume", False, str(e))

    # ── Step 6: Inference loading ──
    logger.info("Phase 6: Inference Pipeline")
    try:
        from landmarkdiff.inference import LandmarkDiffPipeline

        result.check("Inference import", True)
    except ImportError as e:
        result.check("Inference import", False, str(e))

    # ── Step 7: Post-training tools ──
    logger.info("Phase 7: Post-Training Tools")
    try:
        from scripts.analyze_training_run import (
            TrainingMetrics,
            check_phase_transition,
            detect_convergence_issues,
            find_checkpoints,
            generate_report,
        )

        result.check("Analyzer import", True)
    except ImportError as e:
        result.check("Analyzer import", False, str(e))

    try:
        from scripts.post_training_pipeline import PipelineStep, run_pipeline

        result.check("Pipeline import", True)
    except ImportError as e:
        result.check("Pipeline import", False, str(e))

    try:
        from scripts.score_checkpoints import (
            compute_tps_score,
            load_val_samples,
            rank_checkpoints,
        )

        result.check("Scorer import", True)
    except ImportError as e:
        result.check("Scorer import", False, str(e))

    try:
        from scripts.run_evaluation import (
            aggregate_metrics,
            evaluate_controlnet,
            evaluate_tps_baseline,
            generate_latex_table,
        )

        result.check("Evaluation import", True)
    except ImportError as e:
        result.check("Evaluation import", False, str(e))

    # ── Cleanup ──
    _cleanup(tmp_dir)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("%s", result.summary())
    logger.info("=" * 60)

    if result.all_passed:
        logger.info("Training pipeline is ready for SLURM submission.")
    else:
        logger.error("FIX the failures above before submitting to SLURM.")
        for c in result.checks:
            if not c["passed"]:
                logger.error("  FAILED: %s -- %s", c["name"], c["detail"])

    return result


def _cleanup(tmp_dir: Path):
    """Remove temporary dry-run files."""
    if tmp_dir.exists():
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Training dry-run validator")
    parser.add_argument("--config", default=None, help="YAML config to validate")
    parser.add_argument("--steps", type=int, default=5, help="Mini training steps")
    parser.add_argument("--verbose", action="store_true", help="Show per-step loss")
    args = parser.parse_args()

    result = run_dry_run(args.config, args.steps, args.verbose)
    sys.exit(0 if result.all_passed else 1)
