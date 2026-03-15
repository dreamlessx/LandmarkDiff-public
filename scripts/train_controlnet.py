"""ControlNet fine-tuning training loop.

Phase A: Diffusion loss only (synthetic TPS data).
Phase B: Full 4-term loss (FEM/clinical data).

Implements all spec safeguards:
- BF16 only (never FP16)
- VAE frozen
- EMA decay 0.9999
- GroupNorm (not BatchNorm)
- Resume from checkpoint
- WandB logging (offline mode for HPC)
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# DDP support — activated automatically by torchrun
_DDP_ENABLED = "RANK" in os.environ
_RANK = int(os.environ.get("RANK", 0))
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_IS_MAIN = _RANK == 0

# Optional imports with graceful fallback
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from landmarkdiff.synthetic.augmentation import apply_clinical_augmentation

    HAS_CLINICAL_AUG = True
except ImportError:
    HAS_CLINICAL_AUG = False

try:
    from landmarkdiff.augmentation import AugmentationConfig, augment_training_sample

    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False

try:
    from scripts.training_resilience import (
        GradientWatchdog,
        SlurmSignalHandler,
        create_emergency_save_fn,
    )

    HAS_RESILIENCE = True
except ImportError:
    HAS_RESILIENCE = False

try:
    from scripts.experiment_lineage import LineageDB

    HAS_LINEAGE = True
except ImportError:
    HAS_LINEAGE = False

logger = logging.getLogger(__name__)


class SyntheticPairDataset(Dataset):
    """Load pre-generated synthetic training pairs from disk.

    Applies online clinical degradation augmentation to target images
    (simulating clinical photo conditions) while keeping conditioning intact.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        clinical_augment: bool = False,
        geometric_augment: bool = True,
        augmentation_config: dict | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        # Clinical augmentation for Phase B (real clinical photos).
        # Disabled by default for Phase A (synthetic data with mesh conditioning).
        self.clinical_augment = clinical_augment and HAS_CLINICAL_AUG
        # Geometric + photometric augmentation from augmentation.py
        self.geometric_augment = geometric_augment and HAS_AUGMENTATION

        # Find all pairs by looking for *_input.png files
        self.pairs = sorted(self.data_dir.glob("*_input.png"))
        if not self.pairs:
            raise FileNotFoundError(f"No training pairs found in {data_dir}")

        # Load per-sample metadata for curriculum learning
        self._sample_procedures = {}
        self._sample_difficulties = {}
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            try:
                import json as _json

                with open(meta_path) as f:
                    meta = _json.load(f)
                pairs_meta = meta.get("pairs", {})
                for p in self.pairs:
                    prefix = p.stem.replace("_input", "")
                    info = pairs_meta.get(prefix, {})
                    self._sample_procedures[prefix] = info.get("procedure", "unknown")
            except Exception:
                pass

        if self.clinical_augment:
            self._rng = np.random.default_rng()

        if self.geometric_augment:
            aug_kwargs = augmentation_config or {}
            self._aug_config = AugmentationConfig(**aug_kwargs)
            self._aug_rng = np.random.default_rng()

    def get_procedure(self, idx: int) -> str:
        """Get the procedure type for a sample (for curriculum weighting)."""
        prefix = self.pairs[idx].stem.replace("_input", "")
        return self._sample_procedures.get(prefix, "unknown")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        prefix = self.pairs[idx].stem.replace("_input", "")

        # Load all images as numpy uint8 BGR for augmentation pipeline
        input_bgr = self._load_bgr(self.data_dir / f"{prefix}_input.png")
        target_bgr = self._load_bgr(self.data_dir / f"{prefix}_target.png")
        cond_bgr = self._load_bgr(self.data_dir / f"{prefix}_conditioning.png")

        mask_path = self.data_dir / f"{prefix}_mask.png"
        if mask_path.exists():
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_arr is not None:
                mask_arr = cv2.resize(mask_arr, (self.resolution, self.resolution))
                mask_arr = mask_arr.astype(np.float32) / 255.0
            else:
                mask_arr = np.ones((self.resolution, self.resolution), dtype=np.float32)
        else:
            mask_arr = np.ones((self.resolution, self.resolution), dtype=np.float32)

        # Apply geometric + photometric augmentation (preserves landmark correspondence)
        if self.geometric_augment:
            aug_result = augment_training_sample(
                input_image=input_bgr,
                target_image=target_bgr,
                conditioning=cond_bgr,
                mask=mask_arr,
                config=self._aug_config,
                rng=self._aug_rng,
            )
            input_bgr = aug_result["input_image"]
            target_bgr = aug_result["target_image"]
            cond_bgr = aug_result["conditioning"]
            mask_arr = aug_result["mask"]

        # Apply clinical augmentation to target only (domain gap closure)
        if self.clinical_augment and np.random.random() < 0.5:
            target_bgr = apply_clinical_augmentation(target_bgr, rng=self._rng)

        # Convert BGR→RGB and normalize to [0, 1] tensors
        input_img = self._bgr_to_tensor(input_bgr)
        target_img = self._bgr_to_tensor(target_bgr)
        conditioning = self._bgr_to_tensor(cond_bgr)

        # Mask to tensor
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)

        return {
            "input": input_img,
            "target": target_img,
            "conditioning": conditioning,
            "mask": mask,
            "idx": idx,
        }

    def _load_bgr(self, path: Path) -> np.ndarray:
        """Load image as BGR uint8, resized to self.resolution."""
        img = cv2.imread(str(path))
        if img is None:
            # Fallback: create blank image
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        return cv2.resize(img, (self.resolution, self.resolution))

    @staticmethod
    def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 numpy array to RGB [0,1] tensor (C, H, W)."""
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999):
    """Update EMA model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def _generate_samples(
    ema_controlnet: torch.nn.Module,
    vae,
    unet,
    text_embeddings: torch.Tensor,
    noise_scheduler,
    dataset: Dataset,
    device: torch.device,
    weight_dtype: torch.dtype,
    output_dir: Path,
    global_step: int,
    num_samples: int = 4,
) -> None:
    """Generate sample images using EMA weights for visual monitoring."""
    from diffusers import UniPCMultistepScheduler

    sample_dir = output_dir / "samples" / f"step-{global_step}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    ema_controlnet.eval()

    n = min(num_samples, len(dataset))
    if n == 0:
        return
    for i in range(n):
        sample = dataset[i]
        conditioning = sample["conditioning"].unsqueeze(0).to(device, dtype=weight_dtype)
        target = sample["target"].unsqueeze(0).to(device, dtype=weight_dtype)

        # Encode target to get shape reference
        latents = vae.encode(target * 2 - 1).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Start from pure noise
        noise = torch.randn_like(latents)
        scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)
        scheduler.set_timesteps(20, device=device)

        sample_latents = noise * scheduler.init_noise_sigma
        encoder_hidden_states = text_embeddings[:1]

        for t in scheduler.timesteps:
            scaled = scheduler.scale_model_input(sample_latents, t)

            down_samples, mid_sample = ema_controlnet(
                scaled,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning,
                return_dict=False,
            )
            noise_pred = unet(
                scaled,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample

            sample_latents = scheduler.step(noise_pred, t, sample_latents).prev_sample

        # Decode
        decoded = vae.decode(sample_latents / vae.config.scaling_factor).sample
        decoded = ((decoded + 1) / 2).clamp(0, 1)

        # Save as PNG (cast to float32 — BF16 can't convert to numpy)
        img = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        tgt = (target[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cond = (conditioning[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Side-by-side: conditioning | generated | target
        comparison = np.hstack([cond, img, tgt])
        Image.fromarray(comparison).save(sample_dir / f"sample_{i}.png")

    logger.info("Samples saved: %s", sample_dir)
    ema_controlnet.train()


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    controlnet_id: str = "CrucibleAI/ControlNetMediaPipeFace",
    controlnet_subfolder: str = "diffusion_sd15",
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    learning_rate: float = 1e-5,
    train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_steps: int = 10000,
    checkpoint_every: int = 5000,
    log_every: int = 50,
    sample_every: int = 1000,
    ema_decay: float = 0.9999,
    phase: str = "A",
    resume_from_checkpoint: str | None = None,
    resume_phaseA: str | None = None,
    clinical_augment: bool = False,
    geometric_augment: bool = True,
    seed: int = 42,
    wandb_project: str = "landmarkdiff",
    wandb_dir: str | None = None,
) -> None:
    """Main training loop."""

    # DDP initialization
    if _DDP_ENABLED:
        import datetime

        import torch.distributed as dist

        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=1800),
        )
        device = torch.device(f"cuda:{_LOCAL_RANK}")
        torch.cuda.set_device(device)
    else:
        device = get_device()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    import random

    random.seed(seed + _RANK)  # offset per rank for data diversity
    np.random.seed(seed + _RANK)
    torch.manual_seed(seed + _RANK)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + _RANK)

    # Determine dtype — BF16 on CUDA, FP32 on MPS/CPU
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if _IS_MAIN:
        logger.info(
            "Device: %s | Dtype: %s | Phase: %s%s",
            device,
            weight_dtype,
            phase,
            f" | DDP: {_WORLD_SIZE} GPUs" if _DDP_ENABLED else "",
        )

    # ─── Load models ───
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    # Use local_files_only when HF_HUB_OFFLINE is set or token is unavailable
    _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    _load_kw: dict = {"local_files_only": True} if _local_only else {}
    logger.info("Loading models...%s", " (offline mode)" if _local_only else "")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer", **_load_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, subfolder="text_encoder", **_load_kw
    )
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", **_load_kw)
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", **_load_kw)
    # Load ControlNet: from Phase A checkpoint if doing Phase B, else from pretrained
    if resume_phaseA and phase == "B":
        phaseA_path = Path(resume_phaseA)
        if (phaseA_path / "controlnet_ema").exists():
            phaseA_path = phaseA_path / "controlnet_ema"
        logger.info("Initializing Phase B from Phase A checkpoint: %s", phaseA_path)
        controlnet = ControlNetModel.from_pretrained(str(phaseA_path))
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id, subfolder=controlnet_subfolder, **_load_kw
        )
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model_id, subfolder="scheduler", **_load_kw
    )

    # ─── Freeze everything except ControlNet ───
    vae.requires_grad_(False)  # CRITICAL: gradient leak corrupts latent space
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Enable gradient checkpointing to save VRAM (~30% reduction)
    controlnet.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing()

    # Move to device
    # VAE stays in FP32 for decode quality and to avoid dtype mismatch
    # when computing Phase B image-level losses (identity, perceptual)
    vae.to(device, dtype=torch.float32)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    # ─── EMA ───
    ema_controlnet = copy.deepcopy(controlnet)
    ema_controlnet.requires_grad_(False)

    # ─── DDP wrapper ───
    if _DDP_ENABLED:
        from torch.nn.parallel import DistributedDataParallel as DDP

        controlnet = DDP(controlnet, device_ids=[_LOCAL_RANK])
        # Use the unwrapped module for parameter access
        controlnet_module = controlnet.module
    else:
        controlnet_module = controlnet

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(
        controlnet_module.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
    )

    # Cosine schedule with linear warmup — period based on optimizer steps
    total_optimizer_steps = num_train_steps // gradient_accumulation_steps
    warmup_steps = min(1000, total_optimizer_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ─── Data ───
    if _IS_MAIN:
        logger.info("Loading data from %s...", data_dir)
    dataset = SyntheticPairDataset(
        data_dir,
        resolution=512,
        clinical_augment=clinical_augment,
        geometric_augment=geometric_augment,
    )
    num_workers = min(8, len(dataset))

    # ─── Curriculum learning ───
    proc_curriculum = None
    try:
        from landmarkdiff.curriculum import ProcedureCurriculum

        proc_curriculum = ProcedureCurriculum(total_steps=num_train_steps)
        if _IS_MAIN:
            logger.info("Curriculum learning enabled (procedure-weighted sampling)")
    except ImportError:
        pass

    sampler = None
    if _DDP_ENABLED:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, num_replicas=_WORLD_SIZE, rank=_RANK, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    if _IS_MAIN:
        logger.info(
            "Dataset: %d pairs | Batch: %d | Accum: %d%s",
            len(dataset),
            train_batch_size,
            gradient_accumulation_steps,
            f" | Per-GPU batch: {train_batch_size}" if _DDP_ENABLED else "",
        )

    # ─── Text embeddings (constant — "a photo of a person's face") ───
    text_input = tokenizer(
        "a photo of a person's face",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        text_embeddings = text_embeddings.to(dtype=weight_dtype)

    # ─── WandB (main process only) ───
    if HAS_WANDB and _IS_MAIN:
        wandb.init(
            project=wandb_project,
            config={
                "phase": phase,
                "lr": learning_rate,
                "batch": train_batch_size,
                "accum": gradient_accumulation_steps,
                "steps": num_train_steps,
                "ema_decay": ema_decay,
                "device": str(device),
            },
            dir=wandb_dir,
            mode="offline",
        )

    # ─── Experiment Tracker (main process only) ───
    exp_tracker = None
    exp_id = None
    if _IS_MAIN:
        try:
            from landmarkdiff.experiment_tracker import ExperimentTracker

            exp_tracker = ExperimentTracker(str(out / "experiments"))
            exp_id = exp_tracker.start(
                name=f"phase{phase}_{Path(data_dir).name}",
                config={
                    "phase": phase,
                    "lr": learning_rate,
                    "batch": train_batch_size,
                    "accum": gradient_accumulation_steps,
                    "steps": num_train_steps,
                    "ema_decay": ema_decay,
                    "clinical_augment": clinical_augment,
                    "data_dir": data_dir,
                    "ddp": _DDP_ENABLED,
                    "world_size": _WORLD_SIZE,
                },
                tags=[f"phase{phase}", "ddp" if _DDP_ENABLED else "single_gpu"],
            )
        except Exception:
            pass  # experiment tracking is optional

    # ─── Resume ───
    global_step = 0
    if resume_from_checkpoint in ("latest", "auto"):
        # Auto-detect latest checkpoint by step number
        ckpts = sorted(
            out.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        ckpts = [c for c in ckpts if (c / "training_state.pt").exists()]
        if ckpts:
            resume_from_checkpoint = str(ckpts[-1])
            if _IS_MAIN:
                logger.info("Auto-detected checkpoint: %s", resume_from_checkpoint)
        else:
            resume_from_checkpoint = None
            if _IS_MAIN:
                logger.info("No checkpoints found, starting from scratch")

    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        if _IS_MAIN:
            logger.info("Resuming from %s", resume_from_checkpoint)
        state = torch.load(
            Path(resume_from_checkpoint) / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        controlnet_module.load_state_dict(state["controlnet"])
        ema_controlnet.load_state_dict(state["ema_controlnet"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        if _IS_MAIN:
            logger.info("Resumed at step %d", global_step)

    # ─── Validation callback ───
    val_callback = None
    try:
        from landmarkdiff.validation import ValidationCallback

        val_callback = ValidationCallback(
            val_dataset=dataset,
            output_dir=out / "validation",
            num_samples=min(8, len(dataset)),
        )
        logger.info("Validation callback enabled (%d samples)", val_callback.num_samples)
    except ImportError:
        logger.warning("Validation callback unavailable (missing dependencies)")

    # ─── Phase B auxiliary losses ───
    combined_loss = None
    loss_weights = None
    if phase == "B":
        from landmarkdiff.losses import CombinedLoss, LossWeights

        loss_weights = LossWeights()
        combined_loss = CombinedLoss(
            weights=loss_weights,
            phase="B",
            use_differentiable_arcface=True,
        )
        logger.info("Phase B: identity (PyTorch ArcFace) + perceptual (LPIPS) losses enabled")

    # ─── Curriculum sampling weights ───
    # Pre-compute per-sample procedure weights for curriculum-based sampling.
    # These get updated periodically during training.
    #
    # In DDP mode, we can't use WeightedRandomSampler (conflicts with
    # DistributedSampler), so we apply curriculum weights as per-sample loss
    # multipliers instead. This is mathematically equivalent in expectation.
    _curriculum_weights = None
    _curriculum_use_loss_weighting = _DDP_ENABLED  # DDP: loss weighting; single: sampler
    _curriculum_update_freq = max(1000, num_train_steps // 20)  # update ~20 times
    if proc_curriculum and dataset._sample_procedures:
        # Build initial weights (step 0)
        _curriculum_weights = torch.ones(len(dataset))
        for i in range(len(dataset)):
            proc = dataset.get_procedure(i)
            _curriculum_weights[i] = proc_curriculum.get_weight(0, proc)
        if _IS_MAIN:
            weights_summary = proc_curriculum.get_procedure_weights(0)
            logger.info("Curriculum weights (step 0): %s", weights_summary)
            if _DDP_ENABLED:
                logger.info("  (DDP: using per-sample loss weighting instead of sampler)")

    # ─── SLURM signal handler + gradient watchdog ───
    _signal_handler = None
    _grad_watchdog = None
    _global_step_ref = [global_step]  # mutable ref for emergency save closure

    if HAS_RESILIENCE and _IS_MAIN:
        _save_fn = create_emergency_save_fn(
            out,
            controlnet_module,
            ema_controlnet,
            optimizer,
            scheduler,
            _global_step_ref,
        )
        _signal_handler = SlurmSignalHandler(save_fn=_save_fn)
        _signal_handler.register()
        _grad_watchdog = GradientWatchdog()
        logger.info("SLURM signal handler + gradient watchdog enabled")

    # ─── Training loop ───
    logger.info("Starting training from step %d...", global_step)
    logger.info("Optimizer steps: %d | Warmup: %d", total_optimizer_steps, warmup_steps)
    _epoch = 0
    if sampler is not None:
        sampler.set_epoch(_epoch)
    data_iter = iter(dataloader)
    accumulation_loss = 0.0
    import time as _time

    _t0 = _time.time()

    while global_step < num_train_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            _epoch += 1
            if sampler is not None:
                sampler.set_epoch(_epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        target = batch["target"].to(device, dtype=weight_dtype)
        conditioning = batch["conditioning"].to(device, dtype=weight_dtype)

        # Encode target to latents
        with torch.no_grad():
            latents = vae.encode(target * 2 - 1).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Expand text embeddings to batch
        encoder_hidden_states = text_embeddings.expand(latents.shape[0], -1, -1)

        # ControlNet forward
        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conditioning,
            return_dict=False,
        )

        # UNet forward with ControlNet residuals
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # Loss (with optional curriculum weighting for DDP)
        if _curriculum_weights is not None and _curriculum_use_loss_weighting:
            # Per-sample MSE with curriculum weights as loss multipliers
            per_sample_mse = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean(
                dim=(1, 2, 3)
            )  # (B,)
            sample_indices = batch["idx"]
            w = _curriculum_weights[sample_indices].to(device)
            w = w / w.mean().clamp(min=0.1)  # normalize so mean weight ≈ 1
            diffusion_loss = (per_sample_mse * w).mean()
        else:
            diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())

        if phase == "B":
            # Phase B: add auxiliary image-level losses
            # Compute x0-prediction from noise prediction (Tweedie's formula)
            alpha_bar = noise_scheduler.alphas_cumprod.to(device)[timesteps]
            alpha_bar = alpha_bar.view(-1, 1, 1, 1)
            x0_pred = (
                noisy_latents.float() - (1 - alpha_bar).sqrt() * noise_pred.float()
            ) / alpha_bar.sqrt()

            # Decode to image space — NO torch.no_grad() here!
            # VAE params are frozen (requires_grad=False) so they won't be
            # updated, but gradients DO flow through the VAE decoder back to
            # x0_pred → noise_pred → ControlNet. This is critical for the
            # identity and perceptual losses to provide gradient signal.
            pred_image = vae.decode(x0_pred / vae.config.scaling_factor).sample
            pred_image_01 = ((pred_image + 1) / 2).clamp(0, 1)
            target_01 = target  # already in [0, 1]

            # Identity loss (differentiable PyTorch ArcFace)
            id_loss = combined_loss.identity_loss(pred_image_01, target_01)
            # Perceptual loss (LPIPS on non-surgical region)
            mask_b = batch["mask"].to(device, dtype=weight_dtype)
            if mask_b.dim() == 3:
                mask_b = mask_b.unsqueeze(1)
            perc_loss = combined_loss.perceptual_loss(pred_image_01, target_01, mask_b)

            loss = (
                diffusion_loss
                + loss_weights.identity * id_loss
                + loss_weights.perceptual * perc_loss
            )

            # Log Phase B loss components (main process only)
            if global_step % log_every == 0 and _IS_MAIN:
                if HAS_WANDB:
                    wandb.log(
                        {
                            "loss/diffusion": diffusion_loss.item(),
                            "loss/identity": id_loss.item(),
                            "loss/perceptual": perc_loss.item(),
                        },
                        step=global_step,
                    )
        else:
            loss = diffusion_loss

        loss_unscaled = loss.item()
        loss = loss / gradient_accumulation_steps

        # Skip gradient all-reduce on non-final accumulation steps (DDP perf)
        _is_accum_step = (global_step + 1) % gradient_accumulation_steps != 0
        if _DDP_ENABLED and _is_accum_step:
            with controlnet.no_sync():
                loss.backward()
        else:
            loss.backward()

        accumulation_loss += loss_unscaled

        # Gradient watchdog: check for NaN/Inf before stepping
        _skip_step = False
        if _grad_watchdog is not None:
            action = _grad_watchdog.check(
                controlnet_module.parameters(), loss_unscaled, global_step
            )
            if action == "skip":
                optimizer.zero_grad()
                _skip_step = True
            elif action == "alert" and _signal_handler is not None:
                # Severe instability — trigger emergency checkpoint
                logger.warning(
                    "Gradient alert at step %d: saving emergency checkpoint", global_step
                )
                _signal_handler.save_fn()

        # Broadcast skip decision so all ranks agree (prevents param divergence)
        if _DDP_ENABLED:
            import torch.distributed as dist

            _skip_t = torch.tensor([1 if _skip_step else 0], device=device)
            dist.broadcast(_skip_t, src=0)
            _skip_step = _skip_t.item() == 1

        # Step optimizer
        if not _skip_step and (global_step + 1) % gradient_accumulation_steps == 0:
            _last_grad_norm = torch.nn.utils.clip_grad_norm_(controlnet_module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # EMA update (use unwrapped module)
            update_ema(ema_controlnet, controlnet_module, ema_decay)

        global_step += 1
        _global_step_ref[0] = global_step

        # Check for SLURM signal (graceful exit) -- broadcast to all ranks
        _should_exit = False
        if _signal_handler is not None and _signal_handler.should_exit:
            _should_exit = True
        if _DDP_ENABLED:
            import torch.distributed as dist

            _exit_t = torch.tensor([1 if _should_exit else 0], device=device)
            dist.broadcast(_exit_t, src=0)
            _should_exit = _exit_t.item() == 1
        if _should_exit:
            if _IS_MAIN:
                logger.info("Graceful exit at step %d after SLURM signal", global_step)
            break

        # ─── Logging (main process only) ───
        if global_step % log_every == 0 and _IS_MAIN:
            avg_loss = accumulation_loss / log_every
            lr_current = scheduler.get_last_lr()[0]
            elapsed = _time.time() - _t0
            steps_per_sec = global_step / max(elapsed, 1)
            eta_h = (num_train_steps - global_step) / max(steps_per_sec, 0.01) / 3600
            grad_norm = _last_grad_norm if "_last_grad_norm" in dir() else 0.0
            logger.info(
                "Step %d/%d | Loss: %.6f | LR: %.2e | GradNorm: %.2f | %.1f it/s | ETA: %.1fh",
                global_step,
                num_train_steps,
                avg_loss,
                lr_current,
                grad_norm,
                steps_per_sec,
                eta_h,
            )

            if HAS_WANDB:
                _gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                log_dict = {
                    "loss": avg_loss,
                    "lr": lr_current,
                    "grad_norm": _gn,
                    "steps_per_sec": steps_per_sec,
                }
                wandb.log(log_dict, step=global_step)

            # Experiment tracker
            if exp_tracker and exp_id:
                exp_tracker.log_metric(exp_id, step=global_step, loss=avg_loss, lr=lr_current)

            # Curriculum update: periodically recompute weights
            if _curriculum_weights is not None and global_step % _curriculum_update_freq == 0:
                difficulty = proc_curriculum.get_difficulty(global_step)
                for i in range(len(dataset)):
                    proc = dataset.get_procedure(i)
                    _curriculum_weights[i] = proc_curriculum.get_weight(global_step, proc)

                if _curriculum_use_loss_weighting:
                    # DDP: weights are applied as loss multipliers (already updated)
                    # Broadcast from rank 0 to ensure consistency across ranks
                    if _DDP_ENABLED:
                        import torch.distributed as dist

                        dist.broadcast(_curriculum_weights, src=0)
                else:
                    # Single-GPU: rebuild dataloader with WeightedRandomSampler
                    from torch.utils.data import WeightedRandomSampler

                    _wsamp = WeightedRandomSampler(
                        _curriculum_weights, len(dataset), replacement=True
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=train_batch_size,
                        sampler=_wsamp,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                    )
                    data_iter = iter(dataloader)

                weights_s = proc_curriculum.get_procedure_weights(global_step)
                logger.info("Curriculum updated (difficulty=%.2f): %s", difficulty, weights_s)
                if HAS_WANDB:
                    wandb.log({"curriculum/difficulty": difficulty}, step=global_step)

        if global_step % log_every == 0:
            accumulation_loss = 0.0

        # ─── Sample generation + validation ───
        # All ranks check condition so they all reach the barrier.
        # Only rank 0 does the actual work; other ranks wait at barrier.
        if global_step % sample_every == 0 and global_step > 0:
            if _IS_MAIN:
                _generate_samples(
                    ema_controlnet,
                    vae,
                    unet,
                    text_embeddings,
                    noise_scheduler,
                    dataset,
                    device,
                    weight_dtype,
                    out,
                    global_step,
                )
                # Run validation callback (computes SSIM/LPIPS on generated samples)
                if val_callback is not None:
                    try:
                        val_metrics = val_callback.run(
                            ema_controlnet,
                            vae,
                            unet,
                            text_embeddings,
                            noise_scheduler,
                            device,
                            weight_dtype,
                            global_step,
                        )
                        if HAS_WANDB:
                            wandb.log(
                                {
                                    "val/ssim": val_metrics["ssim_mean"],
                                    "val/lpips": val_metrics["lpips_mean"],
                                },
                                step=global_step,
                            )
                    except Exception as val_err:
                        logger.warning("Validation failed at step %d: %s", global_step, val_err)
                        logger.warning("Continuing training...")

            if _DDP_ENABLED:
                import torch.distributed as dist

                dist.barrier()

        # ─── Checkpoint (save before validation to protect progress) ───
        # All ranks check condition; only rank 0 saves; barrier syncs.
        if global_step % checkpoint_every == 0 and global_step > 0:
            if _IS_MAIN:
                # Collect metrics for checkpoint metadata
                _ckpt_metrics = {"loss": accumulation_loss / max(log_every, 1)}
                if val_callback and val_callback.history:
                    _last_val = val_callback.history[-1]
                    _ckpt_metrics["val_ssim"] = _last_val.get("ssim_mean", 0)
                    _ckpt_metrics["val_lpips"] = _last_val.get("lpips_mean", 0)

                # Use CheckpointManager if available, otherwise fallback
                try:
                    from landmarkdiff.checkpoint_manager import CheckpointManager

                    if not hasattr(train, "_ckpt_manager"):
                        train._ckpt_manager = CheckpointManager(
                            output_dir=out,
                            keep_best=3,
                            keep_latest=5,
                            metric="loss",
                            lower_is_better=True,
                        )
                    ckpt_dir = train._ckpt_manager.save(
                        step=global_step,
                        controlnet=controlnet_module,
                        ema_controlnet=ema_controlnet,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=_ckpt_metrics,
                        phase=phase,
                    )
                    logger.info(
                        "Checkpoint saved: %s | %s", ckpt_dir, train._ckpt_manager.summary()
                    )
                except ImportError:
                    # Fallback: save without manager
                    ckpt_dir = out / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    ema_controlnet.save_pretrained(ckpt_dir / "controlnet_ema")
                    torch.save(
                        {
                            "controlnet": controlnet_module.state_dict(),
                            "ema_controlnet": ema_controlnet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        ckpt_dir / "training_state.pt",
                    )
                    logger.info("Checkpoint saved: %s", ckpt_dir)

            if _DDP_ENABLED:
                import torch.distributed as dist

                dist.barrier()

    # ─── Cleanup resilience handlers ───
    if _signal_handler is not None:
        _signal_handler.unregister()
    if _grad_watchdog is not None and _IS_MAIN:
        logger.info("Gradient watchdog: %s", _grad_watchdog.summary())

    # ─── Final save (main process only) ───
    if _IS_MAIN:
        final_dir = out / "final"
        final_dir.mkdir(exist_ok=True)
        ema_controlnet.save_pretrained(final_dir / "controlnet_ema")

        # Plot validation curves
        if val_callback is not None:
            val_callback.plot_history()
            logger.info("Validation curves saved to %s/validation/", out)

        logger.info("Training complete. Final model: %s", final_dir)

        # Record in experiment lineage
        if HAS_LINEAGE:
            try:
                db = LineageDB.load()
                config_path = ""
                # Find config path from args if available
                for cfg_candidate in [
                    f"configs/phase{phase}_production.yaml",
                    f"configs/phase{phase}.yaml",
                ]:
                    if (Path(__file__).resolve().parent.parent / cfg_candidate).exists():
                        config_path = cfg_candidate
                        break

                slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
                final_loss_val = accumulation_loss / max(log_every, 1)
                if config_path:
                    db.record_training(
                        config_path=config_path,
                        checkpoint_path=str(final_dir),
                        steps=global_step,
                        final_loss=final_loss_val,
                        slurm_job_id=slurm_job_id,
                    )
                    db.save()
                    logger.info("Lineage: training recorded (config=%s)", config_path)
            except Exception as e:
                logger.warning("Lineage recording failed: %s", e)

        # Finish experiment tracking
        if exp_tracker and exp_id:
            final_results = {"steps": global_step}
            if val_callback and val_callback.history:
                last = val_callback.history[-1]
                final_results.update(
                    {
                        "ssim": last.get("ssim_mean"),
                        "lpips": last.get("lpips_mean"),
                    }
                )
            exp_tracker.finish(exp_id, results=final_results)

        if HAS_WANDB:
            wandb.finish()

    # DDP cleanup
    if _DDP_ENABLED:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train LandmarkDiff ControlNet")
    parser.add_argument(
        "--config", default=None, help="YAML config file (overrides all other args)"
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--phase", default="A", choices=["A", "B"])
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint dir, 'latest'/'auto' for last, or None",
    )
    parser.add_argument(
        "--resume_phaseA", default=None, help="Phase A checkpoint dir to initialize Phase B from"
    )
    parser.add_argument(
        "--clinical_augment",
        action="store_true",
        help="Enable clinical degradation augmentation (Phase B)",
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable geometric/photometric augmentation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_dir", default=None)
    args = parser.parse_args()

    # If a YAML config is provided, use it to populate args
    if args.config:
        from landmarkdiff.config import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.config)
        # Map config fields to train() kwargs
        train_kwargs = {
            "data_dir": config.data.train_dir,
            "output_dir": config.output_dir,
            "learning_rate": config.training.learning_rate,
            "train_batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_train_steps": config.training.max_train_steps,
            "checkpoint_every": config.training.save_every_n_steps,
            "log_every": 50,
            "sample_every": config.training.validate_every_n_steps,
            "phase": config.training.phase,
            "ema_decay": config.model.ema_decay,
            "resume_from_checkpoint": config.training.resume_from_checkpoint,
            "resume_phaseA": None,
            "clinical_augment": config.training.phase == "B",
            "geometric_augment": True,
            "seed": config.training.seed,
            "wandb_dir": None,
        }
        # CLI args override config (if explicitly set)
        if args.data_dir:
            train_kwargs["data_dir"] = args.data_dir
        logger.info("Loaded config: %s", args.config)
        logger.info("  Experiment: %s", config.experiment_name)
        logger.info("  Phase: %s", config.training.phase)
        train(**train_kwargs)
    else:
        if args.data_dir is None:
            parser.error("--data_dir is required when --config is not provided")
        kwargs = vars(args)
        kwargs.pop("config", None)
        kwargs["geometric_augment"] = not kwargs.pop("no_augment", False)
        train(**kwargs)
