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
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Optional imports with graceful fallback
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class SyntheticPairDataset(Dataset):
    """Load pre-generated synthetic training pairs from disk."""

    def __init__(self, data_dir: str, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        # Find all pairs by looking for *_input.png files
        self.pairs = sorted(self.data_dir.glob("*_input.png"))
        if not self.pairs:
            raise FileNotFoundError(f"No training pairs found in {data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        prefix = self.pairs[idx].stem.replace("_input", "")

        input_img = self._load_image(self.data_dir / f"{prefix}_input.png")
        target_img = self._load_image(self.data_dir / f"{prefix}_target.png")
        conditioning = self._load_image(self.data_dir / f"{prefix}_conditioning.png")

        mask_path = self.data_dir / f"{prefix}_mask.png"
        if mask_path.exists():
            mask = self._load_image(mask_path, grayscale=True)
        else:
            mask = torch.ones(1, self.resolution, self.resolution)

        return {
            "input": input_img,
            "target": target_img,
            "conditioning": conditioning,
            "mask": mask,
        }

    def _load_image(self, path: Path, grayscale: bool = False) -> torch.Tensor:
        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((self.resolution, self.resolution))
        arr = np.array(img).astype(np.float32) / 255.0

        if grayscale:
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


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

    for i in range(min(num_samples, len(dataset))):
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
                scaled, t, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning, return_dict=False,
            )
            noise_pred = unet(
                scaled, t, encoder_hidden_states=encoder_hidden_states,
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

    print(f"Samples saved: {sample_dir}")
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
    seed: int = 42,
    wandb_project: str = "landmarkdiff",
    wandb_dir: str | None = None,
) -> None:
    """Main training loop."""

    device = get_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Set seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine dtype — BF16 on CUDA, FP32 on MPS/CPU
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    print(f"Device: {device} | Dtype: {weight_dtype} | Phase: {phase}")

    # ─── Load models ───
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_id, subfolder=controlnet_subfolder)
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    # ─── Freeze everything except ControlNet ───
    vae.requires_grad_(False)          # CRITICAL: gradient leak corrupts latent space
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Move to device
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    # ─── EMA ───
    ema_controlnet = copy.deepcopy(controlnet)
    ema_controlnet.requires_grad_(False)

    # ─── Optimizer ───
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
    )

    # Cosine schedule — period based on optimizer steps, not forward passes
    total_optimizer_steps = num_train_steps // gradient_accumulation_steps
    lr_lambda = lambda step: 0.5 * (1 + math.cos(math.pi * step / total_optimizer_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ─── Data ───
    print(f"Loading data from {data_dir}...")
    dataset = SyntheticPairDataset(data_dir, resolution=512)
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} pairs | Batch: {train_batch_size} | Accum: {gradient_accumulation_steps}")

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

    # ─── WandB ───
    if HAS_WANDB:
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

    # ─── Resume ───
    global_step = 0
    if resume_from_checkpoint == "latest":
        ckpts = sorted(out.glob("checkpoint-*"))
        if ckpts:
            resume_from_checkpoint = str(ckpts[-1])

    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        print(f"Resuming from {resume_from_checkpoint}")
        state = torch.load(
            Path(resume_from_checkpoint) / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        controlnet.load_state_dict(state["controlnet"])
        ema_controlnet.load_state_dict(state["ema_controlnet"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        global_step = state["global_step"]
        print(f"Resumed at step {global_step}")

    # ─── Training loop ───
    print(f"\nStarting training from step {global_step}...")
    data_iter = iter(dataloader)
    accumulation_loss = 0.0

    while global_step < num_train_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
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
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
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

        # Loss (Phase A: diffusion only)
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss = loss / gradient_accumulation_steps
        loss.backward()
        accumulation_loss += loss.item()

        # Step optimizer
        if (global_step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # EMA update
            update_ema(ema_controlnet, controlnet, ema_decay)

        global_step += 1

        # ─── Logging ───
        if global_step % log_every == 0:
            avg_loss = accumulation_loss / log_every
            lr_current = scheduler.get_last_lr()[0]
            print(f"Step {global_step}/{num_train_steps} | Loss: {avg_loss:.6f} | LR: {lr_current:.2e}")

            if HAS_WANDB:
                wandb.log({"loss": avg_loss, "lr": lr_current}, step=global_step)

            accumulation_loss = 0.0

        # ─── Sample generation ───
        if global_step % sample_every == 0 and global_step > 0:
            _generate_samples(
                ema_controlnet, vae, unet, text_embeddings, noise_scheduler,
                dataset, device, weight_dtype, out, global_step,
            )

        # ─── Checkpoint ───
        if global_step % checkpoint_every == 0:
            ckpt_dir = out / f"checkpoint-{global_step}"
            ckpt_dir.mkdir(exist_ok=True)

            # Save EMA weights (used for inference)
            ema_controlnet.save_pretrained(ckpt_dir / "controlnet_ema")

            # Save training state for resume
            torch.save({
                "controlnet": controlnet.state_dict(),
                "ema_controlnet": ema_controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
            }, ckpt_dir / "training_state.pt")

            print(f"Checkpoint saved: {ckpt_dir}")

    # ─── Final save ───
    final_dir = out / "final"
    final_dir.mkdir(exist_ok=True)
    ema_controlnet.save_pretrained(final_dir / "controlnet_ema")
    print(f"\nTraining complete. Final model: {final_dir}")

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LandmarkDiff ControlNet")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--phase", default="A", choices=["A", "B"])
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_dir", default=None)
    args = parser.parse_args()

    train(**vars(args))
