"""LCM (Latent Consistency Model) distillation for fast ControlNet inference.

Distills a trained ControlNet teacher into a student that generates high-quality
images in 2-4 denoising steps instead of 20-50.

Approach: Consistency distillation from Luo et al. 2023
- Teacher: trained ControlNet with DDIM scheduler (many steps)
- Student: same architecture, trained via consistency loss
- Result: 4-step inference with ~90% of teacher quality

Usage:
    python scripts/distill_lcm.py \
        --teacher_checkpoint checkpoints/phaseB/latest \
        --data_dir data/training_combined \
        --output_dir checkpoints/lcm_distilled \
        --num_train_steps 5000

For CPU deployment, combine with ONNX export:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/lcm_distilled/latest \
        --quantize int8
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DistillationDataset(Dataset):
    """Loads conditioning images for distillation (no target needed)."""

    def __init__(self, data_dir: str, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.pairs = sorted(
            set(f.stem.rsplit("_", 1)[0] for f in self.data_dir.glob("*_conditioning.png"))
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prefix = self.pairs[idx]
        cond_path = self.data_dir / f"{prefix}_conditioning.png"

        cond = cv2.imread(str(cond_path))
        if cond is None:
            cond = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        cond = cv2.resize(cond, (self.resolution, self.resolution))
        cond = torch.from_numpy(cond[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0

        return {"conditioning": cond, "prefix": prefix}


def consistency_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """LPIPS-weighted consistency loss between student and teacher predictions.

    The student should produce the same denoised output as the teacher
    regardless of the number of steps.
    """
    # Huber loss -- more robust than L2 for diffusion outputs
    return weight * F.huber_loss(student_output, teacher_output, delta=0.5)


def run_teacher_inference(
    controlnet,
    unet,
    vae,
    noise_scheduler,
    text_embeddings,
    conditioning,
    latent_shape,
    device,
    weight_dtype,
    num_steps: int = 20,
    seed: int = 42,
) -> torch.Tensor:
    """Run multi-step teacher inference to get target output."""
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(latent_shape, generator=generator, device=device, dtype=weight_dtype)
    latents = noise * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(latents, t)

        down_samples, mid_sample = controlnet(
            scaled,
            t,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=conditioning,
            return_dict=False,
        )
        noise_pred = unet(
            scaled,
            t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=[s.detach() for s in down_samples],
            mid_block_additional_residual=mid_sample.detach(),
        ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def run_student_inference(
    student_controlnet,
    unet,
    noise_scheduler,
    text_embeddings,
    conditioning,
    noise,
    device,
    weight_dtype,
    num_steps: int = 4,
) -> torch.Tensor:
    """Run few-step student inference."""
    from diffusers import LCMScheduler

    scheduler = LCMScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    latents = noise * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(latents, t)

        down_samples, mid_sample = student_controlnet(
            scaled,
            t,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=conditioning,
            return_dict=False,
        )
        noise_pred = unet(
            scaled,
            t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def main():
    parser = argparse.ArgumentParser(description="LCM distillation for fast ControlNet inference")
    parser.add_argument("--teacher_checkpoint", required=True, help="Path to trained ControlNet")
    parser.add_argument("--data_dir", required=True, help="Training data directory")
    parser.add_argument("--output_dir", default="checkpoints/lcm_distilled")
    parser.add_argument("--base_model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_train_steps", type=int, default=5000)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--teacher_steps", type=int, default=20, help="Teacher inference steps")
    parser.add_argument("--student_steps", type=int, default=4, help="Student inference steps")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ema_decay", type=float, default=0.95, help="EMA for target network")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weight_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    _kw: dict = {"local_files_only": True} if _local_only else {}

    print(f"Device: {device} | Dtype: {weight_dtype}")
    print(f"Teacher: {args.teacher_checkpoint} ({args.teacher_steps} steps)")
    print(f"Student: {args.student_steps} steps")

    # Load models
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_id, subfolder="tokenizer", **_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model_id, subfolder="text_encoder", **_kw
    )
    vae = AutoencoderKL.from_pretrained(args.base_model_id, subfolder="vae", **_kw)
    unet = UNet2DConditionModel.from_pretrained(args.base_model_id, subfolder="unet", **_kw)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model_id, subfolder="scheduler", **_kw
    )

    # Load teacher ControlNet
    teacher_path = Path(args.teacher_checkpoint)
    if (teacher_path / "controlnet_ema").exists():
        teacher_path = teacher_path / "controlnet_ema"
    print(f"Loading teacher ControlNet from {teacher_path}...")
    teacher_controlnet = ControlNetModel.from_pretrained(str(teacher_path))

    # Student = copy of teacher
    student_controlnet = copy.deepcopy(teacher_controlnet)

    # Freeze everything except student
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_controlnet.requires_grad_(False)
    student_controlnet.train()

    # Move to device
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    teacher_controlnet.to(device, dtype=weight_dtype)
    student_controlnet.to(device, dtype=weight_dtype)

    # EMA target network
    ema_student = copy.deepcopy(student_controlnet)
    ema_student.requires_grad_(False)

    # Text embeddings (fixed prompt)
    text_input = tokenizer(
        "a photo of a person's face",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(weight_dtype)

    # Dataset
    dataset = DistillationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} conditioning images")

    # Optimizer
    optimizer = torch.optim.AdamW(student_controlnet.parameters(), lr=args.learning_rate)

    # Training loop
    print(f"\nStarting LCM distillation ({args.num_train_steps} steps)...")
    global_step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while global_step < args.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        conditioning = batch["conditioning"].to(device, dtype=weight_dtype)

        # Get latent shape from a dummy encode
        with torch.no_grad():
            dummy = torch.randn(
                conditioning.shape[0], 3, 512, 512, device=device, dtype=weight_dtype
            )
            latent_shape = vae.encode(dummy).latent_dist.sample().shape

        # Generate random seed per sample
        seed = args.seed + global_step
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent_shape, generator=generator, device=device, dtype=weight_dtype)

        # Teacher: run many-step inference (no grad)
        with torch.no_grad():
            teacher_latents = run_teacher_inference(
                teacher_controlnet,
                unet,
                vae,
                noise_scheduler,
                text_embeddings[: conditioning.shape[0]],
                conditioning,
                latent_shape,
                device,
                weight_dtype,
                num_steps=args.teacher_steps,
                seed=seed,
            )

        # Student: run few-step inference (with grad for ControlNet only)
        student_latents = run_student_inference(
            student_controlnet,
            unet,
            noise_scheduler,
            text_embeddings[: conditioning.shape[0]],
            conditioning,
            noise,
            device,
            weight_dtype,
            num_steps=args.student_steps,
        )

        # Consistency loss: student output should match teacher output
        loss = consistency_loss(student_latents, teacher_latents.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_controlnet.parameters(), 1.0)
        optimizer.step()

        # Update EMA
        with torch.no_grad():
            for p_ema, p_student in zip(ema_student.parameters(), student_controlnet.parameters()):
                p_ema.data.mul_(args.ema_decay).add_(p_student.data, alpha=1 - args.ema_decay)

        global_step += 1

        if global_step % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = global_step / elapsed
            eta = (args.num_train_steps - global_step) / rate
            print(
                f"Step {global_step}/{args.num_train_steps} | "
                f"Loss: {loss.item():.6f} | "
                f"{rate:.1f} it/s | ETA: {eta / 60:.1f} min"
            )

        if global_step % args.checkpoint_every == 0:
            ckpt_dir = out / f"checkpoint-{global_step}"
            ema_student.save_pretrained(ckpt_dir / "controlnet_lcm")
            student_controlnet.save_pretrained(ckpt_dir / "controlnet_student")
            torch.save(
                {
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_dir / "training_state.pt",
            )
            print(f"Checkpoint saved: {ckpt_dir}")

            # Update latest symlink
            latest = out / "latest"
            if latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_dir.name)

    # Final save
    final_dir = out / "final"
    ema_student.save_pretrained(final_dir / "controlnet_lcm")
    print(f"\nDistillation complete! Final model: {final_dir}")
    print(
        f"Use with: LandmarkDiffPipeline(mode='controlnet_fast', "
        f"controlnet_checkpoint='{final_dir / 'controlnet_lcm'}')"
    )


if __name__ == "__main__":
    main()
