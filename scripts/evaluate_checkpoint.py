"""Evaluate a trained ControlNet checkpoint on held-out test pairs.

Computes all paper metrics:
  - FID (via clean-fid)
  - LPIPS (AlexNet backbone)
  - NME (Normalized Mean landmark Error, IOD-normalized)
  - ArcFace identity similarity (cosine)
  - SSIM (windowed, scikit-image)

Stratified by procedure and Fitzpatrick skin type.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint checkpoints_phaseA/final/controlnet_ema \
        --test_dir data/test_pairs \
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_fid,
    compute_identity_similarity,
    compute_lpips,
    compute_ssim,
)
from landmarkdiff.landmarks import extract_landmarks


def load_test_pairs(test_dir: Path) -> list[dict]:
    """Load test pairs with metadata."""
    pairs = []
    input_files = sorted(test_dir.glob("*_input.png"))

    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not target_file.exists():
            continue

        # Try to infer procedure from filename
        procedure = "unknown"
        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            if proc in prefix:
                procedure = proc
                break

        pairs.append(
            {
                "input": str(inp_file),
                "target": str(target_file),
                "conditioning": str(cond_file) if cond_file.exists() else str(inp_file),
                "prefix": prefix,
                "procedure": procedure,
            }
        )

    return pairs


@torch.no_grad()
def generate_from_checkpoint(
    checkpoint_path: str,
    conditioning_img: np.ndarray,
    num_steps: int = 20,
    guidance_scale: float = 9.0,
    cn_scale: float = 0.9,
    seed: int = 42,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate an image from a ControlNet checkpoint + conditioning."""
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
        UniPCMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = "runwayml/stable-diffusion-v1-5"

    # Load models (cached after first call via global)
    if not hasattr(generate_from_checkpoint, "_cache"):
        print("Loading models for evaluation...")
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(
            device, dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(
            device, dtype=dtype
        )

        ckpt = Path(checkpoint_path)
        if (ckpt / "controlnet_ema").exists():
            ckpt = ckpt / "controlnet_ema"
        controlnet = ControlNetModel.from_pretrained(str(ckpt)).to(device, dtype=dtype)

        noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

        text_input = tokenizer(
            "a photo of a person's face",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_emb = text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)

        generate_from_checkpoint._cache = {
            "vae": vae,
            "unet": unet,
            "controlnet": controlnet,
            "noise_scheduler": noise_scheduler,
            "text_emb": text_emb,
            "device": device,
            "dtype": dtype,
        }
        print("Models loaded.")

    c = generate_from_checkpoint._cache
    vae, unet, controlnet = c["vae"], c["unet"], c["controlnet"]
    text_emb = c["text_emb"]

    # Prepare conditioning
    cond_tensor = (
        torch.from_numpy(conditioning_img.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device, dtype=dtype)
    )

    # Generate from noise
    scheduler = UniPCMultistepScheduler.from_config(c["noise_scheduler"].config)
    scheduler.set_timesteps(num_steps, device=device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(latents, t)
        down_samples, mid_sample = controlnet(
            scaled,
            t,
            encoder_hidden_states=text_emb,
            controlnet_cond=cond_tensor,
            return_dict=False,
        )
        noise_pred = unet(
            scaled,
            t,
            encoder_hidden_states=text_emb,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    decoded = ((decoded + 1) / 2).clamp(0, 1)
    img = (decoded[0].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def compute_nme(pred_img: np.ndarray, target_img: np.ndarray) -> float | None:
    """Compute NME between predicted and target landmarks."""
    pred_face = extract_landmarks(pred_img)
    target_face = extract_landmarks(target_img)

    if pred_face is None or target_face is None:
        return None

    pred_coords = pred_face.pixel_coords
    target_coords = target_face.pixel_coords

    # IOD (inter-ocular distance) for normalization — in pixel space
    left_eye = target_coords[33]  # left eye inner corner
    right_eye = target_coords[263]  # right eye inner corner
    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1.0:
        return None

    dists = np.linalg.norm(pred_coords - target_coords, axis=1)
    return float(dists.mean() / iod)


def evaluate(
    checkpoint_path: str,
    test_dir: str,
    output_path: str | None = None,
    max_samples: int = 0,
    num_steps: int = 20,
    seed: int = 42,
    save_images: bool = False,
    images_dir: str | None = None,
) -> dict:
    """Run full evaluation suite including FID and identity similarity."""
    import shutil
    import tempfile

    test_pairs = load_test_pairs(Path(test_dir))
    if max_samples > 0:
        test_pairs = test_pairs[:max_samples]

    print(f"Evaluating {len(test_pairs)} test pairs...")

    # Output directory for comparison images
    if save_images:
        comp_dir = Path(images_dir or "eval_images")
        comp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving comparison images to {comp_dir}")

    results = {
        "checkpoint": checkpoint_path,
        "num_pairs": len(test_pairs),
        "metrics": {},
        "per_procedure": {},
        "per_fitzpatrick": {},
    }

    all_ssim = []
    all_lpips = []
    all_nme = []
    all_identity = []
    per_proc = {}
    per_fitz = {}

    # Temp dirs for FID computation (needs directories of images)
    fid_gen_dir = Path(tempfile.mkdtemp(prefix="eval_gen_"))
    fid_real_dir = Path(tempfile.mkdtemp(prefix="eval_real_"))

    try:
        for i, pair in enumerate(test_pairs):
            cond_img = cv2.imread(pair["conditioning"])
            target_img = cv2.imread(pair["target"])
            if cond_img is None or target_img is None:
                continue

            cond_img = cv2.resize(cond_img, (512, 512))
            target_img = cv2.resize(target_img, (512, 512))

            # Generate prediction
            pred_img = generate_from_checkpoint(
                checkpoint_path,
                cond_img,
                num_steps=num_steps,
                seed=seed,
            )

            # Save for FID
            cv2.imwrite(str(fid_gen_dir / f"{i:06d}.png"), pred_img)
            cv2.imwrite(str(fid_real_dir / f"{i:06d}.png"), target_img)

            # Save comparison images: conditioning | generated | target
            if save_images:
                comparison = np.hstack([cond_img, pred_img, target_img])
                cv2.imwrite(str(comp_dir / f"{pair['prefix']}_comparison.png"), comparison)

            proc = pair["procedure"]
            if proc not in per_proc:
                per_proc[proc] = {"ssim": [], "lpips": [], "nme": [], "identity": []}

            # SSIM
            ssim_val = compute_ssim(pred_img, target_img)
            all_ssim.append(ssim_val)
            per_proc[proc]["ssim"].append(ssim_val)

            # LPIPS
            lpips_val = compute_lpips(pred_img, target_img)
            if lpips_val is not None:
                all_lpips.append(lpips_val)
                per_proc[proc]["lpips"].append(lpips_val)

            # NME
            nme_val = compute_nme(pred_img, target_img)
            if nme_val is not None:
                all_nme.append(nme_val)
                per_proc[proc]["nme"].append(nme_val)

            # ArcFace identity similarity
            id_val = compute_identity_similarity(pred_img, target_img)
            all_identity.append(id_val)
            per_proc[proc]["identity"].append(id_val)

            # Fitzpatrick classification
            fitz = classify_fitzpatrick_ita(target_img)
            if fitz not in per_fitz:
                per_fitz[fitz] = {"ssim": [], "lpips": [], "nme": [], "identity": [], "count": 0}
            per_fitz[fitz]["count"] += 1
            per_fitz[fitz]["ssim"].append(ssim_val)
            per_fitz[fitz]["identity"].append(id_val)
            if lpips_val is not None:
                per_fitz[fitz]["lpips"].append(lpips_val)
            if nme_val is not None:
                per_fitz[fitz]["nme"].append(nme_val)

            if (i + 1) % 10 == 0:
                print(
                    f"  [{i + 1}/{len(test_pairs)}] SSIM={np.mean(all_ssim):.4f} "
                    f"LPIPS={np.mean(all_lpips):.4f} NME={np.mean(all_nme):.4f} "
                    f"ID={np.mean(all_identity):.4f}"
                )

        # FID (distribution-level metric)
        fid_val = 0.0
        try:
            fid_val = compute_fid(str(fid_real_dir), str(fid_gen_dir))
            print(f"  FID: {fid_val:.2f}")
        except Exception as e:
            print(f"  FID computation failed: {e}")

    finally:
        shutil.rmtree(fid_gen_dir, ignore_errors=True)
        shutil.rmtree(fid_real_dir, ignore_errors=True)

    # Aggregate
    results["metrics"] = {
        "fid": fid_val,
        "ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        "lpips": float(np.mean(all_lpips)) if all_lpips else 0.0,
        "nme": float(np.mean(all_nme)) if all_nme else 0.0,
        "identity_sim": float(np.mean(all_identity)) if all_identity else 0.0,
    }

    for proc, vals in per_proc.items():
        results["per_procedure"][proc] = {
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else 0.0,
            "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else 0.0,
            "nme": float(np.mean(vals["nme"])) if vals["nme"] else 0.0,
            "identity_sim": float(np.mean(vals["identity"])) if vals["identity"] else 0.0,
            "count": len(vals["ssim"]),
        }

    for fitz, vals in per_fitz.items():
        results["per_fitzpatrick"][fitz] = {
            "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else 0.0,
            "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else 0.0,
            "nme": float(np.mean(vals["nme"])) if vals["nme"] else 0.0,
            "identity_sim": float(np.mean(vals["identity"])) if vals["identity"] else 0.0,
            "count": vals["count"],
        }

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS ({len(test_pairs)} pairs)")
    print(f"{'=' * 50}")
    print(f"  FID:      {results['metrics']['fid']:.2f}")
    print(f"  SSIM:     {results['metrics']['ssim']:.4f}")
    print(f"  LPIPS:    {results['metrics']['lpips']:.4f}")
    print(f"  NME:      {results['metrics']['nme']:.4f}")
    print(f"  ID Sim:   {results['metrics']['identity_sim']:.4f}")
    print("\nBy Procedure:")
    for proc, vals in sorted(results["per_procedure"].items()):
        print(
            f"  {proc}: SSIM={vals['ssim']:.4f} LPIPS={vals['lpips']:.4f} "
            f"NME={vals['nme']:.4f} ID={vals['identity_sim']:.4f} (n={vals['count']})"
        )
    print("\nBy Fitzpatrick Type:")
    for fitz, vals in sorted(results["per_fitzpatrick"].items()):
        print(
            f"  {fitz}: SSIM={vals['ssim']:.4f} LPIPS={vals['lpips']:.4f} "
            f"NME={vals['nme']:.4f} ID={vals['identity_sim']:.4f} (n={vals['count']})"
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ControlNet checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to ControlNet checkpoint directory"
    )
    parser.add_argument("--test_dir", required=True, help="Directory with test pairs")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=0, help="Max test pairs (0 = all)")
    parser.add_argument("--num_steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_images", action="store_true", help="Save comparison images (cond|gen|target)"
    )
    parser.add_argument("--images_dir", default=None, help="Directory for comparison images")
    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        args.test_dir,
        args.output,
        args.max_samples,
        args.num_steps,
        args.seed,
        args.save_images,
        args.images_dir,
    )
