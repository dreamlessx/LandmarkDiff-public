"""Uncertainty estimation via multi-sample inference.

For clinical applications, uncertainty quantification is critical.
This script generates multiple predictions from the same input using
different random seeds, then computes:
  1. Per-pixel variance map (epistemic uncertainty)
  2. Mean prediction (consensus image)
  3. Coefficient of variation map
  4. Agreement heatmap (regions where all samples agree vs. diverge)

The resulting visualization demonstrates where the model is confident
(stable anatomy) vs. uncertain (details, textures, exact positions).

Usage:
    python scripts/uncertainty_estimation.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/fig_uncertainty.png \
        --n-samples 10
"""

from __future__ import annotations

import argparse
import json
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


def generate_samples(
    pipe,
    conditioning_pil: Image.Image,
    n_samples: int = 10,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    cn_scale: float = 1.0,
    base_seed: int = 42,
) -> list[np.ndarray]:
    """Generate multiple predictions from same conditioning with different seeds.

    Returns list of BGR images as numpy arrays.
    """
    samples = []
    for i in range(n_samples):
        seed = base_seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            output = pipe(
                prompt="high quality photo of a face after cosmetic surgery",
                negative_prompt="blurry, distorted, low quality",
                image=conditioning_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=cn_scale,
                generator=gen,
            )
        pred = np.array(output.images[0])
        samples.append(pred)
    return samples


def compute_uncertainty_maps(samples: list[np.ndarray]) -> dict:
    """Compute uncertainty statistics from multiple samples.

    Returns:
        mean_img: pixel-wise mean (consensus prediction)
        std_img: pixel-wise standard deviation
        cv_img: coefficient of variation (std / mean)
        agreement_img: 1 - normalized variance (high = consensus)
        range_img: per-pixel range (max - min across samples)
    """
    stack = np.array(samples, dtype=np.float32)  # (N, H, W, 3)

    mean_img = stack.mean(axis=0)
    std_img = stack.std(axis=0)
    range_img = stack.max(axis=0) - stack.min(axis=0)

    # Coefficient of variation (avoid division by zero)
    cv_img = np.divide(std_img, mean_img + 1e-8)

    # Agreement map: average across channels, normalize to [0, 1]
    std_gray = std_img.mean(axis=-1)
    max_std = std_gray.max() + 1e-8
    agreement_img = 1.0 - (std_gray / max_std)

    return {
        "mean": mean_img.astype(np.uint8),
        "std": std_img,
        "cv": cv_img,
        "agreement": agreement_img,
        "range": range_img,
    }


def make_uncertainty_figure(
    samples_data: list[dict],
    n_show_samples: int = 4,
) -> np.ndarray:
    """Create a comprehensive uncertainty visualization figure.

    samples_data: list of dicts with keys:
        - "conditioning": PIL image
        - "target": PIL image
        - "samples": list of np.ndarray (RGB)
        - "uncertainty": dict from compute_uncertainty_maps
        - "label": str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(samples_data)
    # Columns: conditioning | sample_1..n | mean | std | agreement | target
    n_cols = 1 + n_show_samples + 3 + 1  # cond + samples + mean/std/agree + target

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = (
        ["Conditioning"]
        + [f"Sample {i + 1}" for i in range(n_show_samples)]
        + ["Mean", "Uncertainty", "Agreement", "Target"]
    )

    for row_idx, data in enumerate(samples_data):
        uncertainty = data["uncertainty"]

        # Column 0: conditioning
        ax = axes[row_idx, 0]
        ax.imshow(data["conditioning"])
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=8, fontweight="bold")
        ax.set_ylabel(data.get("label", ""), fontsize=8, rotation=0, labelpad=55, va="center")
        ax.set_xticks([])
        ax.set_yticks([])

        # Sample columns
        for i in range(n_show_samples):
            ax = axes[row_idx, 1 + i]
            if i < len(data["samples"]):
                ax.imshow(data["samples"][i])
            if row_idx == 0:
                ax.set_title(col_titles[1 + i], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Mean
        ax = axes[row_idx, 1 + n_show_samples]
        ax.imshow(uncertainty["mean"])
        if row_idx == 0:
            ax.set_title("Mean", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Uncertainty (std) as heatmap
        ax = axes[row_idx, 2 + n_show_samples]
        std_gray = uncertainty["std"].mean(axis=-1)
        ax.imshow(std_gray, cmap="hot", vmin=0, vmax=std_gray.max())
        if row_idx == 0:
            ax.set_title("Uncertainty", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Agreement
        ax = axes[row_idx, 3 + n_show_samples]
        ax.imshow(uncertainty["agreement"], cmap="RdYlGn", vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title("Agreement", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Target
        ax = axes[row_idx, 4 + n_show_samples]
        ax.imshow(data["target"])
        if row_idx == 0:
            ax.set_title("Target", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Prediction Uncertainty: Multi-Sample Analysis", fontsize=13, y=1.01)
    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def main():
    parser = argparse.ArgumentParser(description="Uncertainty estimation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/fig_uncertainty.png")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples per input for uncertainty estimation",
    )
    parser.add_argument(
        "--n-show", type=int, default=4, help="Number of individual samples to display"
    )
    parser.add_argument("--max-rows", type=int, default=4, help="Number of test cases to analyze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.checkpoint, device)

    # Load test pairs — one per procedure
    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))

    procedures_seen = set()
    selected = []
    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        proc = "unknown"
        for p in [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ]:
            if p in prefix:
                proc = p
                break
        if proc not in procedures_seen:
            procedures_seen.add(proc)
            selected.append((inp_file, proc, prefix))
            if len(selected) >= args.max_rows:
                break

    print(f"Generating {args.n_samples} samples each for {len(selected)} inputs")

    all_data = []
    overall_stats = {}

    for inp_file, proc, prefix in selected:
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not target_file.exists() or not cond_file.exists():
            continue

        print(f"\n  [{len(all_data) + 1}/{len(selected)}] {prefix} ({proc})")

        # Load images
        target_img = cv2.resize(cv2.imread(str(target_file)), (512, 512))
        target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_pil = Image.fromarray(target_rgb)

        conditioning = cv2.resize(cv2.imread(str(cond_file)), (512, 512))
        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        # Generate multiple samples
        samples = generate_samples(
            pipe,
            cond_pil,
            n_samples=args.n_samples,
            base_seed=args.seed,
        )

        # Compute uncertainty
        uncertainty = compute_uncertainty_maps(samples)

        # Per-sample SSIM to target
        from landmarkdiff.evaluation import compute_ssim

        ssims = []
        for s in samples:
            s_bgr = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
            ssims.append(compute_ssim(s_bgr, target_img))

        mean_ssim = np.mean(ssims)
        std_ssim = np.std(ssims)
        print(f"    SSIM: {mean_ssim:.4f} +/- {std_ssim:.4f}")
        print(f"    Mean pixel std: {uncertainty['std'].mean():.1f}")
        print(f"    Mean agreement: {uncertainty['agreement'].mean():.3f}")

        overall_stats[prefix] = {
            "procedure": proc,
            "ssim_mean": float(mean_ssim),
            "ssim_std": float(std_ssim),
            "pixel_std_mean": float(uncertainty["std"].mean()),
            "agreement_mean": float(uncertainty["agreement"].mean()),
        }

        all_data.append(
            {
                "conditioning": cond_pil,
                "target": target_pil,
                "samples": samples[: args.n_show],
                "uncertainty": uncertainty,
                "label": proc.capitalize(),
            }
        )

    if not all_data:
        print("No valid samples found")
        return

    # Generate figure
    grid = make_uncertainty_figure(all_data, n_show_samples=args.n_show)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(str(out_path), quality=95)
    print(f"\nSaved: {out_path}")

    # Save JSON stats
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"Stats: {json_path}")


if __name__ == "__main__":
    main()
