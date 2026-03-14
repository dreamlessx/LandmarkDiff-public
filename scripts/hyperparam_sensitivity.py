"""Hyperparameter sensitivity analysis.

Tests how metrics change with different inference hyperparameters:
- num_inference_steps: [5, 10, 15, 20, 30, 50]
- guidance_scale: [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
- controlnet_conditioning_scale: [0.5, 0.7, 0.9, 1.0, 1.2]

Generates a grid of plots showing metric vs. hyperparameter.
This helps justify the chosen hyperparameter values in the paper.

Usage:
    python scripts/hyperparam_sensitivity.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/hyperparam_sensitivity.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def evaluate_single(
    pipe,
    cond_pil: Image.Image,
    target_img: np.ndarray,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    cn_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run inference with specific hyperparameters and compute metrics."""
    from landmarkdiff.evaluation import compute_identity_similarity, compute_lpips, compute_ssim

    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        output = pipe(
            prompt="high quality photo of a face after cosmetic surgery",
            negative_prompt="blurry, distorted, low quality",
            image=cond_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=cn_scale,
            generator=gen,
        )

    pred_bgr = cv2.cvtColor(np.array(output.images[0]), cv2.COLOR_RGB2BGR)

    return {
        "ssim": compute_ssim(pred_bgr, target_img),
        "lpips": compute_lpips(pred_bgr, target_img),
        "identity_sim": compute_identity_similarity(pred_bgr, target_img),
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/hyperparam_sensitivity.png")
    parser.add_argument("--max-pairs", type=int, default=10, help="Test on subset for speed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.checkpoint, device)

    # Load test pairs
    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))[: args.max_pairs]

    test_data = []
    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"
        if not target_file.exists() or not cond_file.exists():
            continue
        target_img = cv2.resize(cv2.imread(str(target_file)), (512, 512))
        conditioning = cv2.resize(cv2.imread(str(cond_file)), (512, 512))
        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)
        test_data.append({"cond_pil": cond_pil, "target": target_img, "prefix": prefix})

    print(f"Testing on {len(test_data)} pairs")

    # ── Sweep 1: num_inference_steps ─────────────────────────────────────
    steps_values = [5, 10, 15, 20, 30, 50]
    steps_results = defaultdict(lambda: defaultdict(list))

    print("\n[1] Sweeping num_inference_steps...")
    for n_steps in steps_values:
        for _j, td in enumerate(test_data):
            metrics = evaluate_single(
                pipe, td["cond_pil"], td["target"], num_steps=n_steps, seed=args.seed
            )
            for k, v in metrics.items():
                steps_results[n_steps][k].append(v)
        m = {k: np.mean(v) for k, v in steps_results[n_steps].items()}
        print(
            f"  steps={n_steps}: SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f} "
            f"ArcFace={m['identity_sim']:.3f}"
        )

    # ── Sweep 2: guidance_scale ──────────────────────────────────────────
    guidance_values = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    guidance_results = defaultdict(lambda: defaultdict(list))

    print("\n[2] Sweeping guidance_scale...")
    for g in guidance_values:
        for _j, td in enumerate(test_data):
            metrics = evaluate_single(
                pipe, td["cond_pil"], td["target"], guidance_scale=g, seed=args.seed
            )
            for k, v in metrics.items():
                guidance_results[g][k].append(v)
        m = {k: np.mean(v) for k, v in guidance_results[g].items()}
        print(
            f"  guidance={g}: SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f} "
            f"ArcFace={m['identity_sim']:.3f}"
        )

    # ── Sweep 3: controlnet_conditioning_scale ───────────────────────────
    cn_values = [0.5, 0.7, 0.9, 1.0, 1.2]
    cn_results = defaultdict(lambda: defaultdict(list))

    print("\n[3] Sweeping controlnet_conditioning_scale...")
    for cn in cn_values:
        for _j, td in enumerate(test_data):
            metrics = evaluate_single(
                pipe, td["cond_pil"], td["target"], cn_scale=cn, seed=args.seed
            )
            for k, v in metrics.items():
                cn_results[cn][k].append(v)
        m = {k: np.mean(v) for k, v in cn_results[cn].items()}
        print(
            f"  cn_scale={cn}: SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f} "
            f"ArcFace={m['identity_sim']:.3f}"
        )

    # ── Plot ─────────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    metrics_order = ["ssim", "lpips", "identity_sim"]
    metric_labels = ["SSIM ↑", "LPIPS ↓", "ArcFace ↑"]

    sweeps = [
        ("Inference Steps", steps_values, steps_results),
        ("Guidance Scale", guidance_values, guidance_results),
        ("ControlNet Scale", cn_values, cn_results),
    ]

    for row, (sweep_name, values, results) in enumerate(sweeps):
        for col, (metric, label) in enumerate(zip(metrics_order, metric_labels, strict=False)):
            ax = axes[row, col]
            means = [np.mean(results[v][metric]) for v in values]
            stds = [np.std(results[v][metric]) for v in values]

            ax.errorbar(values, means, yerr=stds, marker="o", capsize=4, linewidth=2, markersize=6)

            # Highlight chosen value
            chosen_map = {
                "Inference Steps": 20,
                "Guidance Scale": 7.5,
                "ControlNet Scale": 1.0,
            }
            chosen = chosen_map[sweep_name]
            if chosen in values:
                values.index(chosen)
                ax.axvline(
                    chosen, color="red", linestyle="--", alpha=0.5, label=f"Chosen: {chosen}"
                )

            ax.set_xlabel(sweep_name)
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs {sweep_name}")
            ax.grid(True, alpha=0.3)
            if chosen in values:
                ax.legend(fontsize=8)

    plt.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # Save JSON
    json_path = out_path.with_suffix(".json")
    output_data = {
        "steps_sweep": {
            str(v): {k: {"mean": float(np.mean(r[k])), "std": float(np.std(r[k]))} for k in r}
            for v, r in steps_results.items()
        },
        "guidance_sweep": {
            str(v): {k: {"mean": float(np.mean(r[k])), "std": float(np.std(r[k]))} for k in r}
            for v, r in guidance_results.items()
        },
        "cn_scale_sweep": {
            str(v): {k: {"mean": float(np.mean(r[k])), "std": float(np.std(r[k]))} for k in r}
            for v, r in cn_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
