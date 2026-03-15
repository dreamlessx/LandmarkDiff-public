"""Cross-procedure transfer evaluation.

Tests how well the model generalizes across procedures by evaluating
predictions with different procedure-specific conditioning on the same
input face. This reveals whether the model learns procedure-specific
spatial priors vs. generic face generation.

The key insight: if the model truly learns procedure-specific deformations,
then applying rhinoplasty conditioning to a blepharoplasty target should
produce poor metrics — the model should "know" to modify different regions.

Usage:
    python scripts/cross_procedure_eval.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/cross_procedure_results.json
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


PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def load_pipeline(checkpoint: str, device: torch.device):
    """Load ControlNet pipeline (cached)."""
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


def generate_with_procedure(
    pipe,
    input_img: np.ndarray,
    face_lm,
    procedure: str,
    intensity: float = 1.0,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate a prediction using procedure-specific displacement.

    Applies the specified procedure's deformation field to the input
    landmarks and generates via ControlNet.
    """
    from landmarkdiff.landmarks import render_landmark_image
    from landmarkdiff.manipulation import apply_procedure_preset

    h, w = input_img.shape[:2]

    # Apply procedure-specific deformation via the preset API
    # intensity is 0-100 scale; apply_procedure_preset divides internally
    displaced_lm = apply_procedure_preset(
        face_lm,
        procedure,
        intensity=intensity * 100,
        image_size=max(w, h),
    )

    # Render conditioning
    mesh = render_landmark_image(displaced_lm, w, h)
    mesh_gray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    conditioning = cv2.merge([mesh_gray, canny, np.zeros_like(mesh_gray)])

    cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
    cond_pil = Image.fromarray(cond_rgb)

    prompt = "high quality photo of a face after cosmetic surgery, realistic skin texture"
    neg_prompt = "blurry, distorted, low quality, deformed"

    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=cond_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )

    return cv2.cvtColor(np.array(output.images[0]), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Cross-procedure transfer evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/cross_procedure_results.json")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.checkpoint, device)

    from landmarkdiff.evaluation import (
        compute_identity_similarity,
        compute_lpips,
        compute_ssim,
    )
    from landmarkdiff.landmarks import extract_landmarks

    # Load test pairs
    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))

    pairs = []
    for inp_file in input_files:
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        if not target_file.exists():
            continue
        procedure = "unknown"
        for proc in PROCEDURES:
            if proc in prefix:
                procedure = proc
                break
        pairs.append(
            {
                "input": str(inp_file),
                "target": str(target_file),
                "prefix": prefix,
                "procedure": procedure,
            }
        )

    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    print(f"Cross-procedure evaluation: {len(pairs)} pairs")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # For each test pair, generate with its own procedure AND every other procedure
    # Result: confusion-matrix-like table of metrics
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # results[true_proc][applied_proc][metric] = [values...]

    for i, pair in enumerate(pairs):
        input_img = cv2.resize(cv2.imread(pair["input"]), (512, 512))
        target_img = cv2.resize(cv2.imread(pair["target"]), (512, 512))
        true_proc = pair["procedure"]

        face_lm = extract_landmarks(input_img)
        if face_lm is None:
            continue

        for applied_proc in PROCEDURES:
            pred = generate_with_procedure(
                pipe,
                input_img,
                face_lm,
                applied_proc,
                num_steps=args.num_steps,
                seed=args.seed,
            )

            ssim = compute_ssim(pred, target_img)
            lpips = compute_lpips(pred, target_img)
            id_sim = compute_identity_similarity(pred, target_img)

            results[true_proc][applied_proc]["ssim"].append(ssim)
            results[true_proc][applied_proc]["lpips"].append(lpips)
            results[true_proc][applied_proc]["identity_sim"].append(id_sim)

        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(pairs)}] done")

    # Aggregate and print confusion matrix
    print(f"\n{'=' * 72}")
    print("CROSS-PROCEDURE TRANSFER MATRIX")
    print(f"{'=' * 72}")

    output = {"confusion_matrices": {}}

    for metric in ["ssim", "lpips", "identity_sim"]:
        print(f"\n--- {metric.upper()} ---")
        label = "True \\ Applied"
        header = f"{label:<18s}"
        for p in PROCEDURES:
            header += f" {p[:5]:>8s}"
        print(header)
        print("-" * 60)

        output["confusion_matrices"][metric] = {}
        for true_p in PROCEDURES:
            row = f"{true_p:<18s}"
            output["confusion_matrices"][metric][true_p] = {}
            for applied_p in PROCEDURES:
                vals = results[true_p][applied_p][metric]
                if vals:
                    mean_val = float(np.mean(vals))
                    output["confusion_matrices"][metric][true_p][applied_p] = {
                        "mean": mean_val,
                        "std": float(np.std(vals)),
                        "n": len(vals),
                    }
                    # Bold diagonal (matching procedure)
                    if true_p == applied_p:
                        row += f" *{mean_val:.3f}*"
                    else:
                        row += f"  {mean_val:.3f} "
                else:
                    row += f"  {'N/A':>6s} "
            print(row)

    # Summary: is diagonal always best?
    print(f"\n{'=' * 72}")
    print("DIAGONAL DOMINANCE (does matching procedure always win?)")
    for metric in ["ssim", "identity_sim"]:
        wins = 0
        total = 0
        for true_p in PROCEDURES:
            if true_p not in results:
                continue
            diagonal_vals = results[true_p][true_p][metric]
            if not diagonal_vals:
                continue
            diag_mean = np.mean(diagonal_vals)
            best_for_metric = metric in ["ssim", "identity_sim"]  # higher is better

            is_best = True
            for applied_p in PROCEDURES:
                if applied_p == true_p:
                    continue
                off_vals = results[true_p][applied_p][metric]
                if off_vals:
                    off_mean = np.mean(off_vals)
                    if (best_for_metric and off_mean > diag_mean) or (
                        not best_for_metric and off_mean < diag_mean
                    ):
                        is_best = False
            if is_best:
                wins += 1
            total += 1
        print(f"  {metric}: diagonal wins {wins}/{total} procedures")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
