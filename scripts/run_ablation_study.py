"""Run ablation study experiments for LandmarkDiff paper Table 2.

Tests three dimensions of ablation:
1. Conditioning channels: mesh-only vs mesh+canny vs mesh+canny+mask (full)
2. Compositing strategy: direct output vs mask composite vs mask+LAB color match
3. Clinical augmentation: with vs without

Each ablation runs inference on the test set with one component removed/changed,
then computes metrics. Results populate Table 2 in the paper.

Usage:
    python scripts/run_ablation_study.py \
        --checkpoint checkpoints/phaseA/best \
        --test-dir data/hda_splits/test \
        --output paper/ablation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pipeline(checkpoint_path: Path):
    """Load the ControlNet pipeline."""
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet_subdir = (
        "controlnet_ema" if (checkpoint_path / "controlnet_ema").exists() else "controlnet"
    )
    controlnet = ControlNetModel.from_pretrained(
        str(checkpoint_path / controlnet_subdir),
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def render_conditioning(
    face_lm,
    image: np.ndarray,
    mode: str = "full",
    procedure: str = "rhinoplasty",
) -> np.ndarray:
    """Render conditioning image with different channel configurations.

    Args:
        face_lm: FaceLandmarks object from extract_landmarks().
        image: (H, W, 3) input image for Canny edge extraction.
        mode: "mesh_only", "mesh_canny", or "full" (mesh+canny+mask).
        procedure: Procedure type for surgical mask generation.

    Returns:
        (H, W, 3) uint8 conditioning image.
    """
    from landmarkdiff.landmarks import render_landmark_image

    h, w = image.shape[:2]

    if mode == "mesh_only":
        mesh = render_landmark_image(face_lm, w, h)
        mesh_gray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)
        return cv2.merge([mesh_gray, mesh_gray, mesh_gray])

    elif mode == "mesh_canny":
        mesh = render_landmark_image(face_lm, w, h)
        mesh_gray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)

        zeros = np.zeros_like(mesh_gray)
        return cv2.merge([mesh_gray, canny, zeros])

    elif mode == "full":
        mesh = render_landmark_image(face_lm, w, h)
        mesh_gray = cv2.cvtColor(mesh, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)

        from landmarkdiff.masking import generate_surgical_mask

        try:
            mask = generate_surgical_mask(face_lm, procedure)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            # Convert float32 [0-1] mask to uint8 [0-255] for cv2.merge
            if mask.dtype != np.uint8:
                mask = (mask * 255).clip(0, 255).astype(np.uint8)
        except Exception:
            mask = np.zeros_like(mesh_gray)

        return cv2.merge([mesh_gray, canny, mask])

    raise ValueError(f"Unknown conditioning mode: {mode}")


def apply_compositing(
    prediction: np.ndarray,
    input_img: np.ndarray,
    face_lm,
    procedure: str,
    mode: str = "full",
) -> np.ndarray:
    """Apply compositing with different strategies.

    Args:
        prediction: Raw diffusion output (H, W, 3) BGR.
        input_img: Original input image (H, W, 3) BGR.
        face_lm: FaceLandmarks object for mask generation.
        procedure: Procedure type string.
        mode: "direct", "mask_only", or "full" (mask + LAB color match).

    Returns:
        Composited output (H, W, 3) BGR.
    """
    if mode == "direct":
        return prediction

    from landmarkdiff.masking import generate_surgical_mask

    mask = generate_surgical_mask(face_lm, procedure)
    mask_f = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)

    if mode == "mask_only":
        if mask_f.ndim == 2:
            mask_f = mask_f[:, :, np.newaxis]
        return (mask_f * prediction + (1.0 - mask_f) * input_img).astype(np.uint8)

    elif mode == "full":
        from landmarkdiff.postprocess import histogram_match_skin

        try:
            matched = histogram_match_skin(prediction, input_img, mask_f)
        except Exception:
            matched = prediction

        if mask_f.ndim == 2:
            mask_f = mask_f[:, :, np.newaxis]
        return (mask_f * matched + (1.0 - mask_f) * input_img).astype(np.uint8)

    raise ValueError(f"Unknown compositing mode: {mode}")


def run_single_ablation(
    pipe,
    test_pairs: list[dict],
    conditioning_mode: str = "full",
    compositing_mode: str = "full",
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
    max_pairs: int | None = None,
) -> dict:
    """Run a single ablation configuration on the test set.

    Returns dict with per-procedure and overall metrics.
    """
    from landmarkdiff.evaluation import (
        compute_identity_similarity,
        compute_lpips,
        compute_nme,
        compute_ssim,
    )
    from landmarkdiff.landmarks import extract_landmarks

    results = defaultdict(lambda: defaultdict(list))
    n_processed = 0
    n_failed = 0

    prompt = "high quality photo of a face after cosmetic surgery, realistic skin texture"
    negative_prompt = "blurry, distorted, low quality, deformed"

    pairs = test_pairs[:max_pairs] if max_pairs else test_pairs

    for i, pair in enumerate(pairs):
        input_img = cv2.imread(pair["input"])
        target_img = cv2.imread(pair["target"])
        procedure = pair["procedure"]

        if input_img is None or target_img is None:
            n_failed += 1
            continue

        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        # Extract landmarks
        input_lm = extract_landmarks(input_img)
        target_lm = extract_landmarks(target_img)
        if input_lm is None or target_lm is None:
            n_failed += 1
            continue

        # Render conditioning
        conditioning = render_conditioning(
            target_lm,
            input_img,
            conditioning_mode,
            procedure,
        )
        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        # Run inference
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=cond_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        pred_bgr = cv2.cvtColor(np.array(output.images[0]), cv2.COLOR_RGB2BGR)

        # Apply compositing
        final = apply_compositing(
            pred_bgr,
            input_img,
            target_lm,
            procedure,
            compositing_mode,
        )

        # Compute metrics
        ssim = compute_ssim(final, target_img)
        lpips = compute_lpips(final, target_img)

        pred_lm = extract_landmarks(final)
        if pred_lm is not None:
            nme = compute_nme(pred_lm.pixel_coords, target_lm.pixel_coords)
        else:
            nme = float("nan")

        id_sim = compute_identity_similarity(final, target_img)

        results[procedure]["ssim"].append(ssim)
        results[procedure]["lpips"].append(lpips)
        if not np.isnan(nme):
            results[procedure]["nme"].append(nme)
        if not np.isnan(id_sim):
            results[procedure]["identity_sim"].append(id_sim)

        n_processed += 1

        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(pairs)}] {procedure}: SSIM={ssim:.3f} LPIPS={lpips:.3f}")

    # Aggregate
    output = {"n_processed": n_processed, "n_failed": n_failed}
    all_vals = defaultdict(list)

    for proc in sorted(results.keys()):
        output[proc] = {}
        for metric, values in sorted(results[proc].items()):
            arr = np.array(values)
            output[proc][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }
            all_vals[metric].extend(values)

    # Overall
    output["overall"] = {}
    for metric, values in all_vals.items():
        arr = np.array(values)
        output["overall"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }

    return output


def discover_test_pairs(test_dir: Path) -> list[dict]:
    """Discover test pairs in the data directory."""
    pairs = []
    for f in sorted(test_dir.glob("*_input.png")):
        prefix = f.stem.rsplit("_", 1)[0]
        target = test_dir / f"{prefix}_target.png"
        if not target.exists():
            continue
        procedure = prefix.split("_")[0]
        pairs.append(
            {
                "prefix": prefix,
                "procedure": procedure,
                "input": str(f),
                "target": str(target),
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "phaseA" / "best")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data" / "hda_splits" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "ablation_results.json")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pairs", type=int, default=None, help="Limit pairs per ablation (for quick testing)"
    )
    parser.add_argument(
        "--procedure", type=str, default=None, help="Run ablation on a specific procedure only"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LANDMARKDIFF ABLATION STUDY")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print()

    # Load pipeline
    pipe = load_pipeline(args.checkpoint)

    # Discover test pairs
    pairs = discover_test_pairs(args.test_dir)
    if args.procedure:
        pairs = [p for p in pairs if p["procedure"] == args.procedure]
    print(f"Test pairs: {len(pairs)}")

    # Define ablation configurations
    ablations = {
        # Conditioning ablation
        "mesh_only": {"conditioning": "mesh_only", "compositing": "full"},
        "mesh_canny": {"conditioning": "mesh_canny", "compositing": "full"},
        "full_conditioning": {"conditioning": "full", "compositing": "full"},
        # Compositing ablation
        "direct_output": {"conditioning": "full", "compositing": "direct"},
        "mask_no_color": {"conditioning": "full", "compositing": "mask_only"},
        "mask_lab_color": {"conditioning": "full", "compositing": "full"},
    }

    all_results = {}

    for name, config in ablations.items():
        print(f"\n--- Ablation: {name} ---")
        print(f"  Conditioning: {config['conditioning']}")
        print(f"  Compositing: {config['compositing']}")

        t0 = time.time()
        result = run_single_ablation(
            pipe,
            pairs,
            conditioning_mode=config["conditioning"],
            compositing_mode=config["compositing"],
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            max_pairs=args.max_pairs,
        )
        elapsed = time.time() - t0

        overall = result.get("overall", {})
        print(f"  Results ({elapsed:.0f}s):")
        for metric in ["ssim", "lpips", "nme", "identity_sim"]:
            if metric in overall:
                print(f"    {metric}: {overall[metric]['mean']:.4f} ± {overall[metric]['std']:.4f}")

        all_results[name] = {
            "config": config,
            "results": result,
            "elapsed_seconds": elapsed,
        }

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<25s} {'LPIPS↓':>8s} {'SSIM↑':>8s} {'NME↓':>8s} {'ArcFace↑':>10s}")
    print("-" * 65)

    for name, data in all_results.items():
        overall = data["results"].get("overall", {})
        lpips = overall.get("lpips", {}).get("mean", float("nan"))
        ssim = overall.get("ssim", {}).get("mean", float("nan"))
        nme = overall.get("nme", {}).get("mean", float("nan"))
        arcface = overall.get("identity_sim", {}).get("mean", float("nan"))
        print(f"{name:<25s} {lpips:>8.3f} {ssim:>8.3f} {nme:>8.3f} {arcface:>10.3f}")

    # Generate LaTeX table rows
    print("\n--- LaTeX Table Rows ---")
    latex_names = {
        "mesh_only": "Mesh only",
        "mesh_canny": "Mesh + Canny",
        "full_conditioning": "Mesh + Canny + Mask (full)",
        "direct_output": "Direct output (no composite)",
        "mask_no_color": "Mask composite (no color match)",
        "mask_lab_color": "Mask + LAB color match (full)",
    }
    for name, data in all_results.items():
        overall = data["results"].get("overall", {})
        lpips = overall.get("lpips", {}).get("mean", None)
        ssim = overall.get("ssim", {}).get("mean", None)
        nme = overall.get("nme", {}).get("mean", None)
        arcface = overall.get("identity_sim", {}).get("mean", None)

        latex_name = latex_names.get(name, name)
        fid = "--"  # FID requires distribution-level computation
        lpips_s = f"{lpips:.3f}" if lpips is not None else "--"
        nme_s = f"{nme:.3f}" if nme is not None else "--"
        arcface_s = f"{arcface:.3f}" if arcface is not None else "--"

        print(f"{latex_name:<35s} & {fid} & {lpips_s} & {nme_s} & {arcface_s} \\\\")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
