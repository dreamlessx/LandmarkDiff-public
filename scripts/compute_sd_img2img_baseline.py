"""Compute SD1.5 Img2Img baseline on HDA test set for paper Table 1.

Uses Stable Diffusion 1.5 image-to-image pipeline with TPS-warped input
and mask compositing — NO ControlNet conditioning. This isolates the
contribution of ControlNet in LandmarkDiff.

The pipeline:
1. Extract landmarks from input image
2. Apply procedure-specific RBF displacement to get target landmarks
3. TPS-warp input to approximate post-surgery geometry
4. Run SD1.5 img2img on the TPS-warped image (with procedure text prompt)
5. Composite result using surgical mask + LAB color matching
6. Compare against real post-surgery target

Usage:
    python scripts/compute_sd_img2img_baseline.py \
        --test_dir data/hda_splits/test \
        --output paper/sd_img2img_baseline_results.json
"""

from __future__ import annotations

import argparse
import json

# Add project root
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_identity_similarity,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.postprocess import histogram_match_skin
from landmarkdiff.synthetic.tps_warp import warp_image_tps

# Procedure-specific prompts (same as used in training/inference)
PROCEDURE_PROMPTS = {
    "rhinoplasty": (
        "high quality professional photo of a person's face, "
        "refined nose shape, smooth nasal bridge, natural skin texture"
    ),
    "blepharoplasty": (
        "high quality professional photo of a person's face, "
        "refreshed eye area, smooth eyelids, natural appearance"
    ),
    "rhytidectomy": (
        "high quality professional photo of a person's face, "
        "smooth jawline, tightened skin, youthful appearance"
    ),
    "orthognathic": (
        "high quality professional photo of a person's face, "
        "balanced facial proportions, aligned jaw, natural symmetry"
    ),
}

NEGATIVE_PROMPT = "blurry, distorted, extra features, deformed, low quality, artifacts, unnatural"


def discover_test_pairs(test_dir: Path) -> list[dict]:
    """Discover input/target pairs and their procedures."""
    prefixes = sorted(set(f.stem.rsplit("_", 1)[0] for f in test_dir.glob("*_input.png")))
    pairs = []
    for prefix in prefixes:
        input_path = test_dir / f"{prefix}_input.png"
        target_path = test_dir / f"{prefix}_target.png"
        conditioning_path = test_dir / f"{prefix}_conditioning.png"
        mask_path = test_dir / f"{prefix}_mask.png"

        if not target_path.exists():
            continue

        procedure = prefix.split("_")[0]
        pairs.append(
            {
                "prefix": prefix,
                "procedure": procedure,
                "input": str(input_path),
                "target": str(target_path),
                "conditioning": str(conditioning_path),
                "mask": str(mask_path),
            }
        )
    return pairs


def load_sd_img2img_pipeline():
    """Load Stable Diffusion 1.5 img2img pipeline (no ControlNet)."""
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_sd_img2img(
    pipe,
    tps_warped: np.ndarray,
    procedure: str,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.6,
    seed: int = 42,
) -> np.ndarray:
    """Run SD1.5 img2img on a TPS-warped image."""
    # Convert BGR to RGB PIL
    rgb = cv2.cvtColor(tps_warped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((512, 512))

    prompt = PROCEDURE_PROMPTS.get(procedure, PROCEDURE_PROMPTS["rhinoplasty"])

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil_img,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Convert back to BGR numpy
    output = np.array(result.images[0])
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def composite_with_mask(
    generated: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Composite generated region into original using feathered mask + LAB matching."""
    # Normalize mask to float32 [0, 1]
    mask_f = mask.astype(np.float32) / 255.0

    # LAB histogram matching to match skin tones
    try:
        matched = histogram_match_skin(generated, original, mask_f)
    except Exception:
        matched = generated

    # Feathered composite
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, np.newaxis]

    composited = (mask_f * matched + (1.0 - mask_f) * original).astype(np.uint8)
    return composited


def main():
    parser = argparse.ArgumentParser(description="Compute SD1.5 Img2Img baseline")
    parser.add_argument("--test_dir", default="data/hda_splits/test")
    parser.add_argument("--output", default="paper/sd_img2img_baseline_results.json")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="Img2img denoising strength (0=no change, 1=full denoise)",
    )
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save generated images for qualitative comparison",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    pairs = discover_test_pairs(test_dir)
    print(f"Found {len(pairs)} test pairs")

    # Load SD1.5 img2img pipeline
    print("Loading SD1.5 Img2Img pipeline...")
    pipe = load_sd_img2img_pipeline()
    print("Pipeline loaded on GPU")

    # Output directory for saved images
    if args.save_outputs:
        out_img_dir = Path("paper/sd_img2img_outputs")
        out_img_dir.mkdir(parents=True, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))
    fitzpatrick_results = defaultdict(lambda: defaultdict(list))

    for i, pair in enumerate(pairs):
        t0 = time.time()
        input_img = cv2.imread(pair["input"])
        target_img = cv2.imread(pair["target"])
        proc = pair["procedure"]

        if input_img is None or target_img is None:
            continue

        # Resize to 512x512
        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        # Step 1: Extract landmarks
        input_lm = extract_landmarks(input_img)
        target_lm = extract_landmarks(target_img)

        if input_lm is None or target_lm is None:
            print(f"  [{i + 1}] Skipping {pair['prefix']}: landmark extraction failed")
            continue

        # Step 2: TPS warp input -> approximate target geometry
        try:
            tps_warped = warp_image_tps(input_img, input_lm.pixel_coords, target_lm.pixel_coords)
        except Exception:
            print(f"  [{i + 1}] Skipping {pair['prefix']}: TPS warp failed")
            continue

        # Step 3: Run SD1.5 img2img on TPS-warped image
        sd_output = run_sd_img2img(
            pipe,
            tps_warped,
            proc,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
        )

        # Step 4: Generate surgical mask and composite
        mask_path = Path(pair["mask"])
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (512, 512))
            else:
                mask = generate_surgical_mask(input_lm, proc, (512, 512))
        else:
            mask = generate_surgical_mask(input_lm, proc, (512, 512))

        composited = composite_with_mask(sd_output, input_img, mask)

        # Step 5: Compute metrics against real target
        ssim = compute_ssim(composited, target_img)
        lpips = compute_lpips(composited, target_img)

        # NME
        comp_lm = extract_landmarks(composited)
        if comp_lm is not None:
            nme = compute_nme(comp_lm.pixel_coords, target_lm.pixel_coords)
        else:
            nme = float("nan")

        # Identity similarity
        id_sim = compute_identity_similarity(composited, target_img)

        results[proc]["ssim"].append(ssim)
        results[proc]["lpips"].append(lpips)
        if not np.isnan(nme):
            results[proc]["nme"].append(nme)
        if not np.isnan(id_sim):
            results[proc]["identity_sim"].append(id_sim)

        # Fitzpatrick classification
        fitz = classify_fitzpatrick_ita(input_img)
        fitzpatrick_results[fitz]["ssim"].append(ssim)
        fitzpatrick_results[fitz]["lpips"].append(lpips)
        if not np.isnan(nme):
            fitzpatrick_results[fitz]["nme"].append(nme)
        if not np.isnan(id_sim):
            fitzpatrick_results[fitz]["identity_sim"].append(id_sim)

        elapsed = time.time() - t0
        print(
            f"  [{i + 1}/{len(pairs)}] {pair['prefix']}: "
            f"SSIM={ssim:.4f} LPIPS={lpips:.4f} NME={nme:.4f} "
            f"ID={id_sim:.4f} Fitz={fitz} ({elapsed:.1f}s)"
        )

        # Save outputs for qualitative comparison
        if args.save_outputs:
            cv2.imwrite(str(out_img_dir / f"{pair['prefix']}_sd_img2img.png"), composited)
            cv2.imwrite(str(out_img_dir / f"{pair['prefix']}_tps.png"), tps_warped)

    # Aggregate results
    output = {
        "config": {
            "strength": args.strength,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        }
    }

    for proc in sorted(results.keys()):
        output[proc] = {}
        for metric, values in sorted(results[proc].items()):
            arr = np.array(values)
            output[proc][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "n": len(arr),
            }

    # Overall
    output["overall"] = {}
    for metric_key in ["ssim", "lpips", "nme", "identity_sim"]:
        all_vals = []
        for proc in results:
            all_vals.extend(results[proc].get(metric_key, []))
        if all_vals:
            arr = np.array(all_vals)
            output["overall"][metric_key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    # Fitzpatrick stratification
    output["fitzpatrick"] = {}
    for fitz_type in sorted(fitzpatrick_results.keys()):
        output["fitzpatrick"][fitz_type] = {}
        for metric, values in fitzpatrick_results[fitz_type].items():
            arr = np.array(values)
            output["fitzpatrick"][fitz_type][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    # Print summary
    print("\n" + "=" * 80)
    print("SD1.5 IMG2IMG BASELINE RESULTS")
    print("=" * 80)
    for section in sorted(output.keys()):
        if section == "config":
            continue
        print(f"\n--- {section.upper()} ---")
        if isinstance(output[section], dict):
            for metric, stats in sorted(output[section].items()):
                if isinstance(stats, dict) and "mean" in stats:
                    print(
                        f"  {metric:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})"
                    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
