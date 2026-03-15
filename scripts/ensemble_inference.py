#!/usr/bin/env python3
"""Ensemble inference for LandmarkDiff surgical outcome prediction.

Generates N predictions with different random seeds and combines them to reduce
stochastic variance inherent in diffusion sampling. Three ensemble strategies:

    1. **Latent averaging** (default, most principled): Captures the final latent
       tensor from each seed's denoising trajectory via a pipeline callback,
       averages them in latent space, then decodes once through the VAE. This
       preserves high-frequency detail better than pixel averaging because the
       VAE decoder operates on a coherent mean representation rather than
       blending already-decoded outputs that may disagree on fine structure.

    2. **Pixel averaging**: Decodes each seed independently, then averages the
       uint8 outputs. Simpler and faster (no callback overhead), but can
       introduce slight blur where predictions disagree on edge positions.

    3. **Quality-weighted ensemble**: Weights each prediction by inverse LPIPS
       distance to the input image. Predictions that are perceptually closer
       to the original receive higher weight, automatically downweighting
       outlier samples with artifacts.

After ensemble combination, the standard LandmarkDiff post-processing pipeline
is applied: surgical mask generation, LAB-space skin color matching, and
alpha-composited blending with the original face.

Outputs a comparison table (ensemble vs best-single-seed vs mean-single-seed)
for SSIM, LPIPS, NME, and ArcFace identity similarity.

Pipeline: MediaPipe 478 landmarks --> RBF manipulation --> 3-ch ControlNet
conditioning --> SD1.5 + CrucibleAI ControlNet --> mask compositing.

Usage:
    # Latent ensemble, 5 seeds, default guidance
    python scripts/ensemble_inference.py \
        --checkpoint checkpoints/phaseB/best \
        --data-dir data/hda_splits/test \
        --output-dir paper/ensemble_outputs \
        --n-seeds 5 --method latent

    # Quality-weighted ensemble with stronger guidance
    python scripts/ensemble_inference.py \
        --checkpoint checkpoints/phaseB/best \
        --data-dir data/hda_splits/test \
        --output-dir paper/ensemble_outputs \
        --n-seeds 7 --method weighted --guidance-scale 9.0

    # Quick pixel averaging for debugging
    python scripts/ensemble_inference.py \
        --checkpoint checkpoints/phaseA/best \
        --data-dir data/hda_splits/test \
        --output-dir paper/ensemble_outputs \
        --n-seeds 3 --method pixel --num-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project root setup — standard pattern across all LandmarkDiff scripts
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from landmarkdiff.evaluation import (
    compute_identity_similarity,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.postprocess import histogram_match_skin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]

# Diffusion prompt tuned for CrucibleAI ControlNet conditioned on MediaPipe mesh
PROMPT = "high quality photo of a face after cosmetic surgery, realistic skin texture"
NEGATIVE_PROMPT = "blurry, distorted, low quality, deformed"


# ========================================================================== #
# Pipeline Loading
# ========================================================================== #


def load_controlnet_pipeline(checkpoint_path: Path, device: str = "cuda") -> Any:
    """Load the SD1.5 + ControlNet pipeline from a fine-tuned checkpoint.

    The ControlNet weights live under `<checkpoint>/controlnet_ema/` and were
    trained on CrucibleAI-style MediaPipe tessellation meshes (2556 edges,
    3 channels, white wireframe on black background).

    We use UniPCMultistepScheduler for fast high-quality sampling and disable
    the safety checker (clinical/research use, no NSFW concern).

    Args:
        checkpoint_path: Directory containing `controlnet_ema/` subfolder.
        device: Torch device string.

    Returns:
        Loaded StableDiffusionControlNetPipeline on the specified device.
    """
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet_dir = checkpoint_path / "controlnet_ema"
    if not controlnet_dir.exists():
        # Fall back to checkpoint root if controlnet_ema/ is not a subfolder
        # (some checkpoints store weights directly)
        controlnet_dir = checkpoint_path
        print(f"[WARN] No controlnet_ema/ subfolder, loading from {controlnet_dir}")

    print(f"Loading ControlNet from: {controlnet_dir}")
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        torch_dtype=torch.float16,
    )

    print("Loading SD1.5 base pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,  # Clinical use — no NSFW filter needed
    )

    # UniPC is ~2x faster than default PNDM at comparable quality
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Enable memory optimizations for single-GPU inference
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory-efficient attention")
    except Exception:
        # xformers not installed — not critical, just slower
        pass

    print(f"Pipeline loaded on {device}")
    return pipe


# ========================================================================== #
# Single-seed Inference
# ========================================================================== #


def run_single_seed(
    pipe: Any,
    conditioning_img: np.ndarray,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    capture_latents: bool = False,
) -> dict:
    """Run one inference pass with a specific random seed.

    When `capture_latents=True`, we hook into the pipeline's step callback
    to capture the final-step latent tensor before VAE decoding. This is
    needed for latent-space ensemble averaging.

    Args:
        pipe: Loaded ControlNet pipeline.
        conditioning_img: 3-channel uint8 MediaPipe mesh image (512x512).
        seed: Random seed for the CUDA generator.
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        capture_latents: Whether to capture the final latent tensor.

    Returns:
        Dict with keys:
            - "image": Decoded output as uint8 numpy array (H, W, 3), RGB.
            - "latents": Final latent tensor (only if capture_latents=True).
            - "seed": The seed used.
    """
    import torch
    from PIL import Image

    # Convert conditioning image to PIL for the pipeline
    cond_pil = Image.fromarray(conditioning_img)

    # Build the CUDA generator with our seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)

    result = {"seed": seed, "latents": None}

    if capture_latents:
        # ------------------------------------------------------------------
        # Latent capture callback
        # ------------------------------------------------------------------
        # The callback fires after each denoising step. We only care about
        # the FINAL step's latents — those are the fully-denoised latent
        # representation that the VAE will decode into an image.
        #
        # callback_kwargs["latents"] is the (1, 4, 64, 64) latent tensor.
        # We clone it to avoid mutation by the pipeline's internal buffer.
        # ------------------------------------------------------------------
        captured_latents = []

        def capture_final_latents(pipe_ref, step, timestep, callback_kwargs):
            """Pipeline callback: capture the latent tensor at the final step."""
            if step == num_steps - 1:
                captured_latents.append(callback_kwargs["latents"].clone())
            return callback_kwargs

        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=cond_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=capture_final_latents,
        )

        if captured_latents:
            result["latents"] = captured_latents[0]
    else:
        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=cond_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # Convert PIL output to numpy (RGB uint8)
    result["image"] = np.array(output.images[0])
    return result


# ========================================================================== #
# Latent-space Decoding
# ========================================================================== #


def decode_averaged_latents(pipe: Any, latents_list: list) -> np.ndarray:
    """Decode an averaged set of latent tensors through the VAE.

    This is the core of latent-space ensemble averaging. Instead of averaging
    decoded pixel outputs (which blurs disagreements), we average the latent
    representations and decode once. The VAE decoder then produces a single
    coherent image from the consensus latent.

    Args:
        pipe: Loaded pipeline (we need its VAE and scaling factor).
        latents_list: List of (1, 4, 64, 64) latent tensors from each seed.

    Returns:
        Decoded image as uint8 numpy array (H, W, 3), RGB.
    """
    import torch

    # Stack all latents: (N, 1, 4, 64, 64) -> (1, 4, 64, 64) via mean over N
    stacked = torch.stack(latents_list, dim=0)  # (N, 1, 4, 64, 64)
    avg_latents = torch.mean(stacked, dim=0)  # (1, 4, 64, 64)

    # Decode through VAE
    # The VAE expects latents scaled by 1/scaling_factor (typically 0.18215)
    with torch.no_grad():
        decoded = pipe.vae.decode(avg_latents / pipe.vae.config.scaling_factor).sample

    # Convert from [-1, 1] to [0, 1] range, then to uint8
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


# ========================================================================== #
# Ensemble Combination Strategies
# ========================================================================== #


def ensemble_latent(
    pipe: Any,
    conditioning_img: np.ndarray,
    seeds: list[int],
    num_steps: int,
    guidance_scale: float,
) -> dict:
    """Latent-space ensemble averaging.

    Runs N seeds, captures final latents from each, averages in latent space,
    then decodes once through the VAE. This produces sharper results than
    pixel averaging because the VAE operates on a coherent mean representation.

    Args:
        pipe: Loaded ControlNet pipeline.
        conditioning_img: 3-channel MediaPipe mesh conditioning image.
        seeds: List of random seeds to use.
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        Dict with "ensemble_image" (averaged), "individual_images", "seeds".
    """
    individual_images = []
    latents_list = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {i + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        result = run_single_seed(
            pipe,
            conditioning_img,
            seed,
            num_steps,
            guidance_scale,
            capture_latents=True,
        )
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        individual_images.append(result["image"])
        if result["latents"] is not None:
            latents_list.append(result["latents"])

    # Average latents and decode
    if latents_list:
        print("  Decoding averaged latents...", end=" ", flush=True)
        ensemble_image = decode_averaged_latents(pipe, latents_list)
        print("done")
    else:
        # Fallback: if latent capture failed, fall back to pixel averaging
        print("  [WARN] Latent capture failed, falling back to pixel averaging")
        stacked = np.stack(individual_images, axis=0).astype(np.float32)
        ensemble_image = np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8)

    return {
        "ensemble_image": ensemble_image,
        "individual_images": individual_images,
        "seeds": seeds,
    }


def ensemble_pixel(
    pipe: Any,
    conditioning_img: np.ndarray,
    seeds: list[int],
    num_steps: int,
    guidance_scale: float,
) -> dict:
    """Pixel-space ensemble averaging.

    Simplest approach: decode each seed independently, then average the uint8
    outputs. Fast (no callback overhead), but introduces slight blur where
    predictions disagree on edges.

    Args:
        pipe: Loaded ControlNet pipeline.
        conditioning_img: 3-channel MediaPipe mesh conditioning image.
        seeds: List of random seeds to use.
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        Dict with "ensemble_image", "individual_images", "seeds".
    """
    individual_images = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {i + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        result = run_single_seed(
            pipe,
            conditioning_img,
            seed,
            num_steps,
            guidance_scale,
            capture_latents=False,
        )
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        individual_images.append(result["image"])

    # Simple pixel-space average
    stacked = np.stack(individual_images, axis=0).astype(np.float32)
    ensemble_image = np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8)

    return {
        "ensemble_image": ensemble_image,
        "individual_images": individual_images,
        "seeds": seeds,
    }


def ensemble_weighted(
    pipe: Any,
    conditioning_img: np.ndarray,
    input_img_rgb: np.ndarray,
    seeds: list[int],
    num_steps: int,
    guidance_scale: float,
) -> dict:
    """Quality-weighted ensemble averaging via inverse LPIPS.

    Each prediction is weighted by how perceptually similar it is to the
    input image. Lower LPIPS = closer to the input = higher weight. This
    automatically downweights outlier samples that have diffusion artifacts.

    The intuition: we want the ensemble to be dominated by samples that
    "agree" with the input appearance, while still benefiting from the
    variance reduction of multiple samples.

    Args:
        pipe: Loaded ControlNet pipeline.
        conditioning_img: 3-channel MediaPipe mesh conditioning image.
        input_img_rgb: Original input face image (RGB, uint8) for LPIPS.
        seeds: List of random seeds.
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.

    Returns:
        Dict with "ensemble_image", "individual_images", "seeds", "weights".
    """
    individual_images = []
    lpips_scores = []

    for i, seed in enumerate(seeds):
        print(f"  Seed {i + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        result = run_single_seed(
            pipe,
            conditioning_img,
            seed,
            num_steps,
            guidance_scale,
            capture_latents=False,
        )
        elapsed = time.time() - t0

        img = result["image"]
        individual_images.append(img)

        # Compute LPIPS between this prediction and the input
        # Lower LPIPS = more similar to input = higher quality weight
        lpips_val = compute_lpips(img, input_img_rgb)
        lpips_scores.append(float(lpips_val) if not np.isnan(lpips_val) else 1.0)
        print(f"done ({elapsed:.1f}s, LPIPS={lpips_scores[-1]:.4f})")

    # Convert LPIPS distances to weights: w_i = 1/LPIPS_i (inverse distance)
    # Then normalize so weights sum to 1
    eps = 1e-8  # avoid division by zero
    inverse_lpips = [1.0 / (lp + eps) for lp in lpips_scores]
    total_weight = sum(inverse_lpips)
    weights = [w / total_weight for w in inverse_lpips]

    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")

    # Weighted pixel averaging
    ensemble = np.zeros_like(individual_images[0], dtype=np.float32)
    for img, weight in zip(individual_images, weights, strict=False):
        ensemble += img.astype(np.float32) * weight
    ensemble_image = np.clip(ensemble, 0, 255).astype(np.uint8)

    return {
        "ensemble_image": ensemble_image,
        "individual_images": individual_images,
        "seeds": seeds,
        "weights": weights,
        "lpips_scores": lpips_scores,
    }


# ========================================================================== #
# Post-processing (mask composite + color matching)
# ========================================================================== #


def apply_postprocessing(
    prediction: np.ndarray,
    input_img: np.ndarray,
    face_landmarks: Any,
    procedure_type: str,
) -> np.ndarray:
    """Apply surgical mask compositing and LAB color matching.

    This ensures that:
      1. Only the surgical region is modified (via the feathered mask).
      2. Skin tone transitions smoothly (via histogram matching in LAB space).
      3. The background and unaffected facial areas remain identical to input.

    The prediction and input_img should be in the same color space (BGR or RGB).

    Args:
        prediction: Generated face image (H, W, 3), uint8.
        input_img: Original face image (H, W, 3), uint8.
        face_landmarks: FaceLandmarks object for mask generation.
        procedure_type: One of PROCEDURES (determines mask shape).

    Returns:
        Composited output image, uint8, same shape and color space as input.
    """
    h, w = input_img.shape[:2]

    # Generate the surgical mask (feathered alpha, float32 [0,1])
    mask = generate_surgical_mask(face_landmarks, procedure_type, w, h)

    # Normalize mask to float32 [0, 1]
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0

    # Resize prediction to match input if needed
    if prediction.shape[:2] != (h, w):
        prediction = cv2.resize(prediction, (w, h))

    # LAB-space histogram matching for skin tone consistency
    # This prevents the diffusion model's color drift from being visible
    try:
        matched = histogram_match_skin(prediction, input_img, mask_f)
    except Exception:
        # If histogram matching fails (e.g., tiny mask region), skip it
        matched = prediction

    # Alpha-composite: mask * matched_prediction + (1-mask) * original
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, np.newaxis]

    composited = mask_f * matched.astype(np.float32) + (1.0 - mask_f) * input_img.astype(np.float32)
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    return composited


# ========================================================================== #
# Metrics Computation & Comparison Table
# ========================================================================== #


def compute_metrics_for_image(
    prediction: np.ndarray,
    reference: np.ndarray,
    pred_landmarks: np.ndarray | None = None,
    ref_landmarks: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full metric suite for a single prediction vs reference.

    Args:
        prediction: Generated image (H, W, 3), uint8.
        reference: Ground truth / input image (H, W, 3), uint8.
        pred_landmarks: Optional (N, 2) predicted landmark positions.
        ref_landmarks: Optional (N, 2) reference landmark positions.

    Returns:
        Dict with ssim, lpips, nme, arcface keys.
    """
    metrics = {
        "ssim": float(compute_ssim(prediction, reference)),
        "lpips": float(compute_lpips(prediction, reference)),
    }

    # NME requires landmark correspondences
    if pred_landmarks is not None and ref_landmarks is not None:
        metrics["nme"] = float(compute_nme(pred_landmarks, ref_landmarks))
    else:
        metrics["nme"] = float("nan")

    # ArcFace identity similarity
    try:
        metrics["arcface"] = float(compute_identity_similarity(prediction, reference))
    except Exception:
        metrics["arcface"] = float("nan")

    return metrics


def print_comparison_table(
    ensemble_metrics: dict[str, float],
    individual_metrics: list[dict[str, float]],
) -> str:
    """Print a formatted comparison table: ensemble vs individual seeds.

    Computes best-single-seed and mean-single-seed for each metric, then
    shows the improvement (or degradation) from ensembling.

    Args:
        ensemble_metrics: Metrics dict for the ensemble output.
        individual_metrics: List of metrics dicts, one per seed.

    Returns:
        Formatted table string (also printed to stdout).
    """
    n = len(individual_metrics)
    metric_names = ["ssim", "lpips", "nme", "arcface"]
    # For SSIM and ArcFace, higher is better; for LPIPS and NME, lower is better
    higher_is_better = {"ssim": True, "lpips": False, "nme": False, "arcface": True}

    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"  Ensemble Comparison ({n} seeds)")
    lines.append("=" * 78)
    lines.append(
        f"  {'Metric':<12} {'Ensemble':>12} {'Best Seed':>12} {'Mean Seed':>12} {'Improvement':>14}"
    )
    lines.append("-" * 78)

    for metric in metric_names:
        ens_val = ensemble_metrics.get(metric, float("nan"))
        indiv_vals = [m.get(metric, float("nan")) for m in individual_metrics]

        # Filter out NaN values for aggregation
        valid_vals = [v for v in indiv_vals if not np.isnan(v)]
        if not valid_vals:
            lines.append(f"  {metric:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>14}")
            continue

        mean_val = float(np.mean(valid_vals))

        # "Best" depends on whether higher or lower is better
        if higher_is_better[metric]:
            best_val = float(np.max(valid_vals))
            # Improvement = ensemble - mean (positive = better)
            improvement = ens_val - mean_val
            imp_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
        else:
            best_val = float(np.min(valid_vals))
            # Improvement = mean - ensemble (positive = better, since lower is better)
            improvement = mean_val - ens_val
            imp_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"

        lines.append(
            f"  {metric:<12} {ens_val:>12.4f} {best_val:>12.4f} {mean_val:>12.4f} {imp_str:>14}"
        )

    lines.append("-" * 78)
    lines.append("  (+) = ensemble is better than mean-single-seed")
    lines.append("=" * 78)
    lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


# ========================================================================== #
# Data Loading
# ========================================================================== #


def load_test_pairs(data_dir: Path) -> list[dict]:
    """Load test image pairs from a directory.

    Expects either:
      - A directory of images (each is treated as a standalone input)
      - A directory with before/after subdirectories
      - A manifest JSON file listing pairs

    Args:
        data_dir: Path to the test data directory.

    Returns:
        List of dicts with "input_path", and optionally "target_path",
        "procedure", and "landmarks_path".
    """
    pairs = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Check for a manifest file
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            entries = json.load(f)
        for entry in entries:
            pair = {
                "input_path": data_dir / entry["input"],
            }
            if "target" in entry:
                pair["target_path"] = data_dir / entry["target"]
            if "procedure" in entry:
                pair["procedure"] = entry["procedure"]
            pairs.append(pair)
        return pairs

    # Check for before/after structure
    before_dir = data_dir / "before"
    after_dir = data_dir / "after"
    if before_dir.exists() and after_dir.exists():
        for img_path in sorted(before_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                pair = {"input_path": img_path}
                # Try to find matching after image
                for ext in extensions:
                    after_path = after_dir / (img_path.stem + ext)
                    if after_path.exists():
                        pair["target_path"] = after_path
                        break
                pairs.append(pair)
        return pairs

    # Flat directory of images — each is a standalone input
    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() in extensions and img_path.is_file():
            pairs.append({"input_path": img_path})

    return pairs


# ========================================================================== #
# Main Processing Loop
# ========================================================================== #


def process_test_pair(
    pipe: Any,
    pair: dict,
    output_dir: Path,
    seeds: list[int],
    num_steps: int,
    guidance_scale: float,
    method: str,
) -> dict | None:
    """Process a single test pair through ensemble inference.

    Steps:
      1. Load and resize input image to 512x512.
      2. Extract MediaPipe face landmarks.
      3. Render the conditioning mesh image (2556-edge tessellation).
      4. Run N-seed ensemble inference (latent/pixel/weighted).
      5. Apply post-processing (mask composite + color match).
      6. Compute metrics for ensemble and each individual seed.
      7. Save all outputs and return the comparison results.

    Args:
        pipe: Loaded ControlNet pipeline.
        pair: Dict with "input_path", optional "target_path" and "procedure".
        output_dir: Where to save outputs.
        seeds: List of random seeds.
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        method: Ensemble method ("latent", "pixel", or "weighted").

    Returns:
        Results dict with metrics, or None if face detection fails.
    """
    input_path = Path(pair["input_path"])
    stem = input_path.stem
    procedure = pair.get("procedure", "rhinoplasty")

    print(f"\n--- Processing: {input_path.name} ({procedure}) ---")

    # Load and resize input image
    input_bgr = cv2.imread(str(input_path))
    if input_bgr is None:
        print(f"  [SKIP] Cannot read image: {input_path}")
        return None

    input_bgr = cv2.resize(input_bgr, (512, 512))
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

    # Extract face landmarks
    face = extract_landmarks(input_bgr)
    if face is None:
        print(f"  [SKIP] No face detected in: {input_path.name}")
        return None

    # Render the conditioning image (MediaPipe tessellation mesh)
    # This is what the ControlNet sees as its spatial conditioning signal
    conditioning_img = render_landmark_image(face, 512, 512)

    # ------------------------------------------------------------------
    # Run ensemble inference with the chosen method
    # ------------------------------------------------------------------
    t_start = time.time()

    if method == "latent":
        ensemble_result = ensemble_latent(
            pipe,
            conditioning_img,
            seeds,
            num_steps,
            guidance_scale,
        )
    elif method == "pixel":
        ensemble_result = ensemble_pixel(
            pipe,
            conditioning_img,
            seeds,
            num_steps,
            guidance_scale,
        )
    elif method == "weighted":
        ensemble_result = ensemble_weighted(
            pipe,
            conditioning_img,
            input_rgb,
            seeds,
            num_steps,
            guidance_scale,
        )
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    t_ensemble = time.time() - t_start

    # ------------------------------------------------------------------
    # Post-processing: mask composite + color matching
    # ------------------------------------------------------------------
    # Apply to ensemble output
    ensemble_rgb = ensemble_result["ensemble_image"]
    ensemble_bgr = cv2.cvtColor(ensemble_rgb, cv2.COLOR_RGB2BGR)
    ensemble_composited = apply_postprocessing(
        ensemble_bgr,
        input_bgr,
        face,
        procedure,
    )

    # Apply to each individual prediction too (for fair comparison)
    individual_composited = []
    for indiv_rgb in ensemble_result["individual_images"]:
        indiv_bgr = cv2.cvtColor(indiv_rgb, cv2.COLOR_RGB2BGR)
        composited = apply_postprocessing(indiv_bgr, input_bgr, face, procedure)
        individual_composited.append(composited)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    # Reference image: use target if available, otherwise input
    if "target_path" in pair:
        target_bgr = cv2.imread(str(pair["target_path"]))
        if target_bgr is not None:
            target_bgr = cv2.resize(target_bgr, (512, 512))
            reference = target_bgr
        else:
            reference = input_bgr
    else:
        reference = input_bgr

    # Ensemble metrics — convert BGR→RGB for LPIPS (AlexNet expects RGB)
    ensemble_rgb = cv2.cvtColor(ensemble_composited, cv2.COLOR_BGR2RGB)
    reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    ensemble_metrics = compute_metrics_for_image(ensemble_rgb, reference_rgb)

    # Individual seed metrics
    individual_metrics = []
    for composited in individual_composited:
        metrics = compute_metrics_for_image(
            cv2.cvtColor(composited, cv2.COLOR_BGR2RGB), reference_rgb
        )
        individual_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pair_dir = output_dir / stem
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble output
    cv2.imwrite(str(pair_dir / "ensemble_output.png"), ensemble_composited)

    # Save individual seed outputs
    for i, (composited, seed) in enumerate(
        zip(individual_composited, ensemble_result["seeds"], strict=False)
    ):
        cv2.imwrite(str(pair_dir / f"seed_{seed:05d}.png"), composited)

    # Save input and conditioning for reference
    cv2.imwrite(str(pair_dir / "input.png"), input_bgr)
    cv2.imwrite(str(pair_dir / "conditioning.png"), conditioning_img)

    # Comparison strip: input | seed_0 | seed_1 | ... | ensemble
    strip_panels = [cv2.resize(input_bgr, (256, 256))]
    for composited in individual_composited:
        strip_panels.append(cv2.resize(composited, (256, 256)))
    strip_panels.append(cv2.resize(ensemble_composited, (256, 256)))
    strip = np.hstack(strip_panels)

    # Add labels
    labels = ["Input"] + [f"Seed {s}" for s in ensemble_result["seeds"]] + ["Ensemble"]
    for i, label in enumerate(labels):
        x = i * 256 + 5
        cv2.putText(
            strip, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    cv2.imwrite(str(pair_dir / "comparison_strip.png"), strip)

    # Print per-image comparison table
    print_comparison_table(ensemble_metrics, individual_metrics)

    return {
        "image": stem,
        "procedure": procedure,
        "method": method,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ensemble_metrics": ensemble_metrics,
        "individual_metrics": individual_metrics,
        "elapsed_seconds": round(t_ensemble, 2),
        "weights": ensemble_result.get("weights"),
    }


# ========================================================================== #
# Summary Report
# ========================================================================== #


def generate_summary_report(
    results: list[dict],
    method: str,
    output_dir: Path,
) -> None:
    """Generate and print the final summary report across all test pairs.

    Aggregates metrics and prints a grand comparison table showing the
    benefit of ensembling across the entire test set.

    Args:
        results: List of per-pair result dicts from process_test_pair().
        method: Ensemble method used.
        output_dir: Where to save the JSON report.
    """
    if not results:
        print("\nNo results to summarize (all images failed).")
        return

    n = len(results)

    # Aggregate ensemble metrics
    agg_ensemble = {}
    agg_best_seed = {}
    agg_mean_seed = {}

    metric_names = ["ssim", "lpips", "nme", "arcface"]
    higher_is_better = {"ssim": True, "lpips": False, "nme": False, "arcface": True}

    for metric in metric_names:
        ens_vals = [r["ensemble_metrics"].get(metric, float("nan")) for r in results]
        ens_vals = [v for v in ens_vals if not np.isnan(v)]

        indiv_vals_per_image = []
        for r in results:
            vals = [m.get(metric, float("nan")) for m in r["individual_metrics"]]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                if higher_is_better[metric]:
                    indiv_vals_per_image.append((max(vals), np.mean(vals)))
                else:
                    indiv_vals_per_image.append((min(vals), np.mean(vals)))

        if ens_vals:
            agg_ensemble[metric] = float(np.mean(ens_vals))
        if indiv_vals_per_image:
            agg_best_seed[metric] = float(np.mean([v[0] for v in indiv_vals_per_image]))
            agg_mean_seed[metric] = float(np.mean([v[1] for v in indiv_vals_per_image]))

    # Print grand summary
    print("\n" + "=" * 78)
    print(f"  GRAND SUMMARY: {n} images, method={method}")
    print("=" * 78)
    print(
        f"  {'Metric':<12} {'Ensemble':>12} {'Best Seed':>12} {'Mean Seed':>12} {'Ens vs Mean':>14}"
    )
    print("-" * 78)

    for metric in metric_names:
        ens = agg_ensemble.get(metric, float("nan"))
        best = agg_best_seed.get(metric, float("nan"))
        mean = agg_mean_seed.get(metric, float("nan"))

        if np.isnan(ens) or np.isnan(mean):
            print(f"  {metric:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>14}")
            continue

        improvement = ens - mean if higher_is_better[metric] else mean - ens

        imp_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
        print(f"  {metric:<12} {ens:>12.4f} {best:>12.4f} {mean:>12.4f} {imp_str:>14}")

    print("-" * 78)
    print("  (+) = ensemble outperforms mean-single-seed")
    print("=" * 78)

    # Timing summary
    total_time = sum(r["elapsed_seconds"] for r in results)
    avg_time = total_time / n
    print(f"\n  Total inference time: {total_time:.1f}s")
    print(f"  Average per image:   {avg_time:.1f}s")
    print(f"  Average per seed:    {avg_time / results[0]['n_seeds']:.1f}s")

    # Save full report as JSON
    report = {
        "method": method,
        "n_images": n,
        "n_seeds": results[0]["n_seeds"] if results else 0,
        "aggregate": {
            "ensemble": agg_ensemble,
            "best_single_seed": agg_best_seed,
            "mean_single_seed": agg_mean_seed,
        },
        "per_image": results,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = output_dir / "ensemble_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")


# ========================================================================== #
# SLURM Script Generator
# ========================================================================== #


def print_slurm_command(args: argparse.Namespace) -> None:
    """Print a suggested SLURM sbatch command for this configuration.

    Useful for copying/pasting into a terminal or CI pipeline. The command
    uses the pre-made SLURM wrapper script at slurm/ensemble_inference.sh
    but also shows a standalone sbatch invocation.

    Args:
        args: Parsed command-line arguments.
    """
    print("\n" + "=" * 60)
    print("  Suggested SLURM Submission Command")
    print("=" * 60)
    print()
    print("  # Using the pre-made wrapper script:")
    print("  sbatch slurm/ensemble_inference.sh")
    print()
    print("  # Or, standalone sbatch with custom args:")
    print("  sbatch \\")
    print("    --job-name=surgery_ensemble \\")
    print("    --partition=batch_gpu \\")
    print("    --account=your_account \\")
    print("    --time=4:00:00 \\")
    print("    --gres=gpu:nvidia_rtx_a6000:1 \\")
    print("    --mem=32G \\")
    print("    --cpus-per-task=4 \\")
    print("    --output=logs/ensemble_%j.out \\")
    print("    --error=logs/ensemble_%j.err \\")
    print(
        f"    --wrap='$HOME/miniconda3/envs/landmarkdiff/bin/python "
        f"scripts/ensemble_inference.py "
        f"--checkpoint {args.checkpoint} "
        f"--data-dir {args.data_dir} "
        f"--output-dir {args.output_dir} "
        f"--n-seeds {args.n_seeds} "
        f"--method {args.method} "
        f"--num-steps {args.num_steps} "
        f"--guidance-scale {args.guidance_scale}'"
    )
    print()
    print("=" * 60)


# ========================================================================== #
# CLI Entrypoint
# ========================================================================== #


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble inference for LandmarkDiff — generates N predictions "
            "with different seeds and combines them for improved output quality."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Latent ensemble (default, most principled)\n"
            "  python scripts/ensemble_inference.py \\\n"
            "      --checkpoint checkpoints/phaseB/best \\\n"
            "      --data-dir data/hda_splits/test \\\n"
            "      --output-dir paper/ensemble_outputs\n\n"
            "  # Quality-weighted with 7 seeds\n"
            "  python scripts/ensemble_inference.py \\\n"
            "      --checkpoint checkpoints/phaseB/best \\\n"
            "      --data-dir data/hda_splits/test \\\n"
            "      --output-dir paper/ensemble_outputs \\\n"
            "      --n-seeds 7 --method weighted\n"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing controlnet_ema/ subfolder.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Path to test data directory. Supports flat image dirs, "
            "before/after subdirs, or manifest.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper/ensemble_outputs",
        help="Output directory for ensemble results. Default: paper/ensemble_outputs",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of random seeds for ensemble. Default: 5",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of denoising steps per seed. Default: 30",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale. Default: 7.5",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["latent", "pixel", "weighted"],
        default="latent",
        help=(
            "Ensemble combination method. "
            "'latent' averages in latent space (sharpest), "
            "'pixel' averages decoded outputs (simplest), "
            "'weighted' uses inverse-LPIPS quality weighting. "
            "Default: latent"
        ),
    )
    parser.add_argument(
        "--procedure",
        type=str,
        default=None,
        choices=PROCEDURES,
        help="Override procedure type for all images. Default: auto-detect or rhinoplasty.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Seeds will be [base, base+1, ..., base+n-1]. Default: 42",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of test images to process (for debugging).",
    )
    parser.add_argument(
        "--print-slurm",
        action="store_true",
        help="Print a suggested SLURM submission command and exit.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # If --print-slurm, just show the command and exit
    # ------------------------------------------------------------------
    if args.print_slurm:
        print_slurm_command(args)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate seed list: [base_seed, base_seed+1, ..., base_seed+n-1]
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  LandmarkDiff Ensemble Inference")
    print("=" * 60)
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Data dir:        {data_dir}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Method:          {args.method}")
    print(f"  N seeds:         {args.n_seeds} (seeds={seeds})")
    print(f"  Denoising steps: {args.num_steps}")
    print(f"  Guidance scale:  {args.guidance_scale}")
    if args.procedure:
        print(f"  Procedure:       {args.procedure}")
    if args.max_images:
        print(f"  Max images:      {args.max_images}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load pipeline (once for all images)
    # ------------------------------------------------------------------
    print("\nLoading ControlNet pipeline...")
    pipe = load_controlnet_pipeline(checkpoint_path)

    # ------------------------------------------------------------------
    # Load test pairs
    # ------------------------------------------------------------------
    print(f"\nLoading test pairs from: {data_dir}")
    pairs = load_test_pairs(data_dir)

    if not pairs:
        print(f"ERROR: No test images found in {data_dir}")
        sys.exit(1)

    if args.max_images:
        pairs = pairs[: args.max_images]

    # Override procedure if specified
    if args.procedure:
        for pair in pairs:
            pair["procedure"] = args.procedure

    print(f"Found {len(pairs)} test images")

    # ------------------------------------------------------------------
    # Process each test pair
    # ------------------------------------------------------------------
    results = []
    failed = 0

    for i, pair in enumerate(pairs):
        print(f"\n[{i + 1}/{len(pairs)}] ", end="")
        result = process_test_pair(
            pipe,
            pair,
            output_dir,
            seeds,
            args.num_steps,
            args.guidance_scale,
            args.method,
        )
        if result is None:
            failed += 1
        else:
            results.append(result)

    # ------------------------------------------------------------------
    # Generate summary report
    # ------------------------------------------------------------------
    print(f"\n\nProcessed: {len(results)} / {len(pairs)} images (failed: {failed})")
    generate_summary_report(results, args.method, output_dir)

    # Print SLURM suggestion at the end for convenience
    print_slurm_command(args)


if __name__ == "__main__":
    main()
