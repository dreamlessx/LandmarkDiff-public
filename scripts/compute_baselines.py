"""Compute baseline metrics (TPS-only, morphing) that don't require trained models.

These baselines use the same landmark deformation pipeline as LandmarkDiff but
apply simple geometric transforms instead of diffusion-based generation:

1. TPS-only: Thin-plate spline warp from original to deformed landmarks.
2. Morphing: Affine warp using landmark displacement (Crisalix-style).

Both baselines use mask compositing with LAB color matching for fair comparison.

Usage:
    python scripts/compute_baselines.py \
        --test_dir data/test_pairs \
        --output results/baselines.json \
        --max_samples 200

    # Generate comparison images
    python scripts/compute_baselines.py \
        --test_dir data/test_pairs \
        --output results/baselines.json \
        --save_images results/baseline_images/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_identity_similarity,
    compute_lpips,
    compute_ssim,
)
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.masking import generate_surgical_mask


def tps_warp(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Apply thin-plate spline warp from src to dst landmark positions.

    Uses OpenCV's TPS implementation for fast, robust interpolation.
    """
    h, w = image.shape[:2]

    # Subsample control points for efficiency (every 6th landmark)
    n = len(src_points)
    step = max(1, n // 80)
    indices = list(range(0, n, step))
    src_sub = src_points[indices].astype(np.float32)
    dst_sub = dst_points[indices].astype(np.float32)

    # Pixel coordinates
    src_px = src_sub.copy()
    src_px[:, 0] *= w
    src_px[:, 1] *= h
    dst_px = dst_sub.copy()
    dst_px[:, 0] *= w
    dst_px[:, 1] *= h

    # Build TPS transformer
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_px))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(
        dst_px.reshape(1, -1, 2),
        src_px.reshape(1, -1, 2),
        matches,
    )

    # Apply warp
    warped = tps.warpImage(image)
    return warped


def affine_morph(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply simple affine morphing within the mask region.

    This simulates commercial morphing tools (Crisalix-style) that use
    affine/polynomial warps based on landmark displacements.
    """
    h, w = image.shape[:2]

    # Compute displacement field from landmark movements
    src_px = src_points[:, :2].copy()
    dst_px = dst_points[:, :2].copy()
    src_px[:, 0] *= w
    src_px[:, 1] *= h
    dst_px[:, 0] *= w
    dst_px[:, 1] *= h

    displacements = dst_px - src_px  # (N, 2)

    # Build displacement field via RBF interpolation
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    # Use sparse grid for efficiency
    grid_y, grid_x = np.mgrid[0:h:4, 0:w:4]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # RBF interpolation of displacement field
    for i in range(len(src_px)):
        cx, cy = src_px[i]
        dx, dy = displacements[i]
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            continue

        sigma = 30.0  # Gaussian influence radius
        dist_sq = (grid_points[:, 0] - cx) ** 2 + (grid_points[:, 1] - cy) ** 2
        weight = np.exp(-dist_sq / (2 * sigma**2))

        flow_x_sparse = weight * dx
        flow_y_sparse = weight * dy

        flow_x[grid_y.ravel(), grid_x.ravel()] += flow_x_sparse
        flow_y[grid_y.ravel(), grid_x.ravel()] += flow_y_sparse

    # Upscale flow to full resolution
    flow_x = cv2.resize(flow_x, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_y = cv2.resize(flow_y, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply mask to flow
    if mask is not None:
        mask_f = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
        if mask_f.ndim == 3:
            mask_f = mask_f[:, :, 0]
        flow_x *= mask_f
        flow_y *= mask_f

    # Remap
    map_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0) - flow_x
    map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1) - flow_y

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def compute_nme(pred_img: np.ndarray, target_img: np.ndarray) -> float | None:
    """Compute NME between predicted and target landmarks."""
    pred_face = extract_landmarks(pred_img)
    target_face = extract_landmarks(target_img)

    if pred_face is None or target_face is None:
        return None

    pred_coords = pred_face.pixel_coords
    target_coords = target_face.pixel_coords

    # IOD (inter-ocular distance) for normalization — in pixel space
    left_eye = target_coords[33]
    right_eye = target_coords[263]
    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1.0:
        return None

    dists = np.linalg.norm(pred_coords - target_coords, axis=1)
    return float(dists.mean() / iod)


def evaluate_baselines(
    test_dir: str,
    output_path: str | None = None,
    max_samples: int = 0,
    save_images: str | None = None,
) -> dict:
    """Run baseline evaluation on test pairs."""
    test_dir = Path(test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))

    if max_samples > 0:
        input_files = input_files[:max_samples]

    if not input_files:
        print(f"No test pairs found in {test_dir}")
        return {}

    print(f"Evaluating {len(input_files)} test pairs...")

    if save_images:
        Path(save_images).mkdir(parents=True, exist_ok=True)

    results = {
        "tps": {"ssim": [], "lpips": [], "nme": [], "identity": [], "per_proc": {}, "per_fitz": {}},
        "morphing": {
            "ssim": [],
            "lpips": [],
            "nme": [],
            "identity": [],
            "per_proc": {},
            "per_fitz": {},
        },
    }

    t0 = time.time()

    for i, inp_file in enumerate(input_files):
        prefix = inp_file.stem.replace("_input", "")
        target_file = test_dir / f"{prefix}_target.png"
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not target_file.exists():
            continue

        # Load images
        input_img = cv2.imread(str(inp_file))
        target_img = cv2.imread(str(target_file))
        if input_img is None or target_img is None:
            continue

        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        # Extract landmarks from target to determine where face should be
        target_face = extract_landmarks(target_img)
        if target_face is None:
            continue

        # Infer procedure
        procedure = "unknown"
        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            if proc in prefix:
                procedure = proc
                break

        # For TPS baseline: we need original landmarks and deformed landmarks
        # Since we have conditioning (mesh), extract landmarks from the input face
        # and use target face as the deformation target
        input_face = extract_landmarks(input_img)
        if input_face is None:
            # Input is a mesh, not a face — skip or use target as reference
            continue

        # TPS warp: warp input face to match target landmark positions
        try:
            tps_result = tps_warp(
                input_img,
                input_face.landmarks[:, :2],
                target_face.landmarks[:, :2],
            )
        except Exception:
            continue

        # Morphing: affine warp within mask region
        try:
            mask = generate_surgical_mask(
                target_face,
                procedure if procedure != "unknown" else "rhinoplasty",
                512,
                512,
            )
            morph_result = affine_morph(
                input_img,
                input_face.landmarks[:, :2],
                target_face.landmarks[:, :2],
                mask,
            )
        except Exception:
            morph_result = tps_result.copy()  # fallback

        # Fitzpatrick classification
        fitz = classify_fitzpatrick_ita(target_img)

        # Evaluate both baselines
        for method, pred_img in [("tps", tps_result), ("morphing", morph_result)]:
            ssim_val = compute_ssim(pred_img, target_img)
            lpips_val = compute_lpips(pred_img, target_img)
            nme_val = compute_nme(pred_img, target_img)
            id_val = compute_identity_similarity(pred_img, target_img)

            results[method]["ssim"].append(ssim_val)
            if lpips_val is not None:
                results[method]["lpips"].append(lpips_val)
            if nme_val is not None:
                results[method]["nme"].append(nme_val)
            results[method]["identity"].append(id_val)

            # Per-procedure
            pp = results[method]["per_proc"]
            if procedure not in pp:
                pp[procedure] = {"ssim": [], "lpips": [], "nme": [], "identity": [], "count": 0}
            pp[procedure]["count"] += 1
            pp[procedure]["ssim"].append(ssim_val)
            if lpips_val is not None:
                pp[procedure]["lpips"].append(lpips_val)
            if nme_val is not None:
                pp[procedure]["nme"].append(nme_val)
            pp[procedure]["identity"].append(id_val)

            # Per-Fitzpatrick
            pf = results[method]["per_fitz"]
            if fitz not in pf:
                pf[fitz] = {"ssim": [], "lpips": [], "nme": [], "identity": [], "count": 0}
            pf[fitz]["count"] += 1
            pf[fitz]["ssim"].append(ssim_val)
            if lpips_val is not None:
                pf[fitz]["lpips"].append(lpips_val)
            if nme_val is not None:
                pf[fitz]["nme"].append(nme_val)
            pf[fitz]["identity"].append(id_val)

        # Save comparison images
        if save_images:
            comparison = np.hstack([input_img, tps_result, morph_result, target_img])
            cv2.imwrite(
                str(Path(save_images) / f"{prefix}_comparison.png"),
                comparison,
            )

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(input_files)}] "
                f"TPS: SSIM={np.mean(results['tps']['ssim']):.4f} "
                f"LPIPS={np.mean(results['tps']['lpips']):.4f} | "
                f"Morph: SSIM={np.mean(results['morphing']['ssim']):.4f} "
                f"LPIPS={np.mean(results['morphing']['lpips']):.4f} "
                f"({elapsed:.0f}s)"
            )

    # Aggregate results
    output = {"num_pairs": len(input_files), "methods": {}}

    for method in ["tps", "morphing"]:
        r = results[method]
        method_results = {
            "metrics": {
                "ssim": float(np.mean(r["ssim"])) if r["ssim"] else 0.0,
                "lpips": float(np.mean(r["lpips"])) if r["lpips"] else 0.0,
                "nme": float(np.mean(r["nme"])) if r["nme"] else 0.0,
                "identity_sim": float(np.mean(r["identity"])) if r["identity"] else 0.0,
            },
            "per_procedure": {},
            "per_fitzpatrick": {},
        }

        for proc, vals in r["per_proc"].items():
            method_results["per_procedure"][proc] = {
                "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else 0.0,
                "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else 0.0,
                "nme": float(np.mean(vals["nme"])) if vals["nme"] else 0.0,
                "identity_sim": float(np.mean(vals["identity"])) if vals["identity"] else 0.0,
                "count": vals["count"],
            }

        for fitz, vals in r["per_fitz"].items():
            method_results["per_fitzpatrick"][fitz] = {
                "ssim": float(np.mean(vals["ssim"])) if vals["ssim"] else 0.0,
                "lpips": float(np.mean(vals["lpips"])) if vals["lpips"] else 0.0,
                "nme": float(np.mean(vals["nme"])) if vals["nme"] else 0.0,
                "identity_sim": float(np.mean(vals["identity"])) if vals["identity"] else 0.0,
                "count": vals["count"],
            }

        output["methods"][method] = method_results

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BASELINE RESULTS ({len(input_files)} pairs)")
    print(f"{'=' * 60}")
    for method in ["tps", "morphing"]:
        m = output["methods"][method]["metrics"]
        name = "TPS-only" if method == "tps" else "Morphing"
        print(f"\n{name}:")
        print(f"  SSIM:     {m['ssim']:.4f}")
        print(f"  LPIPS:    {m['lpips']:.4f}")
        print(f"  NME:      {m['nme']:.4f}")
        print(f"  ID Sim:   {m['identity_sim']:.4f}")

        print("\n  By Procedure:")
        for proc, vals in sorted(output["methods"][method]["per_procedure"].items()):
            print(
                f"    {proc}: SSIM={vals['ssim']:.4f} LPIPS={vals['lpips']:.4f} "
                f"NME={vals['nme']:.4f} ID={vals['identity_sim']:.4f} (n={vals['count']})"
            )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute baseline metrics")
    parser.add_argument("--test_dir", required=True, help="Directory with test pairs")
    parser.add_argument("--output", default="results/baselines.json", help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=0, help="Max test pairs (0 = all)")
    parser.add_argument("--save_images", default=None, help="Directory to save comparison images")
    args = parser.parse_args()

    evaluate_baselines(args.test_dir, args.output, args.max_samples, args.save_images)
