#!/usr/bin/env python3
"""Run TPS-baseline ablation studies.

Component ablation on the TPS baseline (no trained model needed).
Generates results showing the contribution of each pipeline component.

Usage:
    python scripts/run_tps_ablation.py \
        --input data/celeba_hq_extracted \
        --output results/tps_ablation \
        --num-images 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import compute_lpips, compute_nme, compute_ssim
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
DEFAULT_INTENSITY = 65.0


def ablation_full(img, face, proc, intensity):
    manip = apply_procedure_preset(face, proc, intensity, image_size=512)
    mask = generate_surgical_mask(face, proc, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, img, mask)


def ablation_no_mask(img, face, proc, intensity):
    manip = apply_procedure_preset(face, proc, intensity, image_size=512)
    return warp_image_tps(img, face.pixel_coords, manip.pixel_coords)


def ablation_no_tps(img, face, proc, intensity):
    mask = generate_surgical_mask(face, proc, 512, 512)
    return mask_composite(img, img, mask)


def ablation_half_intensity(img, face, proc, intensity):
    manip = apply_procedure_preset(face, proc, intensity * 0.5, image_size=512)
    mask = generate_surgical_mask(face, proc, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, img, mask)


def ablation_double_intensity(img, face, proc, intensity):
    manip = apply_procedure_preset(face, proc, min(intensity * 2, 100), image_size=512)
    mask = generate_surgical_mask(face, proc, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, img, mask)


def ablation_gaussian_mask(img, face, proc, intensity):
    manip = apply_procedure_preset(face, proc, intensity, image_size=512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    h, w = img.shape[:2]
    cx = int(face.pixel_coords[1, 0])
    cy = int(face.pixel_coords[1, 1])
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 80**2)).astype(np.float32)
    return mask_composite(warped, img, mask)


ABLATIONS = {
    "Full pipeline": ablation_full,
    "w/o mask composite": ablation_no_mask,
    "w/o TPS warp": ablation_no_tps,
    "50% intensity": ablation_half_intensity,
    "200% intensity": ablation_double_intensity,
    "Gaussian mask": ablation_gaussian_mask,
}


def main():
    parser = argparse.ArgumentParser(description="TPS-baseline ablation study")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/tps_ablation")
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = sorted(f for f in Path(args.input).iterdir() if f.suffix.lower() in exts)
    rng.shuffle(all_files)

    images, faces = [], []
    for f in all_files:
        if len(images) >= args.num_images:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_512 = cv2.resize(img, (512, 512))
        face = extract_landmarks(img_512)
        if face is not None:
            images.append(img_512)
            faces.append(face)

    print(f"Loaded {len(images)} images")

    results = {}
    for abl_name, abl_fn in ABLATIONS.items():
        print(f"\n--- {abl_name} ---")
        ssim_all, lpips_all, nme_all = [], [], []

        for proc in PROCEDURES:
            for img, face in zip(images, faces):
                result = abl_fn(img, face, proc, DEFAULT_INTENSITY)
                if result is None:
                    continue
                ssim_all.append(compute_ssim(result, img))
                lpips_all.append(compute_lpips(result, img))
                manip = apply_procedure_preset(face, proc, DEFAULT_INTENSITY, image_size=512)
                nme_all.append(compute_nme(manip.pixel_coords, face.pixel_coords))

        results[abl_name] = {
            "ssim": float(np.mean(ssim_all)),
            "lpips": float(np.mean(lpips_all)),
            "nme": float(np.mean(nme_all)),
            "n": len(ssim_all),
        }
        print(
            f"  SSIM={results[abl_name]['ssim']:.4f} "
            f"LPIPS={results[abl_name]['lpips']:.4f} "
            f"NME={results[abl_name]['nme']:.4f}"
        )

    with open(output_dir / "tps_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Configuration':<25} {'SSIM':>8} {'LPIPS':>8} {'NME':>8}")
    print("-" * 55)
    for name, vals in results.items():
        print(f"{name:<25} {vals['ssim']:>8.4f} {vals['lpips']:>8.4f} {vals['nme']:>8.4f}")


if __name__ == "__main__":
    main()
