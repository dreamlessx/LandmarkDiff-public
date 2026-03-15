"""Ablation study for paper Table 2: conditioning channels and compositing.

Evaluates ablation variants on the HDA test set using a single checkpoint.
Ablations are applied at inference time (no retraining needed).

Variants:
  1. Conditioning: mesh only / mesh+canny / mesh+canny+mask (full)
  2. Compositing: direct output / mask composite / mask+LAB color (full)
  3. Clinical augmentation: compared via Phase A vs augmented Phase A

Usage:
    python scripts/run_table2_ablation.py \
        --checkpoint checkpoints_v2/final \
        --data-dir data/hda_splits/test \
        --output paper/ablation_results.json \
        --procedure rhinoplasty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.evaluation import (
    compute_identity_similarity,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.inference import LandmarkDiffPipeline
from landmarkdiff.landmarks import extract_landmarks


def evaluate_pair(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute evaluation metrics for a single prediction-target pair."""
    metrics = {}
    metrics["ssim"] = float(compute_ssim(pred, target))
    metrics["lpips"] = float(compute_lpips(pred, target))

    face_pred = extract_landmarks(pred)
    face_target = extract_landmarks(target)
    if face_pred is not None and face_target is not None:
        metrics["nme"] = float(compute_nme(face_pred.pixel_coords, face_target.pixel_coords))
    else:
        metrics["nme"] = float("nan")

    metrics["identity_sim"] = float(compute_identity_similarity(pred, target))
    return metrics


def run_ablation(
    checkpoint: str,
    data_dir: str,
    procedure: str,
    num_steps: int = 20,
    seed: int = 42,
) -> dict:
    """Run all ablation variants and return results."""

    data_path = Path(data_dir)
    pairs = sorted(data_path.glob("*_input.png"))

    # Filter by procedure if specified
    if procedure != "all":
        pairs = [p for p in pairs if procedure in p.stem]

    if not pairs:
        print(f"No test pairs found for procedure={procedure} in {data_dir}")
        return {}

    print(f"Evaluating {len(pairs)} pairs for {procedure}")

    # Load pipeline
    pipeline = LandmarkDiffPipeline(
        mode="controlnet",
        controlnet_checkpoint=checkpoint,
    )
    pipeline.load()

    results = {}

    # Run inference once per pair, then compute compositing variants offline
    print(f"\nRunning inference on {len(pairs)} pairs...")
    inference_cache = []

    for i, pair_path in enumerate(pairs):
        stem = pair_path.stem.replace("_input", "")
        target_path = data_path / f"{stem}_target.png"

        if not target_path.exists():
            continue

        input_img = cv2.imread(str(pair_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.imread(str(target_path))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        result = pipeline.generate(
            image=input_img,
            procedure=procedure if procedure != "all" else "rhinoplasty",
            num_inference_steps=num_steps,
            seed=seed,
            postprocess=False,
        )

        raw = result.get("output_raw")
        mask = result.get("mask")
        inp = result.get("input")
        composited = result.get("output")

        if raw is None:
            continue

        # Validate mask
        if mask is None:
            print(f"  WARNING: No mask returned for {stem}, skipping")
            continue
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
        mask_area = np.sum(mask_f > 0.3)
        if mask_area < 100:
            print(f"  WARNING: Tiny mask for {stem} ({mask_area} px > 0.3)")

        inference_cache.append(
            {
                "raw": raw,
                "mask": mask,
                "input": inp,
                "composited": composited,
                "target": target_img,
            }
        )
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(pairs)}] {stem} "
                f"(mask area={mask_area}, range={mask_f.min():.2f}-{mask_f.max():.2f})"
            )

    print(f"\nCached {len(inference_cache)} inference results")

    # Compositing ablation: reuse cached raw outputs
    from landmarkdiff.inference import mask_composite, mask_to_3channel

    variant_outputs = {}
    for variant_name in ["direct_output", "mask_no_color", "mask_lab_full"]:
        print(f"\n--- Compositing: {variant_name} ---")
        all_metrics = []
        first_pred = None

        for item in inference_cache:
            raw = item["raw"]
            mask = item["mask"]
            inp = item["input"]
            target = item["target"]

            if variant_name == "direct_output":
                pred = raw
            elif variant_name == "mask_no_color":
                # Alpha blend without LAB color matching
                mask_f = mask.astype(np.float32)
                if mask_f.max() > 1.0:
                    mask_f = mask_f / 255.0
                mask_3ch = mask_to_3channel(mask_f)
                pred = (
                    (raw.astype(np.float32) * mask_3ch + inp.astype(np.float32) * (1.0 - mask_3ch))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            else:
                # Full pipeline: mask + LAB color match
                pred = mask_composite(raw, inp, mask)

            if first_pred is None:
                first_pred = pred.copy()

            metrics = evaluate_pair(pred, target)
            all_metrics.append(metrics)

        # Store first output for cross-variant comparison
        variant_outputs[variant_name] = first_pred

        if all_metrics:
            avg = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
                if vals:
                    avg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            results[variant_name] = avg
            print(
                f"  LPIPS={avg.get('lpips', {}).get('mean', 0):.3f} "
                f"NME={avg.get('nme', {}).get('mean', 0):.3f} "
                f"ArcFace={avg.get('identity_sim', {}).get('mean', 0):.3f}"
            )

    # Sanity check: verify compositing variants produce different outputs
    if len(variant_outputs) >= 2:
        names = list(variant_outputs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = variant_outputs[names[i]], variant_outputs[names[j]]
                if a is not None and b is not None:
                    diff = np.mean(np.abs(a.astype(float) - b.astype(float)))
                    same = np.array_equal(a, b)
                    print(f"\n  CHECK: {names[i]} vs {names[j]}: MAE={diff:.2f}, identical={same}")
                    if same:
                        print(f"  BUG: {names[i]} and {names[j]} are byte-identical!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Table 2 ablation study")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--data-dir", required=True, help="Test data directory")
    parser.add_argument("--output", default="paper/ablation_results.json")
    parser.add_argument("--procedure", default="rhinoplasty")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_ablation(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        procedure=args.procedure,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
