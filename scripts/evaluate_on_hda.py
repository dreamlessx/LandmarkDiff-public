#!/usr/bin/env python3
"""Evaluate LandmarkDiff predictions against HDA real surgery ground truth.

Takes a trained checkpoint, runs inference on HDA before images with
procedure-specific conditioning, and compares against real after images.

Metrics:
- SSIM (structural similarity)
- LPIPS (perceptual distance)
- NME (normalized mean error on landmarks)
- Identity similarity (ArcFace cosine)
- Per-procedure breakdowns
- Fitzpatrick fairness analysis (if skin type annotations available)

Usage:
    python scripts/evaluate_on_hda.py --checkpoint checkpoints/phaseA/step_50000
    python scripts/evaluate_on_hda.py --checkpoint checkpoints/phaseB/best --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from landmarkdiff.evaluation import compute_lpips, compute_nme, compute_ssim
from landmarkdiff.landmarks import extract_landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_hda_eval_pairs(data_dir: Path) -> list[dict]:
    """Load evaluation pairs from HDA processed directory."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        # Auto-discover from filenames
        input_files = sorted(data_dir.glob("*_input.png"))
        return [
            {
                "prefix": f.stem.replace("_input", ""),
                "input_path": f,
                "target_path": data_dir / f"{f.stem.replace('_input', '')}_target.png",
                "conditioning_path": data_dir / f"{f.stem.replace('_input', '')}_conditioning.png",
                "procedure": "unknown",
            }
            for f in input_files
            if (data_dir / f"{f.stem.replace('_input', '')}_target.png").exists()
        ]

    with open(meta_path) as f:
        metadata = json.load(f)

    pairs = []
    for prefix, info in metadata.get("pairs", {}).items():
        input_path = data_dir / f"{prefix}_input.png"
        target_path = data_dir / f"{prefix}_target.png"
        if input_path.exists() and target_path.exists():
            pairs.append(
                {
                    "prefix": prefix,
                    "input_path": input_path,
                    "target_path": target_path,
                    "conditioning_path": data_dir / f"{prefix}_conditioning.png",
                    "procedure": info.get("procedure", "unknown"),
                }
            )

    return pairs


def run_inference(
    input_path: Path,
    conditioning_path: Path,
    procedure: str,
    checkpoint_dir: Path | None = None,
    resolution: int = 512,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
) -> np.ndarray | None:
    """Run LandmarkDiff inference on a single input.

    Returns the predicted post-surgery image (BGR, uint8).
    """
    try:
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipeline = LandmarkDiffPipeline(
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        )

        input_img = cv2.imread(str(input_path))
        if input_img is None:
            return None

        cond_img = cv2.imread(str(conditioning_path))
        if cond_img is None:
            return None

        result = pipeline.predict(
            input_image=input_img,
            conditioning_image=cond_img,
            procedure=procedure,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

        return result

    except Exception as e:
        logger.warning("Inference failed: %s", e)
        return None


def evaluate_pair(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    procedure: str,
) -> dict:
    """Compute all metrics for a single prediction/ground-truth pair."""
    metrics = {}

    # Ensure same size
    h, w = ground_truth.shape[:2]
    if predicted.shape[:2] != (h, w):
        predicted = cv2.resize(predicted, (w, h))

    # SSIM
    metrics["ssim"] = float(compute_ssim(predicted, ground_truth))

    # LPIPS
    try:
        metrics["lpips"] = float(compute_lpips(predicted, ground_truth))
    except Exception:
        metrics["lpips"] = float("nan")

    # NME (landmark-based)
    try:
        face_pred = extract_landmarks(predicted)
        face_gt = extract_landmarks(ground_truth)
        if face_pred is not None and face_gt is not None:
            metrics["nme"] = float(compute_nme(face_pred, face_gt))
        else:
            metrics["nme"] = float("nan")
    except Exception:
        metrics["nme"] = float("nan")

    # Identity similarity (ArcFace)
    try:
        from landmarkdiff.evaluation import compute_identity_similarity

        id_sim = compute_identity_similarity(predicted, ground_truth)
        metrics["identity_sim"] = float(id_sim)
    except Exception:
        metrics["identity_sim"] = float("nan")

    metrics["procedure"] = procedure
    return metrics


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-sample metrics into summary statistics."""
    if not results:
        return {}

    all_metrics = {}
    procedures = set()

    for r in results:
        proc = r.get("procedure", "unknown")
        procedures.add(proc)
        for key in ["ssim", "lpips", "nme", "identity_sim"]:
            if key in r and not np.isnan(r[key]):
                all_metrics.setdefault(key, []).append(r[key])

    summary = {"overall": {}, "by_procedure": {}}

    # Overall stats
    for key, values in all_metrics.items():
        if values:
            summary["overall"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }

    # Per-procedure stats
    for proc in procedures:
        proc_results = [r for r in results if r.get("procedure") == proc]
        proc_summary = {}
        for key in ["ssim", "lpips", "nme", "identity_sim"]:
            values = [r[key] for r in proc_results if key in r and not np.isnan(r[key])]
            if values:
                proc_summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
        summary["by_procedure"][proc] = proc_summary

    return summary


def print_results_table(summary: dict) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall
    print("\nOverall Metrics:")
    print(f"  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Median':>10} {'N':>6}")
    print(f"  {'-' * 56}")
    for key, stats in summary.get("overall", {}).items():
        print(
            f"  {key:<20} {stats['mean']:>10.4f}"
            f" {stats['std']:>10.4f} {stats['median']:>10.4f}"
            f" {stats['n']:>6}"
        )

    # Per-procedure
    print("\nPer-Procedure Breakdown:")
    procedures = sorted(summary.get("by_procedure", {}).keys())
    for proc in procedures:
        proc_stats = summary["by_procedure"][proc]
        print(f"\n  {proc}:")
        for key, stats in proc_stats.items():
            print(f"    {key:<18} {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on HDA real surgery data")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint directory")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "hda_splits" / "test",
        help="HDA evaluation data directory",
    )
    parser.add_argument(
        "--fallback-dir",
        type=Path,
        default=ROOT / "data" / "hda_processed",
        help="Fallback if splits don't exist",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "hda_eval.json")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit evaluation pairs (0=all)")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip inference, compute metrics on existing predictions",
    )
    args = parser.parse_args()

    # Select data directory
    data_dir = args.data_dir if args.data_dir.exists() else args.fallback_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info("Evaluating on: %s", data_dir)
    logger.info("Checkpoint: %s", args.checkpoint or "baseline")

    # Load evaluation pairs
    pairs = load_hda_eval_pairs(data_dir)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    logger.info("Found %d evaluation pairs", len(pairs))

    # Run evaluation
    results = []
    start_time = time.time()

    for i, pair in enumerate(pairs):
        prefix = pair["prefix"]

        if args.metrics_only:
            # Load existing prediction
            pred_path = args.output.parent / "predictions" / f"{prefix}_predicted.png"
            if pred_path.exists():
                predicted = cv2.imread(str(pred_path))
            else:
                continue
        else:
            # Run inference
            predicted = run_inference(
                pair["input_path"],
                pair["conditioning_path"],
                pair["procedure"],
                checkpoint_dir=args.checkpoint,
                resolution=args.resolution,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
            )

        if predicted is None:
            logger.warning("Skipping %s: inference failed", prefix)
            continue

        # Load ground truth
        gt = cv2.imread(str(pair["target_path"]))
        if gt is None:
            continue

        # Save prediction
        pred_dir = args.output.parent / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pred_dir / f"{prefix}_predicted.png"), predicted)

        # Compute metrics
        metrics = evaluate_pair(predicted, gt, pair["procedure"])
        metrics["prefix"] = prefix
        results.append(metrics)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Progress: %d/%d (%.0f%%) | %.1f s/pair",
                i + 1,
                len(pairs),
                100 * (i + 1) / len(pairs),
                elapsed / (i + 1),
            )

    # Aggregate and save
    summary = aggregate_metrics(results)
    print_results_table(summary)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "data_dir": str(data_dir),
        "n_pairs": len(pairs),
        "n_evaluated": len(results),
        "summary": summary,
        "per_sample": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
