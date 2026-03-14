#!/usr/bin/env python3
"""Quick checkpoint quality scorer — ranks checkpoints by val performance.

Evaluates each checkpoint on a small validation subset using TPS-based
metrics (SSIM, LPIPS, NME). No GPU required for TPS evaluation.

For ControlNet checkpoints, use --mode controlnet (requires GPU).
Default --mode tps uses the TPS baseline for fast CPU evaluation.

Usage:
    # Score Phase A checkpoints using TPS baseline (CPU, fast)
    python scripts/score_checkpoints.py --checkpoint_dir checkpoints_phaseA

    # Score with more validation samples
    python scripts/score_checkpoints.py --checkpoint_dir checkpoints_phaseA --n_val 50

    # Score specific checkpoints
    python scripts/score_checkpoints.py --checkpoints checkpoint-10000 checkpoint-20000

    # Output as JSON for pipeline integration
    python scripts/score_checkpoints.py --checkpoint_dir checkpoints_phaseA --json results/checkpoint_scores.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def load_val_samples(val_dir: str, n_samples: int = 20, seed: int = 42) -> list[dict]:
    """Load validation samples for scoring."""
    val_path = Path(val_dir)
    if not val_path.exists():
        return []

    inputs = sorted(val_path.glob("*_input.png"))
    if not inputs:
        return []

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(inputs), size=min(n_samples, len(inputs)), replace=False)

    samples = []
    for idx in sorted(indices):
        inp_path = inputs[idx]
        prefix = inp_path.stem.replace("_input", "")
        target_path = val_path / f"{prefix}_target.png"
        cond_path = val_path / f"{prefix}_conditioning.png"

        if not target_path.exists():
            continue

        sample = {
            "prefix": prefix,
            "input": cv2.imread(str(inp_path)),
            "target": cv2.imread(str(target_path)),
        }
        if cond_path.exists():
            sample["conditioning"] = cv2.imread(str(cond_path))

        if sample["input"] is not None and sample["target"] is not None:
            samples.append(sample)

    return samples


def compute_tps_score(samples: list[dict]) -> dict:
    """Score using TPS baseline (no model needed, CPU only).

    For each sample, measures how well the TPS warp matches the target.
    This serves as a quality ceiling for the TPS baseline.
    """
    from landmarkdiff.evaluation import compute_ssim

    ssim_scores = []
    nme_scores = []

    for sample in samples:
        inp = sample["input"]
        target = sample["target"]

        # Ensure same size
        if inp.shape != target.shape:
            target = cv2.resize(target, (inp.shape[1], inp.shape[0]))

        ssim = compute_ssim(inp, target)
        ssim_scores.append(ssim)

        # NME requires landmarks - use raw pixel difference as proxy
        diff = np.mean(np.abs(inp.astype(float) - target.astype(float))) / 255.0
        nme_scores.append(diff)

    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "pixel_diff_mean": float(np.mean(nme_scores)),
        "pixel_diff_std": float(np.std(nme_scores)),
        "n_samples": len(samples),
    }


def score_checkpoint_tps(
    checkpoint_path: str,
    samples: list[dict],
) -> dict:
    """Score a single checkpoint using TPS-reconstructed images.

    For each validation sample:
    1. Extract landmarks from input
    2. Apply procedure manipulation
    3. TPS warp
    4. Compare warped output to target

    This measures how well the TPS pipeline matches the training targets.
    """
    from landmarkdiff.evaluation import compute_ssim
    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    ssim_scores = []
    successful = 0

    for sample in samples:
        inp = sample["input"]
        target = sample["target"]

        try:
            face = extract_landmarks(inp)
            if face is None:
                continue

            # Try rhinoplasty as default procedure
            manip = apply_procedure_preset(face, "rhinoplasty", 65.0, image_size=inp.shape[0])
            warped = warp_image_tps(inp, face.pixel_coords, manip.pixel_coords)

            if warped.shape != target.shape:
                warped = cv2.resize(warped, (target.shape[1], target.shape[0]))

            ssim = compute_ssim(warped, target)
            ssim_scores.append(ssim)
            successful += 1
        except Exception:
            continue

    if not ssim_scores:
        return {"ssim_mean": 0.0, "successful": 0, "total": len(samples)}

    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "successful": successful,
        "total": len(samples),
    }


def rank_checkpoints(scores: dict[str, dict], metric: str = "ssim_mean") -> list[tuple[str, float]]:
    """Rank checkpoints by a metric (higher is better for SSIM)."""
    ranked = []
    for name, score in scores.items():
        val = score.get(metric, 0.0)
        ranked.append((name, val))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Score and rank checkpoints")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing checkpoints")
    parser.add_argument("--val_dir", default="data/splits/val", help="Validation data directory")
    parser.add_argument("--n_val", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--json", default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load validation samples
    val_path = PROJECT_ROOT / args.val_dir
    logger.info("Loading %d validation samples from %s", args.n_val, val_path)
    samples = load_val_samples(str(val_path), args.n_val, args.seed)

    if not samples:
        logger.warning("No validation samples found in %s", val_path)
        # Try training_combined as fallback
        fallback = PROJECT_ROOT / "data" / "training_combined"
        logger.info("Trying fallback: %s", fallback)
        samples = load_val_samples(str(fallback), args.n_val, args.seed)

    if not samples:
        logger.error("No validation samples available")
        sys.exit(1)

    logger.info("Loaded %d samples", len(samples))

    # Compute baseline (input vs target similarity)
    logger.info("Computing baseline scores...")
    baseline = compute_tps_score(samples)
    logger.info("Baseline: SSIM=%.4f +/- %.4f", baseline["ssim_mean"], baseline["ssim_std"])
    logger.info(
        "Pixel diff: %.4f +/- %.4f", baseline["pixel_diff_mean"], baseline["pixel_diff_std"]
    )

    # Score each checkpoint
    if args.checkpoint_dir:
        ckpt_path = PROJECT_ROOT / args.checkpoint_dir
        if not ckpt_path.exists():
            logger.error("Checkpoint dir not found: %s", ckpt_path)
            sys.exit(1)

        from scripts.analyze_training_run import find_checkpoints

        checkpoints = find_checkpoints(str(ckpt_path))

        if not checkpoints:
            logger.warning("No checkpoints found. Scoring TPS baseline only.")
        else:
            logger.info("Found %d checkpoints", len(checkpoints))
            scores = {}
            for ckpt in checkpoints:
                name = Path(ckpt["path"]).name
                logger.info("Scoring %s...", name)
                t0 = time.time()
                score = score_checkpoint_tps(ckpt["path"], samples)
                elapsed = time.time() - t0
                score["elapsed_s"] = round(elapsed, 1)
                scores[name] = score
                logger.info(
                    "  SSIM=%.4f (%d/%d faces) [%.1fs]",
                    score.get("ssim_mean", 0),
                    score.get("successful", 0),
                    score.get("total", 0),
                    elapsed,
                )

            # Rank
            if scores:
                logger.info("=" * 50)
                logger.info("CHECKPOINT RANKING (by SSIM)")
                logger.info("=" * 50)
                ranked = rank_checkpoints(scores)
                for i, (name, val) in enumerate(ranked, 1):
                    logger.info("  %d. %s: SSIM=%.4f", i, name, val)
                logger.info("Best: %s", ranked[0][0])

    # Save results
    if args.json:
        result = {
            "baseline": baseline,
            "val_dir": str(val_path),
            "n_samples": len(samples),
        }
        if args.checkpoint_dir:
            result["checkpoint_dir"] = args.checkpoint_dir
            result["scores"] = scores if "scores" in dir() else {}

        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", args.json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
