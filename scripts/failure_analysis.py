"""Failure case analysis for LandmarkDiff predictions.

Identifies the worst-performing predictions and generates a diagnostic
gallery with per-sample metrics, conditioning overlays, and error maps.
This helps understand failure modes for the Discussion section.

Usage:
    python scripts/failure_analysis.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/failure_analysis/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Error map visualization
# ──────────────────────────────────────────────────────────────────────────────


def compute_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute per-pixel absolute error map, colorized with JET colormap.

    Returns a BGR image of the same size as input, where red = high error,
    blue = low error.
    """
    # Convert to float and compute absolute error per channel
    diff = np.abs(pred.astype(float) - target.astype(float))
    # Average across channels for a single-channel error
    error = diff.mean(axis=2).astype(np.float32)
    # Normalize to 0-255
    max_err = error.max() if error.max() > 0 else 1.0
    error_norm = (error / max_err * 255).astype(np.uint8)
    # Apply colormap
    return cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)


def compute_structural_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute SSIM error map (local structural dissimilarity).

    Returns a single-channel float array where higher = more structural error.
    """
    from skimage.metrics import structural_similarity

    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    _, ssim_map = structural_similarity(pred_gray, target_gray, full=True)
    # Convert dissimilarity to error (1 - SSIM)
    error = 1.0 - ssim_map.astype(np.float32)
    error_norm = (np.clip(error, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)


# ──────────────────────────────────────────────────────────────────────────────
# Landmark error visualization
# ──────────────────────────────────────────────────────────────────────────────


def draw_landmark_errors(
    img: np.ndarray,
    pred_lm: np.ndarray,
    target_lm: np.ndarray,
    iod: float,
) -> np.ndarray:
    """Draw arrows from predicted to target landmark positions.

    Color encodes error magnitude: green = small error, red = large error.
    """
    vis = img.copy()
    errors = np.linalg.norm(pred_lm - target_lm, axis=1) / iod
    max_err = max(errors.max(), 0.01)

    for i in range(len(pred_lm)):
        # Color: green → yellow → red based on error
        ratio = min(errors[i] / max_err, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR

        p1 = tuple(pred_lm[i].astype(int))
        p2 = tuple(target_lm[i].astype(int))

        # Only draw significant errors (top 50% of landmark errors)
        if errors[i] > np.median(errors):
            cv2.arrowedLine(vis, p1, p2, color, 1, tipLength=0.3)
        else:
            cv2.circle(vis, p1, 2, (0, 255, 0), -1)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Failure classification
# ──────────────────────────────────────────────────────────────────────────────

FAILURE_CATEGORIES = {
    "identity_collapse": "ArcFace similarity < 0.2 — model generates a different person",
    "geometric_error": "NME > 0.1 — landmarks significantly misaligned",
    "texture_artifact": "LPIPS > 0.6 — severe perceptual quality degradation",
    "structural_mismatch": "SSIM < 0.3 — overall structural correspondence lost",
    "color_shift": "LAB color distance > 30 — skin tone mismatch after compositing",
}


def classify_failure(metrics: dict) -> list[str]:
    """Classify a sample's failure mode(s) based on metric thresholds."""
    categories = []
    if metrics.get("identity_sim", 1.0) < 0.2:
        categories.append("identity_collapse")
    if metrics.get("nme", 0.0) > 0.1:
        categories.append("geometric_error")
    if metrics.get("lpips", 0.0) > 0.6:
        categories.append("texture_artifact")
    if metrics.get("ssim", 1.0) < 0.3:
        categories.append("structural_mismatch")
    if metrics.get("color_dist", 0.0) > 30:
        categories.append("color_shift")
    return categories


def compute_color_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean LAB color distance in the facial region."""
    pred_lab = cv2.cvtColor(pred, cv2.COLOR_BGR2LAB).astype(float)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)

    # Rough face region (center crop)
    h, w = pred.shape[:2]
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4

    pred_crop = pred_lab[y1:y2, x1:x2]
    target_crop = target_lab[y1:y2, x1:x2]

    dist = np.sqrt(np.sum((pred_crop - target_crop) ** 2, axis=2))
    return float(dist.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────


def analyze_failures(
    checkpoint: str,
    test_dir: str,
    output_dir: str,
    num_steps: int = 20,
    seed: int = 42,
    top_k: int = 10,
) -> dict:
    """Run full failure analysis on test set.

    Generates predictions, computes metrics, identifies worst cases,
    and creates diagnostic visualizations.
    """
    from landmarkdiff.evaluation import (
        compute_identity_similarity,
        compute_lpips,
        compute_ssim,
    )
    from landmarkdiff.landmarks import extract_landmarks
    from scripts.evaluate_checkpoint import (
        compute_nme,
        generate_from_checkpoint,
        load_test_pairs,
    )

    test_pairs = load_test_pairs(Path(test_dir))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(test_pairs)} test pairs for failure cases...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {out}")
    print()

    # ── Evaluate all pairs ───────────────────────────────────────────────
    all_samples = []

    for i, pair in enumerate(test_pairs):
        cond_img = cv2.imread(pair["conditioning"])
        target_img = cv2.imread(pair["target"])
        input_img = cv2.imread(pair["input"])

        if cond_img is None or target_img is None or input_img is None:
            continue

        cond_img = cv2.resize(cond_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))
        input_img = cv2.resize(input_img, (512, 512))

        # Generate prediction
        pred_img = generate_from_checkpoint(
            checkpoint,
            cond_img,
            num_steps=num_steps,
            seed=seed,
        )

        # Compute all metrics
        ssim = compute_ssim(pred_img, target_img)
        lpips = compute_lpips(pred_img, target_img)
        nme_val = compute_nme(pred_img, target_img)
        id_sim = compute_identity_similarity(pred_img, target_img)
        color_dist = compute_color_distance(pred_img, target_img)

        metrics = {
            "ssim": ssim,
            "lpips": lpips,
            "nme": nme_val if nme_val is not None else float("nan"),
            "identity_sim": id_sim,
            "color_dist": color_dist,
        }

        # Classify failure mode(s)
        failures = classify_failure(metrics)

        # Composite quality score (lower = worse)
        # Weighted combination: high SSIM good, low LPIPS good, high ArcFace good
        quality = 0.3 * ssim + 0.3 * (1 - lpips) + 0.4 * max(id_sim, 0)

        sample = {
            "index": i,
            "prefix": pair["prefix"],
            "procedure": pair["procedure"],
            "metrics": metrics,
            "quality_score": quality,
            "failure_categories": failures,
            "paths": {
                "input": pair["input"],
                "target": pair["target"],
                "conditioning": pair["conditioning"],
            },
            "_images": {
                "input": input_img,
                "target": target_img,
                "prediction": pred_img,
                "conditioning": cond_img,
            },
        }
        all_samples.append(sample)

        if (i + 1) % 10 == 0:
            n_fail = sum(1 for s in all_samples if s["failure_categories"])
            print(f"  [{i + 1}/{len(test_pairs)}] quality={quality:.3f} failures_so_far={n_fail}")

    # ── Identify worst cases ─────────────────────────────────────────────
    sorted_samples = sorted(all_samples, key=lambda s: s["quality_score"])
    worst = sorted_samples[:top_k]
    best = sorted_samples[-top_k:]

    print(f"\n{'=' * 60}")
    print("FAILURE ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    print(f"Total pairs: {len(all_samples)}")
    n_failures = sum(1 for s in all_samples if s["failure_categories"])
    print(f"Samples with failures: {n_failures}/{len(all_samples)}")

    # Count by category
    cat_counts = defaultdict(int)
    for s in all_samples:
        for cat in s["failure_categories"]:
            cat_counts[cat] += 1

    print("\nFailure category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({FAILURE_CATEGORIES[cat]})")

    # ── Generate diagnostic gallery for worst cases ──────────────────────
    print(f"\nGenerating diagnostic gallery for top-{top_k} worst cases...")

    for rank, sample in enumerate(worst):
        imgs = sample["_images"]
        m = sample["metrics"]

        # Row 1: input | conditioning | prediction | target
        row1 = np.hstack(
            [
                imgs["input"],
                imgs["conditioning"],
                imgs["prediction"],
                imgs["target"],
            ]
        )

        # Row 2: error map | structural error | landmark errors | blank
        error_map = compute_error_map(imgs["prediction"], imgs["target"])
        struct_error = compute_structural_error(imgs["prediction"], imgs["target"])

        # Landmark error visualization
        pred_lm = extract_landmarks(imgs["prediction"])
        target_lm = extract_landmarks(imgs["target"])
        if pred_lm is not None and target_lm is not None:
            left_eye = target_lm.pixel_coords[33]
            right_eye = target_lm.pixel_coords[263]
            iod = max(np.linalg.norm(left_eye - right_eye), 1.0)
            lm_vis = draw_landmark_errors(
                imgs["prediction"],
                pred_lm.pixel_coords,
                target_lm.pixel_coords,
                iod,
            )
        else:
            lm_vis = np.zeros_like(imgs["prediction"])

        # Info panel
        info = np.zeros((512, 512, 3), dtype=np.uint8)
        text_lines = [
            f"Rank: {rank + 1} / {len(all_samples)}",
            f"Procedure: {sample['procedure']}",
            f"Quality: {sample['quality_score']:.3f}",
            "",
            f"SSIM: {m['ssim']:.4f}",
            f"LPIPS: {m['lpips']:.4f}",
            f"NME: {m['nme']:.4f}",
            f"ArcFace: {m['identity_sim']:.4f}",
            f"Color dist: {m['color_dist']:.1f}",
            "",
            "Failures:",
        ]
        for cat in sample["failure_categories"]:
            text_lines.append(f"  - {cat}")
        if not sample["failure_categories"]:
            text_lines.append("  (none)")

        for j, line in enumerate(text_lines):
            cv2.putText(
                info, line, (10, 25 + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        row2 = np.hstack([error_map, struct_error, lm_vis, info])

        # Stack rows
        gallery = np.vstack([row1, row2])

        # Add column headers
        header_h = 30
        header = np.zeros((header_h, gallery.shape[1], 3), dtype=np.uint8)
        col_labels = [
            "Input",
            "Conditioning",
            "Prediction",
            "Target",
            "Pixel Error",
            "SSIM Error",
            "Landmark Error",
            "Metrics",
        ]
        col_w = 512
        for j, label in enumerate(col_labels):
            x = j * col_w + 10
            0 if j < 4 else header_h // 2
            cv2.putText(header, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        full = np.vstack([header, gallery])

        fname = f"worst_{rank + 1:02d}_{sample['procedure']}_{sample['prefix']}.png"
        cv2.imwrite(str(out / fname), full)
        print(f"  Saved: {fname} (quality={sample['quality_score']:.3f})")

    # ── Generate best cases for comparison ───────────────────────────────
    print(f"\nGenerating gallery for top-{min(top_k, 5)} best cases...")
    for rank, sample in enumerate(best[:5]):
        imgs = sample["_images"]
        row = np.hstack(
            [
                imgs["input"],
                imgs["conditioning"],
                imgs["prediction"],
                imgs["target"],
            ]
        )
        fname = f"best_{rank + 1:02d}_{sample['procedure']}_{sample['prefix']}.png"
        cv2.imwrite(str(out / fname), row)

    # ── Save JSON report ─────────────────────────────────────────────────
    # Strip image data for JSON serialization
    report = {
        "total_pairs": len(all_samples),
        "n_failures": n_failures,
        "failure_categories": dict(cat_counts),
        "category_descriptions": FAILURE_CATEGORIES,
        "quality_distribution": {
            "mean": float(np.mean([s["quality_score"] for s in all_samples])),
            "std": float(np.std([s["quality_score"] for s in all_samples])),
            "min": float(min(s["quality_score"] for s in all_samples)),
            "max": float(max(s["quality_score"] for s in all_samples)),
        },
        "per_procedure": {},
        "worst_cases": [],
        "best_cases": [],
    }

    # Per-procedure quality
    proc_scores = defaultdict(list)
    for s in all_samples:
        proc_scores[s["procedure"]].append(s["quality_score"])
    for proc, scores in sorted(proc_scores.items()):
        report["per_procedure"][proc] = {
            "mean_quality": float(np.mean(scores)),
            "n": len(scores),
            "n_failures": sum(
                1 for s in all_samples if s["procedure"] == proc and s["failure_categories"]
            ),
        }

    # Record worst/best without image data
    for s in worst:
        entry = {k: v for k, v in s.items() if k != "_images"}
        report["worst_cases"].append(entry)
    for s in best[:5]:
        entry = {k: v for k, v in s.items() if k != "_images"}
        report["best_cases"].append(entry)

    report_path = out / "failure_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Failure case analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="paper/failure_analysis")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of worst/best cases to visualize"
    )
    args = parser.parse_args()

    analyze_failures(
        checkpoint=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output,
        num_steps=args.num_steps,
        seed=args.seed,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
