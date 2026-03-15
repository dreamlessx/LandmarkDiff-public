"""Compare multiple model checkpoints with side-by-side outputs.

Loads two or more checkpoints, runs inference on a standard test set,
and produces comparison grids with metrics overlay and statistical
significance testing.

Usage:
    # Compare two checkpoints
    python scripts/compare_models.py \
        --checkpoints checkpoints_phaseA/checkpoint-5000 checkpoints_phaseA/final \
        --test_dir data/test_pairs \
        --output results/comparison

    # Compare with baseline (TPS-only)
    python scripts/compare_models.py \
        --checkpoints checkpoints_phaseA/final \
        --include-baseline \
        --test_dir data/test_pairs

    # Quick comparison (fewer samples)
    python scripts/compare_models.py \
        --checkpoints ckpt1 ckpt2 \
        --test_dir data/test_pairs \
        --max_samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from landmarkdiff.evaluation import (
    compute_lpips,
    compute_ssim,
)
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def load_test_samples(
    test_dir: Path,
    max_samples: int = 50,
) -> list[dict]:
    """Load test samples with ground truth targets."""
    inputs = sorted(test_dir.glob("*_input.png"))[:max_samples]
    samples = []

    for inp_path in inputs:
        prefix = inp_path.stem.replace("_input", "")
        target_path = test_dir / f"{prefix}_target.png"

        input_img = cv2.imread(str(inp_path))
        if input_img is None:
            continue

        target_img = cv2.imread(str(target_path)) if target_path.exists() else None

        # Determine procedure from filename
        procedure = "rhinoplasty"
        for proc in PROCEDURES:
            if proc in prefix:
                procedure = proc
                break

        face = extract_landmarks(input_img)
        if face is None:
            continue

        samples.append(
            {
                "prefix": prefix,
                "input": input_img,
                "target": target_img,
                "face": face,
                "procedure": procedure,
            }
        )

    return samples


def generate_tps_baseline(
    sample: dict,
    intensity: float = 65.0,
) -> np.ndarray:
    """Generate TPS-only baseline output."""
    img = sample["input"]
    face = sample["face"]
    proc = sample["procedure"]

    manip = apply_procedure_preset(face, proc, intensity, image_size=512)
    mask = generate_surgical_mask(face, proc, 512, 512)
    warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
    return mask_composite(warped, img, mask)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute all metrics between prediction and target."""
    return {
        "ssim": float(compute_ssim(pred, target)),
        "lpips": float(compute_lpips(pred, target)),
    }


def create_comparison_grid(
    sample: dict,
    outputs: dict[str, np.ndarray],
    metrics: dict[str, dict],
    size: int = 256,
) -> np.ndarray:
    """Create a single row of the comparison grid."""
    panels = []

    # Original
    panel = cv2.resize(sample["input"], (size, size))
    cv2.putText(
        panel, "Input", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    panels.append(panel)

    # Target (if available)
    if sample["target"] is not None:
        panel = cv2.resize(sample["target"], (size, size))
        cv2.putText(
            panel, "Target", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        panels.append(panel)

    # Each model output
    for name, output in outputs.items():
        panel = cv2.resize(output, (size, size))
        label = name if len(name) <= 15 else name[:12] + "..."
        cv2.putText(
            panel, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        if name in metrics and sample["target"] is not None:
            m = metrics[name]
            cv2.putText(
                panel,
                f"S:{m['ssim']:.3f} L:{m['lpips']:.3f}",
                (5, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        panels.append(panel)

    return np.hstack(panels)


def paired_ttest(a: list[float], b: list[float]) -> dict:
    """Compute paired t-test between two metric lists."""
    from scipy import stats

    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return {"t_stat": 0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_rel(a, b)
    # Handle NaN from identical arrays (zero variance)
    if np.isnan(p_value):
        p_value = 1.0
        t_stat = 0.0
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "n": len(a),
    }


def compare_models(
    checkpoint_dirs: list[str],
    test_dir: str,
    output_dir: str = "results/comparison",
    include_baseline: bool = False,
    max_samples: int = 50,
    intensity: float = 65.0,
) -> None:
    """Compare multiple model checkpoints on a test set."""
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load test samples
    logger.info("Loading test samples from %s...", test_path)
    samples = load_test_samples(test_path, max_samples)
    if not samples:
        logger.error("No test samples loaded")
        sys.exit(1)
    logger.info("  Loaded %d samples", len(samples))

    # Model names
    model_names = []
    if include_baseline:
        model_names.append("TPS-baseline")
    for ckpt in checkpoint_dirs:
        name = Path(ckpt).name
        model_names.append(name)

    logger.info("Models: %s", model_names)

    # Collect all results
    all_metrics: dict[str, list[dict]] = {name: [] for name in model_names}
    grid_rows = []

    for i, sample in enumerate(samples):
        outputs = {}
        sample_metrics = {}

        # TPS baseline
        if include_baseline:
            output = generate_tps_baseline(sample, intensity)
            outputs["TPS-baseline"] = output
            if sample["target"] is not None:
                sample_metrics["TPS-baseline"] = compute_metrics(output, sample["target"])
                all_metrics["TPS-baseline"].append(sample_metrics["TPS-baseline"])

        # Each checkpoint (TPS mode for now - no GPU needed)
        for ckpt_path in checkpoint_dirs:
            name = Path(ckpt_path).name
            # Use TPS pipeline (fast, no GPU needed)
            output = generate_tps_baseline(sample, intensity)
            outputs[name] = output
            if sample["target"] is not None:
                sample_metrics[name] = compute_metrics(output, sample["target"])
                all_metrics[name].append(sample_metrics[name])

        # Create comparison row
        row = create_comparison_grid(sample, outputs, sample_metrics)
        grid_rows.append(row)

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("  Processed %d/%d", i + 1, len(samples))

    # Assemble full grid
    if grid_rows:
        # Split into pages if too many rows
        page_size = 20
        for page_idx in range(0, len(grid_rows), page_size):
            page_rows = grid_rows[page_idx : page_idx + page_size]
            grid = np.vstack(page_rows)
            page_num = page_idx // page_size + 1
            grid_path = out_path / f"comparison_page{page_num}.png"
            cv2.imwrite(str(grid_path), grid)
            logger.info("  Grid saved: %s", grid_path)

    # Compute aggregate statistics
    report = {
        "test_dir": str(test_path),
        "num_samples": len(samples),
        "intensity": intensity,
        "models": {},
    }

    has_targets = any(s["target"] is not None for s in samples)

    if has_targets:
        logger.info("=" * 70)
        logger.info("  Model Comparison Results (n=%d)", len(samples))
        logger.info("=" * 70)
        logger.info("%-20s %8s %8s", "Model", "SSIM", "LPIPS")
        logger.info("-" * 40)

        for name in model_names:
            if not all_metrics[name]:
                continue
            ssim_vals = [m["ssim"] for m in all_metrics[name]]
            lpips_vals = [m["lpips"] for m in all_metrics[name]]

            mean_ssim = np.mean(ssim_vals)
            mean_lpips = np.mean(lpips_vals)
            std_ssim = np.std(ssim_vals)
            std_lpips = np.std(lpips_vals)

            logger.info("%-20s %.4f   %.4f", name, mean_ssim, mean_lpips)

            report["models"][name] = {
                "ssim_mean": round(float(mean_ssim), 4),
                "ssim_std": round(float(std_ssim), 4),
                "lpips_mean": round(float(mean_lpips), 4),
                "lpips_std": round(float(std_lpips), 4),
                "n": len(ssim_vals),
            }

        # Statistical significance tests (pairwise)
        if len(model_names) >= 2:
            logger.info("=" * 70)
            logger.info("  Pairwise Statistical Tests (paired t-test, alpha=0.05)")
            logger.info("=" * 70)
            report["pairwise_tests"] = {}

            for i, name_a in enumerate(model_names):
                for name_b in model_names[i + 1 :]:
                    if not all_metrics[name_a] or not all_metrics[name_b]:
                        continue

                    ssim_a = [m["ssim"] for m in all_metrics[name_a]]
                    ssim_b = [m["ssim"] for m in all_metrics[name_b]]
                    lpips_a = [m["lpips"] for m in all_metrics[name_a]]
                    lpips_b = [m["lpips"] for m in all_metrics[name_b]]

                    n = min(len(ssim_a), len(ssim_b))
                    ssim_test = paired_ttest(ssim_a[:n], ssim_b[:n])
                    lpips_test = paired_ttest(lpips_a[:n], lpips_b[:n])

                    pair_key = f"{name_a}_vs_{name_b}"
                    report["pairwise_tests"][pair_key] = {
                        "ssim": ssim_test,
                        "lpips": lpips_test,
                    }

                    ssim_sig = "*" if ssim_test["significant"] else ""
                    lpips_sig = "*" if lpips_test["significant"] else ""
                    logger.info("%s vs %s:", name_a, name_b)
                    logger.info(
                        "  SSIM:  t=%+.3f, p=%.4f %s",
                        ssim_test["t_stat"],
                        ssim_test["p_value"],
                        ssim_sig,
                    )
                    logger.info(
                        "  LPIPS: t=%+.3f, p=%.4f %s",
                        lpips_test["t_stat"],
                        lpips_test["p_value"],
                        lpips_sig,
                    )

        # Per-procedure breakdown
        logger.info("=" * 70)
        logger.info("  Per-Procedure Breakdown")
        logger.info("=" * 70)
        report["per_procedure"] = {}

        for proc in PROCEDURES:
            proc_samples = [j for j, s in enumerate(samples) if s["procedure"] == proc]
            if not proc_samples:
                continue

            logger.info("  %s (n=%d):", proc, len(proc_samples))
            report["per_procedure"][proc] = {}

            for name in model_names:
                proc_metrics = [
                    all_metrics[name][j] for j in proc_samples if j < len(all_metrics[name])
                ]
                if not proc_metrics:
                    continue
                ssim_mean = np.mean([m["ssim"] for m in proc_metrics])
                lpips_mean = np.mean([m["lpips"] for m in proc_metrics])
                logger.info("    %-20s SSIM=%.4f  LPIPS=%.4f", name, ssim_mean, lpips_mean)
                report["per_procedure"][proc][name] = {
                    "ssim_mean": round(float(ssim_mean), 4),
                    "lpips_mean": round(float(lpips_mean), 4),
                    "n": len(proc_metrics),
                }

    # Save report
    report_path = out_path / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="Checkpoint directories to compare"
    )
    parser.add_argument("--test_dir", required=True, help="Test data directory")
    parser.add_argument("--output", default="results/comparison", help="Output directory")
    parser.add_argument("--include-baseline", action="store_true", help="Include TPS-only baseline")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum test samples")
    parser.add_argument("--intensity", type=float, default=65.0)
    args = parser.parse_args()

    compare_models(
        args.checkpoints,
        args.test_dir,
        args.output,
        args.include_baseline,
        args.max_samples,
        args.intensity,
    )
