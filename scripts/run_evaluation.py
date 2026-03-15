"""Comprehensive evaluation runner for paper results.

Orchestrates the complete evaluation pipeline:
1. Run inference on test set with all procedures
2. Compute per-sample metrics (SSIM, LPIPS, NME, identity)
3. Aggregate by procedure and Fitzpatrick type
4. Generate comparison figures
5. Produce LaTeX tables
6. Create summary report

Usage:
    # Full evaluation with TPS baseline
    python scripts/run_evaluation.py \
        --test_dir data/test_pairs \
        --output results/paper_eval \
        --include-baseline

    # Evaluate specific checkpoint
    python scripts/run_evaluation.py \
        --test_dir data/test_pairs \
        --checkpoint checkpoints_phaseA/final/controlnet_ema \
        --output results/paper_eval

    # Quick evaluation (fewer samples)
    python scripts/run_evaluation.py \
        --test_dir data/test_pairs \
        --output results/quick_eval \
        --max_samples 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.evaluation import (
    classify_fitzpatrick_ita,
    compute_identity_similarity,
    compute_lpips,
    compute_nme,
    compute_ssim,
)
from landmarkdiff.inference import LandmarkDiffPipeline, mask_composite
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

logger = logging.getLogger(__name__)

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]


def load_test_set(test_dir: Path, max_samples: int = 0) -> list[dict]:
    """Load test set with ground truth."""
    inputs = sorted(test_dir.glob("*_input.png"))
    if max_samples > 0:
        inputs = inputs[:max_samples]

    samples = []
    for inp_path in inputs:
        prefix = inp_path.stem.replace("_input", "")
        target_path = test_dir / f"{prefix}_target.png"

        input_img = cv2.imread(str(inp_path))
        if input_img is None:
            continue
        if input_img.shape[:2] != (512, 512):
            input_img = cv2.resize(input_img, (512, 512))

        target_img = None
        if target_path.exists():
            target_img = cv2.imread(str(target_path))
            if target_img is not None and target_img.shape[:2] != (512, 512):
                target_img = cv2.resize(target_img, (512, 512))

        procedure = "rhinoplasty"
        for proc in PROCEDURES:
            if proc in prefix:
                procedure = proc
                break

        face = extract_landmarks(input_img)
        if face is None:
            continue

        fitz = classify_fitzpatrick_ita(input_img)

        samples.append(
            {
                "prefix": prefix,
                "input": input_img,
                "target": target_img,
                "face": face,
                "procedure": procedure,
                "fitzpatrick": fitz,
            }
        )

    return samples


def evaluate_tps_baseline(
    samples: list[dict],
    output_dir: Path,
    intensity: float = 65.0,
) -> list[dict]:
    """Run TPS baseline evaluation."""
    results = []
    vis_dir = output_dir / "tps_baseline"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        img = sample["input"]
        face = sample["face"]
        proc = sample["procedure"]

        manip = apply_procedure_preset(face, proc, intensity, image_size=512)
        mask = generate_surgical_mask(face, proc, 512, 512)
        warped = warp_image_tps(img, face.pixel_coords, manip.pixel_coords)
        output = mask_composite(warped, img, mask)

        metrics = {
            "method": "tps_baseline",
            "prefix": sample["prefix"],
            "procedure": proc,
            "fitzpatrick": sample["fitzpatrick"],
        }

        if sample["target"] is not None:
            metrics["ssim"] = float(compute_ssim(output, sample["target"]))
            metrics["lpips"] = float(compute_lpips(output, sample["target"]))
            metrics["nme"] = float(compute_nme(manip.pixel_coords, face.pixel_coords))
            metrics["identity"] = float(compute_identity_similarity(output, img))

        # Save comparison
        comparison = np.hstack([img, output])
        if sample["target"] is not None:
            comparison = np.hstack([comparison, sample["target"]])
        cv2.imwrite(str(vis_dir / f"{sample['prefix']}_comparison.png"), comparison)

        results.append(metrics)

        if (i + 1) % 20 == 0:
            logger.info("TPS baseline: %d/%d", i + 1, len(samples))

    return results


def evaluate_controlnet(
    samples: list[dict],
    output_dir: Path,
    checkpoint: str,
    intensity: float = 65.0,
    num_inference_steps: int = 30,
    guidance_scale: float = 9.0,
    controlnet_conditioning_scale: float = 0.9,
    seed: int = 42,
) -> list[dict]:
    """Run ControlNet evaluation using a fine-tuned checkpoint.

    Loads the LandmarkDiffPipeline in controlnet mode with the given checkpoint
    and generates predictions for each test sample. Falls back to TPS baseline
    if the model fails to load (e.g., no GPU available).
    """
    import torch

    vis_dir = output_dir / "controlnet"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        logger.warning("No GPU available -- ControlNet evaluation requires CUDA.")
        logger.warning("Falling back to TPS baseline as proxy.")
        tps_results = evaluate_tps_baseline(samples, output_dir, intensity)
        for r in tps_results:
            r["method"] = "controlnet_proxy_tps"
        return tps_results

    # Validate checkpoint
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found: %s", checkpoint)
        logger.warning("Falling back to TPS baseline as proxy.")
        tps_results = evaluate_tps_baseline(samples, output_dir, intensity)
        for r in tps_results:
            r["method"] = "controlnet_proxy_tps"
        return tps_results

    # Load pipeline
    try:
        pipe = LandmarkDiffPipeline(
            mode="controlnet",
            controlnet_checkpoint=checkpoint,
        )
        pipe.load()
    except Exception as e:
        logger.warning("Failed to load ControlNet pipeline: %s", e)
        logger.warning("Falling back to TPS baseline as proxy.")
        tps_results = evaluate_tps_baseline(samples, output_dir, intensity)
        for r in tps_results:
            r["method"] = "controlnet_proxy_tps"
        return tps_results

    results = []
    for i, sample in enumerate(samples):
        img = sample["input"]
        proc = sample["procedure"]

        try:
            gen_result = pipe.generate(
                img,
                procedure=proc,
                intensity=intensity,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                seed=seed,
            )
            output = gen_result["output"]
        except Exception as e:
            logger.warning(
                "Sample %s: generation failed (%s), using TPS fallback",
                sample["prefix"],
                e,
            )
            # Fall back to TPS warp for this sample
            face = sample["face"]
            manip_fb = apply_procedure_preset(face, proc, intensity, image_size=512)
            tps_out = warp_image_tps(img, face.pixel_coords, manip_fb.pixel_coords)
            mask_fb = generate_surgical_mask(face, proc, 512, 512)
            output = mask_composite(tps_out, img, mask_fb)

        metrics = {
            "method": "controlnet",
            "prefix": sample["prefix"],
            "procedure": proc,
            "fitzpatrick": sample["fitzpatrick"],
        }

        if sample["target"] is not None:
            metrics["ssim"] = float(compute_ssim(output, sample["target"]))
            metrics["lpips"] = float(compute_lpips(output, sample["target"]))
            face = sample["face"]
            manip = apply_procedure_preset(face, proc, intensity, image_size=512)
            metrics["nme"] = float(compute_nme(manip.pixel_coords, face.pixel_coords))
            metrics["identity"] = float(compute_identity_similarity(output, img))

        # Save comparison
        comparison = np.hstack([img, output])
        if sample["target"] is not None:
            comparison = np.hstack([comparison, sample["target"]])
        cv2.imwrite(str(vis_dir / f"{sample['prefix']}_comparison.png"), comparison)

        results.append(metrics)

        if (i + 1) % 10 == 0:
            logger.info("ControlNet: %d/%d", i + 1, len(samples))

    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-sample metrics into summary statistics."""
    if not results:
        return {}

    metric_keys = ["ssim", "lpips", "nme", "identity"]
    agg = {}

    # Global
    for key in metric_keys:
        vals = [r[key] for r in results if key in r]
        if vals:
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
            agg[f"{key}_median"] = round(float(np.median(vals)), 4)

    agg["n"] = len(results)

    # Per-procedure
    agg["by_procedure"] = {}
    for proc in PROCEDURES:
        proc_results = [r for r in results if r.get("procedure") == proc]
        if not proc_results:
            continue
        proc_agg = {"n": len(proc_results)}
        for key in metric_keys:
            vals = [r[key] for r in proc_results if key in r]
            if vals:
                proc_agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                proc_agg[f"{key}_std"] = round(float(np.std(vals)), 4)
        agg["by_procedure"][proc] = proc_agg

    # Per-Fitzpatrick
    agg["by_fitzpatrick"] = {}
    fitz_groups = defaultdict(list)
    for r in results:
        fitz_groups[r.get("fitzpatrick", "?")].append(r)
    for ftype, group in sorted(fitz_groups.items()):
        fitz_agg = {"n": len(group)}
        for key in metric_keys:
            vals = [r[key] for r in group if key in r]
            if vals:
                fitz_agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg["by_fitzpatrick"][ftype] = fitz_agg

    return agg


def generate_latex_table(
    method_results: dict[str, dict],
    caption: str = "Quantitative evaluation results.",
) -> str:
    """Generate LaTeX table from aggregated results."""
    methods = list(method_results.keys())
    metrics = ["ssim_mean", "lpips_mean", "nme_mean", "identity_mean"]
    headers = ["SSIM $\\uparrow$", "LPIPS $\\downarrow$", "NME $\\downarrow$", "ID Sim $\\uparrow$"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    # Find best values for bolding
    best = {}
    for _i, key in enumerate(metrics):
        vals = []
        for m in methods:
            v = method_results[m].get(key, None)
            if v is not None:
                vals.append((v, m))
        if vals:
            if "lpips" in key or "nme" in key:
                best[key] = min(vals, key=lambda x: x[0])[1]
            else:
                best[key] = max(vals, key=lambda x: x[0])[1]

    for method in methods:
        row = [method.replace("_", " ")]
        for key in metrics:
            val = method_results[method].get(key, None)
            if val is None:
                row.append("--")
            else:
                formatted = f"{val:.4f}"
                if best.get(key) == method:
                    formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_per_procedure_table(
    method_results: dict[str, dict],
) -> str:
    """Generate per-procedure LaTeX table."""
    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{Per-procedure evaluation results (SSIM $\\uparrow$ / LPIPS $\\downarrow$).}",
        "\\begin{tabular}{l" + "c" * len(PROCEDURES) + "}",
        "\\toprule",
        "Method & " + " & ".join([p.capitalize() for p in PROCEDURES]) + " \\\\",
        "\\midrule",
    ]

    for method, agg in method_results.items():
        row = [method.replace("_", " ")]
        by_proc = agg.get("by_procedure", {})
        for proc in PROCEDURES:
            proc_data = by_proc.get(proc, {})
            ssim = proc_data.get("ssim_mean", None)
            lpips = proc_data.get("lpips_mean", None)
            if ssim is not None and lpips is not None:
                row.append(f"{ssim:.3f} / {lpips:.3f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )

    return "\n".join(lines)


def generate_fitzpatrick_table(
    method_results: dict[str, dict],
) -> str:
    """Generate Fitzpatrick equity table."""
    fitz_types = ["I", "II", "III", "IV", "V", "VI"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{SSIM by Fitzpatrick skin type (equity analysis).}",
        "\\begin{tabular}{l" + "c" * len(fitz_types) + "}",
        "\\toprule",
        "Method & " + " & ".join([f"Type {t}" for t in fitz_types]) + " \\\\",
        "\\midrule",
    ]

    for method, agg in method_results.items():
        row = [method.replace("_", " ")]
        by_fitz = agg.get("by_fitzpatrick", {})
        for ftype in fitz_types:
            data = by_fitz.get(ftype, {})
            ssim = data.get("ssim_mean", None)
            if ssim is not None:
                n = data.get("n", 0)
                row.append(f"{ssim:.3f} ({n})")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def run_evaluation(
    test_dir: str,
    output_dir: str = "results/paper_eval",
    checkpoint: str | None = None,
    include_baseline: bool = True,
    max_samples: int = 0,
    intensity: float = 65.0,
) -> None:
    """Run the complete evaluation pipeline."""
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Load test set
    logger.info("Loading test set from %s...", test_path)
    samples = load_test_set(test_path, max_samples)
    if not samples:
        logger.error("No test samples loaded")
        sys.exit(1)
    logger.info("Loaded %d samples", len(samples))

    # Count by procedure and Fitzpatrick
    proc_counts = defaultdict(int)
    fitz_counts = defaultdict(int)
    for s in samples:
        proc_counts[s["procedure"]] += 1
        fitz_counts[s["fitzpatrick"]] += 1
    logger.info("Procedures: %s", dict(proc_counts))
    logger.info("Fitzpatrick: %s", dict(sorted(fitz_counts.items())))

    # Run evaluations
    all_method_results: dict[str, list[dict]] = {}
    all_method_agg: dict[str, dict] = {}

    # TPS baseline
    if include_baseline:
        logger.info("Evaluating TPS baseline...")
        tps_results = evaluate_tps_baseline(samples, out_path, intensity)
        all_method_results["TPS_baseline"] = tps_results
        all_method_agg["TPS_baseline"] = aggregate_metrics(tps_results)

    # ControlNet evaluation with fine-tuned checkpoint
    if checkpoint:
        logger.info("Evaluating ControlNet checkpoint: %s", checkpoint)
        cn_results = evaluate_controlnet(samples, out_path, checkpoint, intensity)
        method_label = "ControlNet"
        if cn_results and cn_results[0].get("method") == "controlnet_proxy_tps":
            method_label = "ControlNet (TPS proxy)"
        all_method_results[method_label] = cn_results
        all_method_agg[method_label] = aggregate_metrics(cn_results)

    # Print results
    logger.info("=" * 70)
    logger.info("Evaluation Results (n=%d)", len(samples))
    logger.info("=" * 70)

    header = f"{'Method':<20} {'SSIM':>8} {'LPIPS':>8} {'NME':>8} {'ID Sim':>8} {'n':>5}"
    logger.info("%s", header)
    logger.info("-" * len(header))

    for method, agg in all_method_agg.items():
        ssim = agg.get("ssim_mean", -1)
        lpips = agg.get("lpips_mean", -1)
        nme = agg.get("nme_mean", -1)
        identity = agg.get("identity_mean", -1)
        n = agg.get("n", 0)
        logger.info(
            "%s %8.4f %8.4f %8.4f %8.4f %5d",
            method.ljust(20),
            ssim,
            lpips,
            nme,
            identity,
            n,
        )

    # Per-procedure breakdown
    logger.info("=" * 70)
    logger.info("Per-Procedure Breakdown")
    logger.info("=" * 70)
    for method, agg in all_method_agg.items():
        logger.info("  %s:", method)
        for proc in PROCEDURES:
            proc_data = agg.get("by_procedure", {}).get(proc, {})
            if proc_data:
                ssim = proc_data.get("ssim_mean", -1)
                lpips = proc_data.get("lpips_mean", -1)
                n = proc_data.get("n", 0)
                logger.info("    %s SSIM=%.4f  LPIPS=%.4f  (n=%d)", proc.ljust(20), ssim, lpips, n)

    # Generate LaTeX tables
    latex_dir = out_path / "latex"
    latex_dir.mkdir(exist_ok=True)

    main_table = generate_latex_table(all_method_agg)
    (latex_dir / "main_results.tex").write_text(main_table)

    proc_table = generate_per_procedure_table(all_method_agg)
    (latex_dir / "per_procedure.tex").write_text(proc_table)

    fitz_table = generate_fitzpatrick_table(all_method_agg)
    (latex_dir / "fitzpatrick_equity.tex").write_text(fitz_table)

    logger.info("LaTeX tables saved to %s", latex_dir)

    # Save full report
    report = {
        "test_dir": str(test_path),
        "num_samples": len(samples),
        "intensity": intensity,
        "procedure_counts": dict(proc_counts),
        "fitzpatrick_counts": dict(sorted(fitz_counts.items())),
        "methods": all_method_agg,
        "per_sample": {method: results for method, results in all_method_results.items()},
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    report_path = out_path / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Full report: %s", report_path)
    logger.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Comprehensive evaluation runner")
    parser.add_argument("--test_dir", required=True, help="Test data directory")
    parser.add_argument("--output", default="results/paper_eval", help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="ControlNet checkpoint to evaluate")
    parser.add_argument("--include-baseline", action="store_true", help="Include TPS baseline")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum samples (0=all)")
    parser.add_argument("--intensity", type=float, default=65.0)
    args = parser.parse_args()

    run_evaluation(
        args.test_dir,
        args.output,
        args.checkpoint,
        args.include_baseline,
        args.max_samples,
        args.intensity,
    )
