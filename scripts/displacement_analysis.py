#!/usr/bin/env python3
"""Displacement analysis for paper tables — generates reports from training data.

Provides streamlined functions to:
1. Generate displacement_report.json from before/after image pairs
2. Compute per-region displacement statistics for paper figures
3. Generate ablation result templates for loss function experiments

Usage:
    # Generate displacement report from training data
    python scripts/displacement_analysis.py report \
        --pairs_dir data/real_surgery_pairs/pairs \
        --output data/displacement_report.json

    # Compute per-region statistics for paper
    python scripts/displacement_analysis.py regions \
        --model data/displacement_model.npz

    # Generate ablation results template
    python scripts/displacement_analysis.py ablation-template \
        --output results/ablation_results.json

    # Full analysis (all of the above)
    python scripts/displacement_analysis.py full \
        --pairs_dir data/real_surgery_pairs/pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Displacement Report Generation ─────────────────────────────


def generate_displacement_report(
    pairs_dir: str | Path,
    output_path: str | Path = "data/displacement_report.json",
    min_quality: float = 0.3,
) -> dict:
    """Generate displacement_report.json from before/after image pairs.

    Args:
        pairs_dir: Directory containing before/after image pairs.
        output_path: Where to save the JSON report.
        min_quality: Minimum alignment quality to include a pair.

    Returns:
        Report dict with total_pairs and per-procedure stats.
    """
    from scripts.extract_displacements import extract_from_directory

    pairs_path = Path(pairs_dir)
    if not pairs_path.exists():
        return {"error": f"Pairs directory not found: {pairs_dir}", "total_pairs": 0}

    results = extract_from_directory(pairs_path, min_quality=min_quality)

    report: dict = {
        "total_pairs": len(results),
        "procedures": {},
    }

    for r in results:
        proc = r["procedure"]
        if proc not in report["procedures"]:
            report["procedures"][proc] = {"count": 0, "quality_scores": []}
        report["procedures"][proc]["count"] += 1
        report["procedures"][proc]["quality_scores"].append(r["quality_score"])

    # Compute aggregate statistics
    for proc in report["procedures"]:
        qs = report["procedures"][proc]["quality_scores"]
        report["procedures"][proc]["mean_quality"] = float(np.mean(qs)) if qs else 0.0
        report["procedures"][proc]["std_quality"] = float(np.std(qs)) if qs else 0.0
        report["procedures"][proc]["min_quality"] = float(np.min(qs)) if qs else 0.0
        report["procedures"][proc]["max_quality"] = float(np.max(qs)) if qs else 0.0
        report["procedures"][proc]["quality_scores"] = []  # don't bloat JSON

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ── Per-Region Displacement Statistics ──────────────────────────


def compute_region_statistics(
    model_path: str | Path = "data/displacement_model.npz",
) -> dict:
    """Compute per-region displacement statistics from a fitted model.

    Returns dict with per-procedure, per-region displacement magnitudes
    suitable for paper figures and tables.
    """
    from landmarkdiff.displacement_model import DisplacementModel
    from landmarkdiff.manipulation import PROCEDURE_LANDMARKS

    model_path = Path(model_path)
    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    model = DisplacementModel()
    model.load(model_path)

    stats: dict = {}

    for procedure in model.procedure_stats:
        proc_stats = model.procedure_stats[procedure]
        mean_disp = proc_stats.get("mean")
        if mean_disp is None:
            continue

        # Overall stats
        magnitudes = np.linalg.norm(mean_disp, axis=1)
        stats[procedure] = {
            "n_samples": int(proc_stats.get("n_samples", 0)),
            "overall": {
                "mean_magnitude": float(np.mean(magnitudes)),
                "max_magnitude": float(np.max(magnitudes)),
                "median_magnitude": float(np.median(magnitudes)),
                "active_landmarks": int(np.sum(magnitudes > 0.005)),
            },
            "regions": {},
        }

        # Per-region stats (using PROCEDURE_LANDMARKS regions)
        for region_name, region_indices in PROCEDURE_LANDMARKS.items():
            valid_indices = [i for i in region_indices if i < len(magnitudes)]
            if not valid_indices:
                continue

            region_mags = magnitudes[valid_indices]
            stats[procedure]["regions"][region_name] = {
                "n_landmarks": len(valid_indices),
                "mean_magnitude": float(np.mean(region_mags)),
                "max_magnitude": float(np.max(region_mags)),
                "std_magnitude": float(np.std(region_mags)),
            }

    return stats


# ── Ablation Results Template ───────────────────────────────────


def generate_ablation_template(
    output_path: str | Path = "results/ablation_results.json",
) -> dict:
    """Generate ablation results template for loss function experiments.

    Creates a template with the 4 ablation configurations expected by
    generate_paper_tables.py Table 2. Values are placeholders (0.0)
    that should be replaced with real evaluation results.
    """
    template = {
        "_description": (
            "Ablation study results. Replace placeholder values with real "
            "evaluation metrics after running each configuration."
        ),
        "_configurations": {
            "diffusion_only": "Phase A training with diffusion loss only",
            "diff_identity": "Diffusion + ArcFace identity loss",
            "diff_perceptual": "Diffusion + LPIPS perceptual loss",
            "full": "Full 4-term loss (diffusion + landmark + identity + perceptual)",
        },
        "diffusion_only": {
            "ssim": 0.0,
            "lpips": 0.0,
            "nme": 0.0,
            "identity_sim": 0.0,
            "fid": 0.0,
        },
        "diff_identity": {
            "ssim": 0.0,
            "lpips": 0.0,
            "nme": 0.0,
            "identity_sim": 0.0,
            "fid": 0.0,
        },
        "diff_perceptual": {
            "ssim": 0.0,
            "lpips": 0.0,
            "nme": 0.0,
            "identity_sim": 0.0,
            "fid": 0.0,
        },
        "full": {
            "ssim": 0.0,
            "lpips": 0.0,
            "nme": 0.0,
            "identity_sim": 0.0,
            "fid": 0.0,
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(template, f, indent=2)

    return template


def update_ablation_results(
    ablation_path: str | Path,
    config_key: str,
    metrics: dict[str, float],
) -> dict:
    """Update ablation results with real evaluation metrics.

    Args:
        ablation_path: Path to ablation_results.json.
        config_key: One of: diffusion_only, diff_identity, diff_perceptual, full.
        metrics: Dict with ssim, lpips, nme, identity_sim, fid values.
    """
    valid_keys = {"diffusion_only", "diff_identity", "diff_perceptual", "full"}
    if config_key not in valid_keys:
        raise ValueError(f"Invalid config_key: {config_key}. Must be one of {valid_keys}")

    path = Path(ablation_path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = generate_ablation_template(path)

    # Update metrics
    if config_key not in data:
        data[config_key] = {}
    data[config_key].update(metrics)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return data


# ── Summary Report ──────────────────────────────────────────────


def generate_full_analysis(
    pairs_dir: str | Path | None = None,
    model_path: str | Path = "data/displacement_model.npz",
    output_dir: str | Path = "results",
) -> dict:
    """Run full displacement analysis and generate all reports.

    Args:
        pairs_dir: Directory with before/after pairs (optional, skips report if None).
        model_path: Path to fitted displacement model .npz.
        output_dir: Directory for output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"analyses": []}

    # 1. Displacement report (if pairs_dir provided)
    if pairs_dir and Path(pairs_dir).exists():
        report = generate_displacement_report(
            pairs_dir, out.parent / "data" / "displacement_report.json"
        )
        summary["displacement_report"] = {
            "total_pairs": report.get("total_pairs", 0),
            "procedures": list(report.get("procedures", {}).keys()),
        }
        summary["analyses"].append("displacement_report")
        print(f"  Displacement report: {report.get('total_pairs', 0)} pairs")

    # 2. Per-region statistics
    if Path(model_path).exists():
        region_stats = compute_region_statistics(model_path)
        stats_path = out / "displacement_region_stats.json"
        with open(stats_path, "w") as f:
            json.dump(region_stats, f, indent=2)
        summary["region_statistics"] = {
            "procedures": list(region_stats.keys()),
            "output": str(stats_path),
        }
        summary["analyses"].append("region_statistics")
        print(f"  Region stats: {len(region_stats)} procedures")

    # 3. Ablation template
    ablation_path = out / "ablation_results.json"
    if not ablation_path.exists():
        generate_ablation_template(ablation_path)
        summary["analyses"].append("ablation_template")
        print(f"  Ablation template: {ablation_path}")

    return summary


# ── CLI ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Displacement analysis for paper tables")
    sub = parser.add_subparsers(dest="command")

    # report
    rp = sub.add_parser("report", help="Generate displacement report from image pairs")
    rp.add_argument("--pairs_dir", required=True, help="Before/after pairs directory")
    rp.add_argument("--output", default="data/displacement_report.json")
    rp.add_argument("--min_quality", type=float, default=0.3)

    # regions
    rg = sub.add_parser("regions", help="Compute per-region displacement statistics")
    rg.add_argument("--model", default="data/displacement_model.npz")
    rg.add_argument("--output", default="results/displacement_region_stats.json")

    # ablation-template
    at = sub.add_parser("ablation-template", help="Generate ablation results template")
    at.add_argument("--output", default="results/ablation_results.json")

    # update-ablation
    ua = sub.add_parser("update-ablation", help="Update ablation results with real metrics")
    ua.add_argument("--ablation", required=True, help="Ablation results JSON path")
    ua.add_argument(
        "--config",
        required=True,
        choices=["diffusion_only", "diff_identity", "diff_perceptual", "full"],
    )
    ua.add_argument("--ssim", type=float, default=None)
    ua.add_argument("--lpips", type=float, default=None)
    ua.add_argument("--nme", type=float, default=None)
    ua.add_argument("--identity_sim", type=float, default=None)
    ua.add_argument("--fid", type=float, default=None)

    # full
    fl = sub.add_parser("full", help="Full displacement analysis")
    fl.add_argument("--pairs_dir", default=None)
    fl.add_argument("--model", default="data/displacement_model.npz")
    fl.add_argument("--output_dir", default="results")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "report":
        report = generate_displacement_report(args.pairs_dir, args.output, args.min_quality)
        n = report.get("total_pairs", 0)
        procs = list(report.get("procedures", {}).keys())
        print(f"Report: {n} pairs across {len(procs)} procedures")
        print(f"  Procedures: {', '.join(procs)}")
        print(f"  Saved to: {args.output}")

    elif args.command == "regions":
        stats = compute_region_statistics(args.model)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
            sys.exit(1)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)

        for proc, data in stats.items():
            overall = data.get("overall", {})
            print(
                f"  {proc}: mean={overall.get('mean_magnitude', 0):.4f}, "
                f"max={overall.get('max_magnitude', 0):.4f}, "
                f"active={overall.get('active_landmarks', 0)}"
            )
        print(f"  Saved to: {args.output}")

    elif args.command == "ablation-template":
        generate_ablation_template(args.output)
        print(f"Ablation template saved to: {args.output}")
        print("  Replace 0.0 values with real evaluation metrics.")

    elif args.command == "update-ablation":
        metrics = {}
        for key in ["ssim", "lpips", "nme", "identity_sim", "fid"]:
            val = getattr(args, key, None)
            if val is not None:
                metrics[key] = val
        if not metrics:
            print("ERROR: Provide at least one metric (--ssim, --lpips, etc.)")
            sys.exit(1)
        update_ablation_results(args.ablation, args.config, metrics)
        print(f"Updated {args.config} with: {metrics}")

    elif args.command == "full":
        summary = generate_full_analysis(args.pairs_dir, args.model, args.output_dir)
        print(f"\nAnalyses completed: {summary['analyses']}")


if __name__ == "__main__":
    main()
