#!/usr/bin/env python3
"""Generate a comprehensive visual report from batch inference results.

Creates publication-quality visualization grids and metric plots from
the batch_report.json output of batch_inference.py.

Produces:
1. Best/worst examples per procedure (by SSIM)
2. Fitzpatrick diversity mosaic
3. Metric distribution histograms
4. Per-procedure comparison strip

Usage:
    python scripts/generate_report.py --report results/tps_baseline/batch_report.json
    python scripts/generate_report.py --report results/batch/batch_report.json --output demos/report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
FITZ_COLORS = {
    "I": "#FAD6A5",
    "II": "#E8C298",
    "III": "#C8A278",
    "IV": "#A67C52",
    "V": "#8B5A2B",
    "VI": "#543310",
}


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_metric_histograms(results: list[dict], output_dir: Path) -> None:
    """Create histograms for SSIM, LPIPS, and NME distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = {
        "SSIM": [r["ssim"] for r in results],
        "LPIPS": [r["lpips"] for r in results],
        "NME": [r["nme"] for r in results],
    }

    colors = ["#4C72B0", "#C44E52", "#55A868"]
    for ax, (name, vals), color in zip(axes, metrics.items(), colors, strict=False):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(
            np.mean(vals),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {np.mean(vals):.4f}",
        )
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Metric Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "metric_histograms.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_procedure_comparison(results: list[dict], output_dir: Path) -> None:
    """Create per-procedure boxplots for all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metric_names = ["ssim", "lpips", "nme"]
    display_names = ["SSIM (higher=better)", "LPIPS (lower=better)", "NME (lower=better)"]

    for ax, metric, display in zip(axes, metric_names, display_names, strict=False):
        data = []
        labels = []
        for proc in PROCEDURES:
            proc_results = [
                r[metric] for r in results if r["procedure"] == proc and not np.isnan(r[metric])
            ]
            if proc_results:
                data.append(proc_results)
                labels.append(proc[:5].capitalize())

        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
            for patch, color in zip(bp["boxes"], colors[: len(data)], strict=False):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_ylabel(display, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Procedure Metric Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "procedure_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_fitzpatrick_chart(results: list[dict], output_dir: Path) -> None:
    """Create Fitzpatrick skin type distribution and metric chart."""
    fitz_groups: dict[str, list[dict]] = {}
    for r in results:
        ftype = r.get("fitzpatrick", "?")
        if ftype != "?":
            fitz_groups.setdefault(ftype, []).append(r)

    if not fitz_groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution bar chart
    types = sorted(fitz_groups.keys())
    counts = [len(fitz_groups[t]) for t in types]
    colors = [FITZ_COLORS.get(t, "#888888") for t in types]
    axes[0].bar(
        [f"Type {t}" for t in types], counts, color=colors, edgecolor="black", linewidth=0.5
    )
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    axes[0].set_title("Sample Distribution by Fitzpatrick Type", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3)

    # SSIM by Fitzpatrick type
    ssim_means = []
    ssim_stds = []
    for t in types:
        vals = [r["ssim"] for r in fitz_groups[t] if not np.isnan(r["ssim"])]
        ssim_means.append(np.mean(vals) if vals else 0)
        ssim_stds.append(np.std(vals) if vals else 0)

    x = range(len(types))
    axes[1].bar(
        x, ssim_means, yerr=ssim_stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Type {t}" for t in types])
    axes[1].set_ylabel("SSIM", fontsize=11)
    axes[1].set_title("SSIM by Fitzpatrick Type (Equity Check)", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fitzpatrick_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_best_worst_grid(
    results: list[dict],
    results_dir: Path,
    output_dir: Path,
    k: int = 3,
) -> None:
    """Create a grid showing best and worst results per procedure."""
    for proc in PROCEDURES:
        proc_results = [r for r in results if r["procedure"] == proc]
        if len(proc_results) < 2:
            continue

        # Sort by SSIM
        proc_results.sort(key=lambda r: r["ssim"])
        worst = proc_results[:k]
        best = proc_results[-k:][::-1]

        rows = []
        cell_h = 200

        for label, group in [("BEST", best), ("WORST", worst)]:
            for r in group:
                img_name = r["image"]
                ba_path = results_dir / proc / f"{img_name}_before_after.png"
                if ba_path.exists():
                    img = cv2.imread(str(ba_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = cell_h / h
                        resized = cv2.resize(img, (int(w * scale), cell_h))
                        # Add metric label
                        cv2.putText(
                            resized,
                            f"{label}: SSIM={r['ssim']:.3f} LPIPS={r['lpips']:.3f}",
                            (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                        )
                        rows.append(resized)

        if rows:
            # Pad to same width
            max_w = max(r.shape[1] for r in rows)
            padded = []
            for row in rows:
                if row.shape[1] < max_w:
                    pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
                    row = np.hstack([row, pad])
                padded.append(row)

            grid = np.vstack(padded)

            # Header
            header = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                header,
                f"{proc.capitalize()} — Top {k} Best / Top {k} Worst by SSIM",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 255),
                1,
            )
            grid = np.vstack([header, grid])
            cv2.imwrite(str(output_dir / f"best_worst_{proc}.png"), grid)


def make_summary_table(report: dict, output_dir: Path) -> None:
    """Create a summary text file with all metrics."""
    results = report.get("results", [])
    lines = ["LandmarkDiff Batch Inference Report", "=" * 50]

    if "aggregate" in report:
        agg = report["aggregate"]
        lines.append(f"\nOverall Metrics (n={report['total_processed']}):")
        lines.append(f"  SSIM:  {agg.get('ssim_mean', 0):.4f} ± {agg.get('ssim_std', 0):.4f}")
        lines.append(f"  LPIPS: {agg.get('lpips_mean', 0):.4f} ± {agg.get('lpips_std', 0):.4f}")
        lines.append(f"  NME:   {agg.get('nme_mean', 0):.4f}")

    # Per-procedure
    proc_groups: dict[str, list[dict]] = {}
    for r in results:
        proc_groups.setdefault(r["procedure"], []).append(r)

    lines.append("\nPer-Procedure Breakdown:")
    for proc in PROCEDURES:
        if proc in proc_groups:
            group = proc_groups[proc]
            ssim_vals = [r["ssim"] for r in group]
            lpips_vals = [r["lpips"] for r in group]
            nme_vals = [r["nme"] for r in group]
            lines.append(f"\n  {proc.capitalize()} (n={len(group)}):")
            lines.append(f"    SSIM:  {np.nanmean(ssim_vals):.4f} ± {np.nanstd(ssim_vals):.4f}")
            lines.append(f"    LPIPS: {np.nanmean(lpips_vals):.4f} ± {np.nanstd(lpips_vals):.4f}")
            lines.append(f"    NME:   {np.nanmean(nme_vals):.4f} ± {np.nanstd(nme_vals):.4f}")

    # Fitzpatrick
    if "fitzpatrick_breakdown" in report:
        lines.append("\nFitzpatrick Stratification:")
        for ftype, data in sorted(report["fitzpatrick_breakdown"].items()):
            lines.append(
                f"  Type {ftype}: n={data['count']}, "
                f"SSIM={data['ssim_mean']:.4f}, LPIPS={data['lpips_mean']:.4f}"
            )

    text = "\n".join(lines)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Generate visual report from batch inference")
    parser.add_argument("--report", required=True, help="Path to batch_report.json")
    parser.add_argument("--output", default=None, help="Output directory (default: same as report)")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: Report not found: {report_path}")
        sys.exit(1)

    report = load_report(str(report_path))
    results_dir = report_path.parent
    output_dir = Path(args.output) if args.output else results_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = report.get("results", [])
    if not results:
        print("No results found in report.")
        return

    print(f"Generating report from {len(results)} results...")

    # Generate all visualizations
    make_summary_table(report, output_dir)
    make_metric_histograms(results, output_dir)
    make_procedure_comparison(results, output_dir)
    make_fitzpatrick_chart(results, output_dir)
    make_best_worst_grid(results, results_dir, output_dir)

    print(f"\nReport saved to {output_dir}/")
    print("  - summary.txt")
    print("  - metric_histograms.png")
    print("  - procedure_comparison.png")
    print("  - fitzpatrick_analysis.png")
    for proc in PROCEDURES:
        bw = output_dir / f"best_worst_{proc}.png"
        if bw.exists():
            print(f"  - best_worst_{proc}.png")


if __name__ == "__main__":
    main()
