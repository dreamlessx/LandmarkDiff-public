#!/usr/bin/env python3
"""Export batch inference results to LaTeX tables for the paper.

Reads JSON reports from batch_inference.py and generates LaTeX table rows
ready to paste into paper/main.tex.

Usage:
    # Export TPS baseline results
    python scripts/export_results.py \
        --reports results/tps_baseline/batch_report.json \
        --method "TPS-only" \
        --output paper/results_tps.tex

    # Export all methods and generate complete tables
    python scripts/export_results.py \
        --reports results/tps_baseline/batch_report.json \
                 results/controlnet/batch_report.json \
        --methods "TPS-only" "\\method{}" \
        --output paper/results_all.tex

    # Print Fitzpatrick table
    python scripts/export_results.py \
        --reports results/tps_baseline/batch_report.json \
        --fitzpatrick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
FITZ_TYPES = ["I", "II", "III", "IV", "V", "VI"]


def load_report(path: str) -> dict:
    """Load a batch inference report."""
    with open(path) as f:
        return json.load(f)


def compute_per_procedure_metrics(report: dict) -> dict[str, dict[str, float]]:
    """Compute per-procedure metrics from individual results."""
    results = report.get("results", [])
    metrics_by_proc: dict[str, dict[str, list[float]]] = {}

    for r in results:
        proc = r["procedure"]
        if proc not in metrics_by_proc:
            metrics_by_proc[proc] = {"ssim": [], "lpips": [], "nme": []}
        metrics_by_proc[proc]["ssim"].append(r["ssim"])
        metrics_by_proc[proc]["lpips"].append(r["lpips"])
        metrics_by_proc[proc]["nme"].append(r["nme"])

    output = {}
    for proc, vals in metrics_by_proc.items():
        output[proc] = {
            "ssim_mean": float(np.nanmean(vals["ssim"])),
            "ssim_std": float(np.nanstd(vals["ssim"])),
            "lpips_mean": float(np.nanmean(vals["lpips"])),
            "lpips_std": float(np.nanstd(vals["lpips"])),
            "nme_mean": float(np.nanmean(vals["nme"])),
            "nme_std": float(np.nanstd(vals["nme"])),
            "n": len(vals["ssim"]),
        }
    return output


def compute_fitzpatrick_metrics(report: dict) -> dict[str, dict[str, float]]:
    """Compute per-Fitzpatrick-type metrics."""
    results = report.get("results", [])
    fitz_groups: dict[str, dict[str, list[float]]] = {}

    for r in results:
        ftype = r.get("fitzpatrick", "?")
        if ftype == "?":
            continue
        if ftype not in fitz_groups:
            fitz_groups[ftype] = {"ssim": [], "lpips": [], "nme": []}
        fitz_groups[ftype]["ssim"].append(r["ssim"])
        fitz_groups[ftype]["lpips"].append(r["lpips"])
        fitz_groups[ftype]["nme"].append(r["nme"])

    output = {}
    for ftype, vals in fitz_groups.items():
        output[ftype] = {
            "ssim_mean": float(np.nanmean(vals["ssim"])),
            "lpips_mean": float(np.nanmean(vals["lpips"])),
            "nme_mean": float(np.nanmean(vals["nme"])),
            "n": len(vals["ssim"]),
        }
    return output


def format_metric(mean: float, std: float | None = None, fmt: str = ".4f") -> str:
    """Format a metric value for LaTeX."""
    if np.isnan(mean):
        return "--"
    if std is not None and not np.isnan(std):
        return f"${mean:{fmt}} \\pm {std:{fmt}}$"
    return f"${mean:{fmt}}$"


def generate_procedure_table_rows(
    method_name: str,
    metrics_by_proc: dict[str, dict[str, float]],
    bold: bool = False,
) -> list[str]:
    """Generate LaTeX table rows for the per-procedure results table.

    Format: Method & FID & LPIPS & NME & SSIM & ArcFace \\
    """
    rows = []
    for proc in PROCEDURES:
        if proc not in metrics_by_proc:
            name = f"\\textbf{{{method_name}}}" if bold else method_name
            rows.append(f"{name} & -- & -- & -- & -- & -- \\\\")
            continue

        m = metrics_by_proc[proc]
        name = f"\\textbf{{{method_name}}}" if bold else method_name
        fid = "--"  # FID requires directory-level computation
        lpips = format_metric(m["lpips_mean"], m.get("lpips_std"))
        nme = format_metric(m["nme_mean"], m.get("nme_std"))
        ssim = format_metric(m["ssim_mean"], m.get("ssim_std"))
        arcface = "--"  # Would need identity similarity from report
        rows.append(f"{name} & {fid} & {lpips} & {nme} & {ssim} & {arcface} \\\\")

    return rows


def generate_fitzpatrick_table(
    fitz_metrics: dict[str, dict[str, float]],
) -> str:
    """Generate the Fitzpatrick stratification table."""
    lines = []

    # NME row
    nme_vals = []
    for ftype in FITZ_TYPES:
        if ftype in fitz_metrics:
            nme_vals.append(f"${fitz_metrics[ftype]['nme_mean']:.4f}$")
        else:
            nme_vals.append("--")
    lines.append("NME $\\downarrow$ & " + " & ".join(nme_vals) + " \\\\")

    # LPIPS row
    lpips_vals = []
    for ftype in FITZ_TYPES:
        if ftype in fitz_metrics:
            lpips_vals.append(f"${fitz_metrics[ftype]['lpips_mean']:.4f}$")
        else:
            lpips_vals.append("--")
    lines.append("LPIPS $\\downarrow$ & " + " & ".join(lpips_vals) + " \\\\")

    # SSIM row
    ssim_vals = []
    for ftype in FITZ_TYPES:
        if ftype in fitz_metrics:
            ssim_vals.append(f"${fitz_metrics[ftype]['ssim_mean']:.4f}$")
        else:
            ssim_vals.append("--")
    lines.append("SSIM $\\uparrow$ & " + " & ".join(ssim_vals) + " \\\\")

    # Count row
    count_vals = []
    for ftype in FITZ_TYPES:
        if ftype in fitz_metrics:
            count_vals.append(str(fitz_metrics[ftype]["n"]))
        else:
            count_vals.append("0")
    lines.append("\\midrule")
    lines.append("$n$ & " + " & ".join(count_vals) + " \\\\")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export results to LaTeX tables")
    parser.add_argument(
        "--reports", nargs="+", required=True, help="Path(s) to batch_report.json files"
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Method names (one per report)")
    parser.add_argument("--output", default=None, help="Output .tex file (or stdout)")
    parser.add_argument(
        "--fitzpatrick", action="store_true", help="Generate Fitzpatrick stratification table"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    reports = [load_report(r) for r in args.reports]
    method_names = args.methods or [f"Method {i + 1}" for i in range(len(reports))]

    if len(method_names) != len(reports):
        print("ERROR: number of --methods must match number of --reports")
        sys.exit(1)

    output_lines = []

    if args.fitzpatrick:
        # Fitzpatrick table from the first report
        fitz = compute_fitzpatrick_metrics(reports[0])
        output_lines.append(f"% Fitzpatrick table for {method_names[0]}")
        output_lines.append(f"% Generated from {args.reports[0]}")
        output_lines.append(generate_fitzpatrick_table(fitz))
    else:
        # Per-procedure results table
        for proc in PROCEDURES:
            output_lines.append("\\midrule")
            output_lines.append(f"\\multicolumn{{6}}{{c}}{{\\textit{{{proc.capitalize()}}}}} \\\\")
            output_lines.append("\\midrule")

            for report, method in zip(reports, method_names, strict=False):
                metrics = compute_per_procedure_metrics(report)
                is_ours = "method" in method.lower() or method.startswith("\\")
                rows = generate_procedure_table_rows(method, metrics, bold=is_ours)
                # Find the row for this procedure
                proc_idx = PROCEDURES.index(proc)
                if proc_idx < len(rows):
                    output_lines.append(rows[proc_idx])

    output_text = "\n".join(output_lines)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
        print(f"Results exported to {args.output}")
    else:
        print(output_text)

    # Print summary
    if args.verbose:
        for report, method in zip(reports, method_names, strict=False):
            metrics = compute_per_procedure_metrics(report)
            print(f"\n=== {method} ===")
            for proc, m in metrics.items():
                print(
                    f"  {proc}: SSIM={m['ssim_mean']:.4f}±{m['ssim_std']:.4f} "
                    f"LPIPS={m['lpips_mean']:.4f}±{m['lpips_std']:.4f} "
                    f"NME={m['nme_mean']:.4f} (n={m['n']})"
                )


if __name__ == "__main__":
    main()
