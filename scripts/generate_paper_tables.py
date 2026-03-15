"""Generate publication-ready LaTeX tables from evaluation results.

Produces:
  Table 1: Main results (LandmarkDiff vs baselines)
  Table 2: Loss ablation study
  Table 3: Fitzpatrick fairness analysis
  Table 4: Per-procedure breakdown

Usage:
    python scripts/generate_paper_tables.py \
        --results results/eval_results.json \
        --baselines results/baseline_results.json \
        --ablation results/ablation_results.json \
        --output paper/tables/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _bold_best(values: list[float], higher_is_better: bool = True) -> list[str]:
    """Format values, bolding the best one."""
    if not values:
        return []
    best_idx = values.index(max(values)) if higher_is_better else values.index(min(values))
    formatted = []
    for i, v in enumerate(values):
        s = f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
        formatted.append(f"\\textbf{{{s}}}" if i == best_idx else s)
    return formatted


def generate_table1_main(
    results_path: str,
    baselines_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Table 1: Main quantitative results.

    Compares LandmarkDiff (ours) against baselines:
    - TPS Warp (traditional)
    - Morphing (alpha blend)
    - Pix2Pix (if available)
    """
    with open(results_path) as f:
        results = json.load(f)

    baselines = {}
    if baselines_path and Path(baselines_path).exists():
        with open(baselines_path) as f:
            baselines = json.load(f)

    methods = []
    metrics_data = []

    # Baselines
    if "tps" in baselines:
        methods.append("TPS Warp")
        metrics_data.append(baselines["tps"])
    if "morphing" in baselines:
        methods.append("Morphing")
        metrics_data.append(baselines["morphing"])

    # Ours
    methods.append("LandmarkDiff (Ours)")
    metrics_data.append(results.get("metrics", results))

    # Metric columns
    metric_keys = [
        ("fid", "FID $\\downarrow$", False),
        ("lpips", "LPIPS $\\downarrow$", False),
        ("ssim", "SSIM $\\uparrow$", True),
        ("nme", "NME $\\downarrow$", False),
        ("identity_sim", "ID Sim $\\uparrow$", True),
    ]

    # Build table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Quantitative comparison of surgical outcome prediction methods. "
        "Bold indicates best performance. $\\downarrow$ = lower is better, $\\uparrow$ = higher is better.}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{l" + "c" * len(metric_keys) + "}",
        "\\toprule",
    ]

    # Header
    header = "Method & " + " & ".join(name for _, name, _ in metric_keys) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for key, _, higher_is_better in metric_keys:
        col_vals = []
        for md in metrics_data:
            col_vals.append(md.get(key, 0.0))
        # Bold the best
        best_idx = (
            col_vals.index(max(col_vals)) if higher_is_better else col_vals.index(min(col_vals))
        )
        for i, md in enumerate(metrics_data):
            if f"_{key}_formatted" not in md:
                v = md.get(key, 0.0)
                s = f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
                md[f"_{key}_formatted"] = f"\\textbf{{{s}}}" if i == best_idx else s

    for i, method in enumerate(methods):
        row = method
        for key, _, _ in metric_keys:
            row += f" & {metrics_data[i][f'_{key}_formatted']}"
        row += " \\\\"
        if method == methods[-1]:
            pass  # no midrule after last
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)

    return table


def generate_table2_ablation(
    ablation_path: str,
    output_path: str | None = None,
) -> str:
    """Table 2: Loss function ablation study."""
    with open(ablation_path) as f:
        ablation = json.load(f)

    configs = [
        ("$\\mathcal{L}_{\\text{diff}}$", "diffusion_only"),
        ("$\\mathcal{L}_{\\text{diff}} + \\mathcal{L}_{\\text{id}}$", "diff_identity"),
        ("$\\mathcal{L}_{\\text{diff}} + \\mathcal{L}_{\\text{perc}}$", "diff_perceptual"),
        ("Full (Ours)", "full"),
    ]

    metrics = [
        ("ssim", "SSIM $\\uparrow$", True),
        ("lpips", "LPIPS $\\downarrow$", False),
        ("nme", "NME $\\downarrow$", False),
        ("identity_sim", "ID Sim $\\uparrow$", True),
        ("fid", "FID $\\downarrow$", False),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Loss function ablation. Each row adds one loss term to the diffusion baseline. "
        "The full 4-term objective achieves the best overall performance.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Loss Configuration & " + " & ".join(n for _, n, _ in metrics) + " \\\\",
        "\\midrule",
    ]

    # Collect values per metric for bolding
    for mkey, _, higher in metrics:
        vals = []
        for _, config_key in configs:
            if config_key in ablation:
                vals.append(ablation[config_key].get(mkey, 0.0))
            else:
                vals.append(0.0)
        best_idx = vals.index(max(vals)) if higher else vals.index(min(vals))
        for i, (_, config_key) in enumerate(configs):
            if config_key not in ablation:
                continue
            v = vals[i]
            s = f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
            ablation[config_key][f"_{mkey}_fmt"] = f"\\textbf{{{s}}}" if i == best_idx else s

    for label, config_key in configs:
        if config_key not in ablation:
            continue
        row = label
        for mkey, _, _ in metrics:
            row += f" & {ablation[config_key].get(f'_{mkey}_fmt', '--')}"
        row += " \\\\"
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)
    return table


def generate_table3_fairness(
    results_path: str,
    output_path: str | None = None,
) -> str:
    """Table 3: Fitzpatrick skin type fairness analysis."""
    with open(results_path) as f:
        results = json.load(f)

    fitz_data = results.get("per_fitzpatrick", {})
    if not fitz_data:
        return "% No Fitzpatrick data available"

    types = sorted(fitz_data.keys())
    metrics = [
        ("ssim", "SSIM"),
        ("lpips", "LPIPS"),
        ("nme", "NME"),
        ("identity_sim", "ID Sim"),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Performance stratified by Fitzpatrick skin type (I--VI). "
        "Consistent metrics across skin types indicate equitable model performance.}",
        "\\label{tab:fairness}",
        "\\begin{tabular}{l" + "c" * len(types) + "}",
        "\\toprule",
        "Metric & " + " & ".join(f"Type {t}" for t in types) + " \\\\",
        "\\midrule",
    ]

    for mkey, mname in metrics:
        row = mname
        for t in types:
            v = fitz_data[t].get(mkey, 0.0)
            row += f" & {v:.4f}"
        row += " \\\\"
        lines.append(row)

    # Add count row
    count_row = "$n$"
    for t in types:
        count_row += f" & {fitz_data[t].get('count', 0)}"
    count_row += " \\\\"
    lines.append("\\midrule")
    lines.append(count_row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)
    return table


def generate_table4_procedures(
    results_path: str,
    output_path: str | None = None,
) -> str:
    """Table 4: Per-procedure breakdown."""
    with open(results_path) as f:
        results = json.load(f)

    proc_data = results.get("per_procedure", {})
    if not proc_data:
        return "% No per-procedure data available"

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    proc_labels = {
        "rhinoplasty": "Rhinoplasty",
        "blepharoplasty": "Blepharoplasty",
        "rhytidectomy": "Rhytidectomy",
        "orthognathic": "Orthognathic",
        "brow_lift": "Brow Lift",
        "mentoplasty": "Mentoplasty",
    }

    metrics = [
        ("ssim", "SSIM $\\uparrow$"),
        ("lpips", "LPIPS $\\downarrow$"),
        ("nme", "NME $\\downarrow$"),
        ("identity_sim", "ID Sim $\\uparrow$"),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-procedure evaluation results. Each procedure has distinct "
        "deformation patterns affecting different facial regions.}",
        "\\label{tab:per_procedure}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "c}",
        "\\toprule",
        "Procedure & " + " & ".join(n for _, n in metrics) + " & $n$ \\\\",
        "\\midrule",
    ]

    for proc in procedures:
        if proc not in proc_data:
            continue
        pd = proc_data[proc]
        row = proc_labels.get(proc, proc)
        for mkey, _ in metrics:
            v = pd.get(mkey, 0.0)
            row += f" & {v:.4f}"
        row += f" & {pd.get('count', 0)}"
        row += " \\\\"
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)
    return table


def generate_displacement_table(
    displacement_report_path: str,
    output_path: str | None = None,
) -> str:
    """Generate table showing displacement statistics from real surgery data."""
    with open(displacement_report_path) as f:
        report = json.load(f)

    procedures = report.get("procedures", {})

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Summary of real surgical displacement data used to train "
        "the displacement model. Quality scores from face detection confidence.}",
        "\\label{tab:displacement_data}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Procedure & Pairs & Mean Quality \\\\",
        "\\midrule",
    ]

    total = 0
    for proc in [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]:
        if proc not in procedures:
            continue
        pd = procedures[proc]
        n = pd["count"]
        q = pd.get("mean_quality", 0.0)
        total += n
        lines.append(f"{proc.capitalize()} & {n:,} & {q:.3f} \\\\")

    lines.append("\\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total:,}}} & -- \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    table = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper LaTeX tables")
    parser.add_argument(
        "--results", default="results/eval_results.json", help="Main evaluation results JSON"
    )
    parser.add_argument(
        "--baselines", default="results/baseline_results.json", help="Baseline results JSON"
    )
    parser.add_argument(
        "--ablation", default="results/ablation_results.json", help="Ablation results JSON"
    )
    parser.add_argument(
        "--displacement_report",
        default="data/displacement_report.json",
        help="Displacement extraction report",
    )
    parser.add_argument("--output", default="paper/tables", help="Output directory for LaTeX files")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Table 1: Main results
    if Path(args.results).exists():
        t1 = generate_table1_main(args.results, args.baselines, str(out / "table1_main.tex"))
        logger.info("Table 1 (Main results):")
        logger.info(t1)
        logger.info("")

    # Table 2: Ablation
    if Path(args.ablation).exists():
        t2 = generate_table2_ablation(args.ablation, str(out / "table2_ablation.tex"))
        logger.info("Table 2 (Ablation):")
        logger.info(t2)
        logger.info("")

    # Table 3: Fairness
    if Path(args.results).exists():
        t3 = generate_table3_fairness(args.results, str(out / "table3_fairness.tex"))
        logger.info("Table 3 (Fairness):")
        logger.info(t3)
        logger.info("")

    # Table 4: Per-procedure
    if Path(args.results).exists():
        t4 = generate_table4_procedures(args.results, str(out / "table4_procedures.tex"))
        logger.info("Table 4 (Per-procedure):")
        logger.info(t4)
        logger.info("")

    # Displacement data table
    if Path(args.displacement_report).exists():
        t5 = generate_displacement_table(
            args.displacement_report, str(out / "table5_displacement.tex")
        )
        logger.info("Table 5 (Displacement data):")
        logger.info(t5)
        logger.info("")

    logger.info("All tables saved to %s/", out)
