"""Ablation study comparison and visualization.

Compares multiple experiment configurations side-by-side to understand
the contribution of each component. Generates comparison tables and
LaTeX-formatted results for the paper.

Usage:
    python scripts/ablation_study.py --experiments exp1_dir exp2_dir exp3_dir
    python scripts/ablation_study.py --config configs/ablation.yaml
    python scripts/ablation_study.py --tracker experiments/ --ids exp_001 exp_002 exp_003
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def load_experiment_metrics(exp_dir: str | Path) -> dict:
    """Load metrics from an experiment directory.

    Looks for:
    - metrics.json (summary)
    - eval_results/metrics.json
    - results.json
    """
    exp_dir = Path(exp_dir)

    for candidate in [
        exp_dir / "metrics.json",
        exp_dir / "eval_results" / "metrics.json",
        exp_dir / "results.json",
        exp_dir / "summary.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)

    return {}


def load_from_tracker(tracker_dir: str, exp_ids: list[str]) -> dict[str, dict]:
    """Load experiments from an ExperimentTracker directory."""
    from landmarkdiff.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(tracker_dir)
    results = {}

    for exp_id in exp_ids:
        exp = tracker._index["experiments"].get(exp_id)
        if exp:
            results[exp_id] = {
                "name": exp["name"],
                "config": exp["config"],
                "results": exp.get("results", {}),
            }
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

METRIC_INFO = {
    "fid": {"name": "FID", "lower_better": True, "fmt": ".1f"},
    "ssim": {"name": "SSIM", "lower_better": False, "fmt": ".4f"},
    "lpips": {"name": "LPIPS", "lower_better": True, "fmt": ".4f"},
    "nme": {"name": "NME", "lower_better": True, "fmt": ".4f"},
    "identity_sim": {"name": "ID Sim", "lower_better": False, "fmt": ".4f"},
    "psnr": {"name": "PSNR", "lower_better": False, "fmt": ".2f"},
}


def compute_ablation_table(
    experiments: dict[str, dict],
    metrics: list[str] | None = None,
) -> dict:
    """Compute ablation comparison table.

    Args:
        experiments: Dict mapping exp_name -> {results: {metric: value}}.
        metrics: Which metrics to include. None = all available.

    Returns:
        Structured table data.
    """
    if metrics is None:
        # Auto-detect from first experiment
        all_metrics = set()
        for exp in experiments.values():
            results = exp.get("results", exp)
            if isinstance(results, dict):
                all_metrics.update(k for k, v in results.items() if isinstance(v, int | float))
        metrics = sorted(all_metrics & set(METRIC_INFO.keys()))
        if not metrics:
            metrics = sorted(all_metrics)[:5]

    rows = []
    for exp_name, exp_data in experiments.items():
        results = exp_data.get("results", exp_data)
        row = {"name": exp_data.get("name", exp_name)}
        for m in metrics:
            row[m] = results.get(m)
        rows.append(row)

    # Find best values for each metric
    best = {}
    for m in metrics:
        values = [r[m] for r in rows if r[m] is not None]
        if values:
            info = METRIC_INFO.get(m, {"lower_better": True})
            if info.get("lower_better", True):
                best[m] = min(values)
            else:
                best[m] = max(values)

    return {
        "metrics": metrics,
        "rows": rows,
        "best": best,
    }


def compute_deltas(
    table: dict,
    baseline_name: str | None = None,
) -> list[dict]:
    """Compute improvement deltas relative to baseline.

    Args:
        table: Output of compute_ablation_table.
        baseline_name: Name of the baseline experiment. If None, uses first.

    Returns:
        List of rows with delta values added.
    """
    rows = table["rows"]
    metrics = table["metrics"]

    # Find baseline
    baseline = rows[0]
    if baseline_name:
        for r in rows:
            if r["name"] == baseline_name:
                baseline = r
                break

    delta_rows = []
    for row in rows:
        delta = dict(row)
        for m in metrics:
            base_val = baseline.get(m)
            row_val = row.get(m)
            if base_val is not None and row_val is not None:
                info = METRIC_INFO.get(m, {"lower_better": True})
                diff = row_val - base_val
                # Positive delta = improvement
                if info.get("lower_better", True):
                    delta[f"{m}_delta"] = -diff  # Lower is better, so negative diff = improvement
                else:
                    delta[f"{m}_delta"] = diff
            else:
                delta[f"{m}_delta"] = None
        delta_rows.append(delta)

    return delta_rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_text_table(table: dict, deltas: list[dict] | None = None) -> str:
    """Format results as a text table."""
    metrics = table["metrics"]
    best = table["best"]
    rows = deltas or table["rows"]

    # Column widths
    name_width = max(len(r["name"]) for r in rows) + 2
    col_width = 12

    lines = []

    # Header
    header = f"{'Experiment':<{name_width}}"
    for m in metrics:
        info = METRIC_INFO.get(m, {"name": m})
        header += f"{info.get('name', m):>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for row in rows:
        line = f"{row['name']:<{name_width}}"
        for m in metrics:
            val = row.get(m)
            if val is None:
                line += f"{'--':>{col_width}}"
            else:
                info = METRIC_INFO.get(m, {"fmt": ".4f"})
                fmt = info.get("fmt", ".4f")
                val_str = f"{val:{fmt}}"

                # Mark best with asterisk
                if val == best.get(m):
                    val_str = f"*{val_str}"
                else:
                    val_str = f" {val_str}"

                # Add delta if available
                delta = row.get(f"{m}_delta")
                if delta is not None and delta != 0:
                    sign = "+" if delta > 0 else ""
                    val_str += f" ({sign}{delta:{fmt}})"

                line += f"{val_str:>{col_width + 10}}"
        lines.append(line)

    lines.append("")
    lines.append("* = best value")
    return "\n".join(lines)


def format_latex_table(table: dict, caption: str = "Ablation study results") -> str:
    """Format results as a LaTeX table."""
    metrics = table["metrics"]
    best = table["best"]
    rows = table["rows"]

    n_cols = len(metrics) + 1
    cols = "l" + "c" * len(metrics)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:ablation}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
    ]

    # Header
    header_parts = ["Experiment"]
    for m in metrics:
        info = METRIC_INFO.get(m, {"name": m})
        name = info.get("name", m)
        arrow = "$\\downarrow$" if info.get("lower_better", True) else "$\\uparrow$"
        header_parts.append(f"{name} {arrow}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Rows
    for row in rows:
        parts = [row["name"].replace("_", "\\_")]
        for m in metrics:
            val = row.get(m)
            if val is None:
                parts.append("--")
            else:
                info = METRIC_INFO.get(m, {"fmt": ".4f"})
                fmt = info.get("fmt", ".4f")
                val_str = f"{val:{fmt}}"
                if val == best.get(m):
                    val_str = f"\\textbf{{{val_str}}}"
                parts.append(val_str)
        lines.append(" & ".join(parts) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def format_markdown_table(table: dict) -> str:
    """Format results as a Markdown table."""
    metrics = table["metrics"]
    best = table["best"]
    rows = table["rows"]

    header = "| Experiment |"
    separator = "|---|"
    for m in metrics:
        info = METRIC_INFO.get(m, {"name": m})
        header += f" {info.get('name', m)} |"
        separator += "---|"

    lines = [header, separator]

    for row in rows:
        line = f"| {row['name']} |"
        for m in metrics:
            val = row.get(m)
            if val is None:
                line += " -- |"
            else:
                info = METRIC_INFO.get(m, {"fmt": ".4f"})
                fmt = info.get("fmt", ".4f")
                val_str = f"{val:{fmt}}"
                if val == best.get(m):
                    val_str = f"**{val_str}**"
                line += f" {val_str} |"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Ablation study comparison")
    parser.add_argument("--experiments", nargs="+", help="Experiment directories to compare")
    parser.add_argument("--names", nargs="+", help="Display names for experiments")
    parser.add_argument("--tracker", default=None, help="ExperimentTracker directory")
    parser.add_argument("--ids", nargs="+", help="Experiment IDs (with --tracker)")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to compare")
    parser.add_argument(
        "--baseline", default=None, help="Baseline experiment name for delta computation"
    )
    parser.add_argument("--format", default="text", choices=["text", "latex", "markdown", "json"])
    parser.add_argument("--output", default=None, help="Save output to file")
    args = parser.parse_args()

    experiments = OrderedDict()

    if args.tracker and args.ids:
        tracker_data = load_from_tracker(args.tracker, args.ids)
        for exp_id, data in tracker_data.items():
            experiments[exp_id] = data
    elif args.experiments:
        names = args.names or [Path(d).name for d in args.experiments]
        for name, exp_dir in zip(names, args.experiments):
            metrics = load_experiment_metrics(exp_dir)
            experiments[name] = {"name": name, "results": metrics}
    else:
        # Demo mode with example data
        print("No experiments specified. Running demo with example data.\n")
        experiments = OrderedDict(
            {
                "baseline": {
                    "name": "Baseline (TPS only)",
                    "results": {
                        "ssim": 0.812,
                        "lpips": 0.189,
                        "fid": 87.3,
                        "identity_sim": 0.72,
                        "nme": 0.032,
                    },
                },
                "phase_a": {
                    "name": "Phase A",
                    "results": {
                        "ssim": 0.856,
                        "lpips": 0.142,
                        "fid": 62.1,
                        "identity_sim": 0.78,
                        "nme": 0.025,
                    },
                },
                "phase_a_curriculum": {
                    "name": "+ Curriculum",
                    "results": {
                        "ssim": 0.867,
                        "lpips": 0.131,
                        "fid": 55.8,
                        "identity_sim": 0.80,
                        "nme": 0.023,
                    },
                },
                "phase_b": {
                    "name": "+ Phase B (4-loss)",
                    "results": {
                        "ssim": 0.891,
                        "lpips": 0.108,
                        "fid": 42.5,
                        "identity_sim": 0.88,
                        "nme": 0.019,
                    },
                },
                "phase_b_ema": {
                    "name": "+ EMA",
                    "results": {
                        "ssim": 0.897,
                        "lpips": 0.101,
                        "fid": 39.2,
                        "identity_sim": 0.89,
                        "nme": 0.018,
                    },
                },
            }
        )

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    print(f"Comparing {len(experiments)} experiments")

    # Build table
    table = compute_ablation_table(experiments, metrics=args.metrics)
    deltas = compute_deltas(table, baseline_name=args.baseline)

    # Format output
    if args.format == "text":
        output = format_text_table(table, deltas)
    elif args.format == "latex":
        output = format_latex_table(table)
    elif args.format == "markdown":
        output = format_markdown_table(table)
    elif args.format == "json":
        output = json.dumps(
            {
                "table": table,
                "deltas": deltas,
            },
            indent=2,
            default=str,
        )

    print()
    print(output)

    if args.output:
        Path(args.output).write_text(output)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
