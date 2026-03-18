"""Analyze evaluation failures and identify patterns in poor-performing samples.

Parses evaluation results to find worst-performing samples, groups failures
by procedure type and Fitzpatrick skin type, and generates diagnostic reports.

Usage:
    python scripts/analyze_failures.py --results eval_results/metrics.json
    python scripts/analyze_failures.py --results eval_results/ --top-k 20
    python scripts/analyze_failures.py --results eval_results/ --export failures.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_path: str | Path) -> list[dict]:
    """Load per-sample evaluation results.

    Supports:
    - Single JSON file with a "samples" key
    - Directory with per-sample JSON files
    - CSV file with metric columns
    """
    path = Path(results_path)

    if path.is_file() and path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if "samples" in data:
            return data["samples"]
        elif isinstance(data, list):
            return data
        else:
            return [data]

    elif path.is_file() and path.suffix == ".csv":
        samples = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                sample = {}
                for k, v in row.items():
                    try:
                        sample[k] = float(v)
                    except (ValueError, TypeError):
                        sample[k] = v
                samples.append(sample)
        return samples

    elif path.is_dir():
        # Look for metrics files
        samples = []
        for f in sorted(path.glob("*.json")):
            if f.name in ("summary.json", "config.json"):
                continue
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "samples" in data:
                    samples.extend(data["samples"])
                elif isinstance(data, list):
                    samples.extend(data)
            except Exception:
                pass
        return samples

    raise FileNotFoundError(f"Cannot load results from {results_path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_worst_samples(
    samples: list[dict],
    metric: str = "lpips",
    higher_is_worse: bool = True,
    top_k: int = 10,
) -> list[dict]:
    """Find the worst-performing samples by a given metric.

    Args:
        samples: List of per-sample result dicts.
        metric: Metric name to sort by.
        higher_is_worse: If True, higher values are worse (e.g., LPIPS, FID).
        top_k: Number of worst samples to return.

    Returns:
        List of worst samples sorted by metric (worst first).
    """
    scored = [s for s in samples if metric in s and s[metric] is not None]
    scored.sort(key=lambda s: s[metric], reverse=higher_is_worse)
    return scored[:top_k]


def group_by_procedure(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by surgical procedure."""
    groups = defaultdict(list)
    for s in samples:
        proc = s.get("procedure", "unknown")
        groups[proc].append(s)
    return dict(groups)


def group_by_fitzpatrick(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by Fitzpatrick skin type."""
    groups = defaultdict(list)
    for s in samples:
        fitz = s.get("fitzpatrick", s.get("skin_type", "unknown"))
        groups[str(fitz)].append(s)
    return dict(groups)


def compute_group_stats(groups: dict[str, list[dict]], metrics: list[str]) -> dict:
    """Compute per-group statistics for given metrics."""
    stats = {}
    for group_name, group_samples in sorted(groups.items()):
        group_stats = {"n": len(group_samples)}
        for metric in metrics:
            values = [s[metric] for s in group_samples if metric in s and s[metric] is not None]
            if values:
                group_stats[metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                    "median": round(float(np.median(values)), 4),
                }
        stats[group_name] = group_stats
    return stats


def detect_failure_patterns(samples: list[dict], thresholds: dict | None = None) -> dict:
    """Detect common failure patterns.

    Args:
        samples: Per-sample results.
        thresholds: Dict mapping metric -> (threshold, higher_is_failure).
            Default thresholds used if None.

    Returns:
        Dict with failure pattern analysis.
    """
    if thresholds is None:
        thresholds = {
            "lpips": (0.3, True),  # LPIPS > 0.3 is poor
            "ssim": (0.7, False),  # SSIM < 0.7 is poor
            "identity_sim": (0.6, False),  # Identity < 0.6 is poor
            "nme": (0.05, True),  # NME > 0.05 is poor
        }

    failures = defaultdict(list)
    multi_failures = []

    for s in samples:
        sample_failures = []
        for metric, (thresh, higher_is_bad) in thresholds.items():
            if metric not in s or s[metric] is None:
                continue
            val = s[metric]
            if (higher_is_bad and val > thresh) or (not higher_is_bad and val < thresh):
                sample_failures.append(metric)
                failures[metric].append(s)

        if len(sample_failures) > 1:
            multi_failures.append(
                {
                    "sample": s.get("name", s.get("prefix", "unknown")),
                    "failed_metrics": sample_failures,
                }
            )

    return {
        "per_metric_failures": {k: len(v) for k, v in failures.items()},
        "multi_metric_failures": len(multi_failures),
        "worst_multi_failures": sorted(
            multi_failures, key=lambda x: len(x["failed_metrics"]), reverse=True
        )[:10],
        "total_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_report(
    samples: list[dict],
    worst: list[dict],
    proc_stats: dict,
    fitz_stats: dict,
    failure_patterns: dict,
    metric: str,
) -> str:
    """Format a failure analysis report."""
    lines = [
        "=" * 70,
        "LandmarkDiff Failure Analysis Report",
        "=" * 70,
        "",
        f"Total samples analyzed: {len(samples)}",
        f"Primary failure metric: {metric}",
        "",
    ]

    # Worst samples
    lines.append(f"--- Top {len(worst)} Worst Samples (by {metric}) ---")
    for i, s in enumerate(worst):
        name = s.get("name", s.get("prefix", f"sample_{i}"))
        val = s.get(metric, "N/A")
        proc = s.get("procedure", "?")
        lines.append(f"  {i + 1:3d}. {name:<30s} {metric}={val:.4f}  proc={proc}")
    lines.append("")

    # Per-procedure breakdown
    lines.append("--- Per-Procedure Statistics ---")
    metrics_to_show = ["ssim", "lpips", "identity_sim", "nme"]
    for proc, stats in proc_stats.items():
        lines.append(f"\n  {proc} (n={stats['n']}):")
        for m in metrics_to_show:
            if m in stats:
                s = stats[m]
                lines.append(
                    f"    {m:<15s}: {s['mean']:.4f} ± {s['std']:.4f} "
                    f"(range: [{s['min']:.4f}, {s['max']:.4f}])"
                )

    # Per-Fitzpatrick breakdown
    if fitz_stats and any(k != "unknown" for k in fitz_stats):
        lines.append("\n--- Per-Fitzpatrick Type Statistics ---")
        for fitz, stats in fitz_stats.items():
            lines.append(f"\n  Type {fitz} (n={stats['n']}):")
            for m in metrics_to_show:
                if m in stats:
                    s = stats[m]
                    lines.append(f"    {m:<15s}: {s['mean']:.4f} ± {s['std']:.4f}")

    # Failure patterns
    lines.append("\n--- Failure Pattern Analysis ---")
    fp = failure_patterns
    lines.append("  Per-metric failures:")
    for m, count in fp["per_metric_failures"].items():
        pct = (count / fp["total_samples"]) * 100 if fp["total_samples"] > 0 else 0
        lines.append(f"    {m:<20s}: {count:4d} ({pct:.1f}%)")
    lines.append(f"  Multi-metric failures: {fp['multi_metric_failures']}")

    if fp["worst_multi_failures"]:
        lines.append("  Worst multi-failures:")
        for entry in fp["worst_multi_failures"][:5]:
            lines.append(f"    {entry['sample']}: {', '.join(entry['failed_metrics'])}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def export_csv(worst: list[dict], output_path: str, metrics: list[str]) -> None:
    """Export worst samples to CSV."""
    if not worst:
        return

    fieldnames = ["rank", "name", "procedure"] + metrics
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, s in enumerate(worst):
            row = {
                "rank": i + 1,
                "name": s.get("name", s.get("prefix", f"sample_{i}")),
                "procedure": s.get("procedure", "unknown"),
            }
            for m in metrics:
                row[m] = s.get(m, "")
            writer.writerow(row)
    print(f"Exported to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation failures")
    parser.add_argument(
        "--results", required=True, help="Path to evaluation results (JSON, CSV, or directory)"
    )
    parser.add_argument("--metric", default="lpips", help="Primary metric for ranking failures")
    parser.add_argument("--top-k", type=int, default=10, help="Number of worst samples to show")
    parser.add_argument("--export", default=None, help="Export worst samples to CSV")
    parser.add_argument(
        "--higher-is-worse",
        action="store_true",
        default=True,
        help="Whether higher metric values are worse",
    )
    parser.add_argument("--output", default=None, help="Save report to file")
    args = parser.parse_args()

    # Load results
    try:
        samples = load_results(args.results)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not samples:
        print("No samples found in results.")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from {args.results}")

    # Detect available metrics
    all_metrics = set()
    for s in samples:
        for k, v in s.items():
            if isinstance(v, int | float):
                all_metrics.add(k)
    metrics = sorted(all_metrics & {"ssim", "lpips", "identity_sim", "nme", "fid"})
    if not metrics:
        metrics = sorted(all_metrics)[:5]

    print(f"Available metrics: {metrics}")

    # Analysis
    worst = analyze_worst_samples(samples, args.metric, args.higher_is_worse, args.top_k)

    proc_groups = group_by_procedure(samples)
    proc_stats = compute_group_stats(proc_groups, metrics)

    fitz_groups = group_by_fitzpatrick(samples)
    fitz_stats = compute_group_stats(fitz_groups, metrics)

    failure_patterns = detect_failure_patterns(samples)

    # Report
    report = format_report(samples, worst, proc_stats, fitz_stats, failure_patterns, args.metric)
    print()
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")

    if args.export:
        export_csv(worst, args.export, metrics)


if __name__ == "__main__":
    main()
