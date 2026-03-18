"""Metrics aggregation across checkpoints, experiments, and procedures.

Collects evaluation results from multiple sources and computes aggregate
statistics, confidence intervals, and significance tests for paper reporting.

Usage:
    from landmarkdiff.metrics_agg import MetricsAggregator

    agg = MetricsAggregator()
    agg.add("baseline", "rhinoplasty", {"ssim": 0.82, "lpips": 0.18})
    agg.add("ours", "rhinoplasty", {"ssim": 0.91, "lpips": 0.09})
    print(agg.summary_table())
    print(agg.improvement_over("baseline"))
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricRecord:
    """A single evaluation record."""

    experiment: str
    procedure: str
    metrics: dict[str, float]
    checkpoint_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsAggregator:
    """Aggregate and analyze evaluation metrics.

    Supports multiple experiments, procedures, and per-sample results
    for computing confidence intervals and significance.
    """

    HIGHER_BETTER = {
        "ssim": True,
        "psnr": True,
        "identity_sim": True,
        "lpips": False,
        "fid": False,
        "nme": False,
    }

    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def add(
        self,
        experiment: str,
        procedure: str,
        metrics: dict[str, float],
        checkpoint_step: int | None = None,
        **metadata: Any,
    ) -> None:
        """Add a single evaluation record."""
        self.records.append(
            MetricRecord(
                experiment=experiment,
                procedure=procedure,
                metrics=metrics,
                checkpoint_step=checkpoint_step,
                metadata=metadata,
            )
        )

    def add_batch(
        self,
        experiment: str,
        records: list[dict[str, Any]],
    ) -> None:
        """Add multiple records for an experiment.

        Each record dict should have 'procedure' and metric keys.
        """
        for rec in records:
            proc = rec.get("procedure", "all")
            metrics = {
                k: v for k, v in rec.items() if k != "procedure" and isinstance(v, int | float)
            }
            self.add(experiment, proc, metrics)

    @property
    def experiments(self) -> list[str]:
        """Unique experiment names in insertion order."""
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r.experiment, None)
        return list(seen.keys())

    @property
    def procedures(self) -> list[str]:
        """Unique procedure names in insertion order."""
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r.procedure, None)
        return list(seen.keys())

    @property
    def metric_names(self) -> list[str]:
        """All unique metric names."""
        names: set[str] = set()
        for r in self.records:
            names.update(r.metrics.keys())
        return sorted(names)

    def filter(
        self,
        experiment: str | None = None,
        procedure: str | None = None,
    ) -> list[MetricRecord]:
        """Filter records by experiment and/or procedure."""
        results = self.records
        if experiment is not None:
            results = [r for r in results if r.experiment == experiment]
        if procedure is not None:
            results = [r for r in results if r.procedure == procedure]
        return results

    def mean(
        self,
        experiment: str,
        metric: str,
        procedure: str | None = None,
    ) -> float:
        """Compute mean of a metric for an experiment."""
        recs = self.filter(experiment=experiment, procedure=procedure)
        vals = [r.metrics[metric] for r in recs if metric in r.metrics]
        if not vals:
            return float("nan")
        return sum(vals) / len(vals)

    def std(
        self,
        experiment: str,
        metric: str,
        procedure: str | None = None,
    ) -> float:
        """Compute standard deviation of a metric."""
        recs = self.filter(experiment=experiment, procedure=procedure)
        vals = [r.metrics[metric] for r in recs if metric in r.metrics]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var)

    def ci_95(
        self,
        experiment: str,
        metric: str,
        procedure: str | None = None,
    ) -> tuple[float, float]:
        """Compute 95% confidence interval (mean +/- 1.96*SE)."""
        recs = self.filter(experiment=experiment, procedure=procedure)
        vals = [r.metrics[metric] for r in recs if metric in r.metrics]
        if not vals:
            return (float("nan"), float("nan"))
        n = len(vals)
        m = sum(vals) / n
        if n < 2:
            return (m, m)
        var = sum((v - m) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
        return (m - 1.96 * se, m + 1.96 * se)

    def improvement_over(
        self,
        baseline: str,
        metric: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute relative improvement of all experiments over a baseline.

        Returns:
            {experiment: {metric: relative_improvement_pct}}
        """
        metrics = [metric] if metric else self.metric_names
        result: dict[str, dict[str, float]] = {}

        for exp in self.experiments:
            if exp == baseline:
                continue
            improvements: dict[str, float] = {}
            for m in metrics:
                base_val = self.mean(baseline, m)
                exp_val = self.mean(exp, m)
                if math.isnan(base_val) or math.isnan(exp_val) or base_val == 0:
                    continue

                higher_better = self.HIGHER_BETTER.get(m, True)
                if higher_better:
                    pct = (exp_val - base_val) / abs(base_val) * 100
                else:
                    pct = (base_val - exp_val) / abs(base_val) * 100
                improvements[m] = round(pct, 2)

            result[exp] = improvements

        return result

    def best_experiment(
        self,
        metric: str,
        procedure: str | None = None,
    ) -> str | None:
        """Find the experiment with the best mean for a metric."""
        higher_better = self.HIGHER_BETTER.get(metric, True)
        best_exp = None
        best_val = float("-inf") if higher_better else float("inf")

        for exp in self.experiments:
            val = self.mean(exp, metric, procedure)
            if math.isnan(val):
                continue
            if (higher_better and val > best_val) or (not higher_better and val < best_val):
                best_val = val
                best_exp = exp

        return best_exp

    def summary_table(
        self,
        metrics: list[str] | None = None,
        procedure: str | None = None,
        include_std: bool = False,
    ) -> str:
        """Generate a text summary table.

        Args:
            metrics: Metrics to include. None = all.
            procedure: Filter by procedure. None = aggregate.
            include_std: Show mean +/- std.

        Returns:
            Formatted text table.
        """
        metrics = metrics or self.metric_names
        exps = self.experiments

        # Header
        cols = ["Experiment"] + metrics
        header = " | ".join(f"{c:>16s}" for c in cols)
        lines = [header, "-" * len(header)]

        for exp in exps:
            parts = [f"{exp:>16s}"]
            for m in metrics:
                val = self.mean(exp, m, procedure)
                if math.isnan(val):
                    parts.append(f"{'--':>16s}")
                elif include_std:
                    s = self.std(exp, m, procedure)
                    parts.append(f"{val:>8.4f}±{s:<6.4f}")
                else:
                    parts.append(f"{val:>16.4f}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def to_json(self, path: str | Path | None = None) -> str:
        """Export all records as JSON.

        Args:
            path: Optional file path to write to.

        Returns:
            JSON string.
        """
        data = {
            "experiments": self.experiments,
            "procedures": self.procedures,
            "metrics": self.metric_names,
            "records": [
                {
                    "experiment": r.experiment,
                    "procedure": r.procedure,
                    "metrics": r.metrics,
                    "checkpoint_step": r.checkpoint_step,
                    "metadata": r.metadata,
                }
                for r in self.records
            ],
        }
        j = json.dumps(data, indent=2)

        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(j)

        return j

    @staticmethod
    def from_json(path: str | Path) -> MetricsAggregator:
        """Load aggregator from JSON."""
        with open(path) as f:
            data = json.load(f)

        agg = MetricsAggregator()
        for rec in data.get("records", []):
            agg.add(
                experiment=rec["experiment"],
                procedure=rec["procedure"],
                metrics=rec["metrics"],
                checkpoint_step=rec.get("checkpoint_step"),
            )
        return agg
