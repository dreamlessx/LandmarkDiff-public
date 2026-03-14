"""Publication-quality metrics visualization for LandmarkDiff.

Generates figures suitable for MICCAI/medical imaging papers:
- Bar charts comparing procedures and methods
- Radar plots for multi-metric comparison
- Box plots for per-sample distributions
- Heatmaps for Fitzpatrick equity analysis
- Table formatters for LaTeX

Usage:
    from landmarkdiff.metrics_viz import MetricsVisualizer

    viz = MetricsVisualizer(output_dir="paper/figures")

    # Bar chart comparing procedures
    viz.procedure_comparison(metrics_by_procedure)

    # Radar plot for ablation study
    viz.radar_plot(experiments)

    # Equity heatmap
    viz.fitzpatrick_heatmap(metrics_by_type)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MetricsVisualizer:
    """Generate publication-quality figures from evaluation metrics.

    Args:
        output_dir: Directory to save generated figures.
        dpi: Resolution for saved figures.
        style: Matplotlib style preset.
    """

    # Color palette (colorblind-safe, MICCAI-friendly)
    COLORS = {
        "rhinoplasty": "#4C72B0",
        "blepharoplasty": "#55A868",
        "rhytidectomy": "#C44E52",
        "orthognathic": "#8172B2",
        "baseline": "#CCB974",
        "ours": "#4C72B0",
    }

    METRIC_LABELS = {
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "fid": "FID",
        "nme": "NME",
        "identity_sim": "ID Sim.",
        "psnr": "PSNR (dB)",
    }

    METRIC_HIGHER_BETTER = {
        "ssim": True,
        "lpips": False,
        "fid": False,
        "nme": False,
        "identity_sim": True,
        "psnr": True,
    }

    def __init__(
        self,
        output_dir: str | Path = "figures",
        dpi: int = 300,
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.style = style

    def _get_plt(self):
        """Import matplotlib with configuration."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use("seaborn-v0_8")
        # Publication font sizes
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 13,
            }
        )
        return plt

    # ------------------------------------------------------------------
    # Procedure comparison bar chart
    # ------------------------------------------------------------------

    def procedure_comparison(
        self,
        metrics_by_procedure: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        title: str = "Per-Procedure Performance",
        filename: str = "procedure_comparison.pdf",
    ) -> Path:
        """Generate grouped bar chart comparing procedures.

        Args:
            metrics_by_procedure: {procedure: {metric: value}}.
            metrics: Which metrics to show. None = auto-detect.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        plt = self._get_plt()

        if metrics is None:
            all_metrics: set[str] = set()
            for m in metrics_by_procedure.values():
                all_metrics.update(m.keys())
            metrics = sorted(all_metrics & set(self.METRIC_LABELS.keys()))

        procedures = list(metrics_by_procedure.keys())
        n_procs = len(procedures)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics, strict=False):
            values = [metrics_by_procedure[p].get(metric, 0) for p in procedures]
            colors = [self.COLORS.get(p, "#999999") for p in procedures]

            bars = ax.bar(range(n_procs), values, color=colors, width=0.6, edgecolor="white")
            ax.set_xticks(range(n_procs))
            ax.set_xticklabels(
                [p[:5].title() for p in procedures],
                rotation=30,
                ha="right",
            )
            ax.set_ylabel(self.METRIC_LABELS.get(metric, metric))
            ax.set_title(self.METRIC_LABELS.get(metric, metric))

            # Add value labels on bars
            for bar, val in zip(bars, values, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Radar plot for multi-metric comparison
    # ------------------------------------------------------------------

    def radar_plot(
        self,
        experiments: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
        title: str = "Multi-Metric Comparison",
        filename: str = "radar_plot.pdf",
    ) -> Path:
        """Generate radar/spider plot for comparing experiments.

        Args:
            experiments: {experiment_name: {metric: value}}.
            metrics: Which metrics to show.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        plt = self._get_plt()

        if metrics is None:
            metrics = sorted(
                set.intersection(*(set(v.keys()) for v in experiments.values()))
                & set(self.METRIC_LABELS.keys())
            )

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

        colors = list(self.COLORS.values())
        for i, (name, values_dict) in enumerate(experiments.items()):
            raw_values = []
            for m in metrics:
                val = values_dict.get(m, 0)
                # Normalize: for "lower is better" metrics, invert
                if not self.METRIC_HIGHER_BETTER.get(m, True):
                    val = 1 - min(val, 1)  # Invert so higher = better on plot
                raw_values.append(val)

            # Normalize to [0, 1] range
            vals = np.array(raw_values)
            vals = vals / max(vals.max(), 1e-10)
            vals = vals.tolist() + vals[:1].tolist()

            color = colors[i % len(colors)]
            ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, vals, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.METRIC_LABELS.get(m, m) for m in metrics])
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.set_title(title, fontweight="bold", pad=20)

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Fitzpatrick equity heatmap
    # ------------------------------------------------------------------

    def fitzpatrick_heatmap(
        self,
        metrics_by_type: dict[str, dict[str, float]],
        metric: str = "ssim",
        title: str | None = None,
        filename: str = "fitzpatrick_equity.pdf",
    ) -> Path:
        """Generate heatmap showing metric values across Fitzpatrick types and procedures.

        Args:
            metrics_by_type: {fitzpatrick_type: {procedure: value}}.
            metric: Which metric to visualize.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        plt = self._get_plt()

        fitz_types = sorted(metrics_by_type.keys())
        procedures = sorted(set.union(*(set(v.keys()) for v in metrics_by_type.values())))

        # Build matrix
        matrix = np.zeros((len(fitz_types), len(procedures)))
        for i, ft in enumerate(fitz_types):
            for j, proc in enumerate(procedures):
                matrix[i, j] = metrics_by_type[ft].get(proc, 0)

        fig, ax = plt.subplots(
            figsize=(max(6, len(procedures) * 1.5), max(4, len(fitz_types) * 0.8))
        )

        cmap = "RdYlGn" if self.METRIC_HIGHER_BETTER.get(metric, True) else "RdYlGn_r"
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(procedures)))
        ax.set_xticklabels([p.title() for p in procedures], rotation=30, ha="right")
        ax.set_yticks(range(len(fitz_types)))
        ax.set_yticklabels(fitz_types)
        ax.set_ylabel("Fitzpatrick Type")

        # Annotate cells
        for i in range(len(fitz_types)):
            for j in range(len(procedures)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if matrix[i, j] < np.median(matrix) else "black",
                )

        fig.colorbar(im, ax=ax, label=self.METRIC_LABELS.get(metric, metric))

        if title is None:
            title = f"{self.METRIC_LABELS.get(metric, metric)} by Fitzpatrick Type"
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Box plots for per-sample distribution
    # ------------------------------------------------------------------

    def distribution_boxplot(
        self,
        samples_by_group: dict[str, list[float]],
        metric: str = "ssim",
        title: str | None = None,
        filename: str = "distribution.pdf",
    ) -> Path:
        """Generate box plot showing per-sample metric distributions.

        Args:
            samples_by_group: {group_name: [sample_values]}.
            metric: Metric being plotted.
            title: Figure title.
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        plt = self._get_plt()

        groups = list(samples_by_group.keys())
        data = [samples_by_group[g] for g in groups]

        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.5},
        )

        colors = [self.COLORS.get(g, "#4C72B0") for g in groups]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(
            [g.title() for g in groups],
            rotation=30,
            ha="right",
        )
        ax.set_ylabel(self.METRIC_LABELS.get(metric, metric))

        if title is None:
            title = f"{self.METRIC_LABELS.get(metric, metric)} Distribution"
        ax.set_title(title, fontweight="bold")

        # Add sample count annotations
        for i, (_g, vals) in enumerate(zip(groups, data, strict=False)):
            ax.text(
                i + 1,
                ax.get_ylim()[0],
                f"n={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="gray",
            )

        fig.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # LaTeX table formatter
    # ------------------------------------------------------------------

    @staticmethod
    def to_latex_table(
        rows: list[dict[str, Any]],
        metrics: list[str],
        caption: str = "Quantitative results",
        label: str = "tab:results",
        highlight_best: bool = True,
    ) -> str:
        """Format metrics as a LaTeX table.

        Args:
            rows: List of dicts with 'name' and metric values.
            metrics: List of metric names to include.
            caption: Table caption.
            label: LaTeX label.
            highlight_best: Bold the best value per column.

        Returns:
            LaTeX table string.
        """
        metric_labels = MetricsVisualizer.METRIC_LABELS
        higher_better = MetricsVisualizer.METRIC_HIGHER_BETTER

        # Find best values
        best: dict[str, float] = {}
        if highlight_best:
            for m in metrics:
                vals = [r.get(m) for r in rows if r.get(m) is not None]
                if vals:
                    if higher_better.get(m, True):
                        best[m] = max(vals)
                    else:
                        best[m] = min(vals)

        cols = "l" + "c" * len(metrics)
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{cols}}}",
            "\\toprule",
        ]

        # Header
        header = ["Method"]
        for m in metrics:
            name = metric_labels.get(m, m)
            arrow = "$\\uparrow$" if higher_better.get(m, True) else "$\\downarrow$"
            header.append(f"{name} {arrow}")
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        # Data rows
        for row in rows:
            parts = [row.get("name", "").replace("_", "\\_")]
            for m in metrics:
                val = row.get(m)
                if val is None:
                    parts.append("--")
                else:
                    fmt = ".4f" if abs(val) < 10 else ".1f"
                    val_str = f"{val:{fmt}}"
                    if highlight_best and val == best.get(m):
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
