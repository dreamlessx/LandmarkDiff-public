"""Tests for landmarkdiff.metrics_viz."""

from __future__ import annotations

import pytest

from landmarkdiff.metrics_viz import MetricsVisualizer


@pytest.fixture
def viz(tmp_path):
    return MetricsVisualizer(output_dir=tmp_path, dpi=72)


@pytest.fixture
def procedure_metrics():
    return {
        "rhinoplasty": {"ssim": 0.89, "lpips": 0.11, "identity_sim": 0.87},
        "blepharoplasty": {"ssim": 0.91, "lpips": 0.09, "identity_sim": 0.90},
        "rhytidectomy": {"ssim": 0.85, "lpips": 0.15, "identity_sim": 0.82},
        "orthognathic": {"ssim": 0.83, "lpips": 0.18, "identity_sim": 0.79},
    }


@pytest.fixture
def experiment_metrics():
    return {
        "Baseline": {"ssim": 0.80, "lpips": 0.20, "identity_sim": 0.70},
        "Phase A": {"ssim": 0.87, "lpips": 0.13, "identity_sim": 0.82},
        "Phase B": {"ssim": 0.91, "lpips": 0.09, "identity_sim": 0.89},
    }


class TestProcedureComparison:
    def test_creates_figure(self, viz, procedure_metrics):
        path = viz.procedure_comparison(procedure_metrics)
        assert path.exists()
        assert path.suffix == ".pdf"

    def test_custom_filename(self, viz, procedure_metrics):
        path = viz.procedure_comparison(
            procedure_metrics,
            filename="custom.png",
        )
        assert path.name == "custom.png"

    def test_specific_metrics(self, viz, procedure_metrics):
        path = viz.procedure_comparison(
            procedure_metrics,
            metrics=["ssim"],
        )
        assert path.exists()


class TestRadarPlot:
    def test_creates_figure(self, viz, experiment_metrics):
        path = viz.radar_plot(experiment_metrics)
        assert path.exists()

    def test_specific_metrics(self, viz, experiment_metrics):
        path = viz.radar_plot(
            experiment_metrics,
            metrics=["ssim", "lpips"],
        )
        assert path.exists()


class TestFitzpatrickHeatmap:
    def test_creates_figure(self, viz):
        metrics_by_type = {
            "I-II": {"rhinoplasty": 0.91, "blepharoplasty": 0.93},
            "III-IV": {"rhinoplasty": 0.89, "blepharoplasty": 0.90},
            "V-VI": {"rhinoplasty": 0.85, "blepharoplasty": 0.87},
        }
        path = viz.fitzpatrick_heatmap(metrics_by_type)
        assert path.exists()

    def test_different_metric(self, viz):
        metrics_by_type = {
            "I-II": {"rhinoplasty": 0.10},
            "V-VI": {"rhinoplasty": 0.15},
        }
        path = viz.fitzpatrick_heatmap(
            metrics_by_type,
            metric="lpips",
        )
        assert path.exists()


class TestDistributionBoxplot:
    def test_creates_figure(self, viz):
        import numpy as np

        samples = {
            "rhinoplasty": np.random.normal(0.9, 0.05, 50).tolist(),
            "blepharoplasty": np.random.normal(0.85, 0.08, 50).tolist(),
        }
        path = viz.distribution_boxplot(samples)
        assert path.exists()


class TestLatexTable:
    def test_basic_table(self):
        rows = [
            {"name": "Baseline", "ssim": 0.80, "lpips": 0.20},
            {"name": "Ours", "ssim": 0.91, "lpips": 0.09},
        ]
        latex = MetricsVisualizer.to_latex_table(
            rows,
            metrics=["ssim", "lpips"],
        )
        assert "\\begin{table}" in latex
        assert "\\textbf{0.9100}" in latex  # best SSIM
        assert "\\textbf{0.0900}" in latex  # best LPIPS
        assert "SSIM" in latex

    def test_no_highlight(self):
        rows = [
            {"name": "A", "ssim": 0.80},
            {"name": "B", "ssim": 0.90},
        ]
        latex = MetricsVisualizer.to_latex_table(
            rows,
            metrics=["ssim"],
            highlight_best=False,
        )
        assert "\\textbf" not in latex

    def test_custom_caption(self):
        rows = [{"name": "Test", "ssim": 0.5}]
        latex = MetricsVisualizer.to_latex_table(
            rows,
            metrics=["ssim"],
            caption="My custom caption",
            label="tab:custom",
        )
        assert "My custom caption" in latex
        assert "tab:custom" in latex

    def test_missing_values(self):
        rows = [
            {"name": "A", "ssim": 0.80, "fid": None},
        ]
        latex = MetricsVisualizer.to_latex_table(
            rows,
            metrics=["ssim", "fid"],
        )
        assert "--" in latex
