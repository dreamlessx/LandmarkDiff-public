"""Tests for landmarkdiff.metrics_agg."""

from __future__ import annotations

import json
import math

import pytest

from landmarkdiff.metrics_agg import MetricRecord, MetricsAggregator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agg():
    a = MetricsAggregator()
    # Baseline: 2 rhinoplasty records
    a.add("baseline", "rhinoplasty", {"ssim": 0.80, "lpips": 0.20})
    a.add("baseline", "rhinoplasty", {"ssim": 0.82, "lpips": 0.18})
    # Baseline: 2 blepharoplasty records
    a.add("baseline", "blepharoplasty", {"ssim": 0.78, "lpips": 0.22})
    a.add("baseline", "blepharoplasty", {"ssim": 0.80, "lpips": 0.20})
    # Ours: 2 rhinoplasty records
    a.add("ours", "rhinoplasty", {"ssim": 0.90, "lpips": 0.10})
    a.add("ours", "rhinoplasty", {"ssim": 0.92, "lpips": 0.08})
    # Ours: 2 blepharoplasty records
    a.add("ours", "blepharoplasty", {"ssim": 0.88, "lpips": 0.12})
    a.add("ours", "blepharoplasty", {"ssim": 0.86, "lpips": 0.14})
    return a


# ---------------------------------------------------------------------------
# MetricRecord
# ---------------------------------------------------------------------------


class TestMetricRecord:
    def test_fields(self):
        r = MetricRecord(experiment="test", procedure="rhinoplasty", metrics={"ssim": 0.9})
        assert r.experiment == "test"
        assert r.metrics["ssim"] == 0.9
        assert r.checkpoint_step is None

    def test_with_metadata(self):
        r = MetricRecord(
            experiment="test",
            procedure="rhinoplasty",
            metrics={"ssim": 0.9},
            metadata={"note": "best"},
        )
        assert r.metadata["note"] == "best"


# ---------------------------------------------------------------------------
# Add records
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_single(self):
        a = MetricsAggregator()
        a.add("exp1", "rhinoplasty", {"ssim": 0.9})
        assert len(a.records) == 1

    def test_add_batch(self):
        a = MetricsAggregator()
        a.add_batch(
            "exp1",
            [
                {"procedure": "rhinoplasty", "ssim": 0.9, "lpips": 0.1},
                {"procedure": "blepharoplasty", "ssim": 0.85, "lpips": 0.15},
            ],
        )
        assert len(a.records) == 2
        assert a.records[0].procedure == "rhinoplasty"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_experiments(self, agg):
        assert agg.experiments == ["baseline", "ours"]

    def test_procedures(self, agg):
        assert set(agg.procedures) == {"rhinoplasty", "blepharoplasty"}

    def test_metric_names(self, agg):
        assert set(agg.metric_names) == {"ssim", "lpips"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_mean(self, agg):
        assert agg.mean("baseline", "ssim", "rhinoplasty") == pytest.approx(0.81)
        assert agg.mean("ours", "ssim", "rhinoplasty") == pytest.approx(0.91)

    def test_mean_all_procedures(self, agg):
        # Baseline: (0.80 + 0.82 + 0.78 + 0.80) / 4 = 0.80
        assert agg.mean("baseline", "ssim") == pytest.approx(0.80)

    def test_mean_missing_metric(self, agg):
        assert math.isnan(agg.mean("baseline", "fid"))

    def test_std(self, agg):
        s = agg.std("baseline", "ssim", "rhinoplasty")
        assert s > 0

    def test_std_single_record(self):
        a = MetricsAggregator()
        a.add("exp", "rhinoplasty", {"ssim": 0.9})
        assert a.std("exp", "ssim") == 0.0

    def test_ci_95(self, agg):
        low, high = agg.ci_95("baseline", "ssim", "rhinoplasty")
        mean = agg.mean("baseline", "ssim", "rhinoplasty")
        assert low < mean
        assert high > mean
        assert low < high

    def test_ci_95_single(self):
        a = MetricsAggregator()
        a.add("exp", "rhinoplasty", {"ssim": 0.9})
        low, high = a.ci_95("exp", "ssim")
        assert low == high == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Improvement
# ---------------------------------------------------------------------------


class TestImprovement:
    def test_improvement_over(self, agg):
        imp = agg.improvement_over("baseline")
        assert "ours" in imp
        # SSIM: (0.90 - 0.80) / 0.80 = 12.5% for all procedures combined
        # Our mean: (0.90 + 0.92 + 0.88 + 0.86)/4 = 0.89
        # Baseline mean: 0.80
        # (0.89 - 0.80) / 0.80 = 11.25%
        assert imp["ours"]["ssim"] > 0  # positive = improvement

    def test_improvement_lpips(self, agg):
        imp = agg.improvement_over("baseline")
        # LPIPS: lower is better, so improvement = (baseline - ours) / baseline
        assert imp["ours"]["lpips"] > 0  # positive = improvement (we reduced LPIPS)

    def test_improvement_specific_metric(self, agg):
        imp = agg.improvement_over("baseline", metric="ssim")
        assert "ssim" in imp["ours"]
        assert "lpips" not in imp["ours"]

    def test_no_self_comparison(self, agg):
        imp = agg.improvement_over("baseline")
        assert "baseline" not in imp


# ---------------------------------------------------------------------------
# Best experiment
# ---------------------------------------------------------------------------


class TestBestExperiment:
    def test_best_ssim(self, agg):
        assert agg.best_experiment("ssim") == "ours"

    def test_best_lpips(self, agg):
        assert agg.best_experiment("lpips") == "ours"

    def test_best_by_procedure(self, agg):
        assert agg.best_experiment("ssim", procedure="rhinoplasty") == "ours"

    def test_best_unknown_metric(self, agg):
        assert agg.best_experiment("nonexistent") is None


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_table_has_all_experiments(self, agg):
        table = agg.summary_table()
        assert "baseline" in table
        assert "ours" in table

    def test_table_has_metrics(self, agg):
        table = agg.summary_table(metrics=["ssim"])
        assert "ssim" in table

    def test_table_with_std(self, agg):
        table = agg.summary_table(include_std=True)
        assert "±" in table


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestJson:
    def test_to_json(self, agg):
        j = agg.to_json()
        data = json.loads(j)
        assert len(data["records"]) == 8
        assert "baseline" in data["experiments"]

    def test_to_json_file(self, agg, tmp_path):
        path = tmp_path / "metrics.json"
        agg.to_json(path)
        assert path.exists()

    def test_roundtrip(self, agg, tmp_path):
        path = tmp_path / "metrics.json"
        agg.to_json(path)
        loaded = MetricsAggregator.from_json(path)
        assert len(loaded.records) == len(agg.records)
        assert loaded.experiments == agg.experiments

    def test_from_json_preserves_metrics(self, agg, tmp_path):
        path = tmp_path / "metrics.json"
        agg.to_json(path)
        loaded = MetricsAggregator.from_json(path)
        assert loaded.mean("baseline", "ssim") == pytest.approx(agg.mean("baseline", "ssim"))


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_by_experiment(self, agg):
        recs = agg.filter(experiment="baseline")
        assert len(recs) == 4
        assert all(r.experiment == "baseline" for r in recs)

    def test_filter_by_procedure(self, agg):
        recs = agg.filter(procedure="rhinoplasty")
        assert len(recs) == 4

    def test_filter_both(self, agg):
        recs = agg.filter(experiment="ours", procedure="rhinoplasty")
        assert len(recs) == 2

    def test_empty_aggregator(self):
        a = MetricsAggregator()
        assert a.experiments == []
        assert a.metric_names == []
        table = a.summary_table()
        assert "Experiment" in table
