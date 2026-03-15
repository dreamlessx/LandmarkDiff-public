"""Tests for the experiment tracker module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.experiment_tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temporary directory."""
    return ExperimentTracker(str(tmp_path / "experiments"))


class TestTrackerInit:
    """Tests for tracker initialization."""

    def test_creates_directory(self, tmp_path):
        exp_dir = tmp_path / "new_experiments"
        assert not exp_dir.exists()
        ExperimentTracker(str(exp_dir))
        assert exp_dir.exists()

    def test_loads_existing_index(self, tmp_path):
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        index = {"experiments": {}, "counter": 5}
        (exp_dir / "index.json").write_text(json.dumps(index))
        tracker = ExperimentTracker(str(exp_dir))
        assert tracker._index["counter"] == 5

    def test_fresh_index(self, tracker):
        assert tracker._index["counter"] == 0
        assert tracker._index["experiments"] == {}


class TestStartExperiment:
    """Tests for starting experiments."""

    def test_start_returns_id(self, tracker):
        exp_id = tracker.start("test_run", config={"lr": 1e-5})
        assert exp_id == "exp_001"

    def test_start_increments_counter(self, tracker):
        id1 = tracker.start("run1", config={})
        id2 = tracker.start("run2", config={})
        assert id1 == "exp_001"
        assert id2 == "exp_002"

    def test_start_stores_config(self, tracker):
        config = {"lr": 1e-5, "batch": 4, "steps": 1000}
        exp_id = tracker.start("test", config=config)
        exp = tracker._index["experiments"][exp_id]
        assert exp["config"] == config

    def test_start_stores_name(self, tracker):
        exp_id = tracker.start("my_experiment", config={})
        assert tracker._index["experiments"][exp_id]["name"] == "my_experiment"

    def test_start_sets_status_running(self, tracker):
        exp_id = tracker.start("test", config={})
        assert tracker._index["experiments"][exp_id]["status"] == "running"

    def test_start_with_tags(self, tracker):
        exp_id = tracker.start("test", config={}, tags=["phase_a", "v2"])
        assert tracker._index["experiments"][exp_id]["tags"] == ["phase_a", "v2"]

    def test_start_creates_metrics_file(self, tracker):
        exp_id = tracker.start("test", config={})
        metrics_path = tracker.dir / f"{exp_id}_metrics.jsonl"
        assert metrics_path.exists()

    def test_start_persists_to_disk(self, tracker):
        tracker.start("test", config={"lr": 1e-5})
        # Reload from disk
        reloaded = ExperimentTracker(str(tracker.dir))
        assert reloaded._index["counter"] == 1
        assert "exp_001" in reloaded._index["experiments"]


class TestLogMetric:
    """Tests for logging metrics."""

    def test_log_basic_metric(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.log_metric(exp_id, step=100, loss=0.05)

        metrics = tracker.get_metrics(exp_id)
        assert len(metrics) == 1
        assert metrics[0]["step"] == 100
        assert metrics[0]["loss"] == 0.05

    def test_log_multiple_metrics(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.log_metric(exp_id, step=100, loss=0.05, ssim=0.8)
        tracker.log_metric(exp_id, step=200, loss=0.03, ssim=0.85)

        metrics = tracker.get_metrics(exp_id)
        assert len(metrics) == 2
        assert metrics[1]["loss"] == 0.03

    def test_log_unknown_experiment_noop(self, tracker):
        # Should not raise
        tracker.log_metric("nonexistent", step=100, loss=0.1)

    def test_metrics_have_timestamp(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.log_metric(exp_id, step=1, loss=0.1)
        metrics = tracker.get_metrics(exp_id)
        assert "timestamp" in metrics[0]
        assert isinstance(metrics[0]["timestamp"], float)


class TestFinishExperiment:
    """Tests for finishing experiments."""

    def test_finish_sets_completed(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.finish(exp_id, results={"fid": 42.0})
        assert tracker._index["experiments"][exp_id]["status"] == "completed"

    def test_finish_stores_results(self, tracker):
        exp_id = tracker.start("test", config={})
        results = {"fid": 42.0, "ssim": 0.87, "lpips": 0.12}
        tracker.finish(exp_id, results=results)
        assert tracker._index["experiments"][exp_id]["results"] == results

    def test_finish_sets_timestamp(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.finish(exp_id)
        assert tracker._index["experiments"][exp_id]["finished_at"] is not None

    def test_finish_with_custom_status(self, tracker):
        exp_id = tracker.start("test", config={})
        tracker.finish(exp_id, status="failed")
        assert tracker._index["experiments"][exp_id]["status"] == "failed"

    def test_finish_unknown_experiment_noop(self, tracker):
        tracker.finish("nonexistent", results={"fid": 1.0})


class TestListAndCompare:
    """Tests for listing and comparing experiments."""

    def test_list_empty(self, tracker):
        experiments = tracker.list_experiments()
        assert experiments == []

    def test_list_returns_summaries(self, tracker):
        tracker.start("run1", config={"lr": 1e-5})
        tracker.start("run2", config={"lr": 5e-6})
        experiments = tracker.list_experiments()
        assert len(experiments) == 2
        assert experiments[0]["name"] == "run1"
        assert experiments[1]["name"] == "run2"

    def test_list_includes_results_metrics(self, tracker):
        exp_id = tracker.start("run1", config={})
        tracker.finish(exp_id, results={"fid": 42.0, "ssim": 0.87})
        experiments = tracker.list_experiments()
        assert experiments[0]["fid"] == 42.0
        assert experiments[0]["ssim"] == 0.87

    def test_compare_experiments(self, tracker):
        id1 = tracker.start("run1", config={"lr": 1e-5})
        id2 = tracker.start("run2", config={"lr": 5e-6})
        tracker.finish(id1, results={"fid": 42.0})
        tracker.finish(id2, results={"fid": 38.0})

        comparison = tracker.compare([id1, id2])
        assert id1 in comparison
        assert id2 in comparison
        assert comparison[id1]["results"]["fid"] == 42.0
        assert comparison[id2]["results"]["fid"] == 38.0

    def test_compare_missing_experiment(self, tracker):
        id1 = tracker.start("run1", config={})
        comparison = tracker.compare([id1, "nonexistent"])
        assert id1 in comparison
        assert "nonexistent" not in comparison


class TestGetBest:
    """Tests for finding best experiment."""

    def test_get_best_fid(self, tracker):
        id1 = tracker.start("run1", config={})
        id2 = tracker.start("run2", config={})
        id3 = tracker.start("run3", config={})
        tracker.finish(id1, results={"fid": 42.0})
        tracker.finish(id2, results={"fid": 38.0})
        tracker.finish(id3, results={"fid": 45.0})

        best = tracker.get_best("fid", lower_is_better=True)
        assert best == id2  # 38.0 is lowest

    def test_get_best_ssim(self, tracker):
        id1 = tracker.start("run1", config={})
        id2 = tracker.start("run2", config={})
        tracker.finish(id1, results={"ssim": 0.87})
        tracker.finish(id2, results={"ssim": 0.92})

        best = tracker.get_best("ssim", lower_is_better=False)
        assert best == id2  # 0.92 is highest

    def test_get_best_skips_running(self, tracker):
        tracker.start("run1", config={})
        id2 = tracker.start("run2", config={})
        tracker.finish(id2, results={"fid": 50.0})
        # id1 is still running

        best = tracker.get_best("fid")
        assert best == id2  # id1 skipped (running)

    def test_get_best_no_experiments(self, tracker):
        assert tracker.get_best("fid") is None

    def test_get_best_missing_metric(self, tracker):
        id1 = tracker.start("run1", config={})
        tracker.finish(id1, results={"ssim": 0.8})
        assert tracker.get_best("fid") is None  # no fid in results


class TestPrintSummary:
    """Tests for summary printing."""

    def test_print_empty(self, tracker, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="landmarkdiff.experiment_tracker"):
            tracker.print_summary()
        assert "No experiments found" in caplog.text

    def test_print_with_experiments(self, tracker, caplog):
        import logging

        exp_id = tracker.start("test_run", config={})
        tracker.finish(exp_id, results={"fid": 42.0, "ssim": 0.87})
        with caplog.at_level(logging.INFO, logger="landmarkdiff.experiment_tracker"):
            tracker.print_summary()
        assert "test_run" in caplog.text
        assert "completed" in caplog.text
