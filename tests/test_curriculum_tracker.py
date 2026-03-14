"""Tests for curriculum learning and experiment tracker modules."""

import json
import tempfile

import numpy as np


class TestTrainingCurriculum:
    """Tests for TrainingCurriculum schedule."""

    def test_difficulty_warmup(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000, warmup_fraction=0.1, full_difficulty_at=0.5)
        # During warmup (first 10%)
        assert c.get_difficulty(0) == 0.0
        assert c.get_difficulty(50) == 0.0
        assert c.get_difficulty(99) == 0.0

    def test_difficulty_ramp(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000, warmup_fraction=0.1, full_difficulty_at=0.5)
        # Midpoint of ramp should be ~0.5
        d = c.get_difficulty(300)
        assert 0.0 < d < 1.0

    def test_difficulty_full(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000, warmup_fraction=0.1, full_difficulty_at=0.5)
        assert c.get_difficulty(500) == 1.0
        assert c.get_difficulty(999) == 1.0

    def test_monotonic_increase(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000)
        prev = -1
        for step in range(0, 1001, 10):
            d = c.get_difficulty(step)
            assert d >= prev, f"Difficulty decreased at step {step}"
            prev = d

    def test_should_include_easy_always(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000)
        rng = np.random.default_rng(42)
        # Easy samples should always be included
        for step in range(0, 1001, 100):
            assert c.should_include(step, 0.0, rng)

    def test_should_include_hard_rejected_early(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=1000)
        rng = np.random.default_rng(42)
        # Hard samples mostly rejected early
        rejections = sum(1 for _ in range(100) if not c.should_include(0, 0.9, rng))
        assert rejections > 50  # most should be rejected


class TestProcedureCurriculum:
    """Tests for ProcedureCurriculum."""

    def test_easy_procedure_higher_than_hard(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        c = ProcedureCurriculum(total_steps=1000)
        # Blepharoplasty is easiest — should have higher weight than orthognathic
        w_easy = c.get_weight(0, "blepharoplasty")
        w_hard = c.get_weight(0, "orthognathic")
        assert w_easy > w_hard

    def test_hard_procedure_low_weight_early(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        c = ProcedureCurriculum(total_steps=1000)
        # Orthognathic is hardest — should have lower weight early
        w = c.get_weight(0, "orthognathic")
        assert w < 0.5

    def test_all_full_weight_at_end(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        c = ProcedureCurriculum(total_steps=1000)
        weights = c.get_procedure_weights(999)
        for proc, w in weights.items():
            assert w == 1.0, f"{proc} weight is {w}, expected 1.0 at end"

    def test_never_zero_weight(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        c = ProcedureCurriculum(total_steps=1000)
        for step in range(0, 1001, 50):
            weights = c.get_procedure_weights(step)
            for proc, w in weights.items():
                assert w >= 0.1, f"{proc} weight is {w} at step {step}"


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_start_and_list(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            exp_id = tracker.start("test_exp", {"lr": 1e-5, "batch": 4})
            assert exp_id == "exp_001"
            exps = tracker.list_experiments()
            assert len(exps) == 1
            assert exps[0]["name"] == "test_exp"
            assert exps[0]["status"] == "running"

    def test_log_metrics(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            exp_id = tracker.start("test_exp", {})
            tracker.log_metric(exp_id, step=100, loss=0.05, ssim=0.8)
            tracker.log_metric(exp_id, step=200, loss=0.03, ssim=0.85)
            metrics = tracker.get_metrics(exp_id)
            assert len(metrics) == 2
            assert metrics[0]["step"] == 100
            assert metrics[1]["loss"] == 0.03

    def test_finish(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            exp_id = tracker.start("test_exp", {})
            tracker.finish(exp_id, results={"fid": 42.0, "ssim": 0.87})
            exps = tracker.list_experiments()
            assert exps[0]["status"] == "completed"
            assert exps[0]["fid"] == 42.0

    def test_compare(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            id1 = tracker.start("exp_a", {"lr": 1e-5})
            id2 = tracker.start("exp_b", {"lr": 5e-6})
            tracker.finish(id1, results={"fid": 50})
            tracker.finish(id2, results={"fid": 42})
            comparison = tracker.compare([id1, id2])
            assert len(comparison) == 2
            assert comparison[id2]["results"]["fid"] == 42

    def test_get_best(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            id1 = tracker.start("exp_a", {})
            id2 = tracker.start("exp_b", {})
            tracker.finish(id1, results={"fid": 50, "ssim": 0.82})
            tracker.finish(id2, results={"fid": 42, "ssim": 0.87})
            # FID: lower is better
            assert tracker.get_best("fid", lower_is_better=True) == id2
            # SSIM: higher is better
            assert tracker.get_best("ssim", lower_is_better=False) == id2

    def test_persistence(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker1 = ExperimentTracker(tmpdir)
            exp_id = tracker1.start("persistent", {"lr": 1e-5})
            tracker1.finish(exp_id, results={"fid": 40})
            # Reload
            tracker2 = ExperimentTracker(tmpdir)
            exps = tracker2.list_experiments()
            assert len(exps) == 1
            assert exps[0]["fid"] == 40

    def test_multiple_experiments(self):
        from landmarkdiff.experiment_tracker import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            for i in range(5):
                eid = tracker.start(f"exp_{i}", {"i": i})
                tracker.finish(eid, results={"fid": 50 - i * 5})
            exps = tracker.list_experiments()
            assert len(exps) == 5


class TestComputeSampleDifficulty:
    """Tests for compute_sample_difficulty."""

    def test_basic(self):
        from landmarkdiff.curriculum import compute_sample_difficulty

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "pairs": {
                        "000001": {
                            "procedure": "blepharoplasty",
                            "source": "synthetic",
                            "intensity": 0.5,
                        },
                        "000002": {"procedure": "orthognathic", "source": "real", "intensity": 1.5},
                    }
                },
                f,
            )
            f.flush()
            difficulties = compute_sample_difficulty(f.name)
            # Blepharoplasty synthetic should be easier than orthognathic real
            assert difficulties["000001"] < difficulties["000002"]
