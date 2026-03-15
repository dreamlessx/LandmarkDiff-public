"""Tests for curriculum learning, data splitting, and benchmark utilities."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


class TestTrainingCurriculum:
    """Test curriculum learning schedule."""

    def test_warmup_difficulty_zero(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1)
        # During warmup (first 10%), difficulty should be 0
        assert c.get_difficulty(0) == 0.0
        assert c.get_difficulty(500) == 0.0
        assert c.get_difficulty(999) == 0.0

    def test_full_difficulty_one(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, full_difficulty_at=0.5)
        # After full_difficulty_at, difficulty should be 1
        assert c.get_difficulty(5000) == 1.0
        assert c.get_difficulty(10000) == 1.0

    def test_monotonic_increase(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1, full_difficulty_at=0.5)
        prev = 0.0
        for step in range(0, 10001, 100):
            d = c.get_difficulty(step)
            assert d >= prev - 1e-7, f"Difficulty decreased at step {step}: {prev} -> {d}"
            prev = d

    def test_cosine_ramp_midpoint(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, warmup_fraction=0.0, full_difficulty_at=1.0)
        # At midpoint, cosine ramp should give ~0.5
        mid = c.get_difficulty(5000)
        assert abs(mid - 0.5) < 0.01

    def test_should_include_easy_samples(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1)
        rng = np.random.default_rng(42)
        # Easy samples (difficulty 0) should always be included
        for _ in range(20):
            assert c.should_include(0, 0.0, rng) is True

    def test_should_exclude_hard_early(self):
        from landmarkdiff.curriculum import TrainingCurriculum

        c = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1)
        rng = np.random.default_rng(42)
        # Very hard samples (difficulty 0.9) should rarely be included during warmup
        included = sum(c.should_include(0, 0.9, rng) for _ in range(100))
        assert included < 20, f"Too many hard samples included during warmup: {included}/100"


class TestProcedureCurriculum:
    """Test procedure-aware curriculum."""

    def test_weight_bounds(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        pc = ProcedureCurriculum(total_steps=10000)
        for step in range(0, 10001, 1000):
            for proc in [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ]:
                w = pc.get_weight(step, proc)
                assert 0.1 <= w <= 1.0, f"Weight out of bounds for {proc} at step {step}: {w}"

    def test_easy_procs_always_weighted_higher(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        pc = ProcedureCurriculum(total_steps=10000)
        # Blepharoplasty (easy) should always be weighted >= hard procedures
        for step in range(0, 10001, 1000):
            w_easy = pc.get_weight(step, "blepharoplasty")
            w_hard = pc.get_weight(step, "orthognathic")
            assert w_easy >= w_hard, (
                f"Easy proc ({w_easy}) weighted less than hard ({w_hard}) at step {step}"
            )

    def test_all_weights_one_at_end(self):
        from landmarkdiff.curriculum import ProcedureCurriculum

        pc = ProcedureCurriculum(total_steps=10000, warmup_fraction=0.1)
        weights = pc.get_procedure_weights(10000)
        for proc, w in weights.items():
            assert w == 1.0, f"{proc} not at full weight at end: {w}"


class TestDataSplitting:
    """Test stratified dataset splitting."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset with multiple procedures."""
        rng = np.random.default_rng(42)
        for proc in [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ]:
            for i in range(20):
                prefix = f"{proc}_{i:03d}"
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                for suffix in ["input", "target", "conditioning"]:
                    cv2.imwrite(str(tmp_path / f"{prefix}_{suffix}.png"), img)
        return tmp_path

    def test_split_creates_dirs(self, mock_dataset, tmp_path):
        from scripts.split_dataset import stratified_split

        out = tmp_path / "splits"
        stratified_split(str(mock_dataset), str(out), 0.1, 0.1, seed=42)
        assert (out / "train").exists()
        assert (out / "val").exists()
        assert (out / "test").exists()
        assert (out / "split_info.json").exists()

    def test_split_no_overlap(self, mock_dataset, tmp_path):
        from scripts.split_dataset import stratified_split

        out = tmp_path / "splits"
        result = stratified_split(str(mock_dataset), str(out), 0.15, 0.15, seed=42)

        train = set(result["train_prefixes"])
        val = set(result["val_prefixes"])
        test = set(result["test_prefixes"])

        assert len(train & val) == 0, "Train-Val overlap!"
        assert len(train & test) == 0, "Train-Test overlap!"
        assert len(val & test) == 0, "Val-Test overlap!"

    def test_split_covers_all(self, mock_dataset, tmp_path):
        from scripts.split_dataset import stratified_split

        out = tmp_path / "splits"
        result = stratified_split(str(mock_dataset), str(out), 0.1, 0.1, seed=42)

        total = (
            len(result["train_prefixes"])
            + len(result["val_prefixes"])
            + len(result["test_prefixes"])
        )
        # Should cover all 80 pairs (20 per procedure * 4 procedures)
        assert total == 80

    def test_split_reproducible(self, mock_dataset, tmp_path):
        from scripts.split_dataset import stratified_split

        out1 = tmp_path / "splits1"
        out2 = tmp_path / "splits2"
        r1 = stratified_split(str(mock_dataset), str(out1), 0.1, 0.1, seed=42)
        r2 = stratified_split(str(mock_dataset), str(out2), 0.1, 0.1, seed=42)
        assert r1["train_prefixes"] == r2["train_prefixes"]
        assert r1["test_prefixes"] == r2["test_prefixes"]


class TestBenchmarkUtilities:
    """Test benchmark script utilities."""

    def test_load_test_pairs(self, tmp_path):
        """Test pair loading from test directory."""
        # Create mock test pairs
        rng = np.random.default_rng(42)
        for proc in ["rhinoplasty", "blepharoplasty"]:
            for i in range(5):
                prefix = f"{proc}_{i:03d}"
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{prefix}_input.png"), img)
                cv2.imwrite(str(tmp_path / f"{prefix}_target.png"), img)

        from scripts.benchmark_quality import load_test_pairs

        pairs = load_test_pairs(str(tmp_path))
        assert len(pairs) == 10
        assert all(p["procedure"] in ["rhinoplasty", "blepharoplasty"] for p in pairs)

    def test_generate_latex_table(self, tmp_path):
        """Test LaTeX table generation from mock results."""
        from scripts.benchmark_quality import generate_latex_table

        mock_results = {
            "TPS Baseline": {
                "aggregate": {"ssim_mean": 0.85, "lpips_mean": 0.12, "nme_mean": 0.05},
                "per_sample": [],
            },
            "Phase B (Ours)": {
                "aggregate": {"ssim_mean": 0.92, "lpips_mean": 0.06, "nme_mean": 0.02},
                "per_sample": [],
            },
        }

        latex = generate_latex_table(mock_results, tmp_path / "table.tex")
        assert "TPS Baseline" in latex
        assert r"\textbf" in latex  # "Ours" should be bold
        assert (tmp_path / "table.tex").exists()

    def test_significance_test(self):
        """Test statistical significance computation."""
        from scripts.benchmark_quality import significance_test

        rng = np.random.default_rng(42)

        # Create mock method results with clear difference
        method_a = {
            "per_sample": [
                {
                    "prefix": f"img_{i}",
                    "ssim": 0.8 + rng.normal(0, 0.02),
                    "lpips": 0.1 + rng.normal(0, 0.01),
                    "nme": 0.05 + rng.normal(0, 0.005),
                }
                for i in range(50)
            ]
        }
        method_b = {
            "per_sample": [
                {
                    "prefix": f"img_{i}",
                    "ssim": 0.92 + rng.normal(0, 0.02),
                    "lpips": 0.05 + rng.normal(0, 0.01),
                    "nme": 0.02 + rng.normal(0, 0.005),
                }
                for i in range(50)
            ]
        }

        result = significance_test(method_a, method_b)
        assert result["n_common"] == 50
        assert "tests" in result
        # With clear differences, p-value should be significant
        assert result["tests"]["ssim"]["significant_005"]
