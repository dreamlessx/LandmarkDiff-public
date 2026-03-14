"""Tests for displacement model (statistical displacement extraction and modeling)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.displacement_model import (
    NUM_LANDMARKS,
    DisplacementModel,
    _compute_alignment_quality,
    _top_k_landmarks,
    classify_procedure,
)
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS


class TestClassifyProcedure:
    """Tests for procedure classification from displacement vectors."""

    def test_rhinoplasty(self):
        """Displacement in nose region should classify as rhinoplasty."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        nose_indices = PROCEDURE_LANDMARKS.get("rhinoplasty", [])
        for idx in nose_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.01, 0.005]
        result = classify_procedure(displacements)
        assert result == "rhinoplasty"

    def test_blepharoplasty(self):
        """Displacement in eye region should classify as blepharoplasty."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        eye_indices = PROCEDURE_LANDMARKS.get("blepharoplasty", [])
        for idx in eye_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.008, 0.008]
        result = classify_procedure(displacements)
        assert result == "blepharoplasty"

    def test_no_displacement(self):
        """Zero displacement should classify as unknown."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        result = classify_procedure(displacements)
        assert result == "unknown"

    def test_tiny_displacement(self):
        """Sub-threshold displacement should classify as unknown."""
        displacements = np.random.randn(NUM_LANDMARKS, 2) * 0.0001
        result = classify_procedure(displacements)
        assert result == "unknown"


class TestAlignmentQuality:
    """Tests for alignment quality estimation."""

    def test_perfect_alignment(self):
        """Identical landmarks should yield quality = 1.0."""
        landmarks = np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25
        quality = _compute_alignment_quality(landmarks, landmarks)
        assert quality == 1.0

    def test_poor_alignment(self):
        """Large displacement on stable points should yield low quality."""
        before = np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25
        after = before + 0.1  # large shift everywhere
        quality = _compute_alignment_quality(before, after)
        assert quality < 0.5

    def test_quality_range(self):
        """Quality should be in [0, 1]."""
        before = np.random.rand(NUM_LANDMARKS, 2)
        after = before + np.random.randn(NUM_LANDMARKS, 2) * 0.01
        quality = _compute_alignment_quality(before, after)
        assert 0.0 <= quality <= 1.0


class TestTopKLandmarks:
    """Tests for _top_k_landmarks helper."""

    def test_returns_correct_count(self):
        magnitudes = np.random.rand(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=5)
        assert len(result) == 5

    def test_sorted_descending(self):
        magnitudes = np.random.rand(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=10)
        mags = [r["magnitude"] for r in result]
        assert mags == sorted(mags, reverse=True)

    def test_correct_indices(self):
        magnitudes = np.zeros(NUM_LANDMARKS)
        magnitudes[42] = 1.0  # only one non-zero
        result = _top_k_landmarks(magnitudes, k=1)
        assert result[0]["index"] == 42
        assert result[0]["magnitude"] == 1.0


class TestDisplacementModel:
    """Tests for the DisplacementModel class."""

    @pytest.fixture
    def sample_displacements(self):
        """Create sample displacement data for fitting."""
        rng = np.random.default_rng(42)
        data = []
        for _ in range(10):
            disp = np.zeros((NUM_LANDMARKS, 2))
            # Add rhinoplasty-like displacements
            nose_indices = PROCEDURE_LANDMARKS.get("rhinoplasty", [])
            for idx in nose_indices:
                if idx < NUM_LANDMARKS:
                    disp[idx] = rng.normal(0.005, 0.002, 2)
            data.append(
                {
                    "displacements": disp,
                    "procedure": "rhinoplasty",
                }
            )
        for _ in range(5):
            disp = np.zeros((NUM_LANDMARKS, 2))
            eye_indices = PROCEDURE_LANDMARKS.get("blepharoplasty", [])
            for idx in eye_indices:
                if idx < NUM_LANDMARKS:
                    disp[idx] = rng.normal(0.008, 0.003, 2)
            data.append(
                {
                    "displacements": disp,
                    "procedure": "blepharoplasty",
                }
            )
        return data

    def test_unfitted_properties(self):
        model = DisplacementModel()
        assert not model.fitted
        assert model.procedures == []

    def test_fit(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)
        assert model.fitted
        assert "rhinoplasty" in model.procedures
        assert "blepharoplasty" in model.procedures
        assert model.n_samples["rhinoplasty"] == 10
        assert model.n_samples["blepharoplasty"] == 5

    def test_fit_empty_raises(self):
        model = DisplacementModel()
        with pytest.raises(ValueError, match="empty"):
            model.fit([])

    def test_fit_no_valid_data_raises(self):
        model = DisplacementModel()
        with pytest.raises(ValueError, match="No valid"):
            model.fit([{"displacements": None, "procedure": "test"}])

    def test_get_displacement_field(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        field = model.get_displacement_field("rhinoplasty", intensity=1.0)
        assert field.shape == (NUM_LANDMARKS, 2)
        assert field.dtype == np.float32

    def test_get_displacement_field_unfitted(self):
        model = DisplacementModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_displacement_field("rhinoplasty")

    def test_get_displacement_field_unknown_procedure(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)
        with pytest.raises(KeyError, match="not in model"):
            model.get_displacement_field("unknown_procedure")

    def test_displacement_field_intensity_scaling(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        f1 = model.get_displacement_field("rhinoplasty", intensity=1.0, noise_scale=0.0)
        f2 = model.get_displacement_field("rhinoplasty", intensity=2.0, noise_scale=0.0)
        np.testing.assert_allclose(f2, f1 * 2.0, atol=1e-6)

    def test_displacement_field_with_noise(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        f1 = model.get_displacement_field("rhinoplasty", noise_scale=0.5, rng=rng1)
        f2 = model.get_displacement_field("rhinoplasty", noise_scale=0.5, rng=rng2)
        np.testing.assert_array_equal(f1, f2)

    def test_get_summary(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        summary = model.get_summary()
        assert summary["fitted"] is True
        assert "rhinoplasty" in summary["procedures"]
        rhino = summary["procedures"]["rhinoplasty"]
        assert rhino["n_samples"] == 10
        assert "global_mean_magnitude" in rhino
        assert "top_landmarks" in rhino

    def test_get_summary_unfitted(self):
        model = DisplacementModel()
        summary = model.get_summary()
        assert summary["fitted"] is False

    def test_get_summary_single_procedure(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)
        summary = model.get_summary(procedure="rhinoplasty")
        assert "rhinoplasty" in summary["procedures"]
        assert "blepharoplasty" not in summary["procedures"]

    def test_save_load_roundtrip(self, sample_displacements, tmp_path):
        model = DisplacementModel()
        model.fit(sample_displacements)

        save_path = tmp_path / "model.npz"
        model.save(save_path)
        assert save_path.exists()

        loaded = DisplacementModel.load(save_path)
        assert loaded.fitted
        assert loaded.procedures == model.procedures
        assert loaded.n_samples == model.n_samples

        # Check stats match
        for proc in model.procedures:
            np.testing.assert_allclose(
                loaded.stats[proc]["mean"],
                model.stats[proc]["mean"],
                atol=1e-6,
            )

    def test_save_adds_extension(self, sample_displacements, tmp_path):
        model = DisplacementModel()
        model.fit(sample_displacements)
        model.save(tmp_path / "model")  # no .npz
        assert (tmp_path / "model.npz").exists()

    def test_save_unfitted_raises(self, tmp_path):
        model = DisplacementModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.save(tmp_path / "model.npz")

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DisplacementModel.load(tmp_path / "nonexistent.npz")

    def test_displacement_field_deterministic_no_noise(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        f1 = model.get_displacement_field("rhinoplasty", intensity=0.5, noise_scale=0.0)
        f2 = model.get_displacement_field("rhinoplasty", intensity=0.5, noise_scale=0.0)
        np.testing.assert_array_equal(f1, f2)

    def test_stats_keys(self, sample_displacements):
        model = DisplacementModel()
        model.fit(sample_displacements)

        for proc in model.procedures:
            stats = model.stats[proc]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "median" in stats
            assert "mean_magnitude" in stats
            assert stats["mean"].shape == (NUM_LANDMARKS, 2)
            assert stats["mean_magnitude"].shape == (NUM_LANDMARKS,)
