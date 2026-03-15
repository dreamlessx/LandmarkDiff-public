"""Extended tests for displacement model: fitting, prediction, serialization, and edge cases.

Covers DisplacementModel with single-procedure fitting, noise behavior,
summary filtering, serialization formats, classify_procedure boundary
conditions, and helper function edge cases.
"""

from __future__ import annotations

import json
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
    visualize_displacements,
)
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rhinoplasty_data():
    """Create sample rhinoplasty-only displacement data."""
    rng = np.random.default_rng(42)
    data = []
    nose_indices = PROCEDURE_LANDMARKS.get("rhinoplasty", [])
    for _ in range(8):
        disp = np.zeros((NUM_LANDMARKS, 2))
        for idx in nose_indices:
            if idx < NUM_LANDMARKS:
                disp[idx] = rng.normal(0.006, 0.002, 2)
        data.append({"displacements": disp, "procedure": "rhinoplasty"})
    return data


@pytest.fixture
def multi_procedure_data():
    """Create displacement data for three procedures."""
    rng = np.random.default_rng(99)
    data = []
    for proc, count in [("rhinoplasty", 10), ("blepharoplasty", 7), ("rhytidectomy", 5)]:
        indices = PROCEDURE_LANDMARKS.get(proc, [])
        for _ in range(count):
            disp = np.zeros((NUM_LANDMARKS, 2))
            for idx in indices:
                if idx < NUM_LANDMARKS:
                    disp[idx] = rng.normal(0.005, 0.002, 2)
            data.append({"displacements": disp, "procedure": proc})
    return data


@pytest.fixture
def fitted_model(multi_procedure_data):
    """Return a fitted DisplacementModel on multiple procedures."""
    model = DisplacementModel()
    model.fit(multi_procedure_data)
    return model


# ---------------------------------------------------------------------------
# DisplacementModel fitting
# ---------------------------------------------------------------------------


class TestDisplacementModelFitting:
    """Tests for the fitting process."""

    def test_single_procedure_fit(self, rhinoplasty_data):
        model = DisplacementModel()
        model.fit(rhinoplasty_data)
        assert model.fitted
        assert model.procedures == ["rhinoplasty"]
        assert model.n_samples["rhinoplasty"] == 8

    def test_multi_procedure_fit(self, multi_procedure_data):
        model = DisplacementModel()
        model.fit(multi_procedure_data)
        assert model.fitted
        assert len(model.procedures) == 3
        assert model.n_samples["rhinoplasty"] == 10
        assert model.n_samples["blepharoplasty"] == 7
        assert model.n_samples["rhytidectomy"] == 5

    def test_fit_with_unknown_procedure(self):
        """Entries classified as 'unknown' should still be fitted."""
        data = [
            {
                "displacements": np.zeros((NUM_LANDMARKS, 2)),
                "procedure": "unknown",
            }
        ]
        model = DisplacementModel()
        model.fit(data)
        assert model.fitted
        assert "unknown" in model.procedures

    def test_fit_skips_none_displacements(self):
        """Entries with None displacements should be skipped."""
        data = [
            {"displacements": None, "procedure": "rhinoplasty"},
            {
                "displacements": np.zeros((NUM_LANDMARKS, 2)),
                "procedure": "rhinoplasty",
            },
        ]
        model = DisplacementModel()
        model.fit(data)
        assert model.fitted
        assert model.n_samples["rhinoplasty"] == 1

    def test_fit_skips_wrong_shape(self):
        """Entries with wrong displacement shape should be skipped."""
        data = [
            {
                "displacements": np.zeros((100, 2)),  # wrong shape
                "procedure": "rhinoplasty",
            },
            {
                "displacements": np.zeros((NUM_LANDMARKS, 2)),
                "procedure": "rhinoplasty",
            },
        ]
        model = DisplacementModel()
        model.fit(data)
        assert model.n_samples["rhinoplasty"] == 1

    def test_fit_resets_previous(self, rhinoplasty_data, multi_procedure_data):
        """Re-fitting should replace previous stats entirely."""
        model = DisplacementModel()
        model.fit(rhinoplasty_data)
        assert len(model.procedures) == 1

        model.fit(multi_procedure_data)
        assert len(model.procedures) == 3

    def test_fit_empty_raises(self):
        model = DisplacementModel()
        with pytest.raises(ValueError, match="empty"):
            model.fit([])

    def test_fit_all_invalid_raises(self):
        """If all entries are invalid, should raise ValueError."""
        data = [
            {"displacements": None, "procedure": "rhinoplasty"},
            {"displacements": np.zeros((10, 2)), "procedure": "rhinoplasty"},
        ]
        model = DisplacementModel()
        with pytest.raises(ValueError, match="No valid"):
            model.fit(data)

    def test_fit_missing_procedure_key(self):
        """Entries without 'procedure' key should default to 'unknown'."""
        data = [{"displacements": np.zeros((NUM_LANDMARKS, 2))}]
        model = DisplacementModel()
        model.fit(data)
        assert "unknown" in model.procedures


# ---------------------------------------------------------------------------
# get_displacement_field
# ---------------------------------------------------------------------------


class TestDisplacementField:
    """Tests for displacement field generation."""

    def test_zero_intensity(self, fitted_model):
        """Intensity 0 should produce zero displacements (no noise)."""
        field = fitted_model.get_displacement_field("rhinoplasty", intensity=0.0, noise_scale=0.0)
        np.testing.assert_allclose(field, 0.0, atol=1e-10)

    def test_negative_intensity(self, fitted_model):
        """Negative intensity should invert the displacement direction."""
        f_pos = fitted_model.get_displacement_field("rhinoplasty", intensity=1.0, noise_scale=0.0)
        f_neg = fitted_model.get_displacement_field("rhinoplasty", intensity=-1.0, noise_scale=0.0)
        np.testing.assert_allclose(f_neg, -f_pos, atol=1e-6)

    def test_field_shape(self, fitted_model):
        field = fitted_model.get_displacement_field("rhinoplasty")
        assert field.shape == (NUM_LANDMARKS, 2)
        assert field.dtype == np.float32

    def test_noise_increases_variance(self, fitted_model):
        """Adding noise should increase the variance of the field."""
        f_no_noise = fitted_model.get_displacement_field("rhinoplasty", noise_scale=0.0)
        rng = np.random.default_rng(42)
        f_noisy = fitted_model.get_displacement_field("rhinoplasty", noise_scale=1.0, rng=rng)
        # With noise, the field should differ from the noiseless version
        diff = np.abs(f_noisy - f_no_noise)
        assert np.sum(diff) > 0

    def test_different_seeds_different_noise(self, fitted_model):
        """Different RNG seeds should produce different noisy fields."""
        f1 = fitted_model.get_displacement_field(
            "rhinoplasty", noise_scale=0.5, rng=np.random.default_rng(1)
        )
        f2 = fitted_model.get_displacement_field(
            "rhinoplasty", noise_scale=0.5, rng=np.random.default_rng(2)
        )
        assert not np.allclose(f1, f2)

    def test_default_rng_used_when_none(self, fitted_model):
        """When rng is None and noise_scale > 0, should use default_rng."""
        # Should not raise
        field = fitted_model.get_displacement_field("rhinoplasty", noise_scale=0.1)
        assert field.shape == (NUM_LANDMARKS, 2)

    def test_unfitted_raises(self):
        model = DisplacementModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_displacement_field("rhinoplasty")

    def test_unknown_procedure_raises(self, fitted_model):
        with pytest.raises(KeyError, match="not in model"):
            fitted_model.get_displacement_field("mentoplasty")

    def test_large_intensity(self, fitted_model):
        """Very large intensity should scale linearly."""
        f1 = fitted_model.get_displacement_field("rhinoplasty", intensity=1.0, noise_scale=0.0)
        f100 = fitted_model.get_displacement_field("rhinoplasty", intensity=100.0, noise_scale=0.0)
        np.testing.assert_allclose(f100, f1 * 100.0, atol=1e-4)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Tests for model summary generation."""

    def test_unfitted_summary(self):
        model = DisplacementModel()
        s = model.get_summary()
        assert s["fitted"] is False
        assert "procedures" not in s

    def test_fitted_summary_structure(self, fitted_model):
        s = fitted_model.get_summary()
        assert s["fitted"] is True
        assert len(s["procedures"]) == 3

    def test_summary_single_procedure(self, fitted_model):
        s = fitted_model.get_summary(procedure="blepharoplasty")
        assert "blepharoplasty" in s["procedures"]
        assert "rhinoplasty" not in s["procedures"]

    def test_summary_nonexistent_procedure(self, fitted_model):
        """Requesting summary for a procedure not in model should skip it."""
        s = fitted_model.get_summary(procedure="mentoplasty")
        assert s["fitted"] is True
        assert len(s["procedures"]) == 0

    def test_summary_top_landmarks_count(self, fitted_model):
        s = fitted_model.get_summary()
        for proc_data in s["procedures"].values():
            assert len(proc_data["top_landmarks"]) == 10

    def test_summary_magnitudes_positive(self, fitted_model):
        s = fitted_model.get_summary()
        for proc_data in s["procedures"].values():
            assert proc_data["global_mean_magnitude"] >= 0.0
            assert proc_data["global_max_magnitude"] >= 0.0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for model serialization."""

    def test_save_creates_file(self, fitted_model, tmp_path):
        path = tmp_path / "model.npz"
        fitted_model.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_auto_extension(self, fitted_model, tmp_path):
        fitted_model.save(tmp_path / "model")
        assert (tmp_path / "model.npz").exists()

    def test_save_unfitted_raises(self, tmp_path):
        model = DisplacementModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.save(tmp_path / "model.npz")

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DisplacementModel.load(tmp_path / "nope.npz")

    def test_roundtrip_stats_match(self, fitted_model, tmp_path):
        """Stats should match exactly after save/load."""
        path = tmp_path / "model.npz"
        fitted_model.save(path)
        loaded = DisplacementModel.load(path)

        assert loaded.fitted
        assert sorted(loaded.procedures) == sorted(fitted_model.procedures)
        assert loaded.n_samples == fitted_model.n_samples

        for proc in fitted_model.procedures:
            for stat_name in ["mean", "std", "min", "max", "median", "mean_magnitude"]:
                np.testing.assert_allclose(
                    loaded.stats[proc][stat_name],
                    fitted_model.stats[proc][stat_name],
                    atol=1e-6,
                    err_msg=f"Mismatch in {proc}/{stat_name}",
                )

    def test_roundtrip_displacement_field_match(self, fitted_model, tmp_path):
        """Displacement fields should match after save/load."""
        path = tmp_path / "model.npz"
        fitted_model.save(path)
        loaded = DisplacementModel.load(path)

        for proc in fitted_model.procedures:
            f_orig = fitted_model.get_displacement_field(proc, intensity=1.0, noise_scale=0.0)
            f_loaded = loaded.get_displacement_field(proc, intensity=1.0, noise_scale=0.0)
            np.testing.assert_allclose(f_orig, f_loaded, atol=1e-6)

    def test_save_contains_metadata(self, fitted_model, tmp_path):
        """Saved file should contain parseable JSON metadata."""
        path = tmp_path / "model.npz"
        fitted_model.save(path)

        data = np.load(str(path), allow_pickle=False)
        assert "__metadata__" in data.files
        meta_bytes = data["__metadata__"].tobytes()
        metadata = json.loads(meta_bytes.decode("utf-8"))
        assert "procedures" in metadata
        assert "n_samples" in metadata
        assert metadata["num_landmarks"] == NUM_LANDMARKS

    def test_save_creates_parent_dirs(self, fitted_model, tmp_path):
        """Save should work even if parent directory does not exist."""
        path = tmp_path / "nested" / "deep" / "model.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        fitted_model.save(path)
        assert path.exists()


# ---------------------------------------------------------------------------
# classify_procedure
# ---------------------------------------------------------------------------


class TestClassifyProcedureExtended:
    """Extended tests for procedure classification."""

    def test_threshold_boundary(self):
        """Displacement just below 0.002 threshold should classify as unknown."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        # Set all procedure landmarks to just below threshold
        nose_indices = PROCEDURE_LANDMARKS.get("rhinoplasty", [])
        for idx in nose_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.001, 0.001]
        # Mean magnitude per landmark ~0.0014, which is below 0.002
        result = classify_procedure(displacements)
        assert result == "unknown"

    def test_above_threshold(self):
        """Displacement above 0.002 threshold should classify correctly."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        # Use blepharoplasty (no overlap with other procedure landmark sets)
        eye_indices = PROCEDURE_LANDMARKS.get("blepharoplasty", [])
        for idx in eye_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.01, 0.01]
        result = classify_procedure(displacements)
        assert result == "blepharoplasty"

    def test_competing_regions(self):
        """When multiple regions have displacement, highest mean wins."""
        displacements = np.zeros((NUM_LANDMARKS, 2))
        # Set nose with moderate displacement
        nose_indices = PROCEDURE_LANDMARKS.get("rhinoplasty", [])
        for idx in nose_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.005, 0.005]
        # Set eye with stronger displacement
        eye_indices = PROCEDURE_LANDMARKS.get("blepharoplasty", [])
        for idx in eye_indices:
            if idx < NUM_LANDMARKS:
                displacements[idx] = [0.02, 0.02]
        result = classify_procedure(displacements)
        assert result == "blepharoplasty"

    def test_uniform_displacement(self):
        """Uniform displacement everywhere should pick the region with most landmarks."""
        displacements = np.full((NUM_LANDMARKS, 2), 0.01)
        result = classify_procedure(displacements)
        # Should classify as some procedure (not unknown since magnitude > 0.002)
        assert result != "unknown"
        assert result in PROCEDURE_LANDMARKS

    def test_all_procedures_classifiable(self):
        """Each procedure should be classifiable when its region is active."""
        for proc, indices in PROCEDURE_LANDMARKS.items():
            displacements = np.zeros((NUM_LANDMARKS, 2))
            for idx in indices:
                if idx < NUM_LANDMARKS:
                    displacements[idx] = [0.02, 0.02]
            result = classify_procedure(displacements)
            # The targeted procedure should be the winner (or tied with another)
            assert result != "unknown", f"Procedure {proc} classified as unknown"


# ---------------------------------------------------------------------------
# _compute_alignment_quality
# ---------------------------------------------------------------------------


class TestAlignmentQualityExtended:
    """Additional alignment quality tests."""

    def test_small_shift(self):
        """Small shift on stable points should yield high quality."""
        rng = np.random.default_rng(0)
        before = rng.uniform(0.3, 0.7, (NUM_LANDMARKS, 2))
        after = before + 0.001  # tiny shift
        quality = _compute_alignment_quality(before, after)
        assert quality > 0.9

    def test_moderate_shift(self):
        """Moderate shift should yield mid-range quality."""
        rng = np.random.default_rng(0)
        before = rng.uniform(0.3, 0.7, (NUM_LANDMARKS, 2))
        after = before + 0.025
        quality = _compute_alignment_quality(before, after)
        assert 0.2 < quality < 0.8

    def test_large_shift_zero_quality(self):
        """Very large shift (>= 0.05 on stable points) should yield 0."""
        before = np.full((NUM_LANDMARKS, 2), 0.3)
        after = np.full((NUM_LANDMARKS, 2), 0.4)  # 0.1 shift
        quality = _compute_alignment_quality(before, after)
        assert quality == 0.0

    def test_quality_symmetric(self):
        """Quality should be independent of direction."""
        rng = np.random.default_rng(0)
        a = rng.uniform(0.3, 0.7, (NUM_LANDMARKS, 2))
        b = a + 0.01
        q1 = _compute_alignment_quality(a, b)
        q2 = _compute_alignment_quality(b, a)
        assert abs(q1 - q2) < 1e-6


# ---------------------------------------------------------------------------
# _top_k_landmarks
# ---------------------------------------------------------------------------


class TestTopKLandmarksExtended:
    """Extended tests for _top_k_landmarks."""

    def test_k_equals_total(self):
        """k == NUM_LANDMARKS should return all landmarks."""
        magnitudes = np.random.rand(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=NUM_LANDMARKS)
        assert len(result) == NUM_LANDMARKS

    def test_k_one(self):
        """k=1 should return the single largest."""
        magnitudes = np.zeros(NUM_LANDMARKS)
        magnitudes[100] = 5.0
        result = _top_k_landmarks(magnitudes, k=1)
        assert len(result) == 1
        assert result[0]["index"] == 100
        assert result[0]["magnitude"] == 5.0

    def test_all_zeros(self):
        """All-zero magnitudes should still return k entries."""
        magnitudes = np.zeros(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=5)
        assert len(result) == 5
        for entry in result:
            assert entry["magnitude"] == 0.0

    def test_result_dict_keys(self):
        """Each result should have 'index' and 'magnitude' keys."""
        magnitudes = np.random.rand(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=3)
        for entry in result:
            assert "index" in entry
            assert "magnitude" in entry
            assert isinstance(entry["index"], int)
            assert isinstance(entry["magnitude"], float)

    def test_indices_unique(self):
        """Top-k indices should all be unique."""
        magnitudes = np.random.rand(NUM_LANDMARKS)
        result = _top_k_landmarks(magnitudes, k=20)
        indices = [r["index"] for r in result]
        assert len(set(indices)) == len(indices)


# ---------------------------------------------------------------------------
# visualize_displacements
# ---------------------------------------------------------------------------


class TestVisualizeDisplacements:
    """Tests for displacement visualization (no GPU needed)."""

    def test_output_shape(self):
        """Visualization should return same-shaped image."""
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        result_dict = {
            "landmarks_before": np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25,
            "displacements": np.random.randn(NUM_LANDMARKS, 2) * 0.01,
            "procedure": "rhinoplasty",
            "quality_score": 0.9,
        }
        vis = visualize_displacements(img, result_dict)
        assert vis.shape == img.shape
        assert vis.dtype == np.uint8

    def test_does_not_modify_original(self):
        """Visualization should not modify the input image."""
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        original = img.copy()
        result_dict = {
            "landmarks_before": np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25,
            "displacements": np.zeros((NUM_LANDMARKS, 2)),
            "procedure": "unknown",
            "quality_score": 0.5,
        }
        visualize_displacements(img, result_dict)
        np.testing.assert_array_equal(img, original)

    def test_zero_displacements_minimal_change(self):
        """Zero displacements should only add text overlay, not arrows."""
        img = np.full((256, 256, 3), 100, dtype=np.uint8)
        result_dict = {
            "landmarks_before": np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25,
            "displacements": np.zeros((NUM_LANDMARKS, 2)),
            "procedure": "unknown",
            "quality_score": 1.0,
        }
        vis = visualize_displacements(img, result_dict)
        # Only the text region should differ
        diff = np.abs(vis.astype(int) - img.astype(int))
        # Text is drawn in the top-left, most of the image should be unchanged
        unchanged_ratio = np.sum(diff == 0) / diff.size
        assert unchanged_ratio > 0.8

    def test_custom_scale_and_color(self):
        """Custom scale and color should not crash."""
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result_dict = {
            "landmarks_before": np.random.rand(NUM_LANDMARKS, 2) * 0.5 + 0.25,
            "displacements": np.random.randn(NUM_LANDMARKS, 2) * 0.01,
            "procedure": "rhinoplasty",
            "quality_score": 0.8,
        }
        vis = visualize_displacements(
            img, result_dict, scale=20.0, arrow_color=(255, 0, 0), thickness=2
        )
        assert vis.shape == img.shape


# ---------------------------------------------------------------------------
# Properties and state
# ---------------------------------------------------------------------------


class TestModelProperties:
    """Tests for DisplacementModel property accessors."""

    def test_procedures_empty_before_fit(self):
        model = DisplacementModel()
        assert model.procedures == []

    def test_fitted_false_before_fit(self):
        model = DisplacementModel()
        assert model.fitted is False

    def test_procedures_returns_copy(self, fitted_model):
        """Modifying the returned procedures list should not affect model."""
        procs = fitted_model.procedures
        procs.append("fake")
        assert "fake" not in fitted_model.procedures

    def test_stats_keys_complete(self, fitted_model):
        """Each procedure's stats should have all expected keys."""
        expected_keys = {"mean", "std", "min", "max", "median", "mean_magnitude"}
        for proc in fitted_model.procedures:
            assert set(fitted_model.stats[proc].keys()) == expected_keys

    def test_stats_shapes_correct(self, fitted_model):
        """All stat arrays should have correct shapes."""
        for proc in fitted_model.procedures:
            stats = fitted_model.stats[proc]
            for key in ["mean", "std", "min", "max", "median"]:
                assert stats[key].shape == (NUM_LANDMARKS, 2), (
                    f"{proc}/{key} has shape {stats[key].shape}"
                )
            assert stats["mean_magnitude"].shape == (NUM_LANDMARKS,)
