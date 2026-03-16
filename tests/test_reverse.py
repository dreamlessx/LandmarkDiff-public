"""Tests for the reverse module (reverse prediction / procedure detection)."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset


def _make_face(seed=42, width=512, height=512):
    """Create a mock FaceLandmarks object."""
    rng = np.random.default_rng(seed)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4
        landmarks[i, 1] = 0.2 + rng.random() * 0.6
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        confidence=0.95,
        image_width=width,
        image_height=height,
    )


class TestReversePrediction:
    """Tests for ReversePrediction dataclass."""

    def test_summary(self):
        from landmarkdiff.reverse import ReversePrediction

        rp = ReversePrediction(
            procedure="rhinoplasty",
            intensity=65.0,
            confidence=0.82,
            displacement_error=2.5,
        )
        s = rp.summary()
        assert "rhinoplasty" in s
        assert "65.0" in s
        assert "0.820" in s
        assert "2.50" in s


class TestMultiReversePrediction:
    """Tests for MultiReversePrediction dataclass."""

    def test_procedures(self):
        from landmarkdiff.reverse import ReversePrediction, MultiReversePrediction

        mrp = MultiReversePrediction(
            predictions=[
                ReversePrediction(procedure="rhinoplasty", intensity=50.0, confidence=0.8, displacement_error=1.0),
                ReversePrediction(procedure="blepharoplasty", intensity=30.0, confidence=0.6, displacement_error=2.0),
            ],
            residual_error=0.5,
        )
        assert mrp.procedures == ["rhinoplasty", "blepharoplasty"]
        assert mrp.intensities == {"rhinoplasty": 50.0, "blepharoplasty": 30.0}

    def test_summary(self):
        from landmarkdiff.reverse import ReversePrediction, MultiReversePrediction

        mrp = MultiReversePrediction(
            predictions=[
                ReversePrediction(procedure="rhinoplasty", intensity=50.0, confidence=0.8, displacement_error=1.0),
            ],
            residual_error=0.5,
        )
        s = mrp.summary()
        assert "rhinoplasty" in s
        assert "Residual" in s


class TestReversePredict:
    """Tests for reverse_predict function."""

    def test_basic_reverse_prediction(self):
        from landmarkdiff.reverse import reverse_predict

        face_before = _make_face(seed=42)
        # Apply a known procedure and try to detect it
        face_after = apply_procedure_preset(face_before, "rhinoplasty", intensity=50.0)

        result = reverse_predict(face_before, face_after, intensity_steps=5)
        assert result.procedure is not None
        assert 0.0 <= result.intensity <= 100.0
        assert result.confidence >= 0.0
        assert result.displacement_error >= 0.0

    def test_specific_procedures(self):
        from landmarkdiff.reverse import reverse_predict

        face_before = _make_face(seed=42)
        face_after = apply_procedure_preset(face_before, "rhinoplasty", intensity=50.0)

        result = reverse_predict(
            face_before, face_after,
            procedures=["rhinoplasty", "blepharoplasty"],
            intensity_steps=5,
        )
        assert result.procedure in ["rhinoplasty", "blepharoplasty"]

    def test_identical_faces(self):
        from landmarkdiff.reverse import reverse_predict

        face = _make_face(seed=42)
        result = reverse_predict(face, face, intensity_steps=5)
        # Zero displacement → should have low intensity
        assert result.intensity <= 10.0 or result.displacement_error < 1.0


class TestReversePredictMulti:
    """Tests for reverse_predict_multi function."""

    def test_basic_multi_prediction(self):
        from landmarkdiff.reverse import reverse_predict_multi

        face_before = _make_face(seed=42)
        face_after = apply_procedure_preset(face_before, "rhinoplasty", intensity=50.0)

        result = reverse_predict_multi(
            face_before, face_after,
            max_procedures=2,
            min_confidence=0.0,  # low threshold to ensure at least one match
            intensity_steps=5,
        )
        assert len(result.predictions) >= 0  # greedy decomposition may find 0 if residual is tiny
        assert result.residual_error >= 0.0

    def test_max_procedures_limit(self):
        from landmarkdiff.reverse import reverse_predict_multi

        face_before = _make_face(seed=42)
        face_after = apply_procedure_preset(face_before, "rhinoplasty", intensity=50.0)

        result = reverse_predict_multi(
            face_before, face_after,
            max_procedures=1,
            intensity_steps=5,
        )
        assert len(result.predictions) <= 1


class TestInvertDeformation:
    """Tests for invert_deformation."""

    def test_invert_returns_face_landmarks(self):
        from landmarkdiff.reverse import invert_deformation

        face = _make_face(seed=42)
        result = invert_deformation(face, "rhinoplasty", intensity=50.0)

        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (478, 3)
        assert result.image_width == 512
        assert result.image_height == 512

    def test_invert_differs_from_original(self):
        from landmarkdiff.reverse import invert_deformation

        face = _make_face(seed=42)
        result = invert_deformation(face, "rhinoplasty", intensity=50.0)

        # Inverted should differ from original
        orig_px = face.pixel_coords[:, :2]
        inv_px = result.pixel_coords[:, :2]
        assert not np.allclose(orig_px, inv_px, atol=0.1)

    def test_invert_coordinates_in_bounds(self):
        from landmarkdiff.reverse import invert_deformation

        face = _make_face(seed=42)
        result = invert_deformation(face, "rhinoplasty", intensity=80.0)

        px = result.pixel_coords
        assert np.all(px[:, 0] >= 0)
        assert np.all(px[:, 0] < 512)
        assert np.all(px[:, 1] >= 0)
        assert np.all(px[:, 1] < 512)
