"""Tests for landmark manipulation engine."""

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    DeformationHandle,
    apply_procedure_preset,
    gaussian_rbf_deform,
)


def _make_fake_face() -> FaceLandmarks:
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=0.9)


class TestGaussianRBF:
    def test_deformation_returns_new_array(self):
        landmarks = np.random.default_rng(0).uniform(0, 512, (478, 2)).astype(np.float32)
        handle = DeformationHandle(
            landmark_index=1,
            displacement=np.array([10.0, 5.0]),
            influence_radius=30.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        assert result is not landmarks  # immutable
        assert not np.array_equal(result, landmarks)

    def test_deformation_max_at_handle(self):
        landmarks = np.zeros((10, 2), dtype=np.float32)
        landmarks[5] = [100.0, 100.0]
        handle = DeformationHandle(
            landmark_index=5,
            displacement=np.array([20.0, 0.0]),
            influence_radius=30.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        # Handle point should move by full displacement
        assert abs(result[5, 0] - 120.0) < 0.01

    def test_deformation_falls_off(self):
        landmarks = np.array([[0, 0], [100, 0], [200, 0]], dtype=np.float32)
        handle = DeformationHandle(
            landmark_index=1,
            displacement=np.array([50.0, 0.0]),
            influence_radius=20.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        # Point at distance 100 should barely move
        assert abs(result[0, 0]) < 1.0
        # Handle moves full
        assert abs(result[1, 0] - 150.0) < 0.01


class TestProcedurePresets:
    @pytest.mark.parametrize("procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"])
    def test_preset_returns_face_landmarks(self, procedure):
        face = _make_fake_face()
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (478, 3)

    def test_invalid_procedure_raises(self):
        face = _make_fake_face()
        with pytest.raises(ValueError, match="Unknown procedure"):
            apply_procedure_preset(face, "botox")

    def test_zero_intensity_no_change(self):
        face = _make_fake_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=0.0)
        np.testing.assert_array_almost_equal(result.landmarks, face.landmarks, decimal=5)
