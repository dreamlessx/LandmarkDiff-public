"""Tests for the auto_intensity module (auto-detect procedure and intensity)."""

from __future__ import annotations

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


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


def _make_displaced_face(face, seed=99, magnitude=10.0, region_indices=None):
    """Create a face with targeted displacements in a specific region."""
    rng = np.random.default_rng(seed)
    lm = face.landmarks.copy()
    if region_indices is not None:
        for idx in region_indices:
            if idx < len(lm):
                lm[idx, 0] += magnitude / face.image_width
                lm[idx, 1] += magnitude / face.image_height
    else:
        # Global small displacement
        lm[:, 0] += rng.normal(0, magnitude / face.image_width, size=478).astype(np.float32)
        lm[:, 1] += rng.normal(0, magnitude / face.image_height, size=478).astype(np.float32)
    return FaceLandmarks(
        landmarks=lm,
        confidence=face.confidence,
        image_width=face.image_width,
        image_height=face.image_height,
    )


class TestIntensityEstimate:
    """Tests for IntensityEstimate dataclass."""

    def test_summary(self):
        from landmarkdiff.auto_intensity import IntensityEstimate

        est = IntensityEstimate(procedure="rhinoplasty", intensity=60.0, score=0.85)
        s = est.summary()
        assert "rhinoplasty" in s
        assert "60.0" in s
        assert "0.8500" in s


class TestEstimateIntensity:
    """Tests for estimate_intensity."""

    def test_basic_estimation(self):
        from landmarkdiff.auto_intensity import estimate_intensity

        face_before = _make_face(seed=42)
        face_after = _make_displaced_face(face_before, seed=99, magnitude=5.0)

        result = estimate_intensity(face_before, face_after)
        assert result.procedure is not None
        assert 0.0 <= result.intensity <= 100.0
        assert result.score >= 0.0

    def test_specific_procedures(self):
        from landmarkdiff.auto_intensity import estimate_intensity

        face_before = _make_face(seed=42)
        face_after = _make_displaced_face(face_before, seed=99, magnitude=5.0)

        result = estimate_intensity(
            face_before, face_after, procedures=["rhinoplasty", "blepharoplasty"]
        )
        assert result.procedure in ["rhinoplasty", "blepharoplasty"]

    def test_zero_displacement(self):
        from landmarkdiff.auto_intensity import estimate_intensity

        face = _make_face(seed=42)
        result = estimate_intensity(face, face)
        assert result.intensity >= 0.0


class TestEstimateAllProcedures:
    """Tests for estimate_all_procedures."""

    def test_returns_all_procedures(self):
        from landmarkdiff.auto_intensity import estimate_all_procedures
        from landmarkdiff.manipulation import PROCEDURE_LANDMARKS

        face_before = _make_face(seed=42)
        face_after = _make_displaced_face(face_before, seed=99, magnitude=5.0)

        results = estimate_all_procedures(face_before, face_after)
        assert len(results) == len(PROCEDURE_LANDMARKS)

    def test_sorted_by_score_descending(self):
        from landmarkdiff.auto_intensity import estimate_all_procedures

        face_before = _make_face(seed=42)
        face_after = _make_displaced_face(face_before, seed=99, magnitude=5.0)

        results = estimate_all_procedures(face_before, face_after)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_intensities_clamped(self):
        from landmarkdiff.auto_intensity import estimate_all_procedures

        face_before = _make_face(seed=42)
        face_after = _make_displaced_face(face_before, seed=99, magnitude=50.0)

        results = estimate_all_procedures(face_before, face_after)
        for r in results:
            assert 0.0 <= r.intensity <= 100.0
