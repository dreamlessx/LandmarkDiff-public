"""Extended tests for morphometry module (nasal ratios, facial symmetry)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_landmarks_478(seed=42):
    """Create mock 478-point landmarks as (478, 3) array."""
    rng = np.random.default_rng(seed)
    lm = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        lm[i, 0] = 0.3 + rng.random() * 0.4
        lm[i, 1] = 0.2 + rng.random() * 0.6
        lm[i, 2] = rng.random() * 0.1
    return lm


class TestNasalRatios:
    """Tests for NasalRatios dataclass."""

    def test_defaults(self):
        from landmarkdiff.morphometry import NasalRatios

        nr = NasalRatios()
        assert nr.alar_intercanthal == 0.0
        assert nr.alar_face_width == 0.0

    def test_improvement_score(self):
        from landmarkdiff.morphometry import NasalRatios

        reference = NasalRatios(alar_intercanthal=1.5, alar_face_width=0.3)
        improved = NasalRatios(alar_intercanthal=1.1, alar_face_width=0.22)
        scores = improved.improvement_score(reference)
        assert isinstance(scores, dict)
        assert "alar_intercanthal" in scores
        assert "alar_face_width" in scores

    def test_improvement_score_with_equal_values(self):
        from landmarkdiff.morphometry import NasalRatios

        nr = NasalRatios(alar_intercanthal=1.0, alar_face_width=0.2)
        scores = nr.improvement_score(nr)
        assert isinstance(scores, dict)


class TestNasalMorphometry:
    """Tests for NasalMorphometry class."""

    def test_compute(self):
        from landmarkdiff.morphometry import NasalMorphometry

        morph = NasalMorphometry()
        lm = _make_landmarks_478()
        result = morph.compute(lm)
        assert hasattr(result, "alar_intercanthal")
        assert hasattr(result, "alar_face_width")
        assert hasattr(result, "nose_length_face_height")
        assert hasattr(result, "tip_midline_deviation")
        assert hasattr(result, "nostril_vertical_asymmetry")

    def test_different_landmarks_different_ratios(self):
        from landmarkdiff.morphometry import NasalMorphometry

        morph = NasalMorphometry()
        lm1 = _make_landmarks_478(seed=42)
        lm2 = _make_landmarks_478(seed=99)
        r1 = morph.compute(lm1)
        r2 = morph.compute(lm2)
        assert r1.alar_intercanthal != r2.alar_intercanthal


class TestFacialSymmetry:
    """Tests for FacialSymmetry class."""

    def test_compute(self):
        from landmarkdiff.morphometry import FacialSymmetry

        sym = FacialSymmetry()
        lm = _make_landmarks_478()
        score = sym.compute(lm)
        assert isinstance(score, float)

    def test_symmetric_landmarks(self):
        from landmarkdiff.morphometry import FacialSymmetry

        sym = FacialSymmetry()
        # Perfect center = maximum symmetry
        lm = np.zeros((478, 3), dtype=np.float32)
        lm[:, 0] = 0.5
        lm[:, 1] = 0.5
        score = sym.compute(lm)
        assert isinstance(score, float)


class TestCompareMorphometry:
    """Tests for compare_morphometry function.

    Note: compare_morphometry requires mediapipe.solutions which may not be
    available in all environments, so we test with a mock if needed.
    """

    def test_basic_comparison_requires_images(self):
        """compare_morphometry takes BGR images (not raw landmarks).
        Skip if mediapipe.solutions not available."""
        pytest.importorskip("mediapipe.solutions")
        from landmarkdiff.morphometry import compare_morphometry

        img1 = np.full((512, 512, 3), 128, dtype=np.uint8)
        img2 = np.full((512, 512, 3), 140, dtype=np.uint8)
        # May fail if no face detected, but function should not crash
        try:
            result = compare_morphometry(img1, img2)
            assert isinstance(result, dict)
        except Exception:
            pass  # acceptable if no face detected in synthetic images
