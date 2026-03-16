"""Extended tests for morphometry module (nasal ratios, facial symmetry)."""

from __future__ import annotations

import numpy as np


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


class TestNasalRatiosToDict:
    """Tests for NasalRatios.to_dict."""

    def test_returns_dict_with_all_fields(self):
        from landmarkdiff.morphometry import NasalRatios

        nr = NasalRatios(
            alar_intercanthal=1.0,
            alar_face_width=0.2,
            nose_length_face_height=0.3,
            tip_midline_deviation=0.01,
            nostril_vertical_asymmetry=0.005,
        )
        d = nr.to_dict()
        assert len(d) == 5
        assert d["alar_intercanthal"] == 1.0
        assert d["tip_midline_deviation"] == 0.01

    def test_default_to_dict(self):
        from landmarkdiff.morphometry import NasalRatios

        d = NasalRatios().to_dict()
        assert all(v == 0.0 for v in d.values())


class TestNasalRatiosImprovementDetailed:
    """Detailed tests for NasalRatios.improvement_score."""

    def test_deviation_improved(self):
        from landmarkdiff.morphometry import NasalRatios

        ref = NasalRatios(tip_midline_deviation=0.1, nostril_vertical_asymmetry=0.08)
        pred = NasalRatios(tip_midline_deviation=0.02, nostril_vertical_asymmetry=0.01)
        scores = pred.improvement_score(ref)
        assert scores["tip_midline_deviation"] is True
        assert scores["nostril_vertical_asymmetry"] is True

    def test_deviation_worsened(self):
        from landmarkdiff.morphometry import NasalRatios

        ref = NasalRatios(tip_midline_deviation=0.01, nostril_vertical_asymmetry=0.005)
        pred = NasalRatios(tip_midline_deviation=0.15, nostril_vertical_asymmetry=0.1)
        scores = pred.improvement_score(ref)
        assert scores["tip_midline_deviation"] is False
        assert scores["nostril_vertical_asymmetry"] is False


class TestMorphometryFromImage:
    """Tests for compute_from_image methods (mocked mediapipe path)."""

    def test_nasal_morphometry_no_mediapipe(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import NasalMorphometry

        morph = NasalMorphometry()
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        with patch.dict("sys.modules", {"mediapipe": None}):
            result = morph.compute_from_image(img)
        assert result is None

    def test_facial_symmetry_no_mediapipe(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import FacialSymmetry

        sym = FacialSymmetry()
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        with patch.dict("sys.modules", {"mediapipe": None}):
            result = sym.compute_from_image(img)
        assert result is None


class TestCompareMorphometry:
    """Tests for compare_morphometry function using mocks."""

    def test_both_none_ratios(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import (
            FacialSymmetry,
            NasalMorphometry,
            compare_morphometry,
        )

        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        with (
            patch.object(NasalMorphometry, "compute_from_image", return_value=None),
            patch.object(FacialSymmetry, "compute_from_image", return_value=None),
        ):
            result = compare_morphometry(img, img)
        assert result["procedure"] == "rhinoplasty"
        assert result["input_ratios"] is None
        assert result["pred_ratios"] is None
        assert result["improvements"] is None
        assert result["symmetry_improved"] is None

    def test_with_valid_ratios_and_symmetry(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import (
            FacialSymmetry,
            NasalMorphometry,
            NasalRatios,
            compare_morphometry,
        )

        input_ratios = NasalRatios(
            alar_intercanthal=1.5, tip_midline_deviation=0.1, nostril_vertical_asymmetry=0.05
        )
        pred_ratios = NasalRatios(
            alar_intercanthal=1.1, tip_midline_deviation=0.02, nostril_vertical_asymmetry=0.01
        )
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        with (
            patch.object(
                NasalMorphometry, "compute_from_image", side_effect=[input_ratios, pred_ratios]
            ),
            patch.object(FacialSymmetry, "compute_from_image", side_effect=[0.15, 0.08]),
        ):
            result = compare_morphometry(img, img)
        assert result["improvements"] is not None
        assert result["symmetry_improved"] is True

    def test_symmetry_worsened(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import (
            FacialSymmetry,
            NasalMorphometry,
            compare_morphometry,
        )

        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        with (
            patch.object(NasalMorphometry, "compute_from_image", return_value=None),
            patch.object(FacialSymmetry, "compute_from_image", side_effect=[0.05, 0.10]),
        ):
            result = compare_morphometry(img, img)
        assert result["symmetry_improved"] is False

    def test_custom_procedure(self):
        from unittest.mock import patch

        from landmarkdiff.morphometry import (
            FacialSymmetry,
            NasalMorphometry,
            compare_morphometry,
        )

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        with (
            patch.object(NasalMorphometry, "compute_from_image", return_value=None),
            patch.object(FacialSymmetry, "compute_from_image", return_value=None),
        ):
            result = compare_morphometry(img, img, procedure="blepharoplasty")
        assert result["procedure"] == "blepharoplasty"
