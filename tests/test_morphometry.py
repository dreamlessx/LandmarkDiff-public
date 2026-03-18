"""Tests for nasal morphometry and facial symmetry evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.morphometry import (
    CHIN,
    FOREHEAD,
    GLABELLA,
    LEFT_CHEEK,
    LEFT_INNER_EYE,
    LEFT_NOSTRIL,
    LEFT_OUTER_EYE,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_INNER_EYE,
    RIGHT_NOSTRIL,
    RIGHT_OUTER_EYE,
    FacialSymmetry,
    NasalMorphometry,
    NasalRatios,
)


def _make_landmarks(n: int = 478, **overrides: tuple[float, float]) -> np.ndarray:
    """Create synthetic landmarks with controllable key points.

    Produces a roughly symmetric face layout with specified overrides
    for named landmark indices.

    Args:
        n: Number of landmarks (default 478 for MediaPipe)
        **overrides: Mapping from landmark index to (x, y) coordinates

    Returns:
        Array of shape (n, 2) with landmark coordinates
    """
    # Base: spread points in a face-like oval
    rng = np.random.RandomState(42)
    pts = np.zeros((n, 2))

    # Place key anatomical points in a realistic layout (512x512 image space)
    defaults = {
        NOSE_TIP: (256, 320),
        LEFT_NOSTRIL: (235.0, 340),
        RIGHT_NOSTRIL: (277, 340),
        LEFT_INNER_EYE: (220, 240),
        RIGHT_INNER_EYE: (292, 240),
        LEFT_OUTER_EYE: (185, 235.0),
        RIGHT_OUTER_EYE: (327, 235.0),
        LEFT_CHEEK: (150, 310),
        RIGHT_CHEEK: (362, 310),
        FOREHEAD: (256, 120),
        CHIN: (256, 440),
        GLABELLA: (256, 210),
    }

    # Fill all points with noise around center
    for i in range(n):
        pts[i] = (256 + rng.randn() * 30, 280 + rng.randn() * 60)

    # Set key points
    for idx, (x, y) in defaults.items():
        if idx < n:
            pts[idx] = (x, y)

    # Apply overrides
    for idx_name, (x, y) in overrides.items():
        idx = int(idx_name)
        pts[idx] = (x, y)

    return pts


class TestNasalMorphometry:
    """Tests for NasalMorphometry computation."""

    def test_basic_computation(self) -> None:
        landmarks = _make_landmarks()
        morph = NasalMorphometry()
        ratios = morph.compute(landmarks)

        assert isinstance(ratios, NasalRatios)
        assert ratios.alar_intercanthal > 0
        assert ratios.alar_face_width > 0
        assert ratios.nose_length_face_height > 0
        assert ratios.tip_midline_deviation >= 0
        assert ratios.nostril_vertical_asymmetry >= 0

    def test_ideal_nose_ratios(self) -> None:
        """A well-proportioned face should have ratios near ideals."""
        landmarks = _make_landmarks()
        morph = NasalMorphometry()
        ratios = morph.compute(landmarks)

        # Alar width (42) / intercanthal (72) ~ 0.58
        # This is synthetic, just check it's reasonable
        assert 0.1 < ratios.alar_intercanthal < 3.0
        assert 0.05 < ratios.alar_face_width < 0.5

    def test_centered_nose_tip(self) -> None:
        """Centered nose tip should have near-zero deviation."""
        landmarks = _make_landmarks()
        morph = NasalMorphometry()
        ratios = morph.compute(landmarks)

        # Nose tip at x=256, midline at (185+327)/2 = 256
        assert ratios.tip_midline_deviation < 0.01

    def test_asymmetric_nostrils(self) -> None:
        """Vertically offset nostrils increase asymmetry score."""
        symmetric = _make_landmarks()
        asymmetric = _make_landmarks()
        # Move left nostril down by 20px
        asymmetric[LEFT_NOSTRIL] = (235.0, 360.0)

        morph = NasalMorphometry()
        sym_ratios = morph.compute(symmetric)
        asym_ratios = morph.compute(asymmetric)

        assert asym_ratios.nostril_vertical_asymmetry > sym_ratios.nostril_vertical_asymmetry

    def test_deviated_nose_tip(self) -> None:
        """Off-center nose tip increases deviation score."""
        centered = _make_landmarks()
        deviated = _make_landmarks()
        deviated[NOSE_TIP] = (280.0, 320.0)  # shifted right

        morph = NasalMorphometry()
        c_ratios = morph.compute(centered)
        d_ratios = morph.compute(deviated)

        assert d_ratios.tip_midline_deviation > c_ratios.tip_midline_deviation

    def test_to_dict(self) -> None:
        landmarks = _make_landmarks()
        morph = NasalMorphometry()
        ratios = morph.compute(landmarks)
        d = ratios.to_dict()

        assert set(d.keys()) == {
            "alar_intercanthal",
            "alar_face_width",
            "nose_length_face_height",
            "tip_midline_deviation",
            "nostril_vertical_asymmetry",
        }
        for v in d.values():
            assert isinstance(v, float)

    def test_3d_landmarks(self) -> None:
        """Should work with (N, 3) landmarks too."""
        pts_2d = _make_landmarks()
        pts_3d = np.column_stack([pts_2d, np.zeros(len(pts_2d))])

        morph = NasalMorphometry()
        ratios_2d = morph.compute(pts_2d)
        ratios_3d = morph.compute(pts_3d)

        assert ratios_2d.alar_intercanthal == pytest.approx(ratios_3d.alar_intercanthal, abs=1e-6)


class TestNasalRatiosImprovement:
    """Tests for improvement scoring."""

    def test_improvement_toward_ideal(self) -> None:
        before = NasalRatios(
            alar_intercanthal=0.8,
            alar_face_width=0.25,
            tip_midline_deviation=0.05,
            nostril_vertical_asymmetry=0.03,
        )
        after = NasalRatios(
            alar_intercanthal=0.95,  # closer to 1.0
            alar_face_width=0.21,  # closer to 0.20
            tip_midline_deviation=0.02,  # lower
            nostril_vertical_asymmetry=0.01,  # lower
        )
        improvements = after.improvement_score(before)

        assert improvements["alar_intercanthal"] is True
        assert improvements["alar_face_width"] is True
        assert improvements["tip_midline_deviation"] is True
        assert improvements["nostril_vertical_asymmetry"] is True

    def test_no_improvement(self) -> None:
        before = NasalRatios(
            alar_intercanthal=0.98,
            alar_face_width=0.20,
            tip_midline_deviation=0.01,
            nostril_vertical_asymmetry=0.005,
        )
        after = NasalRatios(
            alar_intercanthal=0.7,  # worse
            alar_face_width=0.30,  # worse
            tip_midline_deviation=0.08,  # worse
            nostril_vertical_asymmetry=0.04,  # worse
        )
        improvements = after.improvement_score(before)

        assert improvements["alar_intercanthal"] is False
        assert improvements["alar_face_width"] is False
        assert improvements["tip_midline_deviation"] is False
        assert improvements["nostril_vertical_asymmetry"] is False


class TestFacialSymmetry:
    """Tests for bilateral symmetry scoring."""

    def test_perfect_symmetry(self) -> None:
        """Perfectly symmetric face should score near zero."""
        pts = np.zeros((478, 2))

        # Place symmetric pairs
        midline = 256.0
        for i in range(0, 478, 2):
            offset = (i + 1) * 0.5
            y = 100.0 + i * 0.7
            pts[i] = (midline - offset, y)
            if i + 1 < 478:
                pts[i + 1] = (midline + offset, y)

        # Set eye corners for midline
        pts[LEFT_OUTER_EYE] = (200.0, 235.0)
        pts[RIGHT_OUTER_EYE] = (312.0, 235.0)

        sym = FacialSymmetry()
        score = sym.compute(pts)

        # Should be very low (near-perfect symmetry)
        assert score < 0.5

    def test_asymmetric_face(self) -> None:
        """Deliberately asymmetric face should score higher than symmetric."""
        # Build a nearly-symmetric face
        pts_sym = np.zeros((478, 2))
        midline = 256.0
        for i in range(0, 478, 2):
            offset = (i + 1) * 0.5
            y = 100.0 + i * 0.7
            pts_sym[i] = (midline - offset, y)
            if i + 1 < 478:
                pts_sym[i + 1] = (midline + offset, y)
        pts_sym[LEFT_OUTER_EYE] = (200.0, 235.0)
        pts_sym[RIGHT_OUTER_EYE] = (312.0, 235.0)

        # Make asymmetric copy: shift left-side points (except eye corners)
        pts_asym = pts_sym.copy()
        keep_fixed = {LEFT_OUTER_EYE, RIGHT_OUTER_EYE}
        for i in range(len(pts_asym)):
            if i not in keep_fixed and pts_asym[i][0] < midline:
                pts_asym[i][0] -= 50.0

        sym = FacialSymmetry()
        sym_score = sym.compute(pts_sym)
        asym_score = sym.compute(pts_asym)

        assert asym_score > sym_score

    def test_empty_sides(self) -> None:
        """All points on one side should return 0."""
        pts = np.zeros((478, 2))
        pts[:, 0] = 100.0  # all on left
        pts[LEFT_OUTER_EYE] = (100.0, 235.0)
        pts[RIGHT_OUTER_EYE] = (100.0, 235.0)

        sym = FacialSymmetry()
        score = sym.compute(pts)
        assert score == 0.0
