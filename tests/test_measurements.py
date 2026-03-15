"""Tests for clinical facial measurements module."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.measurements import (
    PROCEDURE_CALIBRATION,
    calibrate_intensity,
    compute_canthal_tilt,
    compute_cephalometric,
    compute_cervicomental_angle,
    compute_dental_show,
    compute_facial_fifths,
    compute_facial_thirds,
    compute_goode_ratio,
    compute_lip_chin_relation,
    compute_mandibular_angle,
    compute_nasofrontal_angle,
    detect_scleral_show,
)


@pytest.fixture
def mock_face():
    """Create a plausible FaceLandmarks for testing."""
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4
        landmarks[i, 1] = 0.2 + rng.random() * 0.6
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=512,
        image_height=512,
        confidence=0.95,
    )


class TestGoodeRatio:
    def test_returns_dataclass(self, mock_face):
        result = compute_goode_ratio(mock_face)
        assert hasattr(result, "ratio")
        assert hasattr(result, "projection_px")
        assert hasattr(result, "length_px")
        assert hasattr(result, "classification")

    def test_ratio_positive(self, mock_face):
        result = compute_goode_ratio(mock_face)
        assert result.ratio > 0
        assert result.projection_px > 0
        assert result.length_px > 0

    def test_classification_valid(self, mock_face):
        result = compute_goode_ratio(mock_face)
        assert result.classification in ("underprojected", "normal", "overprojected")


class TestNasofrontalAngle:
    def test_returns_dataclass(self, mock_face):
        result = compute_nasofrontal_angle(mock_face)
        assert hasattr(result, "angle")
        assert hasattr(result, "classification")

    def test_angle_in_range(self, mock_face):
        result = compute_nasofrontal_angle(mock_face)
        assert 0.0 <= result.angle <= 180.0

    def test_classification_valid(self, mock_face):
        result = compute_nasofrontal_angle(mock_face)
        assert result.classification in ("acute", "normal", "obtuse")


class TestCanthalTilt:
    def test_returns_dataclass(self, mock_face):
        result = compute_canthal_tilt(mock_face)
        assert hasattr(result, "left_angle")
        assert hasattr(result, "right_angle")
        assert hasattr(result, "mean_angle")

    def test_angles_finite(self, mock_face):
        result = compute_canthal_tilt(mock_face)
        assert np.isfinite(result.left_angle)
        assert np.isfinite(result.right_angle)

    def test_classification_valid(self, mock_face):
        result = compute_canthal_tilt(mock_face)
        assert result.classification in ("negative", "neutral", "positive")


class TestCervicoMentalAngle:
    def test_returns_dataclass(self, mock_face):
        result = compute_cervicomental_angle(mock_face)
        assert hasattr(result, "angle")
        assert hasattr(result, "classification")

    def test_angle_positive(self, mock_face):
        result = compute_cervicomental_angle(mock_face)
        assert result.angle > 0

    def test_classification_valid(self, mock_face):
        result = compute_cervicomental_angle(mock_face)
        assert result.classification in ("acute", "ideal", "obtuse")


class TestLipChinRelation:
    def test_returns_dataclass(self, mock_face):
        result = compute_lip_chin_relation(mock_face)
        assert hasattr(result, "h_line_angle")
        assert hasattr(result, "ratio")
        assert hasattr(result, "classification")

    def test_ratio_positive(self, mock_face):
        result = compute_lip_chin_relation(mock_face)
        assert result.ratio > 0
        assert result.lip_to_chin_distance_px > 0

    def test_classification_valid(self, mock_face):
        result = compute_lip_chin_relation(mock_face)
        assert result.classification in (
            "retruded_chin",
            "normal",
            "prominent_chin",
        )


class TestScleralShow:
    def test_returns_dataclass(self, mock_face):
        result = detect_scleral_show(mock_face)
        assert hasattr(result, "left_show_px")
        assert hasattr(result, "right_show_px")
        assert hasattr(result, "has_scleral_show")

    def test_show_nonnegative(self, mock_face):
        result = detect_scleral_show(mock_face)
        assert result.left_show_px >= 0
        assert result.right_show_px >= 0
        assert result.mean_show_px >= 0

    def test_risk_level_valid(self, mock_face):
        result = detect_scleral_show(mock_face)
        assert result.risk_level in ("none", "mild", "significant")


class TestDentalShow:
    def test_returns_dataclass(self, mock_face):
        result = compute_dental_show(mock_face)
        assert hasattr(result, "show_px")
        assert hasattr(result, "classification")

    def test_show_nonnegative(self, mock_face):
        result = compute_dental_show(mock_face)
        assert result.show_px >= 0

    def test_classification_valid(self, mock_face):
        result = compute_dental_show(mock_face)
        assert result.classification in ("insufficient", "normal", "excessive")


class TestMandibularAngle:
    def test_returns_dataclass(self, mock_face):
        result = compute_mandibular_angle(mock_face)
        assert hasattr(result, "left_angle")
        assert hasattr(result, "right_angle")
        assert hasattr(result, "mean_angle")

    def test_angles_positive(self, mock_face):
        result = compute_mandibular_angle(mock_face)
        assert result.left_angle > 0
        assert result.right_angle > 0

    def test_classification_valid(self, mock_face):
        result = compute_mandibular_angle(mock_face)
        assert result.classification in ("low", "normal", "high")


class TestFacialThirds:
    def test_returns_dataclass(self, mock_face):
        result = compute_facial_thirds(mock_face)
        assert hasattr(result, "upper")
        assert hasattr(result, "middle")
        assert hasattr(result, "lower")
        assert hasattr(result, "deviation_from_ideal")

    def test_sum_to_one_ordered_face(self):
        """With properly ordered y-landmarks, thirds sum to 1."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        for i in range(478):
            landmarks[i, 0] = 0.3 + rng.random() * 0.4
            landmarks[i, 1] = 0.3 + rng.random() * 0.4
        # Set key landmarks in proper y-order (top to bottom)
        landmarks[10, 1] = 0.15  # trichion (top)
        landmarks[9, 1] = 0.30  # glabella
        landmarks[94, 1] = 0.55  # subnasale
        landmarks[152, 1] = 0.80  # menton (bottom)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.95,
        )
        result = compute_facial_thirds(face)
        total = result.upper + result.middle + result.lower
        assert abs(total - 1.0) < 0.01

    def test_deviation_nonnegative(self, mock_face):
        result = compute_facial_thirds(mock_face)
        assert result.deviation_from_ideal >= 0


class TestFacialFifths:
    def test_five_segments(self, mock_face):
        result = compute_facial_fifths(mock_face)
        assert len(result.widths) == 5

    def test_sum_close_to_one_ordered_face(self):
        """With properly ordered x-landmarks, fifths sum to 1."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        for i in range(478):
            landmarks[i, 0] = 0.3 + rng.random() * 0.4
            landmarks[i, 1] = 0.3 + rng.random() * 0.4
        # Set key landmarks in proper x-order (left to right)
        landmarks[234, 0] = 0.10  # left temporal
        landmarks[33, 0] = 0.25  # left outer canthus
        landmarks[133, 0] = 0.38  # left inner canthus
        landmarks[362, 0] = 0.62  # right inner canthus
        landmarks[263, 0] = 0.75  # right outer canthus
        landmarks[454, 0] = 0.90  # right temporal
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.95,
        )
        result = compute_facial_fifths(face)
        total = sum(result.widths)
        assert abs(total - 1.0) < 0.01

    def test_deviation_nonnegative(self, mock_face):
        result = compute_facial_fifths(mock_face)
        assert result.deviation_from_ideal >= 0


class TestCalibrateIntensity:
    def test_known_procedure(self, mock_face):
        result = calibrate_intensity("rhinoplasty", 50.0)
        assert result > 0

    def test_zero_intensity(self):
        result = calibrate_intensity("rhinoplasty", 0.0)
        assert result >= 0
        assert result < 1.0

    def test_full_intensity(self):
        result = calibrate_intensity("rhinoplasty", 100.0)
        assert result > 0

    def test_unknown_procedure(self):
        result = calibrate_intensity("unknown_proc", 50.0)
        assert result == 0.5

    def test_with_face_returns_pixels(self, mock_face):
        mm_result = calibrate_intensity("rhinoplasty", 50.0)
        px_result = calibrate_intensity("rhinoplasty", 50.0, face=mock_face)
        # Pixel result should differ from mm result (scaled by IPD)
        assert px_result != mm_result

    def test_linear_mode(self):
        result = calibrate_intensity("rhinoplasty", 50.0, use_sigmoid=False)
        assert result > 0

    def test_all_procedures_have_calibration(self):
        expected = {
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
            "alarplasty",
            "canthoplasty",
            "buccal_fat_removal",
            "dimpleplasty",
            "genioplasty",
            "malarplasty",
            "lip_lift",
            "lip_augmentation",
            "forehead_reduction",
            "submental_liposuction",
            "otoplasty",
        }
        assert set(PROCEDURE_CALIBRATION.keys()) == expected

    def test_monotonic_intensity(self):
        """Higher intensity should produce larger displacement."""
        low = calibrate_intensity("rhinoplasty", 20.0)
        mid = calibrate_intensity("rhinoplasty", 50.0)
        high = calibrate_intensity("rhinoplasty", 80.0)
        assert low < mid < high


class TestCephalometric:
    def test_returns_dataclass(self, mock_face):
        result = compute_cephalometric(mock_face)
        assert hasattr(result, "sna_angle")
        assert hasattr(result, "snb_angle")
        assert hasattr(result, "anb_angle")
        assert hasattr(result, "skeletal_class")

    def test_angles_positive(self, mock_face):
        result = compute_cephalometric(mock_face)
        assert result.sna_angle > 0
        assert result.snb_angle > 0

    def test_anb_is_difference(self, mock_face):
        result = compute_cephalometric(mock_face)
        assert abs(result.anb_angle - (result.sna_angle - result.snb_angle)) < 1e-5

    def test_skeletal_class_valid(self, mock_face):
        result = compute_cephalometric(mock_face)
        assert result.skeletal_class in ("I", "II", "III")

    def test_wits_nonnegative(self, mock_face):
        result = compute_cephalometric(mock_face)
        assert result.wits_appraisal_px >= 0
