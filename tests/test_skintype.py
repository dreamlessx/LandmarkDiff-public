"""Tests for Fitzpatrick skin type auto-detection."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.skintype import (
    PostProcessParams,
    SkinTypeResult,
    detect_skin_type,
    get_postprocess_params,
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
        image_width=256,
        image_height=256,
        confidence=0.95,
    )


def _make_image(brightness: int = 180, size: int = 256) -> np.ndarray:
    """Create a synthetic face image with uniform skin tone."""
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    # Add slight noise
    noise = rng.integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class TestDetectSkinType:
    def test_returns_result(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert isinstance(result, SkinTypeResult)

    def test_fitzpatrick_valid_type(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert result.fitzpatrick_type in ("I", "II", "III", "IV", "V", "VI")

    def test_ita_finite(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert np.isfinite(result.ita_angle)

    def test_confidence_in_range(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert 0.0 <= result.confidence <= 1.0

    def test_sampled_pixels_positive(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert result.sampled_pixels > 0

    def test_light_skin_detected(self, mock_face):
        img = _make_image(220)
        result = detect_skin_type(img, mock_face)
        assert result.fitzpatrick_type in ("I", "II", "III")

    def test_dark_skin_detected(self, mock_face):
        img = _make_image(60)
        result = detect_skin_type(img, mock_face)
        assert result.fitzpatrick_type in ("IV", "V", "VI")

    def test_description_property(self, mock_face):
        img = _make_image(180)
        result = detect_skin_type(img, mock_face)
        assert len(result.description) > 0


class TestGetPostprocessParams:
    @pytest.mark.parametrize("skin_type", ["I", "II", "III", "IV", "V", "VI"])
    def test_all_types_return_params(self, skin_type):
        params = get_postprocess_params(skin_type)
        assert isinstance(params, PostProcessParams)

    def test_unknown_type_defaults(self):
        params = get_postprocess_params("unknown")
        assert isinstance(params, PostProcessParams)

    def test_darker_types_have_stronger_lab_blend(self):
        params_light = get_postprocess_params("I")
        params_dark = get_postprocess_params("VI")
        assert params_dark.lab_blend_weight > params_light.lab_blend_weight

    def test_darker_types_have_weaker_histogram_match(self):
        params_light = get_postprocess_params("I")
        params_dark = get_postprocess_params("VI")
        assert params_dark.histogram_match_strength < params_light.histogram_match_strength

    def test_all_params_in_valid_range(self):
        for t in ["I", "II", "III", "IV", "V", "VI"]:
            p = get_postprocess_params(t)
            assert 0.0 <= p.histogram_match_strength <= 1.0
            assert 0.0 <= p.lab_blend_weight <= 1.0
            assert 0.0 <= p.sharpen_amount <= 1.0
            assert 0.0 <= p.color_correction_strength <= 1.0
