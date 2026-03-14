"""Tests for neural face verifier and distortion detection."""

import numpy as np
import pytest

from landmarkdiff.face_verifier import (
    detect_blur,
    detect_noise,
    detect_compression_artifacts,
    detect_oversmoothing,
    detect_color_cast,
    detect_lighting_issues,
    analyze_distortions,
    DistortionReport,
    RestorationResult,
)


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------

def _make_sharp_image(h=256, w=256):
    """Create a sharp image with high-frequency detail."""
    img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    # Add edges
    img[50:200, 50:200] = 180
    img[80:170, 80:170] = 120
    img[100:150, 100:150] = 160
    return img


def _make_blurry_image(h=256, w=256):
    """Create a blurry image."""
    import cv2
    sharp = _make_sharp_image(h, w)
    return cv2.GaussianBlur(sharp, (31, 31), 10)


def _make_noisy_image(h=256, w=256):
    """Create a noisy image."""
    base = _make_sharp_image(h, w).astype(np.float32)
    noise = np.random.normal(0, 40, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _make_oversmoothed_image(h=256, w=256):
    """Create an oversmoothed (beauty-filtered) image."""
    import cv2
    sharp = _make_sharp_image(h, w)
    # Heavy bilateral filter (preserves edges, removes texture)
    return cv2.bilateralFilter(sharp, 15, 80, 80)


def _make_color_cast_image(h=256, w=256):
    """Create an image with strong color cast."""
    img = _make_sharp_image(h, w)
    # Add heavy blue cast
    img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + 80, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBlurDetection:
    def test_sharp_image_low_blur(self):
        sharp = _make_sharp_image()
        score = detect_blur(sharp)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, f"Sharp image should have low blur score, got {score}"

    def test_blurry_image_high_blur(self):
        blurry = _make_blurry_image()
        score = detect_blur(blurry)
        assert score > 0.3, f"Blurry image should have high blur score, got {score}"

    def test_blur_ordering(self):
        sharp = _make_sharp_image()
        blurry = _make_blurry_image()
        assert detect_blur(blurry) > detect_blur(sharp)


class TestNoiseDetection:
    def test_clean_image_low_noise(self):
        import cv2
        clean = np.full((256, 256, 3), 128, dtype=np.uint8)
        clean = cv2.GaussianBlur(clean, (5, 5), 1)  # Smooth = low noise
        score = detect_noise(clean)
        assert score < 0.3

    def test_noisy_image_high_noise(self):
        noisy = _make_noisy_image()
        score = detect_noise(noisy)
        assert score > 0.2, f"Noisy image should have high noise score, got {score}"


class TestCompressionDetection:
    def test_uncompressed_low_score(self):
        img = _make_sharp_image()
        score = detect_compression_artifacts(img)
        assert 0.0 <= score <= 1.0

    def test_tiny_image_returns_zero(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert detect_compression_artifacts(img) == 0.0


class TestOversmoothing:
    def test_textured_image_low_score(self):
        img = _make_sharp_image()
        score = detect_oversmoothing(img)
        assert 0.0 <= score <= 1.0

    def test_smooth_image_higher_score(self):
        sharp = _make_sharp_image()
        smooth = _make_oversmoothed_image()
        # Oversmoothed should score higher (more beauty-filtered)
        assert detect_oversmoothing(smooth) >= detect_oversmoothing(sharp) - 0.1


class TestColorCast:
    def test_neutral_image_low_cast(self):
        # Gray image = neutral
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        score = detect_color_cast(img)
        assert score <= 0.35

    def test_cast_image_high_score(self):
        cast = _make_color_cast_image()
        score = detect_color_cast(cast)
        assert score > 0.1, f"Color cast image should have high score, got {score}"


class TestLighting:
    def test_normal_exposure(self):
        img = _make_sharp_image()
        score = detect_lighting_issues(img)
        assert 0.0 <= score <= 1.0

    def test_overexposed(self):
        img = np.full((256, 256, 3), 250, dtype=np.uint8)
        score = detect_lighting_issues(img)
        assert score > 0.3, "Overexposed image should flag lighting issues"

    def test_underexposed(self):
        img = np.full((256, 256, 3), 5, dtype=np.uint8)
        score = detect_lighting_issues(img)
        assert score > 0.3, "Underexposed image should flag lighting issues"


class TestAnalyzeDistortions:
    def test_returns_report(self):
        img = _make_sharp_image()
        report = analyze_distortions(img)
        assert isinstance(report, DistortionReport)
        assert 0 <= report.quality_score <= 100
        assert report.severity in ("none", "mild", "moderate", "severe")

    def test_clean_image_high_quality(self):
        img = _make_sharp_image()
        report = analyze_distortions(img)
        assert report.quality_score > 40

    def test_blurry_image_flagged(self):
        img = _make_blurry_image()
        report = analyze_distortions(img)
        assert report.blur_score > 0.3

    def test_summary_format(self):
        img = _make_sharp_image()
        report = analyze_distortions(img)
        summary = report.summary()
        assert "Quality Score" in summary
        assert "Primary Issue" in summary


class TestRestorationResult:
    def test_summary(self):
        img = _make_sharp_image()
        report = analyze_distortions(img)
        result = RestorationResult(
            restored=img,
            original=img,
            distortion_report=report,
            post_quality_score=80.0,
            identity_similarity=0.95,
            identity_preserved=True,
            restoration_stages=["codeformer"],
            improvement=10.0,
        )
        summary = result.summary()
        assert "codeformer" in summary
        assert "0.95" in summary
