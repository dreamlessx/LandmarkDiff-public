"""Tests for clinical safety validation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.safety import SafetyResult, SafetyValidator


class TestSafetyResult:
    """Tests for SafetyResult dataclass."""

    def test_default_passed(self):
        result = SafetyResult()
        assert result.passed
        assert len(result.failures) == 0

    def test_add_failure(self):
        result = SafetyResult()
        result.add_failure("test", "Test failure")
        assert not result.passed
        assert "Test failure" in result.failures
        assert result.checks["test"] is False

    def test_add_pass(self):
        result = SafetyResult()
        result.add_pass("test")
        assert result.passed
        assert result.checks["test"] is True

    def test_summary(self):
        result = SafetyResult()
        result.add_pass("check1")
        result.add_failure("check2", "Failed check 2")
        summary = result.summary()
        assert "FAIL" in summary
        assert "check1" in summary
        assert "check2" in summary


class TestSafetyValidator:
    """Tests for SafetyValidator."""

    @pytest.fixture
    def validator(self):
        return SafetyValidator(
            identity_threshold=0.5,
            max_displacement_fraction=0.05,
            watermark_enabled=True,
        )

    @pytest.fixture
    def test_image(self):
        """Create a realistic-looking test face image."""
        img = np.random.randint(80, 200, (512, 512, 3), dtype=np.uint8)
        # Add skin-like tone (red > blue)
        img[:, :, 2] = np.clip(img[:, :, 2] + 30, 0, 255)  # boost red
        return img

    def test_face_confidence_pass(self, validator):
        result = SafetyResult()
        validator._check_face_confidence(result, 0.9)
        assert result.passed
        assert result.checks.get("face_confidence") is True

    def test_face_confidence_fail(self, validator):
        result = SafetyResult()
        validator._check_face_confidence(result, 0.1)
        assert not result.passed
        assert result.checks.get("face_confidence") is False

    def test_output_quality_pass(self, validator, test_image):
        result = SafetyResult()
        validator._check_output_quality(result, test_image)
        assert result.passed

    def test_output_quality_black(self, validator):
        result = SafetyResult()
        black_img = np.zeros((512, 512, 3), dtype=np.uint8)
        validator._check_output_quality(result, black_img)
        assert not result.passed

    def test_output_quality_white(self, validator):
        result = SafetyResult()
        white_img = np.full((512, 512, 3), 255, dtype=np.uint8)
        validator._check_output_quality(result, white_img)
        assert not result.passed

    def test_anatomical_plausibility_pass(self, validator):
        result = SafetyResult()
        # Small displacement (within threshold)
        orig = np.random.rand(478, 2) * 0.5 + 0.25
        manip = orig + np.random.randn(478, 2) * 0.001
        validator._check_anatomical_plausibility(result, orig, manip, "rhinoplasty")
        assert result.checks.get("anatomical_magnitude") is True

    def test_anatomical_plausibility_fail(self, validator):
        result = SafetyResult()
        orig = np.random.rand(478, 2) * 0.5 + 0.25
        manip = orig + 0.1  # Large displacement
        validator._check_anatomical_plausibility(result, orig, manip, "rhinoplasty")
        assert result.checks.get("anatomical_magnitude") is False

    def test_watermark(self, validator, test_image):
        watermarked = validator.apply_watermark(test_image)
        assert watermarked.shape == test_image.shape
        # Watermarked should differ from original (bottom region)
        bottom_orig = test_image[-20:, :, :].astype(float)
        bottom_wm = watermarked[-20:, :, :].astype(float)
        assert np.abs(bottom_orig - bottom_wm).sum() > 0

    def test_watermark_disabled(self):
        validator = SafetyValidator(watermark_enabled=False)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = validator.apply_watermark(img)
        np.testing.assert_array_equal(result, img)

    def test_ood_check(self, validator, test_image):
        result = SafetyResult()
        validator._check_ood(result, test_image)
        assert result.checks.get("ood_basic") is True

    def test_ood_low_resolution(self, validator):
        result = SafetyResult()
        tiny_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        validator._check_ood(result, tiny_img)
        # Should warn but still pass basic check
        assert len(result.warnings) > 0

    def test_full_validate(self, validator, test_image):
        """Test full validation pipeline."""
        output = test_image.copy()
        output = (output.astype(float) * 0.95).astype(np.uint8)  # slight change

        result = validator.validate(
            input_image=test_image,
            output_image=output,
            face_confidence=0.95,
        )

        assert isinstance(result, SafetyResult)
        assert "face_confidence" in result.checks
        assert "output_quality" in result.checks

    def test_embed_metadata(self, validator, tmp_path):
        """Test metadata embedding."""
        img_path = str(tmp_path / "test_output.png")
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        validator.embed_metadata(
            img_path,
            procedure="rhinoplasty",
            intensity=65.0,
        )

        meta_path = tmp_path / "test_output.meta.json"
        assert meta_path.exists()

        import json

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["procedure"] == "rhinoplasty"
        assert meta["intensity"] == 65.0
        assert "disclaimer" in meta


class TestSafetyResultExtended:
    """Extended tests for SafetyResult."""

    def test_add_warning(self):
        result = SafetyResult()
        result.add_warning("test_warn", "Some warning")
        assert result.passed  # warnings do not cause failure
        assert "Some warning" in result.warnings

    def test_multiple_failures(self):
        result = SafetyResult()
        result.add_failure("a", "Failure A")
        result.add_failure("b", "Failure B")
        assert not result.passed
        assert len(result.failures) == 2
        assert result.checks["a"] is False
        assert result.checks["b"] is False

    def test_summary_pass(self):
        result = SafetyResult()
        result.add_pass("check1")
        summary = result.summary()
        assert "PASS" in summary
        assert "OK" in summary

    def test_summary_with_warnings(self):
        result = SafetyResult()
        result.add_pass("check1")
        result.add_warning("w", "Low res")
        summary = result.summary()
        assert "WARN" in summary
        assert "Low res" in summary


class TestOODDetection:
    """Extended OOD detection tests."""

    @pytest.fixture
    def validator(self):
        return SafetyValidator()

    def test_extreme_aspect_ratio(self, validator):
        """Extremely wide images should trigger a warning."""
        result = SafetyResult()
        wide_img = np.random.randint(50, 200, (64, 512, 3), dtype=np.uint8)
        validator._check_ood(result, wide_img)
        assert any("aspect" in w.lower() for w in result.warnings)

    def test_blue_tinted_image(self, validator):
        """Strongly blue-tinted images should trigger a warning."""
        result = SafetyResult()
        blue_img = np.zeros((256, 256, 3), dtype=np.uint8)
        blue_img[:, :, 0] = 200  # B channel high
        blue_img[:, :, 1] = 50
        blue_img[:, :, 2] = 50  # R channel low
        validator._check_ood(result, blue_img)
        assert any("blue" in w.lower() for w in result.warnings)

    def test_normal_face_no_ood_warnings(self, validator):
        """A normal face-like image should not trigger OOD warnings."""
        result = SafetyResult()
        img = np.full((512, 512, 3), 150, dtype=np.uint8)
        img[:, :, 2] = 180  # R > B, typical skin
        validator._check_ood(result, img)
        assert len(result.warnings) == 0
        assert result.checks.get("ood_basic") is True

    def test_grayscale_image(self, validator):
        """Single-channel image should not crash."""
        result = SafetyResult()
        gray = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        # 2D image, shape has no third channel
        validator._check_ood(result, gray)
        assert result.checks.get("ood_basic") is True


class TestOutputQualityExtended:
    """Extended output quality checks."""

    @pytest.fixture
    def validator(self):
        return SafetyValidator()

    def test_empty_image(self, validator):
        """Zero-size image should fail."""
        result = SafetyResult()
        empty = np.array([], dtype=np.uint8)
        validator._check_output_quality(result, empty)
        assert not result.passed

    def test_low_variance_warning(self, validator):
        """Very uniform images should trigger a warning."""
        result = SafetyResult()
        uniform = np.full((256, 256, 3), 128, dtype=np.uint8)
        # Add tiny noise so it passes the black/white checks
        uniform[0, 0, 0] = 130
        validator._check_output_quality(result, uniform)
        assert any("variance" in w.lower() for w in result.warnings)

    def test_none_image(self, validator):
        """None image should fail."""
        result = SafetyResult()
        validator._check_output_quality(result, None)
        assert not result.passed


class TestAnatomicalPlausibilityExtended:
    """Extended anatomical plausibility tests."""

    @pytest.fixture
    def validator(self):
        return SafetyValidator(max_displacement_fraction=0.05)

    def test_landmark_count_mismatch(self, validator):
        """Mismatched landmark counts should fail."""
        result = SafetyResult()
        orig = np.random.rand(478, 2)
        manip = np.random.rand(300, 2)
        validator._check_anatomical_plausibility(result, orig, manip, None)
        assert not result.passed
        assert any("mismatch" in f.lower() for f in result.failures)

    def test_zero_displacement(self, validator):
        """Identical landmarks should pass."""
        result = SafetyResult()
        lm = np.random.rand(478, 2) * 0.5 + 0.25
        validator._check_anatomical_plausibility(result, lm, lm.copy(), None)
        assert result.checks.get("anatomical_magnitude") is True
        assert result.details["max_displacement"] == 0.0

    def test_with_3d_landmarks(self, validator):
        """3D landmarks (N, 3) should use only the first 2 coords."""
        result = SafetyResult()
        orig_3d = np.random.rand(478, 3)
        manip_3d = orig_3d.copy()
        manip_3d[:, 2] += 0.5  # only z changes, should not affect check
        validator._check_anatomical_plausibility(result, orig_3d, manip_3d, None)
        assert result.checks.get("anatomical_magnitude") is True

    def test_validate_with_landmarks(self):
        """Full validate path with landmarks provided."""
        validator = SafetyValidator(
            max_displacement_fraction=0.05,
        )
        img = np.random.randint(80, 200, (512, 512, 3), dtype=np.uint8)
        img[:, :, 2] = np.clip(img[:, :, 2] + 30, 0, 255)

        orig_lm = np.random.rand(478, 2) * 0.4 + 0.3
        manip_lm = orig_lm + np.random.randn(478, 2) * 0.001

        result = validator.validate(
            input_image=img,
            output_image=img.copy(),
            landmarks_original=orig_lm,
            landmarks_manipulated=manip_lm,
            procedure="rhinoplasty",
            face_confidence=0.95,
        )
        assert isinstance(result, SafetyResult)
        assert "face_confidence" in result.checks
        assert "anatomical_magnitude" in result.checks


class TestMaxDisplacementLimits:
    """Test displacement threshold enforcement."""

    def test_boundary_pass(self):
        """Displacement just below threshold should pass."""
        validator = SafetyValidator(max_displacement_fraction=0.05)
        result = SafetyResult()
        orig = np.array([[0.5, 0.5]])
        manip = np.array([[0.5, 0.549]])  # 0.049, below 0.05
        validator._check_anatomical_plausibility(result, orig, manip, None)
        assert result.checks.get("anatomical_magnitude") is True

    def test_just_over_threshold(self):
        """Displacement just over threshold should fail."""
        validator = SafetyValidator(max_displacement_fraction=0.05)
        result = SafetyResult()
        orig = np.array([[0.5, 0.5]])
        manip = np.array([[0.5, 0.551]])  # 0.051
        validator._check_anatomical_plausibility(result, orig, manip, None)
        assert result.checks.get("anatomical_magnitude") is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        validator = SafetyValidator(max_displacement_fraction=0.1)
        result = SafetyResult()
        orig = np.array([[0.5, 0.5]])
        manip = np.array([[0.5, 0.59]])  # 0.09 < 0.1
        validator._check_anatomical_plausibility(result, orig, manip, None)
        assert result.checks.get("anatomical_magnitude") is True


class TestYawAngle:
    """Test yaw angle parameter is stored correctly."""

    def test_max_yaw_stored(self):
        validator = SafetyValidator(max_yaw_degrees=30.0)
        assert validator.max_yaw_degrees == 30.0


class TestWatermarkExtended:
    """Extended watermark tests."""

    def test_custom_text(self):
        validator = SafetyValidator(watermark_enabled=True)
        img = np.random.randint(80, 200, (256, 256, 3), dtype=np.uint8)
        result = validator.apply_watermark(img, text="CUSTOM TEXT")
        assert result.shape == img.shape
        # Bottom should differ
        assert not np.array_equal(result[-20:], img[-20:])

    def test_custom_opacity(self):
        validator = SafetyValidator(watermark_enabled=True)
        img = np.random.randint(80, 200, (256, 256, 3), dtype=np.uint8)
        r1 = validator.apply_watermark(img, opacity=0.1)
        r2 = validator.apply_watermark(img, opacity=0.9)
        # Different opacities produce different results
        assert not np.array_equal(r1, r2)
