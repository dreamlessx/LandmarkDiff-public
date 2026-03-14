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
            identity_threshold=0.6,
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
