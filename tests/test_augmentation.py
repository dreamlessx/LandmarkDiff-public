"""Tests for training data augmentation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.augmentation import (
    AugmentationConfig,
    FitzpatrickBalancer,
    _adjust_saturation,
    _shift_hue,
    _transform_landmarks,
    augment_skin_tone,
    augment_training_sample,
)


@pytest.fixture
def sample_data():
    """Create a realistic training sample."""
    rng = np.random.default_rng(42)
    h, w = 512, 512
    input_img = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    target_img = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    conditioning = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask = rng.random((h, w)).astype(np.float32)
    landmarks_src = rng.random((478, 2)).astype(np.float32) * 0.5 + 0.25
    landmarks_dst = landmarks_src + rng.normal(0, 0.005, (478, 2)).astype(np.float32)
    return {
        "input_image": input_img,
        "target_image": target_img,
        "conditioning": conditioning,
        "mask": mask,
        "landmarks_src": landmarks_src,
        "landmarks_dst": landmarks_dst,
    }


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_defaults(self):
        cfg = AugmentationConfig()
        assert cfg.random_flip is True
        assert cfg.random_rotation_deg == 5.0
        assert cfg.brightness_range == (0.9, 1.1)
        assert cfg.conditioning_dropout_prob == 0.1

    def test_custom_config(self):
        cfg = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            brightness_range=(1.0, 1.0),
        )
        assert cfg.random_flip is False
        assert cfg.random_rotation_deg == 0.0


class TestAugmentTrainingSample:
    """Tests for augment_training_sample function."""

    def test_output_shapes(self, sample_data):
        result = augment_training_sample(**sample_data, rng=np.random.default_rng(42))
        assert result["input_image"].shape == (512, 512, 3)
        assert result["target_image"].shape == (512, 512, 3)
        assert result["conditioning"].shape == (512, 512, 3)
        assert result["mask"].shape[:2] == (512, 512)

    def test_output_dtypes(self, sample_data):
        result = augment_training_sample(**sample_data, rng=np.random.default_rng(42))
        assert result["input_image"].dtype == np.uint8
        assert result["target_image"].dtype == np.uint8
        assert result["conditioning"].dtype == np.uint8

    def test_landmarks_preserved(self, sample_data):
        result = augment_training_sample(**sample_data, rng=np.random.default_rng(42))
        assert "landmarks_src" in result
        assert "landmarks_dst" in result
        assert result["landmarks_src"].shape == (478, 2)
        assert result["landmarks_dst"].shape == (478, 2)

    def test_no_flip(self, sample_data):
        """With flip disabled, output should only differ by photometric augmentation."""
        config = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            random_scale=(1.0, 1.0),
            random_translate=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_shift_range=0.0,
            conditioning_dropout_prob=0.0,
            conditioning_noise_std=0.0,
        )
        result = augment_training_sample(**sample_data, config=config)
        np.testing.assert_array_equal(result["input_image"], sample_data["input_image"])

    def test_deterministic_with_seed(self, sample_data):
        """Same seed should produce same output."""
        r1 = augment_training_sample(**sample_data, rng=np.random.default_rng(123))
        r2 = augment_training_sample(**sample_data, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(r1["input_image"], r2["input_image"])
        np.testing.assert_array_equal(r1["target_image"], r2["target_image"])

    def test_different_seeds_differ(self, sample_data):
        """Different seeds should produce different outputs."""
        r1 = augment_training_sample(**sample_data, rng=np.random.default_rng(1))
        r2 = augment_training_sample(**sample_data, rng=np.random.default_rng(999))
        # At least one image should differ
        assert not np.array_equal(r1["input_image"], r2["input_image"])

    def test_without_landmarks(self, sample_data):
        """Should work without landmarks."""
        del sample_data["landmarks_src"]
        del sample_data["landmarks_dst"]
        sample_data["landmarks_src"] = None
        sample_data["landmarks_dst"] = None
        result = augment_training_sample(**sample_data, rng=np.random.default_rng(42))
        assert "input_image" in result
        assert result.get("landmarks_src") is None

    def test_conditioning_dropout(self, sample_data):
        """With 100% dropout, conditioning should be all zeros."""
        config = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            random_scale=(1.0, 1.0),
            random_translate=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_shift_range=0.0,
            conditioning_dropout_prob=1.0,  # always dropout
            conditioning_noise_std=0.0,
        )
        result = augment_training_sample(
            **sample_data, config=config, rng=np.random.default_rng(42)
        )
        assert result["conditioning"].sum() == 0


class TestTransformLandmarks:
    """Tests for _transform_landmarks helper."""

    def test_identity_transform(self):
        landmarks = np.array([[0.5, 0.5], [0.25, 0.75]])
        M = np.float64([[1, 0, 0], [0, 1, 0]])  # identity
        result = _transform_landmarks(landmarks, M, 512, 512)
        np.testing.assert_allclose(result, landmarks, atol=1e-6)

    def test_translation(self):
        landmarks = np.array([[0.5, 0.5]])
        M = np.float64([[1, 0, 10], [0, 1, 10]])  # translate 10px
        result = _transform_landmarks(landmarks, M, 512, 512)
        expected_x = (0.5 * 512 + 10) / 512
        expected_y = (0.5 * 512 + 10) / 512
        np.testing.assert_allclose(result[0, 0], expected_x, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], expected_y, atol=1e-6)

    def test_clipping(self):
        """Landmarks should be clipped to [0, 1]."""
        landmarks = np.array([[0.99, 0.99]])
        M = np.float64([[1, 0, 100], [0, 1, 100]])  # large translate
        result = _transform_landmarks(landmarks, M, 512, 512)
        assert result[0, 0] <= 1.0
        assert result[0, 1] <= 1.0


class TestColorAugmentation:
    """Tests for color augmentation helpers."""

    def test_adjust_saturation(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = _adjust_saturation(img, 1.5)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_adjust_saturation_identity(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = _adjust_saturation(img, 1.0)
        # Should be nearly identical (HSV/BGR round-trip has rounding error)
        assert np.abs(result.astype(int) - img.astype(int)).max() <= 5

    def test_shift_hue(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = _shift_hue(img, 30.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_shift_hue_zero(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = _shift_hue(img, 0.0)
        # Zero shift should produce nearly identical result (HSV round-trip rounding)
        assert np.abs(result.astype(int) - img.astype(int)).max() <= 5


class TestSkinToneAugmentation:
    """Tests for augment_skin_tone function."""

    def test_output_shape(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = augment_skin_tone(img, ita_delta=10.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_zero_delta(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = augment_skin_tone(img, ita_delta=0.0)
        # Zero delta should produce nearly identical result (LAB round-trip rounding)
        assert np.abs(result.astype(int) - img.astype(int)).max() <= 8

    def test_positive_delta_lightens(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        lighter = augment_skin_tone(img, ita_delta=20.0)
        # Positive delta should increase lightness on average
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_light = cv2.cvtColor(lighter, cv2.COLOR_BGR2LAB)
        assert lab_light[:, :, 0].mean() > lab_orig[:, :, 0].mean()

    def test_negative_delta_darkens(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        darker = augment_skin_tone(img, ita_delta=-20.0)
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_dark = cv2.cvtColor(darker, cv2.COLOR_BGR2LAB)
        assert lab_dark[:, :, 0].mean() < lab_orig[:, :, 0].mean()


class TestFitzpatrickBalancer:
    """Tests for FitzpatrickBalancer class."""

    def test_default_target_distribution(self):
        balancer = FitzpatrickBalancer()
        assert len(balancer.target) == 6
        for v in balancer.target.values():
            assert abs(v - 1 / 6) < 1e-6

    def test_register_and_weights(self):
        balancer = FitzpatrickBalancer()
        # Register heavily imbalanced samples
        for _ in range(100):
            balancer.register_sample("I")
        for _ in range(10):
            balancer.register_sample("VI")

        types = ["I", "VI"]
        weights = balancer.get_sampling_weights(types)
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-6
        # VI should have higher weight (underrepresented)
        assert weights[1] > weights[0]

    def test_uniform_distribution(self):
        balancer = FitzpatrickBalancer()
        for ft in ["I", "II", "III", "IV", "V", "VI"]:
            for _ in range(100):
                balancer.register_sample(ft)

        types = ["I", "II", "III", "IV", "V", "VI"]
        weights = balancer.get_sampling_weights(types)
        # All weights should be roughly equal
        assert np.std(weights) < 0.01

    def test_custom_target(self):
        target = {"I": 0.5, "VI": 0.5}
        balancer = FitzpatrickBalancer(target_distribution=target)
        assert balancer.target["I"] == 0.5
        assert balancer.target["VI"] == 0.5

    def test_empty_counts(self):
        balancer = FitzpatrickBalancer()
        types = ["I", "II"]
        weights = balancer.get_sampling_weights(types)
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-6
