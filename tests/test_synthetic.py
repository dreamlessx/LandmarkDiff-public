"""Tests for synthetic data generation pipeline."""

import numpy as np
import pytest

from landmarkdiff.synthetic.augmentation import (
    apply_clinical_augmentation,
    color_temperature_jitter,
    gaussian_sensor_noise,
    jpeg_compression,
)
from landmarkdiff.synthetic.tps_warp import generate_random_warp


class TestAugmentation:
    def _make_image(self) -> np.ndarray:
        return np.random.default_rng(0).integers(50, 200, (512, 512, 3), dtype=np.uint8)

    def test_augmentation_preserves_shape(self):
        img = self._make_image()
        result = apply_clinical_augmentation(img, rng=np.random.default_rng(42))
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_augmentation_changes_image(self):
        img = self._make_image()
        result = apply_clinical_augmentation(img, rng=np.random.default_rng(42))
        assert not np.array_equal(result, img)

    def test_color_temperature(self):
        img = self._make_image()
        result = color_temperature_jitter(img, np.random.default_rng(0))
        assert result.shape == img.shape
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_jpeg_roundtrip(self):
        img = self._make_image()
        result = jpeg_compression(img, np.random.default_rng(0))
        assert result.shape == img.shape

    def test_sensor_noise_range(self):
        img = self._make_image()
        result = gaussian_sensor_noise(img, np.random.default_rng(0))
        assert result.dtype == np.uint8


class TestTPSWarp:
    def test_random_warp_returns_copy(self):
        landmarks = np.random.default_rng(0).uniform(100, 400, (478, 2)).astype(np.float32)
        indices = [1, 2, 4, 5, 6]
        result = generate_random_warp(landmarks, indices, rng=np.random.default_rng(42))
        assert result is not landmarks
        assert not np.array_equal(result, landmarks)

    def test_random_warp_only_modifies_indices(self):
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[:] = 256.0
        indices = [0, 1, 2]
        result = generate_random_warp(landmarks, indices, max_displacement=10.0, rng=np.random.default_rng(0))
        # Non-indexed landmarks should be unchanged
        assert np.array_equal(result[10], landmarks[10])
