"""Tests for the FID computation module."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.fid import (
    ImageFolderDataset,
    NumpyArrayDataset,
    _calculate_fid,
    _compute_statistics,
    _imagenet_normalize,
)


class TestImageNetNormalize:
    """Tests for ImageNet normalization."""

    def test_output_shape(self):
        import torch

        t = torch.randn(3, 64, 64)
        out = _imagenet_normalize(t)
        assert out.shape == (3, 64, 64)

    def test_known_values(self):
        import torch

        # All zeros input
        t = torch.zeros(3, 1, 1)
        out = _imagenet_normalize(t)
        # (0 - mean) / std
        expected_r = (0.0 - 0.485) / 0.229
        assert abs(out[0, 0, 0].item() - expected_r) < 1e-5


class TestImageFolderDataset:
    """Tests for the image folder dataset."""

    @pytest.fixture
    def image_dir(self, tmp_path):
        """Create a temp directory with test images."""
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i:04d}.png"), img)
        return tmp_path

    def test_length(self, image_dir):
        ds = ImageFolderDataset(image_dir)
        assert len(ds) == 5

    def test_item_shape(self, image_dir):
        ds = ImageFolderDataset(image_dir, image_size=299)
        item = ds[0]
        assert item.shape == (3, 299, 299)

    def test_custom_size(self, image_dir):
        ds = ImageFolderDataset(image_dir, image_size=128)
        item = ds[0]
        assert item.shape == (3, 128, 128)

    def test_empty_directory(self, tmp_path):
        ds = ImageFolderDataset(tmp_path)
        assert len(ds) == 0

    def test_filters_non_images(self, tmp_path):
        # Create a non-image file
        (tmp_path / "readme.txt").write_text("not an image")
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "real.png"), img)
        ds = ImageFolderDataset(tmp_path)
        assert len(ds) == 1


class TestNumpyArrayDataset:
    """Tests for the numpy array dataset."""

    def test_length(self):
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        ds = NumpyArrayDataset(images)
        assert len(ds) == 3

    def test_item_shape(self):
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
        ds = NumpyArrayDataset(images, image_size=299)
        item = ds[0]
        assert item.shape == (3, 299, 299)

    def test_auto_resize(self):
        images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)]
        ds = NumpyArrayDataset(images, image_size=64)
        item = ds[0]
        assert item.shape == (3, 64, 64)


class TestComputeStatistics:
    """Tests for feature statistics computation."""

    def test_mean_shape(self):
        features = np.random.randn(100, 2048)
        mu, sigma = _compute_statistics(features)
        assert mu.shape == (2048,)
        assert sigma.shape == (2048, 2048)

    def test_covariance_symmetric(self):
        features = np.random.randn(50, 64)
        _, sigma = _compute_statistics(features)
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)

    def test_known_statistics(self):
        # Two identical features -> zero covariance off-diagonal
        features = np.ones((10, 4))
        mu, sigma = _compute_statistics(features)
        np.testing.assert_allclose(mu, np.ones(4))


class TestCalculateFID:
    """Tests for the FID calculation."""

    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 64))
        mu, sigma = _compute_statistics(features)
        fid = _calculate_fid(mu, sigma, mu, sigma)
        assert abs(fid) < 1e-6  # Should be ~0 for same distribution

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        f1 = rng.standard_normal((100, 64))
        f2 = rng.standard_normal((100, 64)) + 5.0  # Shifted
        mu1, s1 = _compute_statistics(f1)
        mu2, s2 = _compute_statistics(f2)
        fid = _calculate_fid(mu1, s1, mu2, s2)
        assert fid > 0  # Should be positive for different distributions

    def test_fid_nonnegative(self):
        rng = np.random.default_rng(123)
        f1 = rng.standard_normal((50, 32))
        f2 = rng.standard_normal((50, 32)) * 2
        mu1, s1 = _compute_statistics(f1)
        mu2, s2 = _compute_statistics(f2)
        fid = _calculate_fid(mu1, s1, mu2, s2)
        assert fid >= -1e-6  # Allow tiny numerical error

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        f1 = rng.standard_normal((80, 32))
        f2 = rng.standard_normal((80, 32)) + 1.0
        mu1, s1 = _compute_statistics(f1)
        mu2, s2 = _compute_statistics(f2)
        fid_12 = _calculate_fid(mu1, s1, mu2, s2)
        fid_21 = _calculate_fid(mu2, s2, mu1, s1)
        assert abs(fid_12 - fid_21) < 1e-4
