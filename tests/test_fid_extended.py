"""Extended tests for fid module -- statistics and FID calculation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestComputeStatistics:
    """Tests for _compute_statistics."""

    def test_basic_stats(self):
        from landmarkdiff.fid import _compute_statistics

        features = np.random.default_rng(42).random((10, 64)).astype(np.float32)
        mu, sigma = _compute_statistics(features)
        assert mu.shape == (64,)
        assert sigma.shape == (64, 64)

    def test_too_few_samples(self):
        from landmarkdiff.fid import _compute_statistics

        features = np.random.default_rng(42).random((1, 64)).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            _compute_statistics(features)


class TestCalculateFID:
    """Tests for _calculate_fid."""

    def test_identical_distributions(self):
        from landmarkdiff.fid import _calculate_fid

        rng = np.random.default_rng(42)
        features = rng.random((50, 16)).astype(np.float64)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        fid = _calculate_fid(mu, sigma, mu, sigma)
        assert fid >= 0.0
        assert fid < 1.0  # Should be near zero

    def test_different_distributions(self):
        from landmarkdiff.fid import _calculate_fid

        rng = np.random.default_rng(42)
        f1 = rng.random((50, 16)).astype(np.float64)
        f2 = rng.random((50, 16)).astype(np.float64) + 5.0
        mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
        mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
        fid = _calculate_fid(mu1, sigma1, mu2, sigma2)
        assert fid > 0.0


class TestImagenetNormalize:
    """Tests for _imagenet_normalize."""

    def test_output_shape(self):
        from landmarkdiff.fid import _imagenet_normalize

        t = torch.rand(3, 32, 32)
        result = _imagenet_normalize(t)
        assert result.shape == (3, 32, 32)


class TestNumpyArrayDataset:
    """Tests for NumpyArrayDataset."""

    def test_length(self):
        from landmarkdiff.fid import NumpyArrayDataset

        images = [np.full((64, 64, 3), i * 10, dtype=np.uint8) for i in range(5)]
        ds = NumpyArrayDataset(images, image_size=32)
        assert len(ds) == 5

    def test_getitem(self):
        from landmarkdiff.fid import NumpyArrayDataset

        images = [np.full((64, 64, 3), 128, dtype=np.uint8)]
        ds = NumpyArrayDataset(images, image_size=32)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 32, 32)

    def test_grayscale_conversion(self):
        from landmarkdiff.fid import NumpyArrayDataset

        images = [np.full((64, 64), 128, dtype=np.uint8)]
        ds = NumpyArrayDataset(images, image_size=32)
        item = ds[0]
        assert item.shape == (3, 32, 32)


class TestImageFolderDataset:
    """Tests for ImageFolderDataset."""

    def test_length(self, tmp_path):
        import cv2

        from landmarkdiff.fid import ImageFolderDataset

        for i in range(3):
            img = np.full((32, 32, 3), i * 50, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        ds = ImageFolderDataset(str(tmp_path), image_size=16)
        assert len(ds) == 3

    def test_getitem(self, tmp_path):
        import cv2

        from landmarkdiff.fid import ImageFolderDataset

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "test.png"), img)
        ds = ImageFolderDataset(str(tmp_path), image_size=16)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 16, 16)

    def test_empty_dir(self, tmp_path):
        from landmarkdiff.fid import ImageFolderDataset

        ds = ImageFolderDataset(str(tmp_path))
        assert len(ds) == 0
