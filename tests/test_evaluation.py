"""Tests for evaluation metrics."""

import numpy as np

from landmarkdiff.evaluation import compute_nme, compute_ssim


class TestNME:
    def test_zero_on_identical(self):
        pts = np.random.default_rng(0).uniform(100, 400, (478, 2)).astype(np.float32)
        assert compute_nme(pts, pts) == 0.0

    def test_positive_on_different(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(100, 400, (478, 2)).astype(np.float32)
        b = a + 5.0
        assert compute_nme(a, b) > 0


class TestSSIM:
    def test_perfect_ssim(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert abs(compute_ssim(img, img) - 1.0) < 1e-6

    def test_low_ssim_on_different(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.full((64, 64, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(a, b)
        assert ssim < 0.1
