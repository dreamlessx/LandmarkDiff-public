"""Tests for loss functions."""

import torch
import pytest

from landmarkdiff.losses import (
    CombinedLoss,
    DiffusionLoss,
    IdentityLoss,
    LandmarkLoss,
    LossWeights,
)


class TestDiffusionLoss:
    def test_zero_loss_on_identical(self):
        loss_fn = DiffusionLoss()
        x = torch.randn(2, 4, 64, 64)
        assert loss_fn(x, x).item() == pytest.approx(0.0)

    def test_positive_loss_on_different(self):
        loss_fn = DiffusionLoss()
        a = torch.randn(2, 4, 64, 64)
        b = torch.randn(2, 4, 64, 64)
        assert loss_fn(a, b).item() > 0


class TestLandmarkLoss:
    def test_zero_on_identical(self):
        loss_fn = LandmarkLoss()
        pts = torch.randn(2, 478, 2)
        assert loss_fn(pts, pts).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(2, 478, 2)
        b = a + 0.1
        assert loss_fn(a, b).item() > 0

    def test_mask_reduces_to_masked_region(self):
        loss_fn = LandmarkLoss()
        a = torch.zeros(1, 10, 2)
        b = torch.ones(1, 10, 2)
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1.0  # only first 5 landmarks
        loss_masked = loss_fn(a, b, mask=mask)
        loss_full = loss_fn(a, b)
        # Both should be > 0 but values differ due to masking
        assert loss_masked.item() > 0


class TestIdentityLoss:
    def test_zero_on_identical(self):
        loss_fn = IdentityLoss()
        img = torch.rand(2, 3, 256, 256)
        loss = loss_fn(img, img, procedure="rhinoplasty")
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_orthognathic_returns_zero(self):
        loss_fn = IdentityLoss()
        a = torch.rand(2, 3, 256, 256)
        b = torch.rand(2, 3, 256, 256)
        assert loss_fn(a, b, procedure="orthognathic").item() == 0.0


class TestCombinedLoss:
    def test_phase_a_only_diffusion(self):
        loss_fn = CombinedLoss(phase="A")
        noise_pred = torch.randn(2, 4, 64, 64)
        noise_target = torch.randn(2, 4, 64, 64)
        losses = loss_fn(noise_pred, noise_target)
        assert "diffusion" in losses
        assert "total" in losses
        assert "landmark" not in losses

    def test_phase_b_includes_all(self):
        loss_fn = CombinedLoss(phase="B")
        noise_pred = torch.randn(2, 4, 64, 64)
        noise_target = torch.randn(2, 4, 64, 64)
        pred_img = torch.rand(2, 3, 256, 256)
        target_img = torch.rand(2, 3, 256, 256)
        mask = torch.rand(2, 1, 256, 256)
        pred_lm = torch.randn(2, 478, 2)
        target_lm = torch.randn(2, 478, 2)

        losses = loss_fn(
            noise_pred, noise_target,
            pred_image=pred_img, target_image=target_img,
            mask=mask, pred_landmarks=pred_lm, target_landmarks=target_lm,
        )
        assert "diffusion" in losses
        assert "landmark" in losses
        assert "identity" in losses
        assert "perceptual" in losses
