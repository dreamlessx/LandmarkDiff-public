"""Tests for the PyTorch-native ArcFace module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.arcface_torch import (
    ArcFaceBackbone,
    ArcFaceLoss,
    IBasicBlock,
    align_face,
    align_face_no_crop,
)


class TestIBasicBlock:
    """Tests for the IResNet basic block."""

    def test_output_shape_no_downsample(self):
        block = IBasicBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 14, 14)
        out = block(x)
        assert out.shape == (2, 64, 14, 14)

    def test_output_shape_with_downsample(self):
        downsample = torch.nn.Conv2d(64, 128, 1, stride=2, bias=True)
        block = IBasicBlock(64, 128, stride=2, downsample=downsample)
        x = torch.randn(2, 64, 14, 14)
        out = block(x)
        assert out.shape == (2, 128, 7, 7)

    def test_residual_connection(self):
        block = IBasicBlock(64, 64, stride=1)
        x = torch.randn(1, 64, 7, 7)
        out = block(x)
        # Output should differ from input due to learned transform
        assert out.shape == x.shape


class TestArcFaceBackbone:
    """Tests for the IResNet-50 backbone."""

    @pytest.fixture
    def backbone(self):
        return ArcFaceBackbone()

    def test_output_shape(self, backbone):
        x = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (2, 512)

    def test_output_l2_normalized(self, backbone):
        x = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            out = backbone(x)
        norms = torch.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=1e-4)

    def test_deterministic_forward(self, backbone):
        backbone.eval()
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self, backbone):
        backbone.eval()
        x = torch.randn(1, 3, 112, 112, requires_grad=True)
        out = backbone(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_single_image(self, backbone):
        backbone.eval()
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (1, 512)

    def test_custom_embedding_dim(self):
        backbone = ArcFaceBackbone(embedding_dim=256)
        backbone.eval()  # BatchNorm1d needs batch>1 in training mode
        x = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (2, 256)


class TestAlignFace:
    """Tests for face alignment utilities."""

    def test_align_face_output_shape(self):
        images = torch.randn(2, 3, 512, 512)
        aligned = align_face(images, size=112)
        assert aligned.shape == (2, 3, 112, 112)

    def test_align_face_noop_if_correct_size(self):
        images = torch.randn(2, 3, 112, 112)
        aligned = align_face(images, size=112)
        torch.testing.assert_close(aligned, images)

    def test_align_face_differentiable(self):
        images = torch.randn(1, 3, 256, 256, requires_grad=True)
        aligned = align_face(images, size=112)
        aligned.sum().backward()
        assert images.grad is not None

    def test_align_face_no_crop_shape(self):
        images = torch.randn(2, 3, 512, 512)
        aligned = align_face_no_crop(images, size=112)
        assert aligned.shape == (2, 3, 112, 112)

    def test_align_face_no_crop_noop(self):
        images = torch.randn(1, 3, 112, 112)
        aligned = align_face_no_crop(images, size=112)
        torch.testing.assert_close(aligned, images)


class TestArcFaceLoss:
    """Tests for the ArcFaceLoss module."""

    @pytest.fixture
    def loss_fn(self):
        # Use random weights (no pretrained) for testing
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loss = ArcFaceLoss(device=torch.device("cpu"), crop_face=True)
            # Force init with random weights
            loss._ensure_initialized(torch.device("cpu"))
        return loss

    def test_loss_returns_scalar(self, loss_fn):
        pred = torch.randn(2, 3, 512, 512).clamp(0, 1)
        target = torch.randn(2, 3, 512, 512).clamp(0, 1)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_identical_images_low_loss(self, loss_fn):
        img = torch.randn(1, 3, 512, 512).clamp(0, 1)
        loss = loss_fn(img, img)
        # Same image should have very low identity loss
        assert loss.item() < 0.1

    def test_orthognathic_returns_zero(self, loss_fn):
        pred = torch.randn(1, 3, 512, 512).clamp(0, 1)
        target = torch.randn(1, 3, 512, 512).clamp(0, 1)
        loss = loss_fn(pred, target, procedure="orthognathic")
        assert loss.item() == 0.0

    def test_loss_differentiable(self, loss_fn):
        pred_raw = torch.randn(1, 3, 256, 256, requires_grad=True)
        pred = pred_raw.clamp(0, 1)
        pred.retain_grad()  # clamp makes non-leaf
        target = torch.randn(1, 3, 256, 256).clamp(0, 1)
        loss = loss_fn(pred, target)
        loss.backward()
        # Gradient flows to the raw input
        assert pred_raw.grad is not None

    def test_procedure_crop_rhinoplasty(self, loss_fn):
        img = torch.randn(1, 3, 512, 512)
        cropped = loss_fn._procedure_crop(img, "rhinoplasty")
        # Should be top 2/3 of image
        assert cropped.shape == (1, 3, 341, 512)

    def test_procedure_crop_blepharoplasty(self, loss_fn):
        img = torch.randn(1, 3, 512, 512)
        cropped = loss_fn._procedure_crop(img, "blepharoplasty")
        assert cropped.shape == (1, 3, 512, 512)  # Full face

    def test_procedure_crop_rhytidectomy(self, loss_fn):
        img = torch.randn(1, 3, 512, 512)
        cropped = loss_fn._procedure_crop(img, "rhytidectomy")
        assert cropped.shape == (1, 3, 384, 512)  # Top 3/4

    def test_get_embedding(self, loss_fn):
        img = torch.randn(2, 3, 256, 256).clamp(0, 1)
        emb = loss_fn.get_embedding(img)
        assert emb.shape == (2, 512)
        # Should be L2-normalized
        norms = torch.norm(emb, dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=1e-4)
