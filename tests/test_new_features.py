"""Tests for new features: ArcFace loss integration, displacement model,
validation callback, batch inference utils."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.losses import CombinedLoss
from landmarkdiff.manipulation import apply_procedure_preset


def _make_fake_face() -> FaceLandmarks:
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=0.9)


class TestDisplacementModel:
    """Test data-driven displacement model."""

    def test_import(self):
        from landmarkdiff.displacement_model import DisplacementModel

        model = DisplacementModel()
        assert not model._fitted

    def test_fit_and_generate(self):
        from landmarkdiff.displacement_model import DisplacementModel

        rng = np.random.default_rng(42)
        model = DisplacementModel()

        # Create synthetic displacement data
        displacements = []
        for _ in range(10):
            displacements.append(
                {
                    "procedure": "rhinoplasty",
                    "displacements": rng.normal(0, 0.01, (478, 2)).astype(np.float32),
                    "quality_score": 0.8,
                }
            )
        for _ in range(8):
            displacements.append(
                {
                    "procedure": "blepharoplasty",
                    "displacements": rng.normal(0, 0.005, (478, 2)).astype(np.float32),
                    "quality_score": 0.7,
                }
            )

        model.fit(displacements)
        assert model._fitted
        assert "rhinoplasty" in model.procedures
        assert "blepharoplasty" in model.procedures
        assert model.n_samples["rhinoplasty"] == 10

        # Generate displacement field
        field = model.get_displacement_field("rhinoplasty", intensity=1.0)
        assert field.shape == (478, 2)
        assert field.dtype == np.float32

        # Intensity scaling
        field_half = model.get_displacement_field("rhinoplasty", intensity=0.5)
        np.testing.assert_allclose(field_half, field * 0.5, atol=1e-6)

    def test_save_load(self):
        from landmarkdiff.displacement_model import DisplacementModel

        rng = np.random.default_rng(42)
        model = DisplacementModel()
        displacements = [
            {
                "procedure": "rhinoplasty",
                "displacements": rng.normal(0, 0.01, (478, 2)).astype(np.float32),
                "quality_score": 0.8,
            }
            for _ in range(5)
        ]
        model.fit(displacements)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            model.save(f.name)
            loaded = DisplacementModel.load(f.name)

        assert loaded._fitted
        assert "rhinoplasty" in loaded.procedures
        field_orig = model.get_displacement_field("rhinoplasty")
        field_loaded = loaded.get_displacement_field("rhinoplasty")
        np.testing.assert_allclose(field_orig, field_loaded, atol=1e-6)

    def test_not_fitted_raises(self):
        from landmarkdiff.displacement_model import DisplacementModel

        model = DisplacementModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_displacement_field("rhinoplasty")


class TestDataDrivenManipulation:
    """Test data-driven displacement integration in manipulation.py."""

    def test_apply_with_displacement_model(self):
        from landmarkdiff.displacement_model import DisplacementModel

        rng = np.random.default_rng(42)
        model = DisplacementModel()
        displacements = [
            {
                "procedure": "rhinoplasty",
                "displacements": rng.normal(0, 0.01, (478, 2)).astype(np.float32),
                "quality_score": 0.8,
            }
            for _ in range(5)
        ]
        model.fit(displacements)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            model.save(f.name)
            model_path = f.name

        face = _make_fake_face()
        result = apply_procedure_preset(
            face,
            "rhinoplasty",
            intensity=50.0,
            displacement_model_path=model_path,
        )
        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == face.landmarks.shape
        # Should be different from original
        assert not np.array_equal(result.landmarks, face.landmarks)


class TestArcFaceTorch:
    """Test PyTorch-native ArcFace module."""

    def test_import(self):
        from landmarkdiff.arcface_torch import ArcFaceBackbone, ArcFaceLoss

        assert ArcFaceLoss is not None
        assert ArcFaceBackbone is not None

    def test_backbone_forward(self):
        from landmarkdiff.arcface_torch import ArcFaceBackbone

        backbone = ArcFaceBackbone()
        x = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            emb = backbone(x)
        assert emb.shape == (2, 512)
        # Should be L2-normalized
        norms = torch.norm(emb, dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-4, rtol=1e-4)

    def test_loss_forward(self):
        import warnings

        from landmarkdiff.arcface_torch import ArcFaceLoss

        warnings.filterwarnings("ignore")
        loss_fn = ArcFaceLoss()
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        loss = loss_fn(pred, target, procedure="rhinoplasty")
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_orthognathic_zero(self):
        import warnings

        from landmarkdiff.arcface_torch import ArcFaceLoss

        warnings.filterwarnings("ignore")
        loss_fn = ArcFaceLoss()
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        loss = loss_fn(pred, target, procedure="orthognathic")
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """The key test: gradients must flow through pred_image."""
        import warnings

        from landmarkdiff.arcface_torch import ArcFaceLoss

        warnings.filterwarnings("ignore")
        loss_fn = ArcFaceLoss()
        pred = torch.rand(1, 3, 112, 112, requires_grad=True)
        target = torch.rand(1, 3, 112, 112)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() > 0


class TestCombinedLossIntegration:
    """Test CombinedLoss with differentiable ArcFace."""

    def test_phase_a_default(self):
        combined = CombinedLoss(phase="A")
        # Should use IdentityLoss (ONNX), not ArcFaceLoss
        from landmarkdiff.losses import IdentityLoss

        assert isinstance(combined.identity_loss, IdentityLoss)

    def test_phase_b_differentiable(self):
        combined = CombinedLoss(phase="B", use_differentiable_arcface=True)
        from landmarkdiff.arcface_torch import ArcFaceLoss

        assert isinstance(combined.identity_loss, ArcFaceLoss)

    def test_phase_b_backward_compat(self):
        """Phase B without flag should still work (uses ONNX)."""
        combined = CombinedLoss(phase="B")
        from landmarkdiff.losses import IdentityLoss

        assert isinstance(combined.identity_loss, IdentityLoss)


class TestValidationCallback:
    """Test validation callback module."""

    def test_import(self):
        from landmarkdiff.validation import ValidationCallback

        assert ValidationCallback is not None


class TestExportResults:
    """Test results export utilities."""

    def test_load_and_format(self):
        """Test that export_results handles a mock report."""
        report = {
            "total_processed": 10,
            "results": [
                {
                    "image": f"test_{i}",
                    "procedure": "rhinoplasty",
                    "intensity": 65.0,
                    "fitzpatrick": "III",
                    "ssim": 0.95 + np.random.default_rng(i).uniform(-0.02, 0.02),
                    "lpips": 0.05 + np.random.default_rng(i).uniform(-0.01, 0.01),
                    "nme": 0.002,
                    "identity_check": {},
                    "elapsed_seconds": 5.0,
                    "neural_postprocess": False,
                }
                for i in range(10)
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            report_path = f.name

        # Import and test
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from export_results import (
            compute_fitzpatrick_metrics,
            compute_per_procedure_metrics,
            load_report,
        )

        loaded = load_report(report_path)
        assert loaded["total_processed"] == 10

        proc_metrics = compute_per_procedure_metrics(loaded)
        assert "rhinoplasty" in proc_metrics
        assert proc_metrics["rhinoplasty"]["n"] == 10
        assert 0.93 < proc_metrics["rhinoplasty"]["ssim_mean"] < 0.97

        fitz_metrics = compute_fitzpatrick_metrics(loaded)
        assert "III" in fitz_metrics
        assert fitz_metrics["III"]["n"] == 10
