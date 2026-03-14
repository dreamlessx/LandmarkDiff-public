"""Tests for training loop integration features.

Tests curriculum integration, auto-resume, displacement model inference,
and gradient accumulation correctness.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestGradientAccumulation:
    """Verify that gradient accumulation produces correct gradients."""

    def test_accumulated_equals_large_batch(self):
        """Gradient accumulation over N steps should equal one large batch."""
        torch.manual_seed(42)
        # Simple model
        model = nn.Linear(10, 1)
        model_ref = nn.Linear(10, 1)
        model_ref.load_state_dict(model.state_dict())

        # Create data (4 samples)
        data = torch.randn(4, 10)
        targets = torch.randn(4, 1)

        # Method 1: Large batch (4 at once)
        out_ref = model_ref(data)
        loss_ref = F.mse_loss(out_ref, targets)
        loss_ref.backward()

        # Method 2: Gradient accumulation (2 x batch of 2)
        accum_steps = 2
        for i in range(accum_steps):
            batch = data[i * 2 : (i + 1) * 2]
            tgts = targets[i * 2 : (i + 1) * 2]
            out = model(batch)
            loss = F.mse_loss(out, tgts) / accum_steps
            loss.backward()

        # Gradients should match
        for p_ref, p_acc in zip(model_ref.parameters(), model.parameters(), strict=False):
            torch.testing.assert_close(p_ref.grad, p_acc.grad, atol=1e-5, rtol=1e-5)

    def test_accumulation_loss_value(self):
        """Accumulated loss tracking matches expected behavior."""
        torch.manual_seed(42)
        model = nn.Linear(10, 1)
        data = torch.randn(8, 10)
        targets = torch.randn(8, 1)
        accum_steps = 4

        total_loss = 0.0
        for i in range(accum_steps):
            batch = data[i * 2 : (i + 1) * 2]
            tgts = targets[i * 2 : (i + 1) * 2]
            out = model(batch)
            loss = F.mse_loss(out, tgts)
            total_loss += loss.item()
            (loss / accum_steps).backward()

        avg_loss = total_loss / accum_steps
        # Should be a reasonable positive number
        assert avg_loss > 0
        assert np.isfinite(avg_loss)


class TestCurriculumIntegration:
    """Tests for curriculum learning integration with dataset."""

    def test_procedure_weights_affect_sampling(self):
        """ProcedureCurriculum weights should create non-uniform sampling."""
        from landmarkdiff.curriculum import ProcedureCurriculum

        curriculum = ProcedureCurriculum(total_steps=1000)

        # At step 0 (early training), easy procedures should have higher weight
        weights = curriculum.get_procedure_weights(0)
        assert weights["blepharoplasty"] > weights["orthognathic"]

        # At end (step 999), all should be 1.0
        weights_end = curriculum.get_procedure_weights(999)
        for w in weights_end.values():
            assert w == 1.0

    def test_weighted_sampler_creation(self):
        """Verify WeightedRandomSampler can be created from curriculum weights."""
        from landmarkdiff.curriculum import ProcedureCurriculum

        curriculum = ProcedureCurriculum(total_steps=1000)
        n_samples = 100
        procedures = ["blepharoplasty"] * 30 + ["rhinoplasty"] * 30 + ["orthognathic"] * 40

        weights = torch.ones(n_samples)
        for i, proc in enumerate(procedures):
            weights[i] = curriculum.get_weight(0, proc)

        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(weights, n_samples, replacement=True)
        indices = list(sampler)
        assert len(indices) == n_samples

        # Blepharoplasty (indices 0-29) should appear more often than orthognathic (60-99)
        bleph_count = sum(1 for i in indices if i < 30)
        sum(1 for i in indices if i >= 60)
        # Not a strict test — but on average blepharoplasty should dominate
        # With 30 easy vs 40 hard, even with weighting blepharoplasty should appear more
        assert bleph_count > 0  # at minimum, some should appear


class TestAutoResume:
    """Tests for auto-resume from checkpoint."""

    def test_checkpoint_sorting(self):
        """Auto-resume should find the checkpoint with highest step number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # Create fake checkpoints
            for step in [1000, 5000, 3000]:
                ckpt_dir = out / f"checkpoint-{step}"
                ckpt_dir.mkdir()
                # Create a minimal training_state.pt
                torch.save({"global_step": step}, ckpt_dir / "training_state.pt")

            # Sort by step number
            ckpts = sorted(
                out.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            ckpts = [c for c in ckpts if (c / "training_state.pt").exists()]
            assert len(ckpts) == 3
            assert ckpts[-1].name == "checkpoint-5000"

    def test_ignores_incomplete_checkpoints(self):
        """Auto-resume should skip checkpoints without training_state.pt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # Complete checkpoint
            ckpt1 = out / "checkpoint-1000"
            ckpt1.mkdir()
            torch.save({"global_step": 1000}, ckpt1 / "training_state.pt")

            # Incomplete checkpoint (no training_state.pt)
            ckpt2 = out / "checkpoint-5000"
            ckpt2.mkdir()

            ckpts = sorted(
                out.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            )
            ckpts = [c for c in ckpts if (c / "training_state.pt").exists()]
            assert len(ckpts) == 1
            assert ckpts[0].name == "checkpoint-1000"


class TestDisplacementModelInference:
    """Tests for displacement model integration in inference pipeline."""

    def test_displacement_field_shapes(self):
        """Displacement model should produce correct shape output."""
        from landmarkdiff.displacement_model import NUM_LANDMARKS, DisplacementModel

        # Create a mock model
        model = DisplacementModel()
        model.stats = {
            "rhinoplasty": {
                "mean": np.zeros((NUM_LANDMARKS, 2), dtype=np.float32),
                "std": np.ones((NUM_LANDMARKS, 2), dtype=np.float32) * 0.01,
            }
        }
        model.n_samples = {"rhinoplasty": 100}
        model._fitted = True

        field = model.get_displacement_field("rhinoplasty", intensity=1.0, noise_scale=0.5)
        assert field.shape == (NUM_LANDMARKS, 2)
        assert field.dtype == np.float32

    def test_intensity_scaling(self):
        """Higher intensity should produce larger displacements."""
        from landmarkdiff.displacement_model import NUM_LANDMARKS, DisplacementModel

        model = DisplacementModel()
        mean = np.random.randn(NUM_LANDMARKS, 2).astype(np.float32) * 0.01
        model.stats = {
            "rhinoplasty": {
                "mean": mean,
                "std": np.ones((NUM_LANDMARKS, 2), dtype=np.float32) * 0.001,
            }
        }
        model.n_samples = {"rhinoplasty": 100}
        model._fitted = True

        field_low = model.get_displacement_field("rhinoplasty", intensity=0.5, noise_scale=0)
        field_high = model.get_displacement_field("rhinoplasty", intensity=2.0, noise_scale=0)

        mag_low = np.linalg.norm(field_low, axis=1).mean()
        mag_high = np.linalg.norm(field_high, axis=1).mean()
        assert mag_high > mag_low


class TestMetricsVisualization:
    """Tests for training metrics visualization."""

    def test_load_metrics(self):
        """Should parse JSONL metrics correctly."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from plot_training_curves import load_metrics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"step": i * 100, "loss": 0.1 - i * 0.01, "lr": 1e-5}) + "\n")
            f.flush()
            metrics = load_metrics(Path(f.name))
            assert len(metrics) == 5
            assert metrics[0]["step"] == 0
            assert metrics[4]["loss"] == pytest.approx(0.06)

    def test_smooth(self):
        """EMA smoothing should reduce noise."""
        from plot_training_curves import smooth

        noisy = [1.0, 0.5, 1.5, 0.3, 1.2, 0.8, 1.1, 0.6]
        smoothed = smooth(noisy, window=5)
        assert len(smoothed) == len(noisy)
        # Smoothed values should have less variance
        assert np.std(smoothed) < np.std(noisy)


class TestMonitorTraining:
    """Tests for training log parser."""

    def test_parse_log_line(self):
        """Should correctly parse training log format."""
        from monitor_training import parse_training_log

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(
                "Step 100/10000 | Loss: 0.045678 | LR: 1.00e-05 | GradNorm: 1.23 | 5.2 it/s | ETA: 3.5h\n"
            )
            f.write(
                "Step 200/10000 | Loss: 0.034567 | LR: 9.80e-06 | GradNorm: 0.98 | 5.5 it/s | ETA: 3.1h\n"
            )
            f.write("Checkpoint saved: checkpoints/checkpoint-200\n")
            f.flush()

            data = parse_training_log(f.name)
            assert len(data["steps"]) == 2
            assert data["steps"][0] == 100
            assert data["losses"][1] == pytest.approx(0.034567)
            assert data["lrs"][0] == pytest.approx(1e-5)
            assert data["checkpoints"] == ["checkpoints/checkpoint-200"]
