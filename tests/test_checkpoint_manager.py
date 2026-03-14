"""Tests for landmarkdiff.checkpoint_manager."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from landmarkdiff.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    _force_symlink,
    _get_state_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def ckpt_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def manager(ckpt_dir):
    return CheckpointManager(
        output_dir=ckpt_dir,
        keep_best=2,
        keep_latest=3,
        metric="loss",
        lower_is_better=True,
    )


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def ema_model():
    return TinyModel()


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4)


# ---------------------------------------------------------------------------
# CheckpointMetadata tests
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    def test_to_dict(self):
        meta = CheckpointMetadata(step=100, timestamp=1.0, metrics={"loss": 0.5})
        d = meta.to_dict()
        assert d["step"] == 100
        assert d["metrics"]["loss"] == 0.5

    def test_from_dict(self):
        d = {"step": 200, "timestamp": 2.0, "metrics": {"fid": 42.0}, "phase": "A"}
        meta = CheckpointMetadata.from_dict(d)
        assert meta.step == 200
        assert meta.phase == "A"

    def test_from_dict_extra_keys(self):
        d = {"step": 1, "timestamp": 0, "unknown_key": "ignored"}
        meta = CheckpointMetadata.from_dict(d)
        assert meta.step == 1

    def test_defaults(self):
        meta = CheckpointMetadata(step=0, timestamp=0)
        assert meta.metrics == {}
        assert meta.epoch is None
        assert meta.is_best is False


# ---------------------------------------------------------------------------
# CheckpointManager basic tests
# ---------------------------------------------------------------------------


class TestCheckpointManagerInit:
    def test_creates_directory(self, ckpt_dir):
        CheckpointManager(output_dir=ckpt_dir)
        assert ckpt_dir.exists()

    def test_loads_existing_index(self, ckpt_dir):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        index = {"checkpoints": {"ckpt-100": {"step": 100, "timestamp": 0, "metrics": {}}}}
        (ckpt_dir / "checkpoint_index.json").write_text(json.dumps(index))
        mgr = CheckpointManager(output_dir=ckpt_dir)
        assert "ckpt-100" in mgr._index["checkpoints"]

    def test_empty_state(self, manager):
        assert manager.list_checkpoints() == []
        assert manager.get_latest_step() == 0
        assert manager.get_best_checkpoint_name() is None
        assert manager.total_size_mb() == 0.0


# ---------------------------------------------------------------------------
# Save and metadata tests
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_creates_directory(self, manager, model, ema_model, optimizer):
        path = manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        assert path.exists()
        assert (path / "training_state.pt").exists()
        assert (path / "metadata.json").exists()

    def test_save_records_metrics(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5, "val_ssim": 0.88},
        )
        ckpts = manager.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0]["metrics"]["loss"] == 0.5
        assert ckpts[0]["metrics"]["val_ssim"] == 0.88

    def test_save_updates_index_file(self, manager, model, ema_model, optimizer):
        manager.save(step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        index_path = manager._index_path()
        assert index_path.exists()
        with open(index_path) as f:
            data = json.load(f)
        assert "checkpoint-100" in data["checkpoints"]

    def test_training_state_loadable(self, manager, model, ema_model, optimizer):
        manager.save(
            step=500,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.1},
        )
        state = torch.load(
            manager.output_dir / "checkpoint-500" / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert state["global_step"] == 500
        assert "controlnet" in state
        assert "ema_controlnet" in state
        assert "optimizer" in state

    def test_save_with_scheduler(self, manager, model, ema_model, optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        state = torch.load(
            manager.output_dir / "checkpoint-100" / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert "scheduler" in state

    def test_save_with_phase(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer, phase="A"
        )
        ckpts = manager.list_checkpoints()
        assert ckpts[0]["phase"] == "A"

    def test_size_tracked(self, manager, model, ema_model, optimizer):
        manager.save(step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        ckpts = manager.list_checkpoints()
        assert ckpts[0]["size_mb"] >= 0  # tiny model may round to 0.0


# ---------------------------------------------------------------------------
# Best checkpoint tracking
# ---------------------------------------------------------------------------


class TestBestTracking:
    def test_single_checkpoint_is_best(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        assert manager.get_best_checkpoint_name() == "checkpoint-100"
        assert manager.get_best_metric_value() == 0.5

    def test_lower_is_better(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        manager.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.3},
        )
        manager.save(
            step=300,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.4},
        )
        assert manager.get_best_checkpoint_name() == "checkpoint-200"
        assert manager.get_best_metric_value() == 0.3

    def test_higher_is_better(self, ckpt_dir, model, ema_model, optimizer):
        mgr = CheckpointManager(
            output_dir=ckpt_dir,
            metric="val_ssim",
            lower_is_better=False,
            keep_best=2,
            keep_latest=3,
        )
        mgr.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"val_ssim": 0.80},
        )
        mgr.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"val_ssim": 0.92},
        )
        mgr.save(
            step=300,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"val_ssim": 0.85},
        )
        assert mgr.get_best_checkpoint_name() == "checkpoint-200"

    def test_no_metric_returns_none(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer, metrics={}
        )
        assert manager.get_best_checkpoint_name() is None
        assert manager.get_best_metric_value() is None


# ---------------------------------------------------------------------------
# Symlink tests
# ---------------------------------------------------------------------------


class TestSymlinks:
    def test_latest_symlink(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        latest = manager.output_dir / "latest"
        assert latest.is_symlink()
        assert latest.resolve().name == "checkpoint-100"

    def test_latest_updates(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        manager.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.3},
        )
        latest = manager.output_dir / "latest"
        assert latest.resolve().name == "checkpoint-200"

    def test_best_symlink(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        manager.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.3},
        )
        best = manager.output_dir / "best"
        assert best.is_symlink()
        assert best.resolve().name == "checkpoint-200"


# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------


class TestPruning:
    def test_prunes_old_checkpoints(self, ckpt_dir, model, ema_model, optimizer):
        mgr = CheckpointManager(
            output_dir=ckpt_dir,
            keep_best=1,
            keep_latest=2,
            metric="loss",
        )
        # Save 4 checkpoints, keep_latest=2 + keep_best=1
        mgr.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        mgr.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.1},
        )  # best
        mgr.save(
            step=300,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.4},
        )
        mgr.save(
            step=400,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.3},
        )

        # Should keep: step 200 (best), step 300 + 400 (latest 2)
        # Should prune: step 100
        ckpts = mgr.list_checkpoints()
        steps = [c["step"] for c in ckpts]
        assert 100 not in steps
        assert 200 in steps  # best
        assert 300 in steps  # latest
        assert 400 in steps  # latest
        assert not (ckpt_dir / "checkpoint-100").exists()

    def test_no_pruning_when_under_limit(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        manager.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.3},
        )
        assert len(manager.list_checkpoints()) == 2

    def test_best_preserved_over_latest(self, ckpt_dir, model, ema_model, optimizer):
        """Best checkpoint survives even if not in latest N."""
        mgr = CheckpointManager(
            output_dir=ckpt_dir,
            keep_best=1,
            keep_latest=1,
            metric="loss",
        )
        mgr.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.01},
        )  # best
        mgr.save(
            step=200,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        mgr.save(
            step=300,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.4},
        )

        ckpts = mgr.list_checkpoints()
        steps = [c["step"] for c in ckpts]
        assert 100 in steps  # best, preserved
        assert 300 in steps  # latest
        assert 200 not in steps  # pruned


# ---------------------------------------------------------------------------
# Summary and queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_summary_empty(self, manager):
        assert "No checkpoints" in manager.summary()

    def test_summary_with_checkpoints(self, manager, model, ema_model, optimizer):
        manager.save(
            step=100,
            controlnet=model,
            ema_controlnet=ema_model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )
        s = manager.summary()
        assert "1 saved" in s
        assert "step 100" in s

    def test_get_latest_step(self, manager, model, ema_model, optimizer):
        manager.save(step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        manager.save(step=200, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        assert manager.get_latest_step() == 200

    def test_get_checkpoint_path(self, manager, model, ema_model, optimizer):
        manager.save(step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        path = manager.get_checkpoint_path("checkpoint-100")
        assert path.exists()

    def test_total_size(self, manager, model, ema_model, optimizer):
        manager.save(step=100, controlnet=model, ema_controlnet=ema_model, optimizer=optimizer)
        # Size is at least 0 (tiny test model may round to 0.0)
        assert manager.total_size_mb() >= 0
        # But checkpoint files should exist on disk
        assert (manager.output_dir / "checkpoint-100" / "training_state.pt").exists()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_state_dict_plain(self):
        m = TinyModel()
        sd = _get_state_dict(m)
        assert "linear.weight" in sd

    def test_get_state_dict_wrapped(self):
        m = TinyModel()

        class Wrapper:
            def __init__(self, module):
                self.module = module

        w = Wrapper(m)
        sd = _get_state_dict(w)
        assert "linear.weight" in sd

    def test_force_symlink_new(self, tmp_path):
        target = tmp_path / "target_dir"
        target.mkdir()
        link = tmp_path / "link"
        _force_symlink(target, link)
        assert link.is_symlink()

    def test_force_symlink_replace(self, tmp_path):
        target1 = tmp_path / "t1"
        target1.mkdir()
        target2 = tmp_path / "t2"
        target2.mkdir()
        link = tmp_path / "link"
        _force_symlink(target1, link)
        _force_symlink(target2, link)
        assert link.resolve().name == "t2"
