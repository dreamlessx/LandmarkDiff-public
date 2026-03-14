"""Tests for landmarkdiff.model_registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from landmarkdiff.model_registry import ModelEntry, ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _create_checkpoint(base_dir: Path, step: int, metrics: dict | None = None,
                       phase: str = "", with_ema: bool = False) -> Path:
    """Helper to create a fake checkpoint directory."""
    ckpt_dir = base_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save training state
    model = TinyModel()
    torch.save({
        "controlnet": model.state_dict(),
        "ema_controlnet": model.state_dict(),
        "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
        "global_step": step,
    }, ckpt_dir / "training_state.pt")

    # Save metadata
    meta = {
        "step": step,
        "timestamp": 1.0,
        "metrics": metrics or {},
        "phase": phase,
        "size_mb": 0.1,
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta))

    if with_ema:
        ema_dir = ckpt_dir / "controlnet_ema"
        ema_dir.mkdir()
        (ema_dir / "config.json").write_text("{}")

    return ckpt_dir


@pytest.fixture
def registry_dir(tmp_path):
    """Create a directory with several checkpoints."""
    base = tmp_path / "checkpoints"
    _create_checkpoint(base, 1000, {"loss": 0.5, "val_ssim": 0.80}, phase="A")
    _create_checkpoint(base, 2000, {"loss": 0.3, "val_ssim": 0.85}, phase="A")
    _create_checkpoint(base, 3000, {"loss": 0.2, "val_ssim": 0.90}, phase="A", with_ema=True)
    _create_checkpoint(base, 4000, {"loss": 0.4, "val_ssim": 0.87}, phase="B")
    return base


@pytest.fixture
def registry(registry_dir):
    return ModelRegistry(registry_dir)


# ---------------------------------------------------------------------------
# ModelEntry tests
# ---------------------------------------------------------------------------

class TestModelEntry:
    def test_inference_path_ema(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "controlnet_ema").mkdir()
        entry = ModelEntry(name="ckpt", path=ckpt, has_ema=True)
        assert entry.inference_path == ckpt / "controlnet_ema"

    def test_inference_path_training_state(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "training_state.pt").touch()
        entry = ModelEntry(name="ckpt", path=ckpt, has_training_state=True)
        assert entry.inference_path == ckpt / "training_state.pt"

    def test_inference_path_none(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        entry = ModelEntry(name="ckpt", path=ckpt)
        assert entry.inference_path is None

    def test_defaults(self):
        entry = ModelEntry(name="test", path=Path("/tmp"))
        assert entry.step == 0
        assert entry.metrics == {}
        assert entry.phase == ""


# ---------------------------------------------------------------------------
# Registry scan and discovery
# ---------------------------------------------------------------------------

class TestScan:
    def test_discovers_checkpoints(self, registry):
        assert len(registry) == 4

    def test_scan_returns_count(self, registry_dir):
        reg = ModelRegistry(registry_dir, scan_on_init=False)
        assert len(reg) == 0
        count = reg.scan()
        assert count == 4

    def test_scan_empty_directory(self, tmp_path):
        reg = ModelRegistry(tmp_path / "nonexistent")
        assert len(reg) == 0

    def test_scan_no_metadata(self, tmp_path):
        """Checkpoints without metadata.json should still be discovered."""
        ckpt_dir = tmp_path / "checkpoints" / "checkpoint-500"
        ckpt_dir.mkdir(parents=True)
        model = TinyModel()
        torch.save({"controlnet": model.state_dict(), "global_step": 500},
                    ckpt_dir / "training_state.pt")

        reg = ModelRegistry(tmp_path / "checkpoints")
        assert len(reg) == 1
        entry = reg.get("checkpoint-500")
        assert entry is not None
        assert entry.step == 500

    def test_multiple_directories(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        _create_checkpoint(dir1, 100)
        _create_checkpoint(dir2, 200)
        reg = ModelRegistry(dir1, dir2)
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# List and query
# ---------------------------------------------------------------------------

class TestQueries:
    def test_list_models_sorted_by_step(self, registry):
        models = registry.list_models(sort_by="step")
        steps = [m.step for m in models]
        assert steps == [1000, 2000, 3000, 4000]

    def test_list_models_sorted_by_name(self, registry):
        models = registry.list_models(sort_by="name")
        assert models[0].name <= models[-1].name

    def test_list_models_sorted_by_metric(self, registry):
        models = registry.list_models(sort_by="loss")
        losses = [m.metrics["loss"] for m in models]
        assert losses == sorted(losses)

    def test_get_existing(self, registry):
        entry = registry.get("checkpoint-2000")
        assert entry is not None
        assert entry.step == 2000
        assert entry.metrics["loss"] == 0.3

    def test_get_missing(self, registry):
        assert registry.get("nonexistent") is None

    def test_contains(self, registry):
        assert "checkpoint-1000" in registry
        assert "checkpoint-9999" not in registry

    def test_get_by_step(self, registry):
        entry = registry.get_by_step(3000)
        assert entry is not None
        assert entry.name == "checkpoint-3000"

    def test_get_by_step_missing(self, registry):
        assert registry.get_by_step(9999) is None


# ---------------------------------------------------------------------------
# Best checkpoint
# ---------------------------------------------------------------------------

class TestBest:
    def test_best_lower_is_better(self, registry):
        best = registry.get_best("loss", lower_is_better=True)
        assert best is not None
        assert best.step == 3000
        assert best.metrics["loss"] == 0.2

    def test_best_higher_is_better(self, registry):
        best = registry.get_best("val_ssim", lower_is_better=False)
        assert best is not None
        assert best.step == 3000

    def test_best_no_metric(self, registry):
        best = registry.get_best("nonexistent_metric")
        assert best is None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_training_state(self, registry):
        state = registry.load("checkpoint-1000")
        assert "controlnet" in state
        assert "global_step" in state
        assert state["global_step"] == 1000

    def test_load_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load("nonexistent")

    def test_load_map_location(self, registry):
        state = registry.load("checkpoint-1000", map_location="cpu")
        assert isinstance(state, dict)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_two(self, registry):
        result = registry.compare(["checkpoint-1000", "checkpoint-3000"])
        assert result["count"] == 2
        assert "loss" in result["metrics"]
        assert len(result["rows"]) == 2

    def test_compare_specific_metrics(self, registry):
        result = registry.compare(
            ["checkpoint-1000", "checkpoint-2000"],
            metrics=["loss"],
        )
        assert result["metrics"] == ["loss"]

    def test_compare_empty(self, registry):
        result = registry.compare(["nonexistent1", "nonexistent2"])
        assert "error" in result

    def test_compare_includes_step(self, registry):
        result = registry.compare(["checkpoint-2000"])
        assert result["rows"][0]["step"] == 2000


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_empty(self, tmp_path):
        reg = ModelRegistry(tmp_path / "empty")
        assert "No models" in reg.summary()

    def test_summary_with_models(self, registry):
        s = registry.summary()
        assert "4 checkpoints" in s
        assert "loss" in s

    def test_summary_shows_step_range(self, registry):
        s = registry.summary()
        assert "1000" in s
        assert "4000" in s
