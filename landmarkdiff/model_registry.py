"""Model registry for checkpoint discovery and management.

Provides a unified interface for finding, loading, and comparing model
checkpoints across local directories and remote sources.

Usage:
    from landmarkdiff.model_registry import ModelRegistry

    registry = ModelRegistry("checkpoints/")

    # Discover all checkpoints
    models = registry.list_models()

    # Get best checkpoint by metric
    best = registry.get_best("loss")

    # Load a specific checkpoint
    state = registry.load("checkpoint-5000")

    # Compare multiple checkpoints
    comparison = registry.compare(["checkpoint-1000", "checkpoint-5000"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ModelEntry:
    """Metadata for a registered model checkpoint."""

    name: str
    path: Path
    step: int = 0
    phase: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    size_mb: float = 0.0
    has_ema: bool = False
    has_training_state: bool = False

    @property
    def inference_path(self) -> Path | None:
        """Path to inference-ready weights (EMA preferred)."""
        ema_dir = self.path / "controlnet_ema"
        if ema_dir.exists():
            return ema_dir
        # Fallback to training state
        state_path = self.path / "training_state.pt"
        if state_path.exists():
            return state_path
        return None


class ModelRegistry:
    """Central registry for discovering and managing model checkpoints.

    Args:
        checkpoint_dirs: One or more directories to scan for checkpoints.
        scan_on_init: Whether to scan directories immediately on creation.
    """

    def __init__(
        self,
        *checkpoint_dirs: str | Path,
        scan_on_init: bool = True,
    ) -> None:
        self.checkpoint_dirs = [Path(d) for d in checkpoint_dirs]
        self._models: dict[str, ModelEntry] = {}

        if scan_on_init:
            self.scan()

    def scan(self) -> int:
        """Scan checkpoint directories and register all found models.

        Returns:
            Number of models found.
        """
        self._models.clear()
        for base_dir in self.checkpoint_dirs:
            if not base_dir.exists():
                continue
            self._scan_directory(base_dir)
        return len(self._models)

    def _scan_directory(self, base_dir: Path) -> None:
        """Scan a single directory for checkpoint subdirectories."""
        # Look for checkpoint-* directories
        for ckpt_dir in sorted(base_dir.glob("checkpoint-*")):
            if not ckpt_dir.is_dir():
                continue
            entry = self._load_entry(ckpt_dir)
            if entry is not None:
                self._models[entry.name] = entry

        # Also check for "final" and "best" directories/symlinks
        for special in ["final", "best", "latest"]:
            special_dir = base_dir / special
            if special_dir.exists() and special_dir.is_dir():
                entry = self._load_entry(special_dir)
                if entry is not None:
                    entry.name = f"{base_dir.name}/{special}"
                    self._models[entry.name] = entry

    def _load_entry(self, ckpt_dir: Path) -> ModelEntry | None:
        """Load metadata for a single checkpoint directory."""
        has_training = (ckpt_dir / "training_state.pt").exists()
        has_ema = (ckpt_dir / "controlnet_ema").exists()

        if not has_training and not has_ema:
            return None

        # Try to load metadata.json (from CheckpointManager)
        meta_path = ckpt_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return ModelEntry(
                name=ckpt_dir.name,
                path=ckpt_dir,
                step=meta.get("step", 0),
                phase=meta.get("phase", ""),
                metrics=meta.get("metrics", {}),
                size_mb=meta.get("size_mb", 0.0),
                has_ema=has_ema,
                has_training_state=has_training,
            )

        # Fallback: extract step from directory name
        step = 0
        parts = ckpt_dir.name.split("-")
        if len(parts) >= 2 and parts[-1].isdigit():
            step = int(parts[-1])

        # Compute size
        size_mb = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / (1024 * 1024)

        return ModelEntry(
            name=ckpt_dir.name,
            path=ckpt_dir,
            step=step,
            size_mb=round(size_mb, 1),
            has_ema=has_ema,
            has_training_state=has_training,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_models(self, sort_by: str = "step") -> list[ModelEntry]:
        """List all registered models.

        Args:
            sort_by: Sort key — "step", "name", or a metric name.

        Returns:
            Sorted list of ModelEntry objects.
        """
        models = list(self._models.values())
        if sort_by == "step":
            models.sort(key=lambda m: m.step)
        elif sort_by == "name":
            models.sort(key=lambda m: m.name)
        else:
            # Sort by metric value
            models.sort(
                key=lambda m: m.metrics.get(sort_by, float("inf")),
            )
        return models

    def get(self, name: str) -> ModelEntry | None:
        """Get a model entry by name."""
        return self._models.get(name)

    def get_best(
        self,
        metric: str = "loss",
        lower_is_better: bool = True,
    ) -> ModelEntry | None:
        """Get the best model by a specific metric.

        Args:
            metric: Metric name to rank by.
            lower_is_better: If True, lower values are better.

        Returns:
            Best ModelEntry, or None if no models have the metric.
        """
        candidates = [m for m in self._models.values() if metric in m.metrics]
        if not candidates:
            return None

        return (
            min(candidates, key=lambda m: m.metrics[metric])
            if lower_is_better
            else max(candidates, key=lambda m: m.metrics[metric])
        )

    def get_by_step(self, step: int) -> ModelEntry | None:
        """Get a model by its training step."""
        for model in self._models.values():
            if model.step == step:
                return model
        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load training state from a checkpoint.

        Args:
            name: Checkpoint name (e.g. "checkpoint-5000").
            map_location: Device to load tensors to.

        Returns:
            State dict containing controlnet, ema_controlnet, optimizer, etc.

        Raises:
            KeyError: If checkpoint not found.
            FileNotFoundError: If training_state.pt missing.
        """
        entry = self._models.get(name)
        if entry is None:
            raise KeyError(f"Checkpoint '{name}' not found in registry")

        state_path = entry.path / "training_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"No training_state.pt in {entry.path}")

        return torch.load(state_path, map_location=map_location, weights_only=True)

    def load_controlnet(
        self,
        name: str,
        use_ema: bool = True,
    ) -> Any:
        """Load a ControlNet model from checkpoint.

        Args:
            name: Checkpoint name.
            use_ema: If True, load EMA weights (preferred for inference).

        Returns:
            ControlNetModel instance.
        """
        from diffusers import ControlNetModel

        entry = self._models.get(name)
        if entry is None:
            raise KeyError(f"Checkpoint '{name}' not found in registry")

        if use_ema and entry.has_ema:
            return ControlNetModel.from_pretrained(str(entry.path / "controlnet_ema"))

        # Fallback: load from training state
        state = self.load(name)
        model = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            subfolder="diffusion_sd15",
        )
        key = "ema_controlnet" if use_ema else "controlnet"
        model.load_state_dict(state[key])
        return model

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        names: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple checkpoints side-by-side.

        Args:
            names: List of checkpoint names to compare.
            metrics: Specific metrics to include. None = all available.

        Returns:
            Dict with comparison table data.
        """
        entries = []
        for name in names:
            entry = self._models.get(name)
            if entry is not None:
                entries.append(entry)

        if not entries:
            return {"error": "No valid checkpoints found"}

        # Collect all available metrics
        if metrics is None:
            all_metrics: set[str] = set()
            for e in entries:
                all_metrics.update(e.metrics.keys())
            metrics = sorted(all_metrics)

        rows = []
        for e in entries:
            row = {
                "name": e.name,
                "step": e.step,
                "phase": e.phase,
                "size_mb": e.size_mb,
            }
            for m in metrics:
                row[m] = e.metrics.get(m)
            rows.append(row)

        return {
            "metrics": metrics,
            "rows": rows,
            "count": len(rows),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary."""
        models = self.list_models()
        if not models:
            return "No models registered."

        total_size = sum(m.size_mb for m in models)
        lines = [
            f"Model Registry: {len(models)} checkpoints ({total_size:.0f} MB)",
            f"  Steps: {models[0].step} — {models[-1].step}",
        ]

        # Show metrics ranges
        all_metrics: set[str] = set()
        for m in models:
            all_metrics.update(m.metrics.keys())

        for metric in sorted(all_metrics):
            values = [m.metrics[metric] for m in models if metric in m.metrics]
            if values:
                lines.append(f"  {metric}: {min(values):.4f} — {max(values):.4f}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
