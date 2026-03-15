"""Checkpoint management with metadata tracking, best-model selection, and pruning.

Provides a central manager for training checkpoints that:
- Tracks per-checkpoint metadata (step, metrics, timestamps)
- Maintains symlinks to best/latest checkpoints
- Prunes old checkpoints to save disk space
- Supports multiple ranking metrics (loss, FID, SSIM, etc.)

Usage:
    manager = CheckpointManager(
        output_dir="checkpoints/phaseA",
        keep_best=3,
        keep_latest=5,
        metric="loss",
        lower_is_better=True,
    )

    # During training loop:
    manager.save(
        step=1000,
        controlnet=controlnet,
        ema_controlnet=ema_controlnet,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"loss": 0.0123, "val_ssim": 0.87},
    )
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointMetadata:
    """Metadata for a single checkpoint."""

    step: int
    timestamp: float
    metrics: dict[str, float] = field(default_factory=dict)
    epoch: int | None = None
    phase: str = ""
    is_best: bool = False
    size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """Manages training checkpoints with pruning and best-model tracking.

    Args:
        output_dir: Base directory for checkpoints.
        keep_best: Number of best checkpoints to retain.
        keep_latest: Number of most recent checkpoints to retain.
        metric: Metric name used to determine "best" checkpoint.
        lower_is_better: If True, lower metric values are better (e.g. loss, FID).
        prefix: Checkpoint directory prefix (default: "checkpoint").
    """

    INDEX_FILE = "checkpoint_index.json"

    def __init__(
        self,
        output_dir: str | Path,
        keep_best: int = 3,
        keep_latest: int = 5,
        metric: str = "loss",
        lower_is_better: bool = True,
        prefix: str = "checkpoint",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_latest = keep_latest
        self.metric = metric
        self.lower_is_better = lower_is_better
        self.prefix = prefix

        self._index: dict[str, Any] = {"checkpoints": {}}
        self._load_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.output_dir / self.INDEX_FILE

    def _load_index(self) -> None:
        path = self._index_path()
        if path.exists():
            with open(path) as f:
                self._index = json.load(f)
            if "checkpoints" not in self._index:
                self._index["checkpoints"] = {}

    def _save_index(self) -> None:
        with open(self._index_path(), "w") as f:
            json.dump(self._index, f, indent=2)

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        controlnet: torch.nn.Module,
        ema_controlnet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        metrics: dict[str, float] | None = None,
        epoch: int | None = None,
        phase: str = "",
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint with metadata.

        Args:
            step: Current training step.
            controlnet: ControlNet model (or any nn.Module).
            ema_controlnet: EMA copy of the model.
            optimizer: Optimizer state.
            scheduler: Optional LR scheduler.
            metrics: Dict of metric values at this step.
            epoch: Optional epoch number.
            phase: Training phase label (e.g. "A", "B").
            extra_state: Any additional state to save.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_name = f"{self.prefix}-{step}"
        ckpt_dir = self.output_dir / ckpt_name
        ckpt_dir.mkdir(exist_ok=True)

        # Save EMA weights (used for inference)
        if hasattr(ema_controlnet, "save_pretrained"):
            ema_controlnet.save_pretrained(ckpt_dir / "controlnet_ema")

        # Save training state for resume
        state = {
            "controlnet": _get_state_dict(controlnet),
            "ema_controlnet": _get_state_dict(ema_controlnet),
            "optimizer": optimizer.state_dict(),
            "global_step": step,
        }
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if extra_state:
            state.update(extra_state)

        torch.save(state, ckpt_dir / "training_state.pt")

        # Compute checkpoint size
        size_mb = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / (1024 * 1024)

        # Create metadata
        meta = CheckpointMetadata(
            step=step,
            timestamp=time.time(),
            metrics=metrics or {},
            epoch=epoch,
            phase=phase,
            size_mb=round(size_mb, 1),
        )

        # Save metadata alongside checkpoint
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        # Update index
        self._index["checkpoints"][ckpt_name] = meta.to_dict()
        self._update_best()
        self._save_index()

        # Update symlinks
        self._update_symlinks()

        # Prune old checkpoints
        self._prune()

        return ckpt_dir

    # ------------------------------------------------------------------
    # Best / latest tracking
    # ------------------------------------------------------------------

    def _update_best(self) -> None:
        """Recompute which checkpoints are 'best'."""
        entries = []
        for name, meta in self._index["checkpoints"].items():
            val = meta.get("metrics", {}).get(self.metric)
            if val is not None:
                entries.append((name, val, meta))

        if not entries:
            return

        # Sort by metric
        entries.sort(key=lambda x: x[1], reverse=not self.lower_is_better)

        # Mark best
        best_names = {e[0] for e in entries[: self.keep_best]}
        for name, meta in self._index["checkpoints"].items():
            meta["is_best"] = name in best_names

    def _update_symlinks(self) -> None:
        """Update 'latest' and 'best' symlinks."""
        checkpoints = self._sorted_by_step()
        if not checkpoints:
            return

        # Latest symlink
        latest_name = checkpoints[-1]
        latest_link = self.output_dir / "latest"
        _force_symlink(self.output_dir / latest_name, latest_link)

        # Best symlink
        best_name = self.get_best_checkpoint_name()
        if best_name:
            best_link = self.output_dir / "best"
            _force_symlink(self.output_dir / best_name, best_link)

    def get_best_checkpoint_name(self) -> str | None:
        """Return the name of the best checkpoint by tracked metric."""
        best = None
        best_val = None
        for name, meta in self._index["checkpoints"].items():
            val = meta.get("metrics", {}).get(self.metric)
            if val is None:
                continue
            if (
                best_val is None
                or (self.lower_is_better and val < best_val)
                or (not self.lower_is_better and val > best_val)
            ):
                best, best_val = name, val
        return best

    def get_best_metric_value(self) -> float | None:
        """Return the best value of the tracked metric."""
        name = self.get_best_checkpoint_name()
        if name is None:
            return None
        return self._index["checkpoints"][name]["metrics"].get(self.metric)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _sorted_by_step(self) -> list[str]:
        """Return checkpoint names sorted by step (ascending)."""
        items = list(self._index["checkpoints"].items())
        items.sort(key=lambda x: x[1].get("step", 0))
        return [name for name, _ in items]

    def _prune(self) -> None:
        """Remove old checkpoints, keeping best N and latest M."""
        all_names = self._sorted_by_step()
        if len(all_names) <= self.keep_latest:
            return

        # Determine which to keep
        keep = set()

        # Keep latest
        for name in all_names[-self.keep_latest :]:
            keep.add(name)

        # Keep best
        for name, meta in self._index["checkpoints"].items():
            if meta.get("is_best", False):
                keep.add(name)

        # Delete the rest
        for name in all_names:
            if name not in keep:
                ckpt_dir = self.output_dir / name
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                self._index["checkpoints"].pop(name, None)

        self._save_index()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return metadata for all tracked checkpoints, sorted by step."""
        result = []
        for name in self._sorted_by_step():
            meta = self._index["checkpoints"][name]
            result.append({"name": name, **meta})
        return result

    def get_checkpoint_path(self, name: str) -> Path:
        """Return the filesystem path for a checkpoint by name."""
        return self.output_dir / name

    def get_latest_step(self) -> int:
        """Return the step of the most recent checkpoint, or 0."""
        names = self._sorted_by_step()
        if not names:
            return 0
        return self._index["checkpoints"][names[-1]].get("step", 0)

    def total_size_mb(self) -> float:
        """Return total disk size of all tracked checkpoints."""
        return sum(meta.get("size_mb", 0.0) for meta in self._index["checkpoints"].values())

    def summary(self) -> str:
        """Return a human-readable summary of checkpoint state."""
        ckpts = self.list_checkpoints()
        if not ckpts:
            return "No checkpoints saved."

        lines = [
            f"Checkpoints: {len(ckpts)} saved ({self.total_size_mb():.0f} MB total)",
            f"Latest: step {self.get_latest_step()}",
        ]

        best_name = self.get_best_checkpoint_name()
        best_val = self.get_best_metric_value()
        if best_name and best_val is not None:
            lines.append(f"Best ({self.metric}): {best_val:.6f} @ {best_name}")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_state_dict(module: torch.nn.Module) -> dict:
    """Extract state dict, handling DDP wrapper."""
    if hasattr(module, "module"):
        return module.module.state_dict()
    return module.state_dict()


def _force_symlink(target: Path, link: Path) -> None:
    """Create or replace a symlink."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)
