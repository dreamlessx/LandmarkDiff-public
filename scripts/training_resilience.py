#!/usr/bin/env python3
"""Training fault tolerance for SLURM HPC environments.

Provides signal handling, OOM recovery, and checkpoint integrity validation
to prevent losing GPU hours when training jobs fail unexpectedly.

Key features:
1. SLURM signal handling (SIGTERM/SIGUSR1 → emergency checkpoint)
2. GPU OOM detection with automatic batch size reduction
3. Checkpoint integrity validation on resume
4. NaN/Inf gradient detection with auto-recovery

Integration:
    from scripts.training_resilience import (
        SlurmSignalHandler,
        OOMHandler,
        validate_checkpoint,
        GradientWatchdog,
    )

    # In training script:
    handler = SlurmSignalHandler(save_fn=my_save_function)
    handler.register()

    oom = OOMHandler(initial_batch_size=4)
    batch_size = oom.current_batch_size

    # Before resume:
    if validate_checkpoint(ckpt_path):
        load_checkpoint(ckpt_path)

Usage:
    # Validate a checkpoint
    python scripts/training_resilience.py --validate checkpoints/checkpoint-5000

    # Check all checkpoints in a directory
    python scripts/training_resilience.py --validate-all checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ── SLURM Signal Handler ────────────────────────────────────────


@dataclass
class SlurmSignalHandler:
    """Catch SLURM preemption/timeout signals and trigger emergency checkpoint.

    SLURM sends SIGTERM (or SIGUSR1 with --signal=USR1@<seconds>) before
    killing a job. This handler catches the signal, calls a user-provided
    save function, then exits cleanly.

    Usage:
        def save_checkpoint():
            torch.save(state, "emergency_checkpoint.pt")

        handler = SlurmSignalHandler(save_fn=save_checkpoint)
        handler.register()

        # In training loop:
        if handler.should_exit:
            break  # exit gracefully after current step
    """

    save_fn: Callable[[], None] | None = None
    should_exit: bool = False
    signal_received: str | None = None
    _original_sigterm: object = None
    _original_sigusr1: object = None

    def register(self) -> None:
        """Register signal handlers for SLURM-aware graceful shutdown."""
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # SIGUSR1 is commonly used with --signal=USR1@120 (2 min warning)
        if hasattr(signal, "SIGUSR1"):
            self._original_sigusr1 = signal.getsignal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, self._handle_signal)

    def unregister(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigusr1 is not None and hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._original_sigusr1)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle termination signal — save checkpoint and flag exit."""
        sig_name = signal.Signals(signum).name
        self.signal_received = sig_name
        self.should_exit = True

        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        logger.warning("=" * 60)
        logger.warning("SIGNAL RECEIVED: %s (SLURM job %s)", sig_name, job_id)
        logger.warning("Saving emergency checkpoint...")

        if self.save_fn is not None:
            try:
                t0 = time.time()
                self.save_fn()
                elapsed = time.time() - t0
                logger.info("Emergency checkpoint saved in %.1fs", elapsed)
            except Exception as e:
                logger.error("Error saving checkpoint: %s", e)

        logger.warning("Exiting gracefully after signal %s", sig_name)
        logger.warning("=" * 60)

    def check_and_exit(self) -> None:
        """Call in training loop — exits if signal was received.

        Use `should_exit` for a non-exiting check instead.
        """
        if self.should_exit:
            sys.exit(0)


# ── OOM Handler ──────────────────────────────────────────────────


@dataclass
class OOMHandler:
    """Detect and recover from GPU out-of-memory errors.

    Automatically halves the batch size on OOM, clears CUDA cache,
    and signals the training loop to rebuild the dataloader.

    Usage:
        oom = OOMHandler(initial_batch_size=4, min_batch_size=1)

        while training:
            try:
                loss = forward_backward(batch)
            except RuntimeError as e:
                if oom.handle_oom(e):
                    rebuild_dataloader(oom.current_batch_size)
                    continue
                raise  # re-raise if not OOM
    """

    initial_batch_size: int = 4
    min_batch_size: int = 1
    current_batch_size: int = 0
    oom_count: int = 0
    max_oom_retries: int = 3
    _batch_size_history: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if self.current_batch_size == 0:
            self.current_batch_size = self.initial_batch_size

    def is_oom_error(self, error: RuntimeError) -> bool:
        """Check if a RuntimeError is a CUDA OOM error."""
        msg = str(error).lower()
        return any(
            phrase in msg
            for phrase in [
                "cuda out of memory",
                "out of memory",
                "cudnn error",
                "cublas error",
            ]
        )

    def handle_oom(self, error: RuntimeError) -> bool:
        """Handle an OOM error — returns True if recovery is possible.

        Returns False if the error is not OOM or retries are exhausted.
        """
        if not self.is_oom_error(error):
            return False

        self.oom_count += 1

        if self.oom_count > self.max_oom_retries:
            logger.error("OOM recovery exhausted (%d retries). Giving up.", self.max_oom_retries)
            return False

        old_bs = self.current_batch_size
        new_bs = max(self.min_batch_size, old_bs // 2)

        if new_bs == old_bs and old_bs == self.min_batch_size:
            logger.error("Already at minimum batch size (%d). Cannot recover.", self.min_batch_size)
            return False

        self.current_batch_size = new_bs
        self._batch_size_history.append((self.oom_count, new_bs))

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.warning("OOM RECOVERY #%d: batch size %d -> %d", self.oom_count, old_bs, new_bs)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.warning("CUDA memory: %.1fGB allocated, %.1fGB reserved", allocated, reserved)

        return True

    def summary(self) -> str:
        """Return summary of OOM events."""
        if self.oom_count == 0:
            return "No OOM events"
        history = ", ".join(f"#{n}→bs{bs}" for n, bs in self._batch_size_history)
        return f"{self.oom_count} OOM events: {history}"


# ── Checkpoint Integrity Validation ──────────────────────────────


@dataclass
class CheckpointValidation:
    """Result of checkpoint validation."""

    path: str
    valid: bool
    step: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    size_mb: float = 0.0

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        msg = f"[{status}] {self.path} (step {self.step}, {self.size_mb:.1f}MB)"
        if self.errors:
            msg += "\n  Errors: " + "; ".join(self.errors)
        if self.warnings:
            msg += "\n  Warnings: " + "; ".join(self.warnings)
        return msg


def validate_checkpoint(checkpoint_path: str | Path) -> CheckpointValidation:
    """Validate a training checkpoint for integrity.

    Checks:
    1. File exists and is non-empty
    2. Can be loaded without errors
    3. Contains required keys (controlnet, ema_controlnet, optimizer, global_step)
    4. No NaN/Inf in model weights
    5. State dict shapes are consistent
    """
    ckpt_path = Path(checkpoint_path)
    result = CheckpointValidation(path=str(ckpt_path), valid=False)

    # Check training_state.pt exists
    state_file = ckpt_path / "training_state.pt"
    if not state_file.exists():
        # Maybe it's the state file itself
        if ckpt_path.suffix == ".pt" and ckpt_path.exists():
            state_file = ckpt_path
        else:
            result.errors.append(f"training_state.pt not found in {ckpt_path}")
            return result

    # Check file size
    size_bytes = state_file.stat().st_size
    result.size_mb = size_bytes / (1024 * 1024)
    if size_bytes < 1024:  # Less than 1KB
        result.errors.append(f"File too small ({size_bytes} bytes) — likely corrupted")
        return result

    # Try to load
    try:
        state = torch.load(state_file, map_location="cpu", weights_only=True)
    except Exception as e:
        result.errors.append(f"Failed to load: {e}")
        return result

    # Check required keys
    required_keys = ["controlnet", "ema_controlnet", "optimizer", "global_step"]
    for key in required_keys:
        if key not in state:
            result.errors.append(f"Missing key: {key}")

    if result.errors:
        return result

    result.step = state["global_step"]

    # Check for NaN/Inf in model weights
    for model_key in ["controlnet", "ema_controlnet"]:
        state_dict = state[model_key]
        nan_params = []
        inf_params = []
        for param_name, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                nan_params.append(param_name)
            if torch.isinf(tensor).any():
                inf_params.append(param_name)

        if nan_params:
            result.errors.append(f"{model_key}: NaN in {len(nan_params)} params: {nan_params[:3]}")
        if inf_params:
            result.warnings.append(
                f"{model_key}: Inf in {len(inf_params)} params: {inf_params[:3]}"
            )

    # Check ControlNet and EMA have same shapes
    cn_shapes = {k: v.shape for k, v in state["controlnet"].items()}
    ema_shapes = {k: v.shape for k, v in state["ema_controlnet"].items()}

    if cn_shapes.keys() != ema_shapes.keys():
        missing_in_ema = set(cn_shapes) - set(ema_shapes)
        missing_in_cn = set(ema_shapes) - set(cn_shapes)
        if missing_in_ema:
            result.errors.append(f"EMA missing {len(missing_in_ema)} params from ControlNet")
        if missing_in_cn:
            result.errors.append(f"ControlNet missing {len(missing_in_cn)} params from EMA")
    else:
        mismatched = [k for k in cn_shapes if cn_shapes[k] != ema_shapes[k]]
        if mismatched:
            result.errors.append(f"Shape mismatch in {len(mismatched)} params: {mismatched[:3]}")

    # Check optimizer state isn't empty
    opt_state = state["optimizer"]
    if not opt_state.get("state"):
        result.warnings.append("Optimizer state is empty (may be fresh init)")

    # Check scheduler state
    if "scheduler" not in state:
        result.warnings.append("No scheduler state saved")

    result.valid = len(result.errors) == 0
    return result


def validate_all_checkpoints(checkpoint_dir: str | Path) -> list[CheckpointValidation]:
    """Validate all checkpoints in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    results = []

    # Find checkpoint directories (checkpoint-NNNNN)
    ckpt_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )

    if not ckpt_dirs:
        logger.warning("No checkpoints found in %s", ckpt_dir)
        return []

    for d in ckpt_dirs:
        result = validate_checkpoint(d)
        results.append(result)
        status = "OK" if result.valid else "FAIL"
        logger.info("[%s] %s -- step %d, %.1fMB", status, d.name, result.step, result.size_mb)
        for err in result.errors:
            logger.error("  %s", err)
        for warn in result.warnings:
            logger.warning("  %s", warn)

    valid = sum(1 for r in results if r.valid)
    logger.info("%d/%d checkpoints valid", valid, len(results))

    return results


# ── Gradient Watchdog ────────────────────────────────────────────


@dataclass
class GradientWatchdog:
    """Monitor gradients for NaN/Inf and loss spikes during training.

    Detects training instability early and can trigger corrective actions:
    - Skip update on NaN/Inf gradients
    - Reduce learning rate on loss spikes
    - Alert when gradients vanish

    Usage:
        watchdog = GradientWatchdog()

        # After loss.backward():
        action = watchdog.check(model.parameters(), loss.item(), step)
        if action == "skip":
            optimizer.zero_grad()  # skip this update
            continue
    """

    window_size: int = 100
    spike_threshold: float = 10.0
    vanish_threshold: float = 1e-8
    nan_tolerance: int = 3  # consecutive NaN before alerting
    _loss_history: list[float] = field(default_factory=list)
    _grad_norm_history: list[float] = field(default_factory=list)
    _consecutive_nan: int = 0
    _total_nan: int = 0
    _total_skipped: int = 0

    def check(
        self,
        parameters,
        loss_value: float,
        step: int,
    ) -> str:
        """Check gradient health. Returns action: 'ok', 'skip', 'alert'.

        'ok' — training is healthy
        'skip' — skip this optimizer step (NaN/Inf gradients)
        'alert' — serious issue (consecutive NaN, loss spike)
        """
        # Check for NaN/Inf loss
        if not _is_finite(loss_value):
            self._consecutive_nan += 1
            self._total_nan += 1
            if self._consecutive_nan >= self.nan_tolerance:
                logger.error(
                    "ALERT: %d consecutive NaN/Inf losses at step %d",
                    self._consecutive_nan,
                    step,
                )
                return "alert"
            logger.warning("NaN/Inf loss at step %d (#%d)", step, self._consecutive_nan)
            self._total_skipped += 1
            return "skip"

        self._consecutive_nan = 0

        # Check gradient norms
        total_norm = _compute_grad_norm(parameters)

        if not _is_finite(total_norm):
            self._total_nan += 1
            self._total_skipped += 1
            logger.warning("NaN/Inf gradient norm at step %d", step)
            return "skip"

        # Track history
        self._loss_history.append(loss_value)
        self._grad_norm_history.append(total_norm)

        # Keep window
        if len(self._loss_history) > self.window_size:
            self._loss_history = self._loss_history[-self.window_size :]
        if len(self._grad_norm_history) > self.window_size:
            self._grad_norm_history = self._grad_norm_history[-self.window_size :]

        # Check for loss spike
        if len(self._loss_history) >= 10:
            recent_mean = sum(self._loss_history[-10:]) / 10
            older_mean = sum(self._loss_history[:-10]) / max(1, len(self._loss_history) - 10)
            if older_mean > 0 and recent_mean / older_mean > self.spike_threshold:
                logger.warning(
                    "Loss spike at step %d (%.4f vs %.4f)", step, recent_mean, older_mean
                )
                return "alert"

        # Check for vanishing gradients
        if total_norm < self.vanish_threshold and len(self._grad_norm_history) >= 5:
            recent_norms = self._grad_norm_history[-5:]
            if all(n < self.vanish_threshold for n in recent_norms):
                logger.warning("Vanishing gradients at step %d (norm=%.2e)", step, total_norm)
                return "alert"

        return "ok"

    def summary(self) -> str:
        """Return summary of gradient monitoring."""
        parts = [f"NaN events: {self._total_nan}, skipped steps: {self._total_skipped}"]
        if self._grad_norm_history:
            avg_norm = sum(self._grad_norm_history) / len(self._grad_norm_history)
            parts.append(f"avg grad norm: {avg_norm:.4f}")
        if self._loss_history:
            parts.append(f"last loss: {self._loss_history[-1]:.6f}")
        return " | ".join(parts)


def _is_finite(value: float) -> bool:
    """Check if a value is finite (not NaN or Inf)."""
    import math

    return math.isfinite(value)


def _compute_grad_norm(parameters) -> float:
    """Compute total gradient norm across parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm**0.5


# ── Emergency Checkpoint Helper ──────────────────────────────────


def create_emergency_save_fn(
    output_dir: str | Path,
    controlnet_module,
    ema_controlnet,
    optimizer,
    scheduler,
    global_step_ref: list[int],
) -> Callable[[], None]:
    """Create a save function for SlurmSignalHandler.

    Args:
        global_step_ref: A mutable list [step] so the closure captures
                        the current value at save time.
    """
    out = Path(output_dir)

    def _save():
        step = global_step_ref[0]
        ckpt_dir = out / f"emergency-checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save training state
        torch.save(
            {
                "controlnet": controlnet_module.state_dict(),
                "ema_controlnet": ema_controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": step,
                "emergency": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            },
            ckpt_dir / "training_state.pt",
        )

        # Also save EMA model in diffusers format for immediate use
        ema_controlnet.save_pretrained(ckpt_dir / "controlnet_ema")

        logger.info("Emergency checkpoint: %s (step %d)", ckpt_dir, step)

    return _save


# ── CLI ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Training resilience utilities")
    parser.add_argument("--validate", default=None, help="Validate a single checkpoint directory")
    parser.add_argument(
        "--validate-all", default=None, help="Validate all checkpoints in a directory"
    )
    args = parser.parse_args()

    if args.validate:
        result = validate_checkpoint(args.validate)
        logger.info("%s", result)
        sys.exit(0 if result.valid else 1)

    if args.validate_all:
        results = validate_all_checkpoints(args.validate_all)
        sys.exit(0 if all(r.valid for r in results) else 1)

    parser.print_help()
