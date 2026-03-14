#!/usr/bin/env python3
"""Pre-training validation — verifies everything is ready for Phase A/B training.

Checks:
  1. Dataset exists and has enough pairs
  2. Metadata present for curriculum learning
  3. Val/test splits created
  4. Config file valid and references existing paths
  5. Dependencies installed (torch, diffusers, etc.)
  6. GPU available and has enough memory
  7. Disk space sufficient for checkpoints
  8. Previous checkpoints (for resume)
  9. WandB configured

Usage:
    python scripts/preflight_training.py --config configs/phaseA_production.yaml
    python scripts/preflight_training.py --config configs/phaseB_production.yaml --phase B
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class PreflightCheck:
    """A single preflight check with pass/fail/warn status."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.warning = False
        self.message = ""
        self.details: list[str] = []

    def pass_(self, msg: str):
        self.passed = True
        self.message = msg

    def warn(self, msg: str):
        self.passed = True
        self.warning = True
        self.message = msg

    def fail(self, msg: str):
        self.passed = False
        self.message = msg

    def detail(self, msg: str):
        self.details.append(msg)

    def __str__(self):
        icon = "PASS" if self.passed and not self.warning else "WARN" if self.warning else "FAIL"
        s = f"[{icon}] {self.name}: {self.message}"
        for d in self.details:
            s += f"\n       {d}"
        return s


def check_dataset(config: dict) -> PreflightCheck:
    """Check training dataset exists and has enough pairs."""
    check = PreflightCheck("Training Dataset")

    train_dir = config.get("data", {}).get("train_dir", "data/training_combined")
    train_path = PROJECT_ROOT / train_dir
    if not train_path.exists():
        check.fail(f"Training directory not found: {train_dir}")
        return check

    # Count pairs
    inputs = list(train_path.glob("*_input.png"))
    n_pairs = len(inputs)

    if n_pairs == 0:
        check.fail(f"No training pairs in {train_dir}")
    elif n_pairs < 1000:
        check.warn(f"{n_pairs} pairs in {train_dir} (< 1000 recommended)")
    else:
        check.pass_(f"{n_pairs:,} training pairs")

    # Check conditioning files
    n_cond = len(list(train_path.glob("*_conditioning.png")))
    if n_cond < n_pairs:
        check.detail(f"Missing conditioning: {n_pairs - n_cond} files")

    return check


def check_metadata(config: dict) -> PreflightCheck:
    """Check metadata.json exists for curriculum learning."""
    check = PreflightCheck("Metadata")

    train_dir = config.get("data", {}).get("train_dir", "data/training_combined")
    meta_path = PROJECT_ROOT / train_dir / "metadata.json"

    if not meta_path.exists():
        check.warn("No metadata.json — curriculum learning disabled")
        check.detail("Run: python scripts/reconstruct_metadata.py")
        return check

    with open(meta_path) as f:
        meta = json.load(f)

    n_pairs = meta.get("total_pairs", 0)
    n_entries = len(meta.get("pairs", {}))
    proc_dist = meta.get("procedure_distribution", {})

    if n_entries < n_pairs * 0.9:
        check.warn(
            f"Metadata covers {n_entries}/{n_pairs} pairs ({n_entries / n_pairs * 100:.0f}%)"
        )
    else:
        check.pass_(f"Metadata: {n_entries} entries, {len(proc_dist)} procedures")
        for proc, count in sorted(proc_dist.items()):
            check.detail(f"{proc}: {count} ({count / n_entries * 100:.1f}%)")

    return check


def check_splits(config: dict) -> PreflightCheck:
    """Check train/val/test splits exist."""
    check = PreflightCheck("Data Splits")

    val_dir = config.get("data", {}).get("val_dir", "data/splits/val")
    val_path = PROJECT_ROOT / val_dir

    if not val_path.exists():
        check.warn(f"Validation directory not found: {val_dir}")
        check.detail(
            "Run: python scripts/split_dataset.py --data_dir data/training_combined --output_dir data/splits"
        )
        return check

    n_val = len(list(val_path.glob("*_input.png")))
    test_path = PROJECT_ROOT / "data/splits/test"
    n_test = len(list(test_path.glob("*_input.png"))) if test_path.exists() else 0

    if n_val == 0:
        check.fail("Validation set is empty")
    else:
        check.pass_(f"Val: {n_val} pairs, Test: {n_test} pairs")

    return check


def check_config(config: dict, config_path: str) -> PreflightCheck:
    """Check config file is valid using schema validator."""
    check = PreflightCheck("Config File")

    required_keys = ["training", "data"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        check.fail(f"Missing config sections: {missing}")
        return check

    training = config.get("training", {})
    phase = training.get("phase", "A")
    lr = training.get("learning_rate", 0)
    batch = training.get("batch_size", 0)
    accum = training.get("gradient_accumulation_steps", 1)
    steps = training.get("max_train_steps", 0)

    effective_batch = batch * accum

    # Run schema validator for deep validation
    try:
        from scripts.validate_config import validate_config as schema_validate

        schema_result = schema_validate(config_path)

        if schema_result.errors:
            check.fail(f"Schema validation: {len(schema_result.errors)} error(s)")
            for err in schema_result.errors:
                check.detail(err)
            return check

        if schema_result.warnings:
            check.warn(
                f"Phase {phase}: {steps:,} steps, LR={lr}, "
                f"effective_batch={effective_batch} "
                f"({len(schema_result.warnings)} warning(s))"
            )
            for warn in schema_result.warnings:
                check.detail(warn)
        else:
            check.pass_(
                f"Phase {phase}: {steps:,} steps, LR={lr}, effective_batch={effective_batch}"
            )
    except ImportError:
        # Fallback to basic validation if schema validator not available
        check.pass_(f"Phase {phase}: {steps:,} steps, LR={lr}, effective_batch={effective_batch}")
        if lr > 1e-3:
            check.detail("WARNING: Learning rate > 1e-3 may be too high")
        if steps < 5000:
            check.detail("WARNING: < 5000 steps may not converge")

    check.detail(f"Config: {config_path}")
    return check


def check_dependencies() -> PreflightCheck:
    """Check required Python packages are installed."""
    check = PreflightCheck("Dependencies")

    required = {
        "torch": "PyTorch",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "accelerate": "Accelerate",
        "cv2": "OpenCV",
        "mediapipe": "MediaPipe",
    }
    missing = []
    versions = {}

    for pkg, name in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            versions[name] = ver
        except ImportError:
            missing.append(name)

    if missing:
        check.fail(f"Missing packages: {', '.join(missing)}")
    else:
        check.pass_(f"All {len(required)} dependencies installed")

    # Show key versions
    for name, ver in sorted(versions.items()):
        check.detail(f"{name}: {ver}")

    return check


def check_gpu() -> PreflightCheck:
    """Check GPU availability and memory."""
    check = PreflightCheck("GPU")

    try:
        import torch

        if not torch.cuda.is_available():
            check.warn("No GPU detected — training will be very slow")
            return check

        n_gpu = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9

        if gpu_mem < 16:
            check.warn(f"{n_gpu}x {gpu_name} ({gpu_mem:.0f}GB) — may need smaller batch size")
        else:
            check.pass_(f"{n_gpu}x {gpu_name} ({gpu_mem:.0f}GB)")

        # Check CUDA version
        check.detail(f"CUDA: {torch.version.cuda}")
        check.detail(f"cuDNN: {torch.backends.cudnn.version()}")

    except Exception as e:
        check.warn(f"GPU check failed: {e}")

    return check


def check_disk_space(config: dict) -> PreflightCheck:
    """Check sufficient disk space for checkpoints."""
    check = PreflightCheck("Disk Space")

    output_dir = config.get("output_dir", "checkpoints")
    out_path = PROJECT_ROOT / output_dir

    # Check available space
    try:
        usage = shutil.disk_usage(str(out_path.parent))
        free_gb = usage.free / 1e9

        # Estimate space needed: ~2GB per checkpoint, save every 10K steps
        steps = config.get("training", {}).get("max_train_steps", 50000)
        save_every = config.get("training", {}).get("save_every_n_steps", 10000)
        n_checkpoints = steps // save_every + 1
        estimated_gb = n_checkpoints * 2  # ~2GB per SD1.5 checkpoint

        if free_gb < estimated_gb:
            check.warn(
                f"{free_gb:.0f}GB free, need ~{estimated_gb}GB for {n_checkpoints} checkpoints"
            )
        else:
            check.pass_(f"{free_gb:.0f}GB free (need ~{estimated_gb}GB)")
            check.detail(f"Output: {output_dir}")

    except Exception as e:
        check.warn(f"Disk check failed: {e}")

    return check


def check_resume(config: dict) -> PreflightCheck:
    """Check for existing checkpoints to resume from."""
    check = PreflightCheck("Resume")

    output_dir = config.get("output_dir", "checkpoints")
    out_path = PROJECT_ROOT / output_dir

    if not out_path.exists():
        check.pass_("No previous checkpoints (fresh start)")
        return check

    checkpoints = sorted(out_path.glob("checkpoint-*"))
    if not checkpoints:
        check.pass_("Output dir exists but no checkpoints")
        return check

    latest = checkpoints[-1]
    step_m = None
    import re

    step_m = re.search(r"checkpoint-(\d+)", latest.name)
    step = int(step_m.group(1)) if step_m else 0

    resume = config.get("training", {}).get("resume_from_checkpoint", "")
    if resume == "auto":
        check.pass_(f"Will auto-resume from checkpoint-{step}")
    else:
        check.warn(f"Existing checkpoint at step {step} but resume not set to 'auto'")
        check.detail("Set resume_from_checkpoint: auto to continue")

    return check


def check_phase_b(config: dict) -> PreflightCheck:
    """Phase B specific: check Phase A checkpoint exists."""
    check = PreflightCheck("Phase A Checkpoint")

    phase = config.get("training", {}).get("phase", "A")
    if phase != "B":
        check.pass_("Not Phase B — skipping")
        return check

    # Look for Phase A checkpoints
    phase_a_dir = PROJECT_ROOT / "checkpoints_phaseA"
    if not phase_a_dir.exists():
        check.fail("Phase A checkpoint directory not found: checkpoints_phaseA/")
        check.detail("Complete Phase A training first")
        return check

    checkpoints = sorted(phase_a_dir.glob("checkpoint-*"))
    ema_checkpoints = sorted(phase_a_dir.glob("checkpoint-*-ema"))

    if ema_checkpoints:
        latest = ema_checkpoints[-1]
        check.pass_(f"Phase A EMA checkpoint: {latest.name}")
    elif checkpoints:
        latest = checkpoints[-1]
        check.warn(f"Phase A checkpoint: {latest.name} (no EMA version)")
    else:
        check.fail("No Phase A checkpoints found")

    return check


def run_preflight(config_path: str, phase: str = "A") -> bool:
    """Run all preflight checks and print report."""
    import yaml

    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        logger.error("Config not found: %s", config_path)
        return False

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("PREFLIGHT CHECK -- Phase %s", phase)
    logger.info("Config: %s", config_path)
    logger.info("=" * 60)

    checks = [
        check_config(config, config_path),
        check_dataset(config),
        check_metadata(config),
        check_splits(config),
        check_dependencies(),
        check_gpu(),
        check_disk_space(config),
        check_resume(config),
    ]

    if phase == "B":
        checks.append(check_phase_b(config))

    all_pass = True
    n_warn = 0
    for check in checks:
        logger.info("%s", check)
        if not check.passed:
            all_pass = False
        if check.warning:
            n_warn += 1

    passed = sum(1 for c in checks if c.passed)
    failed = sum(1 for c in checks if not c.passed)

    logger.info("=" * 60)
    if all_pass and n_warn == 0:
        logger.info("ALL CHECKS PASSED (%d/%d)", passed, len(checks))
        logger.info("Ready to submit training!")
    elif all_pass:
        logger.warning(
            "PASSED WITH WARNINGS (%d/%d passed, %d warnings)", passed, len(checks), n_warn
        )
        logger.warning("Training can proceed but check warnings above.")
    else:
        logger.error("PREFLIGHT FAILED (%d failures, %d warnings)", failed, n_warn)
        logger.error("Fix failures before training.")
    logger.info("=" * 60)

    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pre-training validation")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--phase", default="A", choices=["A", "B"])
    args = parser.parse_args()
    success = run_preflight(args.config, args.phase)
    sys.exit(0 if success else 1)
