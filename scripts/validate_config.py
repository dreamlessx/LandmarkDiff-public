#!/usr/bin/env python3
"""Training config schema validator: catches typos and invalid values.

Validates YAML training configs against the ExperimentConfig schema,
checking for unknown keys, type mismatches, out-of-range values, and
cross-field consistency (e.g., Phase B requires Phase A checkpoint).

The existing config loader silently ignores unknown keys; this validator
catches those as potential typos before GPU hours are wasted.

Usage:
    # Validate a single config
    python scripts/validate_config.py configs/phaseA_production.yaml

    # Validate all configs in a directory
    python scripts/validate_config.py configs/

    # Strict mode (warnings → errors)
    python scripts/validate_config.py configs/phaseA_production.yaml --strict
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.config import (
    DataConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    SafetyConfig,
    SlurmConfig,
    TrainingConfig,
    WandbConfig,
)

# ── Schema: valid keys per section ───────────────────────────────

# Map section names to their dataclass
SECTION_CLASSES: dict[str, type] = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "data": DataConfig,
    "inference": InferenceConfig,
    "evaluation": EvaluationConfig,
    "wandb": WandbConfig,
    "slurm": SlurmConfig,
    "safety": SafetyConfig,
}

# Top-level keys (not in any section)
TOP_LEVEL_KEYS = {
    "experiment_name",
    "description",
    "version",
    "output_dir",
    # Section names
    *SECTION_CLASSES.keys(),
}


# ── Value range constraints ──────────────────────────────────────

RANGE_CONSTRAINTS: dict[str, dict[str, tuple[float, float]]] = {
    "training": {
        "learning_rate": (1e-8, 1e-1),
        "batch_size": (1, 128),
        "gradient_accumulation_steps": (1, 128),
        "max_train_steps": (1, 10_000_000),
        "warmup_steps": (0, 100_000),
        "seed": (0, 2**32 - 1),
        "max_grad_norm": (0.01, 100.0),
        "identity_loss_weight": (0.0, 10.0),
        "perceptual_loss_weight": (0.0, 10.0),
        "save_every_n_steps": (1, 1_000_000),
        "validate_every_n_steps": (1, 1_000_000),
        "num_validation_samples": (1, 1000),
    },
    "model": {
        "ema_decay": (0.9, 1.0),
        "controlnet_conditioning_scale": (0.0, 2.0),
    },
    "data": {
        "image_size": (64, 2048),
        "num_workers": (0, 64),
    },
    "inference": {
        "num_inference_steps": (1, 1000),
        "guidance_scale": (0.0, 50.0),
        "controlnet_conditioning_scale": (0.0, 5.0),
        "codeformer_fidelity": (0.0, 1.0),
        "sharpen_strength": (0.0, 2.0),
        "identity_threshold": (0.0, 1.0),
    },
    "safety": {
        "identity_threshold": (0.0, 1.0),
        "max_displacement_fraction": (0.001, 0.5),
        "ood_confidence_threshold": (0.0, 1.0),
        "min_face_confidence": (0.0, 1.0),
        "max_yaw_degrees": (0.0, 180.0),
    },
}

# Valid choices for enum-like fields
VALID_CHOICES: dict[str, dict[str, list[str]]] = {
    "training": {
        "phase": ["A", "B"],
        "mixed_precision": ["fp16", "bf16", "no"],
        "optimizer": ["adamw", "adam8bit", "prodigy"],
        "lr_scheduler": ["cosine", "linear", "constant", "constant_with_warmup"],
        "resume_from_checkpoint": None,  # None means any string is ok
    },
    "inference": {
        "scheduler": ["ddpm", "ddim", "dpmsolver++", "dpm++_2m_karras", "unipc"],
        "restore_mode": ["codeformer", "gfpgan", "none"],
    },
    "wandb": {
        "mode": ["online", "offline", "disabled"],
    },
}


# ── Validation Result ────────────────────────────────────────────


@dataclasses.dataclass
class ConfigValidation:
    """Result of config validation."""

    path: str
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        parts = []
        status = "PASS" if self.valid else "FAIL"
        parts.append(f"[{status}] {self.path}")
        if self.errors:
            parts.append(f"  {len(self.errors)} error(s):")
            for e in self.errors:
                parts.append(f"    ERROR: {e}")
        if self.warnings:
            parts.append(f"  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                parts.append(f"    WARN: {w}")
        if self.valid and not self.warnings:
            parts.append("  All checks passed")
        return "\n".join(parts)


# ── Core Validation Logic ────────────────────────────────────────


def _get_field_names(cls: type) -> set[str]:
    """Get valid field names from a dataclass."""
    return {f.name for f in dataclasses.fields(cls)}


def validate_config(
    config_path: str | Path,
    strict: bool = False,
) -> ConfigValidation:
    """Validate a YAML training config against the schema.

    Checks:
    1. YAML syntax
    2. Unknown keys at top level and in each section
    3. Type checking (basic: int, float, str, bool)
    4. Value range validation
    5. Valid choices for enum-like fields
    6. Cross-field consistency (Phase B checks)
    7. Path existence (data dirs, checkpoints)

    Args:
        config_path: Path to the YAML config file.
        strict: If True, treat warnings as errors.
    """
    config_path = Path(config_path)
    result = ConfigValidation(path=str(config_path))

    # 1. File existence
    if not config_path.exists():
        result.errors.append(f"File not found: {config_path}")
        return result

    # 2. YAML parse
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.errors.append(f"YAML parse error: {e}")
        return result

    if not isinstance(raw, dict):
        result.errors.append(f"Config must be a YAML mapping, got {type(raw).__name__}")
        return result

    if not raw:
        result.warnings.append("Config file is empty")
        return result

    # 3. Check top-level keys
    for key in raw:
        if key not in TOP_LEVEL_KEYS:
            result.warnings.append(
                f"Unknown top-level key: '{key}'; possible typo? "
                f"Valid keys: {sorted(TOP_LEVEL_KEYS)}"
            )

    # 4. Check section keys
    for section_name, section_cls in SECTION_CLASSES.items():
        section_data = raw.get(section_name)
        if section_data is None:
            continue
        if not isinstance(section_data, dict):
            result.errors.append(
                f"Section '{section_name}' must be a mapping, got {type(section_data).__name__}"
            )
            continue

        valid_fields = _get_field_names(section_cls)
        for key in section_data:
            if key not in valid_fields:
                # Find closest match for typo suggestion
                suggestion = _closest_match(key, valid_fields)
                msg = f"Unknown key '{section_name}.{key}'"
                if suggestion:
                    msg += f"; did you mean '{suggestion}'?"
                else:
                    msg += f"; valid keys: {sorted(valid_fields)}"
                result.warnings.append(msg)

    # 5. Type checking
    for section_name, section_cls in SECTION_CLASSES.items():
        section_data = raw.get(section_name)
        if not isinstance(section_data, dict):
            continue

        for f in dataclasses.fields(section_cls):
            if f.name not in section_data:
                continue
            value = section_data[f.name]
            _check_type(result, f"{section_name}.{f.name}", value, f.type)

    # 6. Range validation
    for section_name, constraints in RANGE_CONSTRAINTS.items():
        section_data = raw.get(section_name)
        if not isinstance(section_data, dict):
            continue

        for key, (lo, hi) in constraints.items():
            if key in section_data:
                val = section_data[key]
                if isinstance(val, int | float) and (val < lo or val > hi):
                    result.errors.append(f"{section_name}.{key}={val} is out of range [{lo}, {hi}]")

    # 7. Choice validation
    for section_name, choices in VALID_CHOICES.items():
        section_data = raw.get(section_name)
        if not isinstance(section_data, dict):
            continue

        for key, valid_values in choices.items():
            if valid_values is None:
                continue  # any value ok
            if key in section_data:
                val = section_data[key]
                if isinstance(val, str) and val not in valid_values:
                    result.errors.append(
                        f"{section_name}.{key}='{val}' is not valid. Must be one of: {valid_values}"
                    )

    # 8. Cross-field consistency
    training = raw.get("training", {})
    if isinstance(training, dict):
        phase = training.get("phase", "A")

        # Phase B checks
        if phase == "B":
            if "resume_phaseA" not in training and "resume_from_checkpoint" not in training:
                result.warnings.append(
                    "Phase B config has no 'resume_phaseA' or 'resume_from_checkpoint'. "
                    "will start from scratch (likely not intended)"
                )
            if training.get("batch_size", 4) > 4:
                result.warnings.append(
                    f"Phase B typically needs batch_size <= 4 for VRAM "
                    f"(got {training['batch_size']})"
                )

        # Mixed precision on non-Ampere
        if training.get("mixed_precision") == "bf16":
            result.warnings.append(
                "BF16 requires Ampere+ GPU (A6000/A100/H100). "
                "Will auto-fallback to FP32 on older GPUs."
            )

        # Validate checkpoint interval vs total steps
        save_every = training.get("save_every_n_steps", 5000)
        max_steps = training.get("max_train_steps", 50000)
        if save_every > max_steps:
            result.warnings.append(
                f"save_every_n_steps ({save_every}) > max_train_steps ({max_steps}). "
                "no intermediate checkpoints will be saved"
            )

    # 9. Path existence checks
    data = raw.get("data", {})
    if isinstance(data, dict):
        for path_key in ["train_dir", "val_dir", "test_dir"]:
            path_val = data.get(path_key)
            if path_val and not Path(path_val).exists():
                # Check relative to project root too
                if not (PROJECT_ROOT / path_val).exists():
                    result.warnings.append(
                        f"data.{path_key}='{path_val}' does not exist "
                        f"(checked absolute and relative to project root)"
                    )

    # 10. experiment_name format
    exp_name = raw.get("experiment_name", "")
    if exp_name and not exp_name.replace("_", "").replace("-", "").replace(".", "").isalnum():
        result.warnings.append(
            f"experiment_name='{exp_name}' contains special characters; "
            "may cause issues in file paths and SLURM job names"
        )

    # Strict mode: promote warnings to errors
    if strict:
        result.errors.extend(result.warnings)
        result.warnings = []

    return result


def validate_all_configs(
    config_dir: str | Path,
    strict: bool = False,
) -> list[ConfigValidation]:
    """Validate all YAML configs in a directory."""
    config_dir = Path(config_dir)
    results = []

    yaml_files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        return []

    for yaml_file in yaml_files:
        result = validate_config(yaml_file, strict=strict)
        results.append(result)
        print(result.summary())
        print()

    # Summary
    valid = sum(1 for r in results if r.valid)
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    print(f"{'=' * 50}")
    print(
        f"{valid}/{len(results)} configs valid | {total_errors} errors | {total_warnings} warnings"
    )

    return results


def _check_type(
    result: ConfigValidation,
    key: str,
    value: Any,
    type_hint: str,
) -> None:
    """Basic type checking for config values."""
    type_str = str(type_hint)

    if value is None:
        return  # None is always allowed (optional fields)

    # Simple type checks
    if "float" in type_str and isinstance(value, bool):
        result.errors.append(f"{key}: expected float, got bool ({value})")
    elif "int" in type_str and not isinstance(value, int):
        if isinstance(value, float) and value == int(value):
            result.warnings.append(f"{key}: got float {value}, expected int")
        elif not isinstance(value, bool) and "float" not in type_str:
            result.errors.append(f"{key}: expected int, got {type(value).__name__}")
    elif "bool" in type_str and not isinstance(value, bool):
        result.errors.append(f"{key}: expected bool, got {type(value).__name__} ({value})")
    elif "str" in type_str and "list" not in type_str and not isinstance(value, str):
        if not isinstance(value, int | float | bool):  # numbers are ok-ish
            result.errors.append(f"{key}: expected str, got {type(value).__name__}")


def _closest_match(key: str, valid_keys: set[str], max_distance: int = 3) -> str | None:
    """Find the closest matching key using edit distance."""
    best = None
    best_dist = max_distance + 1

    for valid in valid_keys:
        dist = _edit_distance(key.lower(), valid.lower())
        if dist < best_dist:
            best_dist = dist
            best = valid

    return best if best_dist <= max_distance else None


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[len(a)]


# ── CLI ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate training configs")
    parser.add_argument("path", help="Config file or directory")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        results = validate_all_configs(path, strict=args.strict)
        sys.exit(0 if all(r.valid for r in results) else 1)
    else:
        result = validate_config(path, strict=args.strict)
        print(result.summary())
        sys.exit(0 if result.valid else 1)
