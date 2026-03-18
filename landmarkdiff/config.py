"""YAML-based experiment configuration for reproducible training and evaluation.

Provides typed dataclasses that can be loaded from YAML files, enabling
reproducible experiments with version-tracked configs.

Usage:
    from landmarkdiff.config import ExperimentConfig
    config = ExperimentConfig.from_yaml("configs/rhinoplasty_phaseA.yaml")
    print(config.training.learning_rate)

    # Or create programmatically
    config = ExperimentConfig(
        experiment_name="rhino_v1",
        training=TrainingConfig(phase="A", learning_rate=1e-5),
    )
    config.to_yaml("configs/rhino_v1.yaml")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """ControlNet and base model configuration."""

    base_model: str = "runwayml/stable-diffusion-v1-5"
    controlnet_conditioning_channels: int = 3
    controlnet_conditioning_scale: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    phase: str = "A"  # "A" or "B"
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 50000
    warmup_steps: int = 500
    mixed_precision: str = "bf16"
    seed: int = 42
    ema_decay: float = 0.9999

    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam8bit", "prodigy"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0

    # LR scheduler
    lr_scheduler: str = "cosine"
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    # Logging intervals
    log_every: int = 100
    sample_every: int = 1000

    # Phase B specific
    identity_loss_weight: float = 0.1
    perceptual_loss_weight: float = 0.05
    use_differentiable_arcface: bool = False
    arcface_weights_path: str | None = None

    # Loss weights (alternative to individual weights)
    loss_weights: dict[str, float] = field(default_factory=dict)

    # Checkpointing
    save_every_n_steps: int = 5000
    resume_from_checkpoint: str | None = None
    resume_phase_a: str | None = None

    # Validation
    validate_every_n_steps: int = 2500
    num_validation_samples: int = 4


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_dir: str = "data/training_combined"
    val_dir: str = "data/splits/val"
    test_dir: str = "data/splits/test"
    image_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    random_flip: bool = True
    random_rotation: float = 5.0  # degrees
    color_jitter: float = 0.1
    clinical_augment: bool = False
    geometric_augment: bool = True

    # Procedure filtering
    procedures: list[str] = field(
        default_factory=lambda: [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ]
    )
    intensity_range: tuple[float, float] = (30.0, 100.0)

    # Data-driven displacement
    displacement_model_path: str | None = None
    noise_scale: float = 0.1


@dataclass
class InferenceConfig:
    """Inference / generation configuration."""

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    scheduler: str = "dpmsolver++"  # "ddpm", "ddim", "dpmsolver++"
    controlnet_conditioning_scale: float = 1.0

    # Post-processing
    use_neural_postprocess: bool = False
    restore_mode: str = "codeformer"
    codeformer_fidelity: float = 0.7
    use_realesrgan: bool = True
    use_laplacian_blend: bool = True
    sharpen_strength: float = 0.25

    # Identity verification
    verify_identity: bool = True
    identity_threshold: float = 0.5


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    compute_fid: bool = True
    compute_lpips: bool = True
    compute_nme: bool = True
    compute_identity: bool = True
    compute_ssim: bool = True
    stratify_fitzpatrick: bool = True
    stratify_procedure: bool = True
    max_eval_samples: int = 0  # 0 = all


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "landmarkdiff"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # "online", "offline", "disabled"


@dataclass
class SlurmConfig:
    """SLURM job submission parameters."""

    partition: str = "batch_gpu"
    account: str = ""  # Set via YAML or SLURM_ACCOUNT env var
    gpu_type: str = "nvidia_rtx_a6000"
    num_gpus: int = 1
    mem: str = "48G"
    cpus_per_task: int = 8
    time_limit: str = "48:00:00"
    job_prefix: str = "surgery_"


@dataclass
class SafetyConfig:
    """Clinical safety and responsible AI parameters."""

    identity_threshold: float = 0.5
    max_displacement_fraction: float = 0.05
    watermark_enabled: bool = True
    watermark_text: str = "AI-GENERATED PREDICTION"
    ood_detection_enabled: bool = True
    ood_confidence_threshold: float = 0.3
    min_face_confidence: float = 0.5
    max_yaw_degrees: float = 45.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "default"
    description: str = ""
    version: str = "0.3.2"

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Output
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from a YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        return cls(
            experiment_name=raw.get("experiment_name", "default"),
            description=raw.get("description", ""),
            version=raw.get("version", "0.3.2"),
            model=_from_dict(ModelConfig, raw.get("model", {})),
            training=_from_dict(TrainingConfig, raw.get("training", {})),
            data=_from_dict(DataConfig, raw.get("data", {})),
            inference=_from_dict(InferenceConfig, raw.get("inference", {})),
            evaluation=_from_dict(EvaluationConfig, raw.get("evaluation", {})),
            wandb=_from_dict(WandbConfig, raw.get("wandb", {})),
            slurm=_from_dict(SlurmConfig, raw.get("slurm", {})),
            safety=_from_dict(SafetyConfig, raw.get("safety", {})),
            output_dir=raw.get("output_dir", "outputs"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = _convert_tuples(asdict(self))
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


_FIELD_ALIASES: dict[str, str] = {
    # YAML name -> dataclass field name
    "max_steps": "max_train_steps",
    "save_interval": "save_every_n_steps",
    "sample_interval": "sample_every",
    "log_interval": "log_every",
    "adam_weight_decay": "weight_decay",
    "lr_warmup_steps": "warmup_steps",
    "resume_from": "resume_from_checkpoint",
}


def _from_dict(cls: type, d: dict) -> Any:
    """Create a dataclass from a dict, ignoring unknown keys.

    Supports field aliases so YAML configs using train_controlnet.py-style
    names (e.g. max_steps) map to dataclass fields (max_train_steps).
    """
    import dataclasses

    field_map = {f.name: f for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in d.items():
        # Resolve aliases
        canonical = _FIELD_ALIASES.get(k, k)
        if canonical not in field_map:
            continue
        # Don't overwrite if the canonical name was already set explicitly
        if canonical in filtered:
            continue
        # Convert lists back to tuples where the field type is tuple
        f = field_map[canonical]
        if isinstance(v, list) and "tuple" in str(f.type):
            v = tuple(v)
        filtered[canonical] = v
    return cls(**filtered)


def _convert_tuples(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _convert_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_convert_tuples(item) for item in obj]
    return obj


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> ExperimentConfig:
    """Load config with optional dot-notation overrides.

    Args:
        config_path: Path to YAML config. None returns defaults.
        overrides: Dict of "section.key" -> value overrides.
            E.g., {"training.learning_rate": 5e-6}

    Returns:
        ExperimentConfig with overrides applied.
    """
    config = ExperimentConfig.from_yaml(config_path) if config_path else ExperimentConfig()

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            obj = config
            resolved = True
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    resolved = False
                    break
            if resolved and hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)

    return config


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate config and return list of warnings."""
    warnings = []

    if config.training.phase == "B" and not config.training.resume_from_checkpoint:
        warnings.append("Phase B should resume from a Phase A checkpoint")

    eff_batch = config.training.batch_size * config.training.gradient_accumulation_steps
    if eff_batch < 8:
        warnings.append(f"Effective batch size {eff_batch} < 8 may cause instability")

    if config.training.learning_rate > 1e-4:
        warnings.append("Learning rate > 1e-4 is unusually high for fine-tuning")

    if config.data.image_size != 512:
        warnings.append(f"Image size {config.data.image_size} != 512; SD1.5 expects 512")

    if config.safety.identity_threshold < 0.3:
        warnings.append("Identity threshold < 0.3 may pass poor quality outputs")

    return warnings
