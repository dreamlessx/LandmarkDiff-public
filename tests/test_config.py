"""Tests for YAML-based experiment configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    SafetyConfig,
    SlurmConfig,
    TrainingConfig,
    WandbConfig,
    _convert_tuples,
    _from_dict,
    load_config,
    validate_config,
)


class TestDataclasses:
    """Tests for config dataclass defaults and fields."""

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        assert cfg.base_model == "runwayml/stable-diffusion-v1-5"
        assert cfg.controlnet_conditioning_channels == 3
        assert cfg.use_ema is True

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.phase == "A"
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.mixed_precision == "bf16"
        assert cfg.optimizer == "adamw"

    def test_training_config_phase_b(self):
        cfg = TrainingConfig(
            phase="B",
            identity_loss_weight=0.2,
            use_differentiable_arcface=True,
        )
        assert cfg.phase == "B"
        assert cfg.identity_loss_weight == 0.2
        assert cfg.use_differentiable_arcface is True

    def test_data_config_defaults(self):
        cfg = DataConfig()
        assert cfg.image_size == 512
        assert "rhinoplasty" in cfg.procedures
        assert cfg.intensity_range == (30.0, 100.0)

    def test_inference_config_defaults(self):
        cfg = InferenceConfig()
        assert cfg.num_inference_steps == 30
        assert cfg.guidance_scale == 7.5
        assert cfg.scheduler == "dpmsolver++"
        assert cfg.restore_mode == "codeformer"

    def test_evaluation_config_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.compute_fid is True
        assert cfg.stratify_fitzpatrick is True

    def test_wandb_config_defaults(self):
        cfg = WandbConfig()
        assert cfg.enabled is True
        assert cfg.project == "landmarkdiff"

    def test_slurm_config_defaults(self):
        cfg = SlurmConfig()
        assert cfg.partition == "batch_gpu"
        assert cfg.num_gpus == 1
        assert cfg.gpu_type == "nvidia_rtx_a6000"

    def test_safety_config_defaults(self):
        cfg = SafetyConfig()
        assert cfg.identity_threshold == 0.5
        assert cfg.watermark_enabled is True
        assert cfg.min_face_confidence == 0.5

    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.experiment_name == "default"
        assert cfg.version == "0.3.2"
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.safety, SafetyConfig)
        assert isinstance(cfg.slurm, SlurmConfig)


class TestYAMLSerialization:
    """Tests for YAML save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        cfg = ExperimentConfig(
            experiment_name="test_experiment",
            description="A test config",
            training=TrainingConfig(learning_rate=5e-6, phase="B"),
            data=DataConfig(image_size=256),
        )
        yaml_path = tmp_path / "test_config.yaml"
        cfg.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.experiment_name == "test_experiment"
        assert loaded.training.learning_rate == 5e-6
        assert loaded.training.phase == "B"
        assert loaded.data.image_size == 256

    def test_load_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.experiment_name == "default"

    def test_load_partial_yaml(self, tmp_path):
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "partial_test",
                    "training": {"learning_rate": 1e-4},
                }
            )
        )
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.experiment_name == "partial_test"
        assert cfg.training.learning_rate == 1e-4
        assert cfg.training.batch_size == 4  # default preserved

    def test_unknown_keys_ignored(self, tmp_path):
        yaml_path = tmp_path / "extra.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "extra",
                    "training": {
                        "learning_rate": 1e-4,
                        "nonexistent_key": 999,
                    },
                }
            )
        )
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.training.learning_rate == 1e-4
        assert not hasattr(cfg.training, "nonexistent_key")

    def test_tuple_roundtrip(self, tmp_path):
        """Tuples should survive YAML round-trip (via list conversion)."""
        cfg = ExperimentConfig(
            data=DataConfig(intensity_range=(20.0, 80.0)),
        )
        yaml_path = tmp_path / "tuple_test.yaml"
        cfg.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.data.intensity_range == (20.0, 80.0)

    def test_to_dict(self):
        cfg = ExperimentConfig(experiment_name="dict_test")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["experiment_name"] == "dict_test"
        assert isinstance(d["training"], dict)
        assert d["training"]["learning_rate"] == 1e-5

    def test_creates_parent_dirs(self, tmp_path):
        yaml_path = tmp_path / "nested" / "deep" / "config.yaml"
        cfg = ExperimentConfig()
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()


class TestLoadConfig:
    """Tests for load_config with overrides."""

    def test_load_defaults(self):
        cfg = load_config()
        assert cfg.experiment_name == "default"
        assert cfg.training.learning_rate == 1e-5

    def test_load_with_overrides(self):
        cfg = load_config(
            overrides={
                "training.learning_rate": 5e-6,
                "training.batch_size": 8,
                "data.image_size": 256,
            }
        )
        assert cfg.training.learning_rate == 5e-6
        assert cfg.training.batch_size == 8
        assert cfg.data.image_size == 256

    def test_load_from_yaml_with_overrides(self, tmp_path):
        yaml_path = tmp_path / "base.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "base",
                    "training": {"learning_rate": 1e-5},
                }
            )
        )
        cfg = load_config(
            config_path=yaml_path,
            overrides={"training.learning_rate": 3e-6},
        )
        assert cfg.experiment_name == "base"
        assert cfg.training.learning_rate == 3e-6

    def test_override_nonexistent_key(self):
        """Non-existent override keys should be silently ignored."""
        cfg = load_config(overrides={"nonexistent.key": 42})
        assert cfg.experiment_name == "default"

    def test_override_safety_config(self):
        cfg = load_config(
            overrides={
                "safety.identity_threshold": 0.8,
                "safety.watermark_enabled": False,
            }
        )
        assert cfg.safety.identity_threshold == 0.8
        assert cfg.safety.watermark_enabled is False


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_config(self):
        cfg = ExperimentConfig()
        warnings = validate_config(cfg)
        assert isinstance(warnings, list)

    def test_phase_b_without_checkpoint(self):
        cfg = ExperimentConfig(
            training=TrainingConfig(phase="B"),
        )
        warnings = validate_config(cfg)
        assert any("Phase B" in w for w in warnings)

    def test_small_effective_batch(self):
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
        )
        warnings = validate_config(cfg)
        assert any("batch size" in w.lower() for w in warnings)

    def test_high_lr_warning(self):
        cfg = ExperimentConfig(
            training=TrainingConfig(learning_rate=1e-3),
        )
        warnings = validate_config(cfg)
        assert any("Learning rate" in w for w in warnings)

    def test_non_512_image_size(self):
        cfg = ExperimentConfig(
            data=DataConfig(image_size=256),
        )
        warnings = validate_config(cfg)
        assert any("512" in w for w in warnings)

    def test_low_identity_threshold(self):
        cfg = ExperimentConfig(
            safety=SafetyConfig(identity_threshold=0.2),
        )
        warnings = validate_config(cfg)
        assert any("Identity threshold" in w for w in warnings)

    def test_good_config_no_warnings(self):
        cfg = ExperimentConfig(
            training=TrainingConfig(
                phase="A",
                learning_rate=1e-5,
                batch_size=4,
                gradient_accumulation_steps=4,
            ),
        )
        warnings = validate_config(cfg)
        assert len(warnings) == 0


class TestHelpers:
    """Tests for helper functions."""

    def test_from_dict_basic(self):
        result = _from_dict(TrainingConfig, {"learning_rate": 2e-5, "phase": "B"})
        assert result.learning_rate == 2e-5
        assert result.phase == "B"
        assert result.batch_size == 4  # default

    def test_from_dict_ignores_unknown(self):
        result = _from_dict(TrainingConfig, {"learning_rate": 1e-5, "unknown": 42})
        assert result.learning_rate == 1e-5

    def test_from_dict_tuple_conversion(self):
        result = _from_dict(DataConfig, {"intensity_range": [10.0, 90.0]})
        assert result.intensity_range == (10.0, 90.0)

    def test_convert_tuples(self):
        data = {"a": (1, 2), "b": {"c": (3, 4)}, "d": [5, (6, 7)]}
        result = _convert_tuples(data)
        assert result["a"] == [1, 2]
        assert result["b"]["c"] == [3, 4]
        assert result["d"] == [5, [6, 7]]
