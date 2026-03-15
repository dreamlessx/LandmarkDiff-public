"""Extended tests for config loading, validation, defaults, and edge cases.

Covers ExperimentConfig construction, YAML edge cases, validate_config
boundary conditions, load_config override mechanics, and _from_dict /
_convert_tuples helpers more thoroughly than test_config.py.
"""

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

# ---------------------------------------------------------------------------
# Dataclass field validation
# ---------------------------------------------------------------------------


class TestModelConfigFields:
    """Deeper tests for ModelConfig field behavior."""

    def test_custom_base_model(self):
        cfg = ModelConfig(base_model="stabilityai/sd-turbo")
        assert cfg.base_model == "stabilityai/sd-turbo"

    def test_custom_conditioning_channels(self):
        cfg = ModelConfig(controlnet_conditioning_channels=1)
        assert cfg.controlnet_conditioning_channels == 1

    def test_ema_disabled(self):
        cfg = ModelConfig(use_ema=False)
        assert cfg.use_ema is False

    def test_gradient_checkpointing_default(self):
        cfg = ModelConfig()
        assert cfg.gradient_checkpointing is True

    def test_conditioning_scale_custom(self):
        cfg = ModelConfig(controlnet_conditioning_scale=0.5)
        assert cfg.controlnet_conditioning_scale == 0.5


class TestTrainingConfigFields:
    """Additional training config field tests."""

    def test_optimizer_options(self):
        for opt in ["adamw", "adam8bit", "prodigy"]:
            cfg = TrainingConfig(optimizer=opt)
            assert cfg.optimizer == opt

    def test_lr_scheduler_kwargs(self):
        cfg = TrainingConfig(lr_scheduler_kwargs={"num_cycles": 3})
        assert cfg.lr_scheduler_kwargs["num_cycles"] == 3

    def test_resume_from_checkpoint_none(self):
        cfg = TrainingConfig()
        assert cfg.resume_from_checkpoint is None

    def test_seed_default(self):
        cfg = TrainingConfig()
        assert cfg.seed == 42

    def test_custom_max_grad_norm(self):
        cfg = TrainingConfig(max_grad_norm=0.5)
        assert cfg.max_grad_norm == 0.5


class TestInferenceConfigFields:
    """Inference config edge cases."""

    def test_scheduler_options(self):
        for sched in ["ddpm", "ddim", "dpmsolver++"]:
            cfg = InferenceConfig(scheduler=sched)
            assert cfg.scheduler == sched

    def test_codeformer_fidelity_range(self):
        """Fidelity should accept boundary values."""
        cfg_low = InferenceConfig(codeformer_fidelity=0.0)
        cfg_high = InferenceConfig(codeformer_fidelity=1.0)
        assert cfg_low.codeformer_fidelity == 0.0
        assert cfg_high.codeformer_fidelity == 1.0

    def test_identity_threshold_custom(self):
        cfg = InferenceConfig(identity_threshold=0.8)
        assert cfg.identity_threshold == 0.8

    def test_all_postprocess_flags(self):
        cfg = InferenceConfig(
            use_neural_postprocess=True,
            use_realesrgan=False,
            use_laplacian_blend=False,
        )
        assert cfg.use_neural_postprocess is True
        assert cfg.use_realesrgan is False
        assert cfg.use_laplacian_blend is False


class TestDataConfigFields:
    """Data config edge cases."""

    def test_procedures_list_mutable(self):
        """Default procedures list should be independent per instance."""
        c1 = DataConfig()
        c2 = DataConfig()
        c1.procedures.append("otoplasty")
        assert "otoplasty" not in c2.procedures

    def test_intensity_range_custom(self):
        cfg = DataConfig(intensity_range=(10.0, 50.0))
        assert cfg.intensity_range == (10.0, 50.0)

    def test_displacement_model_path_none(self):
        cfg = DataConfig()
        assert cfg.displacement_model_path is None

    def test_augmentation_defaults(self):
        cfg = DataConfig()
        assert cfg.random_flip is True
        assert cfg.random_rotation == 5.0
        assert cfg.color_jitter == 0.1


class TestSafetyConfigFields:
    """Safety config boundary tests."""

    def test_max_displacement_fraction(self):
        cfg = SafetyConfig(max_displacement_fraction=0.1)
        assert cfg.max_displacement_fraction == 0.1

    def test_ood_detection_disabled(self):
        cfg = SafetyConfig(ood_detection_enabled=False)
        assert cfg.ood_detection_enabled is False

    def test_max_yaw_degrees(self):
        cfg = SafetyConfig(max_yaw_degrees=30.0)
        assert cfg.max_yaw_degrees == 30.0


# ---------------------------------------------------------------------------
# YAML edge cases
# ---------------------------------------------------------------------------


class TestYAMLEdgeCases:
    """YAML serialization and deserialization edge cases."""

    def test_load_yaml_with_only_top_level_fields(self, tmp_path):
        """YAML with only top-level fields, no sub-configs."""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml.dump({"experiment_name": "minimal", "output_dir": "/tmp/out"}))
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.experiment_name == "minimal"
        assert cfg.output_dir == "/tmp/out"
        # Sub-configs should have defaults
        assert cfg.training.learning_rate == 1e-5
        assert cfg.model.base_model == "runwayml/stable-diffusion-v1-5"

    def test_load_yaml_with_null_values(self, tmp_path):
        """YAML with explicit null/None values."""
        yaml_path = tmp_path / "nulls.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "nulltest",
                    "training": {
                        "resume_from_checkpoint": None,
                        "arcface_weights_path": None,
                    },
                }
            )
        )
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.training.resume_from_checkpoint is None
        assert cfg.training.arcface_weights_path is None

    def test_roundtrip_preserves_all_sections(self, tmp_path):
        """All config sections should survive a save/load roundtrip."""
        cfg = ExperimentConfig(
            experiment_name="full_roundtrip",
            model=ModelConfig(base_model="custom/model"),
            training=TrainingConfig(phase="B", learning_rate=2e-5),
            data=DataConfig(image_size=256, procedures=["rhinoplasty"]),
            inference=InferenceConfig(num_inference_steps=50),
            evaluation=EvaluationConfig(compute_fid=False),
            wandb=WandbConfig(enabled=False, project="test"),
            slurm=SlurmConfig(partition="gpu_dev"),
            safety=SafetyConfig(identity_threshold=0.7),
            output_dir="/tmp/outputs",
        )
        yaml_path = tmp_path / "full.yaml"
        cfg.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)

        assert loaded.experiment_name == "full_roundtrip"
        assert loaded.model.base_model == "custom/model"
        assert loaded.training.phase == "B"
        assert loaded.training.learning_rate == 2e-5
        assert loaded.data.image_size == 256
        assert loaded.data.procedures == ["rhinoplasty"]
        assert loaded.inference.num_inference_steps == 50
        assert loaded.evaluation.compute_fid is False
        assert loaded.wandb.enabled is False
        assert loaded.slurm.partition == "gpu_dev"
        assert loaded.safety.identity_threshold == 0.7
        assert loaded.output_dir == "/tmp/outputs"

    def test_to_dict_nested_structure(self):
        """to_dict should produce fully nested dicts."""
        cfg = ExperimentConfig()
        d = cfg.to_dict()
        assert isinstance(d["model"], dict)
        assert isinstance(d["data"], dict)
        assert isinstance(d["safety"], dict)
        assert d["model"]["base_model"] == "runwayml/stable-diffusion-v1-5"

    def test_to_dict_includes_all_keys(self):
        """to_dict output should have all top-level config sections."""
        cfg = ExperimentConfig()
        d = cfg.to_dict()
        expected_keys = {
            "experiment_name",
            "description",
            "version",
            "model",
            "training",
            "data",
            "inference",
            "evaluation",
            "wandb",
            "slurm",
            "safety",
            "output_dir",
        }
        assert set(d.keys()) == expected_keys

    def test_yaml_with_extra_top_level_keys(self, tmp_path):
        """Extra top-level keys in YAML should not cause errors."""
        yaml_path = tmp_path / "extra_top.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "extra",
                    "unknown_section": {"foo": "bar"},
                    "also_unknown": 42,
                }
            )
        )
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.experiment_name == "extra"

    def test_yaml_with_numeric_experiment_name(self, tmp_path):
        """Numeric experiment names should be handled."""
        yaml_path = tmp_path / "numeric.yaml"
        yaml_path.write_text(yaml.dump({"experiment_name": 12345}))
        cfg = ExperimentConfig.from_yaml(yaml_path)
        # yaml.safe_load will give int, from_yaml just assigns
        assert cfg.experiment_name == 12345

    def test_empty_sub_config_dict(self, tmp_path):
        """Empty sub-config dicts should produce default configs."""
        yaml_path = tmp_path / "empty_sub.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "training": {},
                    "model": {},
                    "safety": {},
                }
            )
        )
        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.training.learning_rate == 1e-5
        assert cfg.model.base_model == "runwayml/stable-diffusion-v1-5"
        assert cfg.safety.identity_threshold == 0.5


# ---------------------------------------------------------------------------
# validate_config edge cases
# ---------------------------------------------------------------------------


class TestValidateConfigExtended:
    """More detailed validate_config tests."""

    def test_boundary_lr_1e4(self):
        """Learning rate at exactly 1e-4 should not warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(
                learning_rate=1e-4,
                batch_size=4,
                gradient_accumulation_steps=4,
            ),
        )
        warnings = validate_config(cfg)
        assert not any("Learning rate" in w for w in warnings)

    def test_lr_just_above_1e4(self):
        """Learning rate just above 1e-4 should produce a warning."""
        cfg = ExperimentConfig(
            training=TrainingConfig(
                learning_rate=1.1e-4,
                batch_size=4,
                gradient_accumulation_steps=4,
            ),
        )
        warnings = validate_config(cfg)
        assert any("Learning rate" in w for w in warnings)

    def test_batch_size_boundary_8(self):
        """Effective batch size of exactly 8 should not warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=2, gradient_accumulation_steps=4),
        )
        warnings = validate_config(cfg)
        assert not any("batch size" in w.lower() for w in warnings)

    def test_batch_size_boundary_7(self):
        """Effective batch size of 7 should warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=7, gradient_accumulation_steps=1),
        )
        warnings = validate_config(cfg)
        assert any("batch size" in w.lower() for w in warnings)

    def test_phase_b_with_checkpoint_no_warning(self):
        """Phase B with checkpoint should not warn about checkpoint."""
        cfg = ExperimentConfig(
            training=TrainingConfig(
                phase="B",
                resume_from_checkpoint="checkpoints/step_50000",
                batch_size=4,
                gradient_accumulation_steps=4,
            ),
        )
        warnings = validate_config(cfg)
        assert not any("Phase B" in w for w in warnings)

    def test_identity_threshold_boundary_0_3(self):
        """Identity threshold at exactly 0.3 should not warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=4, gradient_accumulation_steps=4),
            safety=SafetyConfig(identity_threshold=0.3),
        )
        warnings = validate_config(cfg)
        assert not any("Identity threshold" in w for w in warnings)

    def test_identity_threshold_0_29(self):
        """Identity threshold at 0.29 should warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=4, gradient_accumulation_steps=4),
            safety=SafetyConfig(identity_threshold=0.29),
        )
        warnings = validate_config(cfg)
        assert any("Identity threshold" in w for w in warnings)

    def test_image_size_512_no_warning(self):
        """Image size 512 should not warn."""
        cfg = ExperimentConfig(
            training=TrainingConfig(batch_size=4, gradient_accumulation_steps=4),
        )
        warnings = validate_config(cfg)
        assert not any("512" in w for w in warnings)

    def test_multiple_warnings(self):
        """Should return multiple warnings for multiple issues."""
        cfg = ExperimentConfig(
            training=TrainingConfig(
                phase="B",
                learning_rate=1e-3,
                batch_size=1,
                gradient_accumulation_steps=1,
            ),
            data=DataConfig(image_size=256),
            safety=SafetyConfig(identity_threshold=0.1),
        )
        warnings = validate_config(cfg)
        assert len(warnings) >= 4

    def test_validate_returns_list(self):
        """validate_config should always return a list."""
        cfg = ExperimentConfig()
        result = validate_config(cfg)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# load_config with overrides
# ---------------------------------------------------------------------------


class TestLoadConfigExtended:
    """Extended tests for load_config override mechanics."""

    def test_override_deeply_nested(self):
        """Should handle dotted overrides for all config sections."""
        cfg = load_config(
            overrides={
                "model.base_model": "custom/model",
                "inference.guidance_scale": 12.0,
                "evaluation.compute_fid": False,
            }
        )
        assert cfg.model.base_model == "custom/model"
        assert cfg.inference.guidance_scale == 12.0
        assert cfg.evaluation.compute_fid is False

    def test_override_multiple_in_same_section(self):
        """Multiple overrides in the same section should all apply."""
        cfg = load_config(
            overrides={
                "training.learning_rate": 3e-5,
                "training.batch_size": 16,
                "training.phase": "B",
            }
        )
        assert cfg.training.learning_rate == 3e-5
        assert cfg.training.batch_size == 16
        assert cfg.training.phase == "B"

    def test_override_top_level(self):
        """Top-level fields (no dot) should be overridable."""
        cfg = load_config(overrides={"experiment_name": "overridden", "output_dir": "/new/dir"})
        assert cfg.experiment_name == "overridden"
        assert cfg.output_dir == "/new/dir"

    def test_override_empty_dict(self):
        """Empty overrides dict should return defaults."""
        cfg = load_config(overrides={})
        assert cfg.experiment_name == "default"

    def test_override_none(self):
        """None overrides should return defaults."""
        cfg = load_config(overrides=None)
        assert cfg.experiment_name == "default"

    def test_load_from_yaml_then_override(self, tmp_path):
        """Overrides should take precedence over YAML values."""
        yaml_path = tmp_path / "base.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "experiment_name": "yaml_base",
                    "training": {"learning_rate": 1e-5, "phase": "A"},
                }
            )
        )
        cfg = load_config(
            config_path=yaml_path,
            overrides={
                "experiment_name": "overridden",
                "training.phase": "B",
            },
        )
        assert cfg.experiment_name == "overridden"
        assert cfg.training.phase == "B"
        # Non-overridden values from YAML preserved
        assert cfg.training.learning_rate == 1e-5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestFromDictExtended:
    """Extended tests for _from_dict."""

    def test_empty_dict(self):
        """Empty dict should produce default config."""
        result = _from_dict(TrainingConfig, {})
        assert result.learning_rate == 1e-5
        assert result.phase == "A"

    def test_all_known_keys(self):
        """Should accept all known field names."""
        result = _from_dict(
            TrainingConfig,
            {
                "phase": "B",
                "learning_rate": 3e-5,
                "batch_size": 8,
                "seed": 123,
            },
        )
        assert result.phase == "B"
        assert result.learning_rate == 3e-5
        assert result.batch_size == 8
        assert result.seed == 123

    def test_list_to_tuple_for_procedures(self):
        """Lists should be converted to tuples for tuple-typed fields."""
        result = _from_dict(DataConfig, {"intensity_range": [20.0, 80.0]})
        assert isinstance(result.intensity_range, tuple)
        assert result.intensity_range == (20.0, 80.0)

    def test_list_stays_list_for_list_fields(self):
        """List fields should remain lists."""
        result = _from_dict(DataConfig, {"procedures": ["rhinoplasty", "brow_lift"]})
        assert isinstance(result.procedures, list)
        assert result.procedures == ["rhinoplasty", "brow_lift"]

    def test_mixed_known_unknown_keys(self):
        """Known keys accepted, unknown keys silently dropped."""
        result = _from_dict(
            ModelConfig,
            {
                "base_model": "foo/bar",
                "unknown_key": 42,
                "another_unknown": "hello",
            },
        )
        assert result.base_model == "foo/bar"
        assert not hasattr(result, "unknown_key")


class TestConvertTuplesExtended:
    """Extended tests for _convert_tuples."""

    def test_nested_dicts(self):
        data = {"a": {"b": {"c": (1, 2, 3)}}}
        result = _convert_tuples(data)
        assert result["a"]["b"]["c"] == [1, 2, 3]

    def test_mixed_types(self):
        data = {"num": 42, "text": "hello", "tup": (1, 2), "lst": [3, 4]}
        result = _convert_tuples(data)
        assert result["num"] == 42
        assert result["text"] == "hello"
        assert result["tup"] == [1, 2]
        assert result["lst"] == [3, 4]

    def test_empty_tuple(self):
        data = {"empty": ()}
        result = _convert_tuples(data)
        assert result["empty"] == []

    def test_none_passthrough(self):
        data = {"val": None}
        result = _convert_tuples(data)
        assert result["val"] is None

    def test_scalar_passthrough(self):
        assert _convert_tuples(42) == 42
        assert _convert_tuples("hello") == "hello"
        assert _convert_tuples(3.14) == 3.14
