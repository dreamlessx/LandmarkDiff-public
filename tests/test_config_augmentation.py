"""Tests for config system, augmentation pipeline, and FID computation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---- Config tests ----

class TestExperimentConfig:
    """Test YAML config system."""

    def test_default_config(self):
        from landmarkdiff.config import ExperimentConfig
        config = ExperimentConfig()
        assert config.experiment_name == "default"
        assert config.training.phase == "A"
        assert config.training.learning_rate == 1e-5
        assert config.model.base_model == "runwayml/stable-diffusion-v1-5"

    def test_round_trip_yaml(self, tmp_path):
        from landmarkdiff.config import ExperimentConfig, TrainingConfig
        config = ExperimentConfig(
            experiment_name="test_roundtrip",
            training=TrainingConfig(
                phase="B",
                learning_rate=5e-6,
                batch_size=2,
                identity_loss_weight=0.15,
            ),
        )

        yaml_path = tmp_path / "test.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.experiment_name == "test_roundtrip"
        assert loaded.training.phase == "B"
        assert loaded.training.learning_rate == 5e-6
        assert loaded.training.batch_size == 2
        assert loaded.training.identity_loss_weight == 0.15

    def test_from_yaml_with_defaults(self, tmp_path):
        """Partial YAML should fill defaults for missing keys."""
        from landmarkdiff.config import ExperimentConfig
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("experiment_name: partial_test\ntraining:\n  phase: B\n")

        config = ExperimentConfig.from_yaml(yaml_path)
        assert config.experiment_name == "partial_test"
        assert config.training.phase == "B"
        # All other fields should be defaults
        assert config.training.learning_rate == 1e-5
        assert config.model.gradient_checkpointing is True

    def test_to_dict(self):
        from landmarkdiff.config import ExperimentConfig
        config = ExperimentConfig(experiment_name="dict_test")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["experiment_name"] == "dict_test"
        assert "training" in d
        assert "model" in d

    def test_load_phaseA_yaml(self):
        """Test loading the actual phaseA config file."""
        from landmarkdiff.config import ExperimentConfig
        path = Path(__file__).parent.parent / "configs" / "phaseA_default.yaml"
        if path.exists():
            config = ExperimentConfig.from_yaml(path)
            assert config.training.phase == "A"
            assert config.training.max_train_steps == 50000

    def test_load_phaseB_yaml(self):
        """Test loading the actual phaseB config file."""
        from landmarkdiff.config import ExperimentConfig
        path = Path(__file__).parent.parent / "configs" / "phaseB_identity.yaml"
        if path.exists():
            config = ExperimentConfig.from_yaml(path)
            assert config.training.phase == "B"
            assert config.training.use_differentiable_arcface is True


# ---- Augmentation tests ----

class TestAugmentation:
    """Test training data augmentation."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic training sample."""
        rng = np.random.default_rng(42)
        h, w = 512, 512
        return {
            "input_image": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            "target_image": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            "conditioning": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            "mask": rng.random((h, w)).astype(np.float32),
            "landmarks_src": rng.random((478, 2)).astype(np.float32),
            "landmarks_dst": rng.random((478, 2)).astype(np.float32),
        }

    def test_augment_preserves_shapes(self, sample_data):
        from landmarkdiff.augmentation import augment_training_sample, AugmentationConfig
        config = AugmentationConfig(seed=42)
        result = augment_training_sample(
            **sample_data, config=config,
        )
        assert result["input_image"].shape == (512, 512, 3)
        assert result["target_image"].shape == (512, 512, 3)
        assert result["conditioning"].shape == (512, 512, 3)
        assert result["mask"].shape == (512, 512)
        assert result["landmarks_src"].shape == (478, 2)
        assert result["landmarks_dst"].shape == (478, 2)

    def test_augment_deterministic_with_seed(self, sample_data):
        from landmarkdiff.augmentation import augment_training_sample, AugmentationConfig
        config = AugmentationConfig(seed=123)
        r1 = augment_training_sample(**sample_data, config=config,
                                      rng=np.random.default_rng(123))
        r2 = augment_training_sample(**sample_data, config=config,
                                      rng=np.random.default_rng(123))
        np.testing.assert_array_equal(r1["input_image"], r2["input_image"])

    def test_no_augmentation(self, sample_data):
        """With all augmentations disabled, output should match input."""
        from landmarkdiff.augmentation import augment_training_sample, AugmentationConfig
        config = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            random_scale=(1.0, 1.0),
            random_translate=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_shift_range=0.0,
            conditioning_dropout_prob=0.0,
            conditioning_noise_std=0.0,
        )
        result = augment_training_sample(**sample_data, config=config)
        np.testing.assert_array_equal(result["input_image"], sample_data["input_image"])

    def test_landmarks_bounded(self, sample_data):
        from landmarkdiff.augmentation import augment_training_sample, AugmentationConfig
        config = AugmentationConfig(seed=42)
        result = augment_training_sample(**sample_data, config=config)
        assert np.all(result["landmarks_src"] >= 0)
        assert np.all(result["landmarks_src"] <= 1)
        assert np.all(result["landmarks_dst"] >= 0)
        assert np.all(result["landmarks_dst"] <= 1)

    def test_conditioning_dropout(self, sample_data):
        """With dropout=1.0, conditioning should be all zeros."""
        from landmarkdiff.augmentation import augment_training_sample, AugmentationConfig
        config = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            random_scale=(1.0, 1.0),
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_shift_range=0.0,
            conditioning_dropout_prob=1.0,
            conditioning_noise_std=0.0,
        )
        result = augment_training_sample(**sample_data, config=config)
        assert np.all(result["conditioning"] == 0)


class TestSkinToneAugmentation:
    def test_augment_skin_tone_no_change(self):
        from landmarkdiff.augmentation import augment_skin_tone
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = augment_skin_tone(img, ita_delta=0.0)
        # With zero delta, should be very close to original
        assert np.allclose(result.astype(float), img.astype(float), atol=2)

    def test_augment_skin_tone_lighter(self):
        from landmarkdiff.augmentation import augment_skin_tone
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        result = augment_skin_tone(img, ita_delta=20.0)
        # Should be lighter (higher L channel)
        assert result.mean() > img.mean()


class TestFitzpatrickBalancer:
    def test_uniform_weights(self):
        from landmarkdiff.augmentation import FitzpatrickBalancer
        balancer = FitzpatrickBalancer()
        for ft in ["I", "II", "III", "IV", "V", "VI"]:
            for _ in range(10):
                balancer.register_sample(ft)

        weights = balancer.get_sampling_weights(
            ["I", "II", "III", "IV", "V", "VI"]
        )
        # With equal counts, weights should be approximately equal
        assert np.allclose(weights, weights[0], atol=0.01)

    def test_imbalanced_upweighting(self):
        from landmarkdiff.augmentation import FitzpatrickBalancer
        balancer = FitzpatrickBalancer()
        # Imbalanced: 100 Type I, 10 Type VI
        for _ in range(100):
            balancer.register_sample("I")
        for _ in range(10):
            balancer.register_sample("VI")

        weights = balancer.get_sampling_weights(["I", "VI"])
        # Type VI should get higher weight
        assert weights[1] > weights[0]


# ---- FID tests ----

class TestFID:
    def test_import(self):
        from landmarkdiff.fid import compute_fid_from_dirs, compute_fid_from_arrays
        assert callable(compute_fid_from_dirs)
        assert callable(compute_fid_from_arrays)

    def test_statistics_computation(self):
        from landmarkdiff.fid import _compute_statistics
        features = np.random.randn(100, 2048)
        mu, sigma = _compute_statistics(features)
        assert mu.shape == (2048,)
        assert sigma.shape == (2048, 2048)

    def test_fid_same_distribution(self):
        from landmarkdiff.fid import _compute_statistics, _calculate_fid
        rng = np.random.default_rng(42)
        features = rng.standard_normal((200, 64))  # small dim for speed
        mu, sigma = _compute_statistics(features)
        # FID of distribution with itself should be ~0
        fid = _calculate_fid(mu, sigma, mu, sigma)
        assert abs(fid) < 1e-6

    def test_fid_different_distributions(self):
        from landmarkdiff.fid import _compute_statistics, _calculate_fid
        rng = np.random.default_rng(42)
        feat1 = rng.standard_normal((200, 64))
        feat2 = rng.standard_normal((200, 64)) + 5.0  # shifted mean
        mu1, sigma1 = _compute_statistics(feat1)
        mu2, sigma2 = _compute_statistics(feat2)
        fid = _calculate_fid(mu1, sigma1, mu2, sigma2)
        # Should be large since distributions differ
        assert fid > 100


# ---- Training dataset with augmentation integration tests ----

class TestTrainingDatasetAugmentation:
    """Test SyntheticPairDataset integration with augmentation pipeline."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a minimal mock dataset with 3 synthetic pairs."""
        import cv2
        rng = np.random.default_rng(42)
        for i in range(3):
            prefix = f"{i:06d}"
            # Create fake 64x64 images (small for speed)
            for suffix in ["input", "target", "conditioning"]:
                img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{prefix}_{suffix}.png"), img)
            # Mask
            mask = np.ones((64, 64), dtype=np.uint8) * 255
            cv2.imwrite(str(tmp_path / f"{prefix}_mask.png"), mask)
        return tmp_path

    def test_dataset_loads_with_augmentation(self, mock_dataset_dir):
        """Dataset should load and apply augmentation without errors."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        # Import Dataset class directly from the training script module
        from importlib import import_module
        spec = import_module("train_controlnet")
        ds = spec.SyntheticPairDataset(
            str(mock_dataset_dir),
            resolution=64,
            geometric_augment=True,
        )
        assert len(ds) == 3
        sample = ds[0]
        assert sample["input"].shape == (3, 64, 64)
        assert sample["target"].shape == (3, 64, 64)
        assert sample["conditioning"].shape == (3, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)

    def test_dataset_augmentation_varies(self, mock_dataset_dir):
        """Repeated access to same index should yield different results (stochastic aug)."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from importlib import import_module
        spec = import_module("train_controlnet")
        ds = spec.SyntheticPairDataset(
            str(mock_dataset_dir),
            resolution=64,
            geometric_augment=True,
        )
        s1 = ds[0]["target"].numpy()
        s2 = ds[0]["target"].numpy()
        # With random augmentation, outputs should differ (flips, rotations, etc.)
        # There's a small chance they match exactly, so allow that
        # But over multiple samples it should vary
        any_different = False
        for _ in range(5):
            s_new = ds[0]["target"].numpy()
            if not np.allclose(s1, s_new, atol=1e-5):
                any_different = True
                break
        assert any_different, "Augmentation should produce varying outputs"

    def test_dataset_no_augmentation(self, mock_dataset_dir):
        """With augmentation disabled, repeated access gives same result."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from importlib import import_module
        spec = import_module("train_controlnet")
        ds = spec.SyntheticPairDataset(
            str(mock_dataset_dir),
            resolution=64,
            geometric_augment=False,
            clinical_augment=False,
        )
        s1 = ds[0]["target"].numpy()
        s2 = ds[0]["target"].numpy()
        np.testing.assert_allclose(s1, s2, atol=1e-6)

    def test_dataset_values_in_range(self, mock_dataset_dir):
        """All output tensors should be in [0, 1] range."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from importlib import import_module
        spec = import_module("train_controlnet")
        ds = spec.SyntheticPairDataset(
            str(mock_dataset_dir),
            resolution=64,
            geometric_augment=True,
        )
        for i in range(len(ds)):
            sample = ds[i]
            for key in ["input", "target", "conditioning"]:
                assert sample[key].min() >= 0.0, f"{key} has values < 0"
                assert sample[key].max() <= 1.0, f"{key} has values > 1"
            assert sample["mask"].min() >= 0.0
            assert sample["mask"].max() <= 1.0
