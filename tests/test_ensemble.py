"""Tests for ensemble inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestEnsembleInference:
    """Tests for the EnsembleInference class."""

    def test_pixel_average(self):
        """Test pixel-space averaging."""
        from landmarkdiff.ensemble import EnsembleInference

        ensemble = EnsembleInference(mode="tps", n_samples=3)

        # Create test outputs
        outputs = [
            np.full((64, 64, 3), 100, dtype=np.uint8),
            np.full((64, 64, 3), 200, dtype=np.uint8),
            np.full((64, 64, 3), 150, dtype=np.uint8),
        ]

        result = ensemble._pixel_average(outputs)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        # Average of 100, 200, 150 = 150
        assert abs(result.mean() - 150) < 1

    def test_pixel_median(self):
        """Test pixel-wise median."""
        from landmarkdiff.ensemble import EnsembleInference

        ensemble = EnsembleInference(mode="tps", n_samples=3)

        outputs = [
            np.full((64, 64, 3), 50, dtype=np.uint8),
            np.full((64, 64, 3), 200, dtype=np.uint8),
            np.full((64, 64, 3), 100, dtype=np.uint8),
        ]

        result = ensemble._pixel_median(outputs)
        assert result.shape == (64, 64, 3)
        # Median of 50, 200, 100 = 100
        assert abs(result.mean() - 100) < 1

    def test_weighted_average(self):
        """Test quality-weighted averaging."""
        from landmarkdiff.ensemble import EnsembleInference

        ensemble = EnsembleInference(mode="tps", n_samples=2)

        reference = np.full((64, 64, 3), 128, dtype=np.uint8)
        outputs = [
            np.full((64, 64, 3), 130, dtype=np.uint8),  # close to reference
            np.full((64, 64, 3), 200, dtype=np.uint8),  # far from reference
        ]

        result, scores = ensemble._weighted_average(outputs, reference)
        assert result.shape == (64, 64, 3)
        assert len(scores) == 2
        # The one closer to reference should have higher score
        assert scores[0] > scores[1]

    def test_best_of_n(self):
        """Test best-of-N selection."""
        from landmarkdiff.ensemble import EnsembleInference

        ensemble = EnsembleInference(mode="tps", n_samples=3)

        # Create outputs with varying quality
        reference = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
        outputs = [
            reference.copy(),  # identical to reference (best)
            np.zeros((64, 64, 3), dtype=np.uint8),  # black (worst)
            np.full((64, 64, 3), 128, dtype=np.uint8),  # gray (medium)
        ]

        result, scores, selected_idx = ensemble._best_of_n(outputs, reference)
        assert result.shape == (64, 64, 3)
        assert len(scores) == 3
        # The identical output should be selected
        assert selected_idx == 0

    def test_init_params(self):
        """Test initialization parameters."""
        from landmarkdiff.ensemble import EnsembleInference

        ensemble = EnsembleInference(
            mode="controlnet",
            n_samples=7,
            strategy="median",
            base_seed=123,
        )

        assert ensemble.mode == "controlnet"
        assert ensemble.n_samples == 7
        assert ensemble.strategy == "median"
        assert ensemble.base_seed == 123
        assert not ensemble.is_loaded

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        from landmarkdiff.ensemble import EnsembleInference

        EnsembleInference(mode="tps", strategy="invalid")
        # Strategy validation happens during generate(), not init


class TestSplitScript:
    """Tests for create_test_split.py."""

    def test_extract_source_id_basic(self):
        """Test source ID extraction."""
        from scripts.create_test_split import extract_source_id

        # No metadata: strip aug suffix
        assert extract_source_id("rhinoplasty_000001_aug3", {}) == "rhinoplasty_000001"
        assert extract_source_id("rhinoplasty_000001", {}) == "rhinoplasty_000001"

    def test_extract_source_id_with_meta(self):
        """Test source ID from metadata."""
        from scripts.create_test_split import extract_source_id

        meta = {
            "rhinoplasty_000001": {"source_image": "celeba_00123"},
            "rhinoplasty_000001_aug0": {
                "source": "augmented",
                "original_prefix": "rhinoplasty_000001",
            },
        }

        assert extract_source_id("rhinoplasty_000001", meta) == "celeba_00123"
        assert extract_source_id("rhinoplasty_000001_aug0", meta) == "rhinoplasty_000001"

    def test_split_dry_run(self, tmp_path):
        """Test split in dry-run mode."""
        import cv2
        from scripts.create_test_split import create_split

        # Create mock dataset
        for proc in ["rhinoplasty", "blepharoplasty"]:
            for i in range(20):
                prefix = f"{proc}_{i:06d}"
                img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{prefix}_input.png"), img)
                cv2.imwrite(str(tmp_path / f"{prefix}_target.png"), img)

        result = create_split(
            tmp_path,
            test_dir=tmp_path / "test",
            val_dir=None,
            test_fraction=0.1,
            dry_run=True,
        )

        assert result["total"] == 40
        assert result["test"] > 0
        # Files should not have been moved
        assert len(list(tmp_path.glob("*_input.png"))) == 40
