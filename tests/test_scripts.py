"""Tests for utility scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBatchInference:
    """Tests for batch_inference.py."""

    def test_collect_images(self, tmp_path):
        """Test image collection from directory."""
        from scripts.batch_inference import collect_images

        # Create test images
        for i in range(5):
            cv2.imwrite(
                str(tmp_path / f"img_{i:03d}.png"),
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            )
        # Create non-image file
        (tmp_path / "readme.txt").write_text("not an image")

        images = collect_images(tmp_path)
        assert len(images) == 5
        assert all(p.suffix == ".png" for p in images)

    def test_collect_images_empty(self, tmp_path):
        """Test empty directory."""
        from scripts.batch_inference import collect_images

        images = collect_images(tmp_path)
        assert len(images) == 0

    def test_create_intensity_grid_no_face(self, tmp_path):
        """Test grid creation with no-face image."""
        from scripts.batch_inference import create_intensity_grid

        # Create a solid color image (no face)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img_path = tmp_path / "no_face.png"
        cv2.imwrite(str(img_path), img)

        out_path = tmp_path / "grid.png"
        result = create_intensity_grid(
            img_path,
            "rhinoplasty",
            [25, 50, 75],
            out_path,
        )
        assert result is False


class TestComputeFID:
    """Tests for compute_fid.py."""

    def test_collect_target_images(self, tmp_path):
        """Test target image collection."""
        from scripts.compute_fid import collect_target_images

        # Create *_target.png files
        for i in range(3):
            cv2.imwrite(
                str(tmp_path / f"sample_{i:03d}_target.png"),
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            )

        targets = collect_target_images(tmp_path)
        assert len(targets) == 3

    def test_collect_output_images(self, tmp_path):
        """Test output image collection (inference format)."""
        from scripts.compute_fid import collect_target_images

        for i in range(4):
            cv2.imwrite(
                str(tmp_path / f"sample_{i:03d}_output.png"),
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            )

        targets = collect_target_images(tmp_path)
        assert len(targets) == 4

    def test_prepare_fid_dir(self, tmp_path):
        """Test FID directory preparation with resizing."""
        from scripts.compute_fid import prepare_fid_dir

        images = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(p), np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            images.append(p)

        out_dir = tmp_path / "fid_prep"
        prepare_fid_dir(images, out_dir, size=299)

        prepped = list(out_dir.glob("*.png"))
        assert len(prepped) == 3
        # Check size
        img = cv2.imread(str(prepped[0]))
        assert img.shape[:2] == (299, 299)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestClinicalDemo:
    """Tests for clinical_demo.py."""

    def test_create_header_panel(self):
        """Test header panel creation."""
        from scripts.clinical_demo import create_header_panel

        panel = create_header_panel("Test Header", 512, 60)
        assert panel.shape == (60, 512, 3)
        assert panel.dtype == np.uint8

    def test_create_label_panel(self):
        """Test label panel creation."""
        from scripts.clinical_demo import create_label_panel

        panel = create_label_panel("Test Label", 256, 30)
        assert panel.shape == (30, 256, 3)

    def test_procedure_descriptions(self):
        """Test all procedures have descriptions."""
        from scripts.clinical_demo import PROCEDURE_DESCRIPTIONS, PROCEDURES

        for proc in PROCEDURES:
            assert proc in PROCEDURE_DESCRIPTIONS
            assert len(PROCEDURE_DESCRIPTIONS[proc]) > 0


class TestAugmentationPreview:
    """Tests for augmentation_preview.py."""

    def test_load_training_sample(self, tmp_path):
        """Test loading a training sample."""
        from scripts.augmentation_preview import load_training_sample

        # Create mock training sample
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "test_000_input.png"), img)
        cv2.imwrite(str(tmp_path / "test_000_target.png"), img)
        cv2.imwrite(str(tmp_path / "test_000_conditioning.png"), img)
        mask = np.ones((512, 512), dtype=np.uint8) * 255
        cv2.imwrite(str(tmp_path / "test_000_mask.png"), mask)

        sample = load_training_sample(tmp_path, 0)
        assert sample is not None
        assert "input_image" in sample
        assert "target_image" in sample
        assert "conditioning" in sample
        assert "mask" in sample
        assert sample["prefix"] == "test_000"

    def test_load_training_sample_missing(self, tmp_path):
        """Test loading from empty directory."""
        from scripts.augmentation_preview import load_training_sample

        sample = load_training_sample(tmp_path, 0)
        assert sample is None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCompareModels:
    """Tests for compare_models.py."""

    def test_paired_ttest(self):
        """Test paired t-test."""
        from scripts.compare_models import paired_ttest

        # Identical distributions: not significant
        a = [0.9, 0.85, 0.88, 0.92, 0.87]
        result = paired_ttest(a, a)
        assert not result["significant"]
        assert result["p_value"] == 1.0

    def test_paired_ttest_different(self):
        """Test paired t-test with different distributions."""
        from scripts.compare_models import paired_ttest

        a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.89, 0.86, 0.93, 0.88]
        b = [0.5, 0.45, 0.48, 0.52, 0.47, 0.51, 0.49, 0.46, 0.53, 0.48]
        result = paired_ttest(a, b)
        assert result["significant"]
        assert result["p_value"] < 0.001

    def test_paired_ttest_insufficient_data(self):
        """Test paired t-test with too few samples."""
        from scripts.compare_models import paired_ttest

        result = paired_ttest([0.9], [0.5])
        assert not result["significant"]
        assert result["p_value"] == 1.0


class TestSubmitJob:
    """Tests for submit_job.py."""

    def test_check_data_exists(self, tmp_path):
        """Test data existence check."""
        from scripts.submit_job import check_data_exists

        # Empty directory
        assert not check_data_exists(str(tmp_path / "nonexistent"))

    def test_check_checkpoint_exists(self, tmp_path):
        """Test checkpoint existence check."""
        from scripts.submit_job import check_checkpoint_exists

        (tmp_path / "test_ckpt").mkdir()
        assert check_checkpoint_exists(str(tmp_path / "test_ckpt"))
        assert not check_checkpoint_exists(str(tmp_path / "nonexistent"))
