"""Integration tests for the full LandmarkDiff pipeline.

Tests that all components work together end-to-end:
1. Face detection → landmarks → conditioning → manipulation → warping → composite
2. Dataset loading → batch creation → loss computation
3. Ablation experiment loading and evaluation
4. Paper figure generation
5. Metadata generation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_face_image():
    """Create a synthetic face-like image for testing."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Simple face approximation
    cv2.ellipse(img, (256, 256), (150, 200), 0, 0, 360, (180, 150, 120), -1)
    cv2.circle(img, (200, 220), 20, (60, 60, 60), -1)  # left eye
    cv2.circle(img, (312, 220), 20, (60, 60, 60), -1)  # right eye
    cv2.ellipse(img, (256, 280), (20, 30), 0, 0, 360, (140, 120, 100), -1)  # nose
    cv2.ellipse(img, (256, 340), (50, 15), 0, 0, 360, (120, 80, 80), -1)  # mouth
    return img


@pytest.fixture
def mock_dataset_dir(tmp_path, sample_face_image):
    """Create a mock dataset directory."""
    for i in range(5):
        for proc in ["rhinoplasty", "blepharoplasty"]:
            prefix = f"{proc}_{i:04d}"
            # Input: the face
            cv2.imwrite(str(tmp_path / f"{prefix}_input.png"), sample_face_image)
            # Target: slightly different
            target = sample_face_image.copy()
            target = cv2.GaussianBlur(target, (5, 5), 1)
            cv2.imwrite(str(tmp_path / f"{prefix}_target.png"), target)
            # Conditioning: wireframe-like
            cond = np.zeros_like(sample_face_image)
            cv2.ellipse(cond, (256, 256), (150, 200), 0, 0, 360, (255, 255, 255), 1)
            cv2.imwrite(str(tmp_path / f"{prefix}_conditioning.png"), cond)
    return tmp_path


class TestEndToEndPipeline:
    """Test that core modules work together."""

    def test_landmarks_to_conditioning(self, sample_face_image):
        """Landmark extraction feeds into conditioning generation."""
        from landmarkdiff.conditioning import generate_conditioning
        from landmarkdiff.landmarks import extract_landmarks

        face = extract_landmarks(sample_face_image)
        if face is None:
            pytest.skip("No face detected in synthetic image")

        landmark_img, canny, wireframe = generate_conditioning(face, 512, 512)
        assert landmark_img.shape == (512, 512, 3)
        assert canny.shape == (512, 512)
        assert wireframe.shape == (512, 512)

    def test_landmarks_to_manipulation(self, sample_face_image):
        """Landmark extraction feeds into procedure manipulation."""
        from landmarkdiff.landmarks import extract_landmarks
        from landmarkdiff.manipulation import apply_procedure_preset

        face = extract_landmarks(sample_face_image)
        if face is None:
            pytest.skip("No face detected")

        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            manip = apply_procedure_preset(face, proc, 65.0, image_size=512)
            assert manip is not None
            assert manip.pixel_coords.shape == face.pixel_coords.shape

    def test_full_tps_pipeline(self, sample_face_image):
        """Full TPS baseline: detect → manipulate → mask → warp → composite."""
        from landmarkdiff.inference import mask_composite
        from landmarkdiff.landmarks import extract_landmarks
        from landmarkdiff.manipulation import apply_procedure_preset
        from landmarkdiff.masking import generate_surgical_mask
        from landmarkdiff.synthetic.tps_warp import warp_image_tps

        face = extract_landmarks(sample_face_image)
        if face is None:
            pytest.skip("No face detected")

        manip = apply_procedure_preset(face, "rhinoplasty", 65.0, image_size=512)
        mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        warped = warp_image_tps(sample_face_image, face.pixel_coords, manip.pixel_coords)
        composite = mask_composite(warped, sample_face_image, mask)

        assert composite.shape == (512, 512, 3)
        assert composite.dtype == np.uint8


class TestDatasetPipeline:
    """Test dataset loading and batch creation."""

    def test_dataset_loads(self, mock_dataset_dir):
        """SyntheticPairDataset loads pairs from directory."""
        from scripts.train_controlnet import SyntheticPairDataset

        ds = SyntheticPairDataset(str(mock_dataset_dir), resolution=256, geometric_augment=False)
        assert len(ds) == 10  # 5 rhinoplasty + 5 blepharoplasty
        sample = ds[0]
        assert "input" in sample
        assert "target" in sample
        assert "conditioning" in sample
        assert "mask" in sample
        assert "idx" in sample
        assert sample["input"].shape == (3, 256, 256)

    def test_dataset_returns_valid_tensors(self, mock_dataset_dir):
        """Dataset returns properly normalized tensors."""
        from scripts.train_controlnet import SyntheticPairDataset

        ds = SyntheticPairDataset(str(mock_dataset_dir), resolution=256, geometric_augment=False)
        sample = ds[0]
        assert sample["input"].min() >= 0.0
        assert sample["input"].max() <= 1.0
        assert sample["mask"].min() >= 0.0
        assert sample["mask"].max() <= 1.0

    def test_dataloader_batches(self, mock_dataset_dir):
        """DataLoader can create batches from dataset."""
        from torch.utils.data import DataLoader

        from scripts.train_controlnet import SyntheticPairDataset

        ds = SyntheticPairDataset(str(mock_dataset_dir), resolution=256, geometric_augment=False)
        dl = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(dl))
        assert batch["input"].shape == (2, 3, 256, 256)
        assert batch["target"].shape == (2, 3, 256, 256)
        assert batch["idx"].shape == (2,)


class TestLossPipeline:
    """Test loss computation pipeline."""

    def test_diffusion_loss(self):
        """Diffusion loss computes on random tensors."""
        from landmarkdiff.losses import DiffusionLoss

        loss_fn = DiffusionLoss()
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_combined_loss_phase_a(self):
        """Phase A combined loss uses only diffusion."""
        from landmarkdiff.losses import CombinedLoss

        loss_fn = CombinedLoss(phase="A")
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)
        losses = loss_fn(pred, target)
        assert "diffusion" in losses
        assert "total" in losses
        assert "landmark" not in losses
        assert "identity" not in losses

    def test_landmark_loss(self):
        """Landmark loss computes normalized distance."""
        from landmarkdiff.losses import LandmarkLoss

        loss_fn = LandmarkLoss()
        pred = torch.randn(2, 478, 2)
        target = pred + 0.01  # small displacement
        iod = torch.tensor([100.0, 100.0])
        loss = loss_fn(pred, target, iod=iod)
        assert loss.item() > 0
        assert loss.item() < 1.0  # should be small for small displacement


class TestMetadataPipeline:
    """Test metadata generation pipeline."""

    def test_metadata_generation(self, mock_dataset_dir):
        """Metadata generator produces valid JSON."""
        from scripts.generate_metadata import generate_metadata

        out = str(mock_dataset_dir / "metadata.json")
        generate_metadata(str(mock_dataset_dir), output_path=out)

        assert Path(out).exists()
        with open(out) as f:
            data = json.load(f)
        assert "pairs" in data
        assert "total_pairs" in data
        assert data["total_pairs"] == 10

    def test_metadata_enables_curriculum(self, mock_dataset_dir):
        """Generated metadata enables curriculum learning in dataset."""
        from scripts.generate_metadata import generate_metadata

        generate_metadata(
            str(mock_dataset_dir), output_path=str(mock_dataset_dir / "metadata.json")
        )

        from scripts.train_controlnet import SyntheticPairDataset

        ds = SyntheticPairDataset(str(mock_dataset_dir), resolution=256, geometric_augment=False)
        # Should have loaded procedure info from metadata
        proc = ds.get_procedure(0)
        assert proc in ["rhinoplasty", "blepharoplasty", "unknown"]


class TestEvaluationMetrics:
    """Test that evaluation metrics produce reasonable values."""

    def test_ssim_identical_images(self):
        """SSIM of identical images should be ~1.0."""
        from landmarkdiff.evaluation import compute_ssim

        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim > 0.99

    def test_ssim_different_images(self):
        """SSIM of very different images should be low."""
        from landmarkdiff.evaluation import compute_ssim

        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.full((256, 256, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(img1, img2)
        assert ssim < 0.1

    def test_nme_zero_displacement(self):
        """NME with zero displacement should be 0."""
        from landmarkdiff.evaluation import compute_nme

        landmarks = np.random.rand(478, 2) * 512
        nme = compute_nme(landmarks, landmarks)
        assert nme < 1e-10

    def test_nme_small_displacement(self):
        """NME with small displacement should be small."""
        from landmarkdiff.evaluation import compute_nme

        landmarks = np.random.rand(478, 2) * 512
        displaced = landmarks + np.random.randn(478, 2) * 2  # 2px noise
        nme = compute_nme(landmarks, displaced)
        assert nme > 0
        assert nme < 0.1  # should be small relative to IOD
