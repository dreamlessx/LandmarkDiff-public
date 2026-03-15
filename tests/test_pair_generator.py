"""Tests for landmarkdiff.synthetic.pair_generator."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.synthetic.pair_generator import (
    PROCEDURES,
    TrainingPair,
    generate_pair,
    save_pair,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face(size: int = 512) -> FaceLandmarks:
    """Create a dummy FaceLandmarks for testing."""
    rng = np.random.default_rng(0)
    landmarks = rng.uniform(0.2, 0.8, (478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=size,
        image_height=size,
        confidence=0.95,
    )


def _make_image(size: int = 512) -> np.ndarray:
    return np.random.default_rng(0).integers(50, 200, (size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TrainingPair dataclass
# ---------------------------------------------------------------------------


class TestTrainingPair:
    def test_fields(self):
        pair = TrainingPair(
            input_image=np.zeros((512, 512, 3), dtype=np.uint8),
            target_image=np.zeros((512, 512, 3), dtype=np.uint8),
            conditioning=np.zeros((512, 512, 3), dtype=np.uint8),
            canny=np.zeros((512, 512), dtype=np.uint8),
            mask=np.zeros((512, 512), dtype=np.float32),
            procedure="rhinoplasty",
            intensity=65.0,
        )
        assert pair.procedure == "rhinoplasty"
        assert pair.intensity == 65.0

    def test_frozen(self):
        pair = TrainingPair(
            input_image=np.zeros((10, 10, 3), dtype=np.uint8),
            target_image=np.zeros((10, 10, 3), dtype=np.uint8),
            conditioning=np.zeros((10, 10, 3), dtype=np.uint8),
            canny=np.zeros((10, 10), dtype=np.uint8),
            mask=np.zeros((10, 10), dtype=np.float32),
            procedure="test",
            intensity=50.0,
        )
        with pytest.raises(AttributeError):
            pair.procedure = "other"


class TestProcedures:
    def test_all_procedures_defined(self):
        assert len(PROCEDURES) == 10
        expected = {
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
            "alarplasty",
            "canthoplasty",
            "buccal_fat_removal",
            "dimpleplasty",
        }
        assert set(PROCEDURES) == expected


# ---------------------------------------------------------------------------
# generate_pair
# ---------------------------------------------------------------------------


class TestGeneratePair:
    @patch("landmarkdiff.synthetic.pair_generator.extract_landmarks")
    @patch("landmarkdiff.synthetic.pair_generator.render_landmark_image")
    @patch("landmarkdiff.synthetic.pair_generator.generate_conditioning")
    @patch("landmarkdiff.synthetic.pair_generator.generate_surgical_mask")
    @patch("landmarkdiff.synthetic.pair_generator.apply_procedure_preset")
    @patch("landmarkdiff.synthetic.pair_generator.warp_image_tps")
    @patch("landmarkdiff.synthetic.pair_generator.apply_clinical_augmentation")
    def test_returns_pair(
        self,
        mock_aug,
        mock_warp,
        mock_preset,
        mock_mask,
        mock_cond,
        mock_render,
        mock_extract,
    ):
        face = _make_face()
        mock_extract.return_value = face
        mock_preset.return_value = face
        mock_render.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cond.return_value = (
            np.zeros((512, 512, 3), dtype=np.uint8),
            np.zeros((512, 512), dtype=np.uint8),
            np.zeros((512, 512, 3), dtype=np.uint8),
        )
        mock_mask.return_value = np.zeros((512, 512), dtype=np.float32)
        mock_warp.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_aug.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        img = _make_image()
        pair = generate_pair(img, procedure="rhinoplasty", intensity=65.0)

        assert pair is not None
        assert isinstance(pair, TrainingPair)
        assert pair.procedure == "rhinoplasty"
        assert pair.intensity == 65.0

    @patch("landmarkdiff.synthetic.pair_generator.extract_landmarks")
    def test_returns_none_on_no_face(self, mock_extract):
        mock_extract.return_value = None
        img = _make_image()
        result = generate_pair(img)
        assert result is None

    @patch("landmarkdiff.synthetic.pair_generator.extract_landmarks")
    @patch("landmarkdiff.synthetic.pair_generator.render_landmark_image")
    @patch("landmarkdiff.synthetic.pair_generator.generate_conditioning")
    @patch("landmarkdiff.synthetic.pair_generator.generate_surgical_mask")
    @patch("landmarkdiff.synthetic.pair_generator.apply_procedure_preset")
    @patch("landmarkdiff.synthetic.pair_generator.warp_image_tps")
    @patch("landmarkdiff.synthetic.pair_generator.apply_clinical_augmentation")
    def test_random_procedure(
        self,
        mock_aug,
        mock_warp,
        mock_preset,
        mock_mask,
        mock_cond,
        mock_render,
        mock_extract,
    ):
        face = _make_face()
        mock_extract.return_value = face
        mock_preset.return_value = face
        mock_render.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_cond.return_value = (
            np.zeros((512, 512, 3), dtype=np.uint8),
            np.zeros((512, 512), dtype=np.uint8),
            np.zeros((512, 512, 3), dtype=np.uint8),
        )
        mock_mask.return_value = np.zeros((512, 512), dtype=np.float32)
        mock_warp.return_value = np.zeros((512, 512, 3), dtype=np.uint8)
        mock_aug.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        img = _make_image()
        pair = generate_pair(img, rng=np.random.default_rng(42))

        assert pair is not None
        assert pair.procedure in PROCEDURES
        assert 30.0 <= pair.intensity <= 90.0

    @patch("landmarkdiff.synthetic.pair_generator.extract_landmarks")
    @patch("landmarkdiff.synthetic.pair_generator.render_landmark_image")
    @patch("landmarkdiff.synthetic.pair_generator.generate_conditioning")
    @patch("landmarkdiff.synthetic.pair_generator.generate_surgical_mask")
    @patch("landmarkdiff.synthetic.pair_generator.apply_procedure_preset")
    @patch("landmarkdiff.synthetic.pair_generator.warp_image_tps")
    @patch("landmarkdiff.synthetic.pair_generator.apply_clinical_augmentation")
    def test_resizes_to_target(
        self,
        mock_aug,
        mock_warp,
        mock_preset,
        mock_mask,
        mock_cond,
        mock_render,
        mock_extract,
    ):
        face = _make_face(256)
        mock_extract.return_value = face
        mock_preset.return_value = face
        mock_render.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_cond.return_value = (
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256), dtype=np.uint8),
            np.zeros((256, 256, 3), dtype=np.uint8),
        )
        mock_mask.return_value = np.zeros((256, 256), dtype=np.float32)
        mock_warp.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_aug.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        # Input is 1024x1024, target_size=256
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        pair = generate_pair(img, procedure="rhinoplasty", intensity=50.0, target_size=256)
        assert pair is not None


# ---------------------------------------------------------------------------
# save_pair
# ---------------------------------------------------------------------------


class TestSavePair:
    def test_saves_all_files(self, tmp_path):
        pair = TrainingPair(
            input_image=np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            target_image=np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            conditioning=np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8),
            canny=np.random.default_rng(3).integers(0, 255, (64, 64), dtype=np.uint8),
            mask=np.random.default_rng(4).random((64, 64)).astype(np.float32),
            procedure="rhinoplasty",
            intensity=65.0,
        )
        save_pair(pair, tmp_path, index=42)

        expected_files = [
            "000042_input.png",
            "000042_target.png",
            "000042_conditioning.png",
            "000042_canny.png",
            "000042_mask.png",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing {fname}"

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "sub" / "dir"
        pair = TrainingPair(
            input_image=np.zeros((32, 32, 3), dtype=np.uint8),
            target_image=np.zeros((32, 32, 3), dtype=np.uint8),
            conditioning=np.zeros((32, 32, 3), dtype=np.uint8),
            canny=np.zeros((32, 32), dtype=np.uint8),
            mask=np.zeros((32, 32), dtype=np.float32),
            procedure="test",
            intensity=50.0,
        )
        save_pair(pair, out, index=0)
        assert out.exists()
        assert (out / "000000_input.png").exists()

    def test_mask_saved_as_uint8(self, tmp_path):
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        pair = TrainingPair(
            input_image=np.zeros((64, 64, 3), dtype=np.uint8),
            target_image=np.zeros((64, 64, 3), dtype=np.uint8),
            conditioning=np.zeros((64, 64, 3), dtype=np.uint8),
            canny=np.zeros((64, 64), dtype=np.uint8),
            mask=mask,
            procedure="test",
            intensity=50.0,
        )
        save_pair(pair, tmp_path, index=1)
        loaded = cv2.imread(str(tmp_path / "000001_mask.png"), cv2.IMREAD_GRAYSCALE)
        # mask * 255 = 127 or 128 (rounding)
        assert np.all(loaded > 100)
