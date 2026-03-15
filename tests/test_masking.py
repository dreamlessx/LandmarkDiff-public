"""Tests for surgical mask generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.masking import (
    MASK_CONFIG,
    generate_surgical_mask,
    mask_to_3channel,
)


@pytest.fixture
def mock_face():
    """Create a mock FaceLandmarks object for testing."""
    # Generate plausible landmarks in a face-like arrangement
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Place landmarks roughly in center of image
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4  # x: 0.3-0.7
        landmarks[i, 1] = 0.2 + rng.random() * 0.6  # y: 0.2-0.8
        landmarks[i, 2] = rng.random() * 0.1  # z: small depth

    face = FaceLandmarks(
        landmarks=landmarks,
        confidence=0.95,
        image_width=512,
        image_height=512,
    )
    return face


class TestMaskConfig:
    """Tests for mask configuration."""

    def test_all_procedures_defined(self):
        expected = [
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
        ]
        assert set(expected) == set(MASK_CONFIG.keys())
        for proc in expected:
            assert "landmark_indices" in MASK_CONFIG[proc]
            assert "dilation_px" in MASK_CONFIG[proc]
            assert "feather_sigma" in MASK_CONFIG[proc]

    def test_indices_in_range(self):
        for proc, cfg in MASK_CONFIG.items():
            for idx in cfg["landmark_indices"]:
                assert 0 <= idx <= 477, f"{proc}: index {idx} out of 0-477 range"

    def test_each_procedure_has_nonempty_indices(self):
        for proc, cfg in MASK_CONFIG.items():
            assert len(cfg["landmark_indices"]) > 0, f"{proc}: empty landmark_indices"

    def test_dilation_and_sigma_positive(self):
        for proc, cfg in MASK_CONFIG.items():
            assert cfg["dilation_px"] > 0, f"{proc}: dilation_px must be positive"
            assert cfg["feather_sigma"] > 0, f"{proc}: feather_sigma must be positive"


class TestGenerateSurgicalMask:
    """Tests for generate_surgical_mask function."""

    def test_rhinoplasty_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "rhinoplasty")
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_blepharoplasty_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "blepharoplasty")
        assert mask.shape == (512, 512)
        assert mask.max() > 0  # should have non-zero region

    def test_rhytidectomy_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "rhytidectomy")
        assert mask.shape == (512, 512)

    def test_orthognathic_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "orthognathic")
        assert mask.shape == (512, 512)

    def test_brow_lift_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "brow_lift")
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.max() > 0  # non-empty mask

    def test_mentoplasty_mask(self, mock_face):
        mask = generate_surgical_mask(mock_face, "mentoplasty")
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.max() > 0  # non-empty mask

    @pytest.mark.parametrize("procedure", list(MASK_CONFIG.keys()))
    def test_all_procedures_generate_valid_mask(self, mock_face, procedure):
        mask = generate_surgical_mask(mock_face, procedure)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min()
        assert mask.max() <= 1.0
        assert mask.max() > 0

    def test_custom_dimensions(self, mock_face):
        mask = generate_surgical_mask(mock_face, "rhinoplasty", width=256, height=256)
        assert mask.shape == (256, 256)

    def test_unknown_procedure_raises(self, mock_face):
        with pytest.raises(ValueError, match="Unknown procedure"):
            generate_surgical_mask(mock_face, "nonexistent_procedure")

    def test_feathered_edges(self, mock_face):
        mask = generate_surgical_mask(mock_face, "rhinoplasty")
        # Should have intermediate values (feathered, not binary)
        unique_vals = len(np.unique(np.round(mask, 2)))
        assert unique_vals > 3  # more than just 0 and 1

    def test_different_procedures_differ(self, mock_face):
        m1 = generate_surgical_mask(mock_face, "rhinoplasty")
        m2 = generate_surgical_mask(mock_face, "blepharoplasty")
        assert not np.array_equal(m1, m2)


class TestMaskTo3Channel:
    """Tests for mask_to_3channel helper."""

    def test_shape(self):
        mask = np.random.rand(64, 64).astype(np.float32)
        result = mask_to_3channel(mask)
        assert result.shape == (64, 64, 3)

    def test_channels_identical(self):
        mask = np.random.rand(64, 64).astype(np.float32)
        result = mask_to_3channel(mask)
        np.testing.assert_array_equal(result[:, :, 0], mask)
        np.testing.assert_array_equal(result[:, :, 1], mask)
        np.testing.assert_array_equal(result[:, :, 2], mask)
