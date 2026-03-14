"""Tests for clinical edge case handling."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.clinical import (
    ClinicalFlags,
    adjust_mask_for_vitiligo,
    get_bells_palsy_side_indices,
    adjust_mask_for_keloid,
)


class TestClinicalFlags:
    """Tests for ClinicalFlags dataclass."""

    def test_default_no_conditions(self):
        flags = ClinicalFlags()
        assert not flags.has_any()
        assert not flags.vitiligo
        assert not flags.bells_palsy
        assert not flags.keloid_prone
        assert not flags.ehlers_danlos

    def test_single_condition(self):
        flags = ClinicalFlags(vitiligo=True)
        assert flags.has_any()
        assert flags.vitiligo

    def test_multiple_conditions(self):
        flags = ClinicalFlags(vitiligo=True, keloid_prone=True)
        assert flags.has_any()

    def test_bells_palsy_side(self):
        flags = ClinicalFlags(bells_palsy=True, bells_palsy_side="right")
        assert flags.bells_palsy
        assert flags.bells_palsy_side == "right"

    def test_keloid_regions(self):
        flags = ClinicalFlags(keloid_prone=True, keloid_regions=["jawline", "nose"])
        assert len(flags.keloid_regions) == 2
        assert "jawline" in flags.keloid_regions


class TestVitiligoMaskAdjustment:
    """Tests for vitiligo-aware mask adjustment."""

    def test_no_patches_no_change(self):
        mask = np.ones((64, 64), dtype=np.float32) * 0.8
        no_patches = np.zeros((64, 64), dtype=np.uint8)
        result = adjust_mask_for_vitiligo(mask, no_patches)
        np.testing.assert_array_equal(result, mask)

    def test_patches_reduce_mask(self):
        mask = np.ones((64, 64), dtype=np.float32) * 0.8
        patches = np.zeros((64, 64), dtype=np.uint8)
        patches[20:40, 20:40] = 255  # patch region
        result = adjust_mask_for_vitiligo(mask, patches, preservation_factor=0.5)
        # Mask should be reduced in patch area
        assert result[30, 30] < mask[30, 30]
        # Mask should be unchanged outside
        assert result[0, 0] == mask[0, 0]

    def test_full_preservation(self):
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        patches = np.full((64, 64), 255, dtype=np.uint8)
        result = adjust_mask_for_vitiligo(mask, patches, preservation_factor=1.0)
        # With factor=1, mask should drop to 0 where patches exist
        assert result.max() <= 0.0 + 1e-6


class TestBellsPalsySideIndices:
    """Tests for Bell's palsy side-specific indices."""

    def test_left_side(self):
        indices = get_bells_palsy_side_indices("left")
        assert "eye" in indices
        assert "eyebrow" in indices
        assert "mouth_corner" in indices
        assert "jawline" in indices
        assert len(indices["eye"]) > 0

    def test_right_side(self):
        indices = get_bells_palsy_side_indices("right")
        assert "eye" in indices
        assert len(indices["eye"]) > 0

    def test_sides_different(self):
        left = get_bells_palsy_side_indices("left")
        right = get_bells_palsy_side_indices("right")
        # Left and right should have different indices
        assert set(left["eye"]) != set(right["eye"])


class TestKeloidMaskAdjustment:
    """Tests for keloid-aware mask adjustment."""

    def test_no_keloid_region(self):
        mask = np.ones((64, 64), dtype=np.float32) * 0.8
        keloid_mask = np.zeros((64, 64), dtype=np.float32)
        result = adjust_mask_for_keloid(mask, keloid_mask)
        # No keloid region => output should approximate input
        np.testing.assert_allclose(result, mask, atol=0.05)

    def test_keloid_reduces_mask(self):
        mask = np.ones((128, 128), dtype=np.float32) * 0.9
        keloid_mask = np.zeros((128, 128), dtype=np.float32)
        keloid_mask[40:80, 40:80] = 1.0
        result = adjust_mask_for_keloid(mask, keloid_mask, reduction_factor=0.5)
        # Keloid region should have reduced mask
        assert result[60, 60] < mask[60, 60]

    def test_output_range(self):
        mask = np.random.rand(128, 128).astype(np.float32)
        keloid_mask = np.random.rand(128, 128).astype(np.float32)
        result = adjust_mask_for_keloid(mask, keloid_mask)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
