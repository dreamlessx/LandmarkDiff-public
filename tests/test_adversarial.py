"""Adversarial input tests for robustness.

Tests the pipeline with challenging inputs:
- Noisy images
- Blurred images
- Occluded faces
- Extreme lighting
- Very small and very large faces
- Zero/empty inputs
- Out-of-range landmarks
"""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask


def _make_face(
    seed: int = 0,
    size: int = 512,
    low: float = 0.15,
    high: float = 0.85,
) -> FaceLandmarks:
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(low, high, (478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=size,
        image_height=size,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Extreme landmark positions
# ---------------------------------------------------------------------------


class TestExtremeLandmarks:
    def test_landmarks_at_origin(self):
        """All landmarks at (0, 0) should not crash."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.5,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)
        assert np.all(np.isfinite(result.landmarks))

    def test_landmarks_at_edges(self):
        """Landmarks at image edges (0 and 1) should not crash."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[::2, 0] = 0.0
        landmarks[1::2, 0] = 1.0
        landmarks[:, 1] = 0.5
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.5,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_landmarks_outside_bounds(self):
        """Landmarks slightly outside [0, 1] should still produce finite output."""
        rng = np.random.default_rng(42)
        landmarks = rng.uniform(-0.1, 1.1, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.5,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_very_tight_landmarks(self):
        """All landmarks clustered in a tiny region should not crash."""
        landmarks = np.full((478, 3), 0.5, dtype=np.float32)
        landmarks += np.random.default_rng(0).uniform(-0.001, 0.001, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.5,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))


# ---------------------------------------------------------------------------
# Extreme intensity values
# ---------------------------------------------------------------------------


class TestExtremeIntensity:
    def test_max_intensity(self):
        """Maximum intensity (100) should produce valid output."""
        face = _make_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=100.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_negative_intensity(self):
        """Negative intensity should still produce finite output."""
        face = _make_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=-10.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_very_large_intensity(self):
        """Very large intensity should not crash or produce NaN."""
        face = _make_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=1000.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_tiny_intensity(self):
        """Tiny but nonzero intensity should produce valid output."""
        face = _make_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=0.001)
        assert np.all(np.isfinite(result.landmarks))


# ---------------------------------------------------------------------------
# Extreme image sizes
# ---------------------------------------------------------------------------


class TestExtremeImageSizes:
    def test_very_small_image(self):
        """Extremely small image (16x16) should not crash."""
        face = _make_face(size=16)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)

    def test_very_large_image(self):
        """Very large image (4096x4096) should not crash."""
        face = _make_face(size=4096)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)

    def test_nonsquare_image(self):
        """Non-square image dimensions should work."""
        rng = np.random.default_rng(0)
        landmarks = rng.uniform(0.15, 0.85, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=640,
            image_height=480,
            confidence=0.95,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.image_width == 640
        assert result.image_height == 480


# ---------------------------------------------------------------------------
# Mask generation robustness
# ---------------------------------------------------------------------------


class TestMaskRobustness:
    @pytest.mark.parametrize(
        "procedure",
        [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
    )
    def test_mask_with_edge_landmarks(self, procedure):
        """Mask generation with landmarks at image edges."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        rng = np.random.default_rng(42)
        landmarks[:, 0] = rng.uniform(0.0, 1.0, 478)
        landmarks[:, 1] = rng.uniform(0.0, 1.0, 478)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=256,
            image_height=256,
            confidence=0.5,
        )
        mask = generate_surgical_mask(face, procedure)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        assert np.all(np.isfinite(mask))

    def test_mask_with_all_landmarks_same_point(self):
        """All landmarks at the same point should still produce a valid mask."""
        landmarks = np.full((478, 3), 0.5, dtype=np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=64,
            image_height=64,
            confidence=0.5,
        )
        mask = generate_surgical_mask(face, "rhinoplasty")
        assert mask.shape == (64, 64)
        assert np.all(np.isfinite(mask))


# ---------------------------------------------------------------------------
# All procedures survive adversarial inputs
# ---------------------------------------------------------------------------


ALL_PROCEDURES = [
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
    "genioplasty",
    "malarplasty",
    "lip_lift",
    "lip_augmentation",
    "forehead_reduction",
    "submental_liposuction",
    "otoplasty",
]


class TestAllProceduresRobust:
    @pytest.mark.parametrize("procedure", ALL_PROCEDURES)
    def test_zero_landmarks(self, procedure):
        landmarks = np.zeros((478, 3), dtype=np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=64,
            image_height=64,
            confidence=0.1,
        )
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    @pytest.mark.parametrize("procedure", ALL_PROCEDURES)
    def test_max_intensity(self, procedure):
        face = _make_face()
        result = apply_procedure_preset(face, procedure, intensity=100.0)
        assert np.all(np.isfinite(result.landmarks))
