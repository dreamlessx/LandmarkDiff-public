"""Tests for accessory detection and handling (glasses, hats, masks).

Verifies that the pipeline handles faces with accessories gracefully:
- Glasses detection via edge density
- Teeth mask generation from inner lip contour
- Combined accessory mask
- Deformation robustness with accessory-occluded landmarks
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from landmarkdiff.landmarks import (
    FaceLandmarks,
    detect_glasses_region,
    get_accessory_mask,
    get_teeth_mask,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask


@pytest.fixture
def mock_face():
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4
        landmarks[i, 1] = 0.2 + rng.random() * 0.6
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=256,
        image_height=256,
        confidence=0.95,
    )


def _make_face_image(size: int = 256, skin_color: int = 180) -> np.ndarray:
    """Create a synthetic face image."""
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), skin_color, dtype=np.uint8)
    noise = rng.integers(-5, 5, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_glasses_image(size: int = 256) -> np.ndarray:
    """Create an image with simulated glasses (strong edges in eye region)."""
    img = _make_face_image(size)
    # Draw thick black rectangles simulating glasses frames
    cv2.rectangle(img, (50, 70), (120, 110), (0, 0, 0), 3)
    cv2.rectangle(img, (136, 70), (206, 110), (0, 0, 0), 3)
    # Bridge
    cv2.line(img, (120, 90), (136, 90), (0, 0, 0), 3)
    return img


# ---------------------------------------------------------------------------
# Teeth mask
# ---------------------------------------------------------------------------


class TestTeethMask:
    def test_returns_correct_shape(self, mock_face):
        mask = get_teeth_mask(mock_face, (256, 256))
        assert mask.shape == (256, 256)

    def test_dtype_float32(self, mock_face):
        mask = get_teeth_mask(mock_face, (256, 256))
        assert mask.dtype == np.float32

    def test_values_in_range(self, mock_face):
        mask = get_teeth_mask(mock_face, (256, 256))
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_different_sizes(self, mock_face):
        for size in [(128, 128), (512, 512), (640, 480)]:
            mask = get_teeth_mask(mock_face, size)
            assert mask.shape == size


# ---------------------------------------------------------------------------
# Glasses detection
# ---------------------------------------------------------------------------


class TestGlassesDetection:
    def test_no_glasses_plain_image(self, mock_face):
        img = _make_face_image()
        result = detect_glasses_region(mock_face, img)
        assert isinstance(result, bool)

    def test_returns_bool(self, mock_face):
        img = _make_glasses_image()
        result = detect_glasses_region(mock_face, img)
        assert isinstance(result, bool)

    def test_threshold_parameter(self, mock_face):
        img = _make_face_image()
        # Very high threshold should not detect glasses
        result = detect_glasses_region(mock_face, img, threshold=1000.0)
        assert result is False


# ---------------------------------------------------------------------------
# Combined accessory mask
# ---------------------------------------------------------------------------


class TestAccessoryMask:
    def test_returns_correct_shape(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(mock_face, img)
        assert mask.shape == (256, 256)

    def test_dtype_float32(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(mock_face, img)
        assert mask.dtype == np.float32

    def test_values_in_range(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(mock_face, img)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_teeth_only(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(
            mock_face,
            img,
            include_glasses=False,
            include_teeth=True,
        )
        assert mask.shape == (256, 256)

    def test_glasses_only(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(
            mock_face,
            img,
            include_glasses=True,
            include_teeth=False,
        )
        assert mask.shape == (256, 256)

    def test_neither(self, mock_face):
        img = _make_face_image()
        mask = get_accessory_mask(
            mock_face,
            img,
            include_glasses=False,
            include_teeth=False,
        )
        assert mask.shape == (256, 256)
        assert mask.max() == 0.0  # no regions detected


# ---------------------------------------------------------------------------
# Deformation with accessory-like scenarios
# ---------------------------------------------------------------------------


PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "brow_lift",
    "mentoplasty",
]


class TestDeformationWithAccessories:
    @pytest.mark.parametrize("procedure", PROCEDURES)
    def test_deformation_with_glasses_occluded_face(self, procedure, mock_face):
        """Deformation should work even if glasses occlude eye landmarks."""
        result = apply_procedure_preset(mock_face, procedure, intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    @pytest.mark.parametrize("procedure", PROCEDURES)
    def test_mask_with_accessory_face(self, procedure, mock_face):
        mask = generate_surgical_mask(mock_face, procedure, width=256, height=256)
        assert mask.shape == (256, 256)
        assert np.all(np.isfinite(mask))

    def test_sequential_procedures_with_accessories(self, mock_face):
        """Multiple procedures on a face with accessories."""
        face = mock_face
        for proc in PROCEDURES:
            face = apply_procedure_preset(face, proc, intensity=30.0)
        assert np.all(np.isfinite(face.landmarks))
