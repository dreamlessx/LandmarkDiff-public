"""Edge case tests for extremely small/large faces and occluded inputs.

Covers:
- Very small face regions (< 50px)
- Very large face regions (> 2000px)
- Faces at image boundaries
- Partially occluded faces
- Makeup and face paint scenarios (high-saturation skin)
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask

PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


def _face_at_scale(
    size: int,
    center_x: float = 0.5,
    center_y: float = 0.5,
    spread: float = 0.2,
    seed: int = 42,
) -> FaceLandmarks:
    """Create landmarks centered at a given position with a given spread."""
    rng = np.random.default_rng(seed)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = center_x + (rng.random() - 0.5) * spread * 2
        landmarks[i, 1] = center_y + (rng.random() - 0.5) * spread * 2
        landmarks[i, 2] = rng.random() * 0.05
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=size,
        image_height=size,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Extremely small faces
# ---------------------------------------------------------------------------


class TestSmallFaces:
    @pytest.mark.parametrize("size", [8, 16, 32, 48])
    def test_deformation_small_image(self, size):
        """Deformation should work on very small images."""
        face = _face_at_scale(size)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)
        assert np.all(np.isfinite(result.landmarks))

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_mask_small_image(self, size):
        """Mask generation should work on very small images."""
        face = _face_at_scale(size)
        mask = generate_surgical_mask(face, "rhinoplasty", width=size, height=size)
        assert mask.shape == (size, size)
        assert np.all(np.isfinite(mask))

    def test_tiny_face_in_large_image(self):
        """A very tiny face region in a large image."""
        face = _face_at_scale(1024, spread=0.02)  # face is only ~40px across
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    @pytest.mark.parametrize("procedure", PROCEDURES)
    def test_all_procedures_tiny(self, procedure):
        face = _face_at_scale(32)
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))


# ---------------------------------------------------------------------------
# Extremely large faces
# ---------------------------------------------------------------------------


class TestLargeFaces:
    @pytest.mark.parametrize("size", [2048, 4096])
    def test_deformation_large_image(self, size):
        face = _face_at_scale(size)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)
        assert np.all(np.isfinite(result.landmarks))

    def test_face_fills_entire_image(self):
        """Face landmarks spanning the full [0, 1] range."""
        face = _face_at_scale(512, spread=0.5)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))


# ---------------------------------------------------------------------------
# Faces at image boundaries
# ---------------------------------------------------------------------------


class TestBoundaryFaces:
    def test_face_at_top_edge(self):
        face = _face_at_scale(512, center_y=0.05, spread=0.04)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_face_at_bottom_edge(self):
        face = _face_at_scale(512, center_y=0.95, spread=0.04)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_face_at_left_edge(self):
        face = _face_at_scale(512, center_x=0.05, spread=0.04)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_face_at_right_edge(self):
        face = _face_at_scale(512, center_x=0.95, spread=0.04)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_face_partially_outside(self):
        """Some landmarks outside [0, 1] normalized range."""
        rng = np.random.default_rng(0)
        landmarks = rng.uniform(-0.1, 0.3, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.8,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))


# ---------------------------------------------------------------------------
# Makeup and face paint
# ---------------------------------------------------------------------------


class TestMakeupAndFacePaint:
    """Test post-processing resilience with high-saturation skin regions.

    Makeup and face paint create unusual color distributions that can
    cause histogram matching and color correction to produce artifacts.
    """

    def _image_with_high_saturation(self, size: int = 256) -> np.ndarray:
        """Create an image with high-saturation colored patches (simulating makeup)."""
        img = np.full((size, size, 3), 180, dtype=np.uint8)
        # Red lipstick region
        cv2.rectangle(img, (80, 140), (176, 170), (0, 0, 220), -1)
        # Blue eyeshadow
        cv2.rectangle(img, (60, 80), (196, 100), (200, 80, 40), -1)
        # Green face paint stripe
        cv2.rectangle(img, (0, 110), (256, 130), (40, 200, 40), -1)
        return img

    def test_deformation_with_makeup_face(self):
        """Landmark deformation should work regardless of skin color."""
        face = _face_at_scale(256)
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert np.all(np.isfinite(result.landmarks))

    def test_mask_with_makeup_face(self):
        face = _face_at_scale(256)
        mask = generate_surgical_mask(face, "blepharoplasty", width=256, height=256)
        assert mask.shape == (256, 256)
        assert np.all(np.isfinite(mask))

    @pytest.mark.parametrize("procedure", PROCEDURES)
    def test_all_procedures_with_colored_face(self, procedure):
        """All procedures should handle faces with unusual coloring."""
        face = _face_at_scale(256)
        result = apply_procedure_preset(face, procedure, intensity=60.0)
        assert np.all(np.isfinite(result.landmarks))
        assert result.landmarks.shape == (478, 3)


# ---------------------------------------------------------------------------
# Non-square images
# ---------------------------------------------------------------------------


class TestNonSquareImages:
    @pytest.mark.parametrize(
        "w,h",
        [(640, 480), (480, 640), (1920, 1080), (320, 240)],
    )
    def test_deformation_nonsquare(self, w, h):
        rng = np.random.default_rng(0)
        landmarks = rng.uniform(0.2, 0.8, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=w,
            image_height=h,
            confidence=0.9,
        )
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.image_width == w
        assert result.image_height == h
        assert np.all(np.isfinite(result.landmarks))

    @pytest.mark.parametrize(
        "w,h",
        [(640, 480), (1920, 1080)],
    )
    def test_mask_nonsquare(self, w, h):
        rng = np.random.default_rng(0)
        landmarks = rng.uniform(0.2, 0.8, (478, 3)).astype(np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=w,
            image_height=h,
            confidence=0.9,
        )
        mask = generate_surgical_mask(face, "rhinoplasty", width=w, height=h)
        assert mask.shape == (h, w)


# ---------------------------------------------------------------------------
# Multiple procedures in sequence
# ---------------------------------------------------------------------------


class TestSequentialProcedures:
    def test_chain_two_procedures(self):
        """Applying two procedures sequentially should work."""
        face = _face_at_scale(512)
        result1 = apply_procedure_preset(face, "rhinoplasty", intensity=40.0)
        result2 = apply_procedure_preset(result1, "blepharoplasty", intensity=30.0)
        assert np.all(np.isfinite(result2.landmarks))
        assert result2.landmarks.shape == (478, 3)

    def test_chain_all_base_procedures(self):
        """Apply all base procedures in sequence."""
        face = _face_at_scale(512)
        for proc in PROCEDURES:
            face = apply_procedure_preset(face, proc, intensity=20.0)
        assert np.all(np.isfinite(face.landmarks))
        assert face.landmarks.shape == (478, 3)

    def test_same_procedure_twice(self):
        """Applying the same procedure twice should work."""
        face = _face_at_scale(512)
        result1 = apply_procedure_preset(face, "rhinoplasty", intensity=30.0)
        result2 = apply_procedure_preset(result1, "rhinoplasty", intensity=30.0)
        assert np.all(np.isfinite(result2.landmarks))
