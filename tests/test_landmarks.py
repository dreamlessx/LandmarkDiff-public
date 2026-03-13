"""Tests for landmark extraction and rendering."""

import numpy as np
import pytest

from landmarkdiff.landmarks import (
    FaceLandmarks,
    LANDMARK_REGIONS,
    render_landmark_image,
)


def _make_fake_face(n_landmarks: int = 478) -> FaceLandmarks:
    """Create synthetic landmarks for testing without MediaPipe."""
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, size=(n_landmarks, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=512,
        image_height=512,
        confidence=0.9,
    )


class TestFaceLandmarks:
    def test_pixel_coords_shape(self):
        face = _make_fake_face()
        coords = face.pixel_coords
        assert coords.shape == (478, 2)

    def test_pixel_coords_range(self):
        face = _make_fake_face()
        coords = face.pixel_coords
        assert np.all(coords >= 0)
        assert np.all(coords[:, 0] <= 512)
        assert np.all(coords[:, 1] <= 512)

    def test_get_region_returns_subset(self):
        face = _make_fake_face()
        nose = face.get_region("nose")
        assert len(nose) == len(LANDMARK_REGIONS["nose"])

    def test_immutability(self):
        face = _make_fake_face()
        with pytest.raises(AttributeError):
            face.landmarks = np.zeros((478, 3))


class TestRendering:
    def test_render_landmark_image_shape(self):
        face = _make_fake_face()
        img = render_landmark_image(face, 512, 512)
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8

    def test_render_has_nonzero_pixels(self):
        face = _make_fake_face()
        img = render_landmark_image(face, 512, 512)
        assert np.any(img > 0)

    def test_render_custom_size(self):
        face = _make_fake_face()
        img = render_landmark_image(face, 256, 256)
        assert img.shape == (256, 256, 3)
