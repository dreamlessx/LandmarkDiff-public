"""Tests for landmark extraction and rendering."""

import numpy as np
import pytest

from landmarkdiff.landmarks import (
    LANDMARK_REGIONS,
    FaceLandmarks,
    extract_tps_landmarks,
    render_landmark_image,
)


def _make_fake_face(
    n_landmarks: int = 478,
    width: int = 512,
    height: int = 512,
) -> FaceLandmarks:
    """Create synthetic landmarks for testing without MediaPipe."""
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, size=(n_landmarks, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
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


class TestExtractTPSLandmarks:
    def test_valid_face_returns_stable_contract(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        face = _make_fake_face(width=320, height=240)

        result = extract_tps_landmarks(
            image,
            extractor=lambda *_args, **_kwargs: face,
        )

        assert result.detected is True
        assert result.reason is None
        assert result.coords.shape == (478, 3)
        assert result.coords.dtype == np.float32
        assert result.image_size == (320, 240)
        np.testing.assert_array_equal(result.coords, face.landmarks)

    def test_no_face_detected_returns_explicit_result(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)

        result = extract_tps_landmarks(
            image,
            extractor=lambda *_args, **_kwargs: None,
        )

        assert result.detected is False
        assert result.reason == "no_face_detected"
        assert result.coords.shape == (0, 3)
        assert result.coords.dtype == np.float32
        assert result.confidence == 0.0
        assert result.image_size == (100, 200)

    def test_invalid_image_returns_controlled_result(self):
        invalid = np.array([], dtype=np.uint8)
        result = extract_tps_landmarks(invalid)

        assert result.detected is False
        assert result.reason == "invalid_image"
        assert result.coords.shape == (0, 3)
        assert result.image_size == (0, 0)

    def test_parity_with_legacy_face_payload(self):
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        legacy_face = _make_fake_face(width=128, height=96)

        def fake_legacy_extractor(img, min_det, min_track):
            assert img.shape == image.shape
            assert min_det == 0.6
            assert min_track == 0.7
            return legacy_face

        result = extract_tps_landmarks(
            image,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7,
            extractor=fake_legacy_extractor,
        )

        assert result.detected is True
        np.testing.assert_allclose(result.coords, legacy_face.landmarks, atol=0.0)
        assert result.confidence == legacy_face.confidence
        assert result.image_size == (legacy_face.image_width, legacy_face.image_height)

        reconstructed = result.to_face_landmarks()
        assert reconstructed is not None
        np.testing.assert_array_equal(reconstructed.landmarks, legacy_face.landmarks)
