"""Extended tests for facial landmark extraction and rendering.

Covers FaceLandmarks dataclass methods, visualize_landmarks, render_landmark_image
edge cases, load_image, region definitions, and pixel coordinate conversion.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from landmarkdiff.landmarks import (
    LANDMARK_REGIONS,
    REGION_COLORS,
    FaceLandmarks,
    load_image,
    render_landmark_image,
    visualize_landmarks,
)


def _make_face(
    n_landmarks: int = 478,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> FaceLandmarks:
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.1, 0.9, size=(n_landmarks, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# FaceLandmarks — pixel_coords
# ---------------------------------------------------------------------------


class TestPixelCoords:
    def test_shape(self):
        face = _make_face()
        coords = face.pixel_coords
        assert coords.shape == (478, 2)

    def test_scaling_x(self):
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[0] = [0.5, 0.25, 0.0]
        face = FaceLandmarks(landmarks=landmarks, image_width=640, image_height=480, confidence=1.0)
        coords = face.pixel_coords
        assert abs(coords[0, 0] - 320.0) < 0.01
        assert abs(coords[0, 1] - 120.0) < 0.01

    def test_origin_maps_to_zero(self):
        landmarks = np.zeros((478, 3), dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        coords = face.pixel_coords
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0])

    def test_one_maps_to_max_index(self):
        """Normalized 1.0 maps to width-1 / height-1 (clamped to valid index)."""
        landmarks = np.ones((478, 3), dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=640, image_height=480, confidence=1.0)
        coords = face.pixel_coords
        assert abs(coords[0, 0] - 639.0) < 0.01
        assert abs(coords[0, 1] - 479.0) < 0.01

    def test_is_copy_not_view(self):
        face = _make_face()
        coords = face.pixel_coords
        coords[0, 0] = -999.0
        # Original landmarks should be untouched
        assert face.landmarks[0, 0] != -999.0


# ---------------------------------------------------------------------------
# FaceLandmarks — get_region
# ---------------------------------------------------------------------------


class TestGetRegion:
    @pytest.mark.parametrize("region", list(LANDMARK_REGIONS.keys()))
    def test_valid_regions_return_correct_count(self, region):
        face = _make_face()
        result = face.get_region(region)
        assert len(result) == len(LANDMARK_REGIONS[region])

    def test_unknown_region_returns_empty(self):
        face = _make_face()
        result = face.get_region("nonexistent_region")
        assert len(result) == 0

    def test_returned_values_match_landmarks(self):
        face = _make_face()
        nose = face.get_region("nose")
        indices = LANDMARK_REGIONS["nose"]
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(nose[i], face.landmarks[idx])


# ---------------------------------------------------------------------------
# FaceLandmarks — immutability
# ---------------------------------------------------------------------------


class TestFaceLandmarksImmutability:
    def test_frozen_dataclass(self):
        face = _make_face()
        with pytest.raises(AttributeError):
            face.image_width = 1024

    def test_confidence_frozen(self):
        face = _make_face()
        with pytest.raises(AttributeError):
            face.confidence = 0.5


# ---------------------------------------------------------------------------
# Region data validation
# ---------------------------------------------------------------------------


class TestRegionData:
    def test_all_region_indices_in_range(self):
        for region, indices in LANDMARK_REGIONS.items():
            for idx in indices:
                assert 0 <= idx < 478, f"{region}: index {idx} out of range"

    def test_region_colors_cover_all_regions(self):
        for region in LANDMARK_REGIONS:
            assert region in REGION_COLORS, f"Missing color for region: {region}"

    def test_colors_are_bgr_tuples(self):
        for region, color in REGION_COLORS.items():
            assert len(color) == 3, f"{region}: color must be (B, G, R) tuple"
            for c in color:
                assert 0 <= c <= 255, f"{region}: color values must be 0-255"

    def test_iris_landmarks_are_highest_indices(self):
        left_iris = LANDMARK_REGIONS["iris_left"]
        right_iris = LANDMARK_REGIONS["iris_right"]
        assert min(left_iris) >= 468
        assert min(right_iris) >= 473


# ---------------------------------------------------------------------------
# visualize_landmarks
# ---------------------------------------------------------------------------


class TestVisualizeLandmarks:
    def test_output_shape_matches_input(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = visualize_landmarks(img, face)
        assert result.shape == img.shape

    def test_does_not_modify_input(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        original = img.copy()
        visualize_landmarks(img, face)
        np.testing.assert_array_equal(img, original)

    def test_draws_nonzero_pixels(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = visualize_landmarks(img, face)
        assert np.any(result > 0)

    def test_draw_regions_false_all_white(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = visualize_landmarks(img, face, draw_regions=False)
        nonzero = result[result > 0]
        assert len(nonzero) > 0
        # All drawn pixels should be white (255)
        assert np.all(nonzero == 255)

    def test_draw_regions_true_has_color(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = visualize_landmarks(img, face, draw_regions=True)
        # Should have more than one unique non-zero color
        nonzero_mask = np.any(result > 0, axis=2)
        if np.sum(nonzero_mask) > 0:
            colors = result[nonzero_mask]
            unique_colors = set(map(tuple, colors))
            assert len(unique_colors) > 1

    def test_custom_radius(self):
        face = _make_face()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        small = visualize_landmarks(img, face, radius=1)
        large = visualize_landmarks(img, face, radius=5)
        assert np.sum(large > 0) > np.sum(small > 0)


# ---------------------------------------------------------------------------
# render_landmark_image
# ---------------------------------------------------------------------------


class TestRenderLandmarkImage:
    def test_default_dimensions_from_face(self):
        face = _make_face(width=384, height=256)
        img = render_landmark_image(face)
        assert img.shape == (256, 384, 3)

    def test_custom_dimensions(self):
        face = _make_face()
        img = render_landmark_image(face, 256, 256)
        assert img.shape == (256, 256, 3)

    def test_black_background(self):
        landmarks = np.full((478, 3), 0.5, dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=64, image_height=64, confidence=1.0)
        img = render_landmark_image(face, 64, 64)
        # Corners should be black
        assert np.all(img[0, 0] == 0)
        assert np.all(img[0, -1] == 0)

    def test_dtype_is_uint8(self):
        face = _make_face()
        img = render_landmark_image(face)
        assert img.dtype == np.uint8

    def test_deterministic(self):
        face = _make_face(seed=99)
        img1 = render_landmark_image(face, 512, 512)
        img2 = render_landmark_image(face, 512, 512)
        np.testing.assert_array_equal(img1, img2)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_loads_existing_image(self, tmp_path):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(str(path))
        assert loaded.shape == (64, 64, 3)
        assert loaded.dtype == np.uint8

    def test_loads_pathlib_path(self, tmp_path):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        path = tmp_path / "test2.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(path)
        assert loaded.shape == (32, 32, 3)

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="Could not load image"):
            load_image("/nonexistent/path/to/image.png")

    def test_round_trip_fidelity(self, tmp_path):
        """Written and loaded image should match (lossless PNG)."""
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "roundtrip.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(str(path))
        np.testing.assert_array_equal(loaded, img)

    def test_jpeg_loads(self, tmp_path):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        loaded = load_image(str(path))
        assert loaded.shape == (64, 64, 3)
