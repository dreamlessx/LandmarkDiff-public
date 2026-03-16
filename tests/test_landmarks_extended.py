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
# FaceLandmarks: pixel_coords
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
# FaceLandmarks: get_region
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
# FaceLandmarks: immutability
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
# FaceLandmarks: face_rotation
# ---------------------------------------------------------------------------


class TestFaceRotation:
    def test_upright_face_near_zero(self):
        """Horizontally aligned eyes should give ~0 degrees rotation."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        # Left eye outer corner (33) and right eye outer corner (263) at same y
        landmarks[33] = [0.3, 0.4, 0.0]
        landmarks[263] = [0.7, 0.4, 0.0]
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        assert abs(face.face_rotation) < 1.0

    def test_tilted_face_positive(self):
        """Right eye lower than left eye gives positive rotation."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[33] = [0.3, 0.4, 0.0]
        landmarks[263] = [0.7, 0.5, 0.0]  # right eye lower
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        assert face.face_rotation > 5.0

    def test_tilted_face_negative(self):
        """Right eye higher than left eye gives negative rotation."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[33] = [0.3, 0.5, 0.0]  # left eye lower
        landmarks[263] = [0.7, 0.4, 0.0]
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        assert face.face_rotation < -5.0

    def test_returns_float(self):
        face = _make_face()
        assert isinstance(face.face_rotation, float)


# ---------------------------------------------------------------------------
# FaceLandmarks: face_bbox
# ---------------------------------------------------------------------------


class TestFaceBbox:
    def test_returns_four_ints(self):
        face = _make_face()
        bbox = face.face_bbox
        assert len(bbox) == 4
        for val in bbox:
            assert isinstance(val, int)

    def test_bbox_within_image_bounds(self):
        face = _make_face()
        x_min, y_min, x_max, y_max = face.face_bbox
        assert x_min >= 0
        assert y_min >= 0
        assert x_max < face.image_width
        assert y_max < face.image_height

    def test_bbox_contains_all_landmarks(self):
        face = _make_face()
        coords = face.pixel_coords
        x_min, y_min, x_max, y_max = face.face_bbox
        assert np.all(coords[:, 0] >= x_min)
        assert np.all(coords[:, 1] >= y_min)
        assert np.all(coords[:, 0] <= x_max)
        assert np.all(coords[:, 1] <= y_max)

    def test_rotated_face_gets_more_padding(self):
        """Rotated faces should get wider bounding boxes."""
        # Upright face
        lm_up = np.full((478, 3), 0.5, dtype=np.float32)
        lm_up[:, 0] = np.linspace(0.3, 0.7, 478)
        lm_up[:, 1] = np.linspace(0.2, 0.8, 478)
        lm_up[33] = [0.3, 0.4, 0.0]
        lm_up[263] = [0.7, 0.4, 0.0]  # eyes level
        face_up = FaceLandmarks(
            landmarks=lm_up,
            image_width=512,
            image_height=512,
            confidence=1.0,
        )

        # Rotated face (same landmarks, eyes tilted 30+ degrees)
        lm_rot = lm_up.copy()
        lm_rot[33] = [0.3, 0.3, 0.0]
        lm_rot[263] = [0.7, 0.6, 0.0]  # strong tilt
        face_rot = FaceLandmarks(
            landmarks=lm_rot,
            image_width=512,
            image_height=512,
            confidence=1.0,
        )

        bbox_up = face_up.face_bbox
        bbox_rot = face_rot.face_bbox
        area_up = (bbox_up[2] - bbox_up[0]) * (bbox_up[3] - bbox_up[1])
        area_rot = (bbox_rot[2] - bbox_rot[0]) * (bbox_rot[3] - bbox_rot[1])
        # Rotated face bbox should be larger due to extra padding
        assert area_rot > area_up


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


# ---------------------------------------------------------------------------
# FaceLandmarks: face_yaw
# ---------------------------------------------------------------------------


def _make_frontal_face() -> FaceLandmarks:
    """Create a face with symmetric eye/nose positions (frontal)."""
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[:, 0] = 0.5
    lm[:, 1] = 0.5
    # Symmetric eyes
    lm[33] = [0.35, 0.4, 0.0]  # left eye outer
    lm[263] = [0.65, 0.4, 0.0]  # right eye outer
    lm[1] = [0.5, 0.55, 0.0]  # nose tip centered
    lm[152] = [0.5, 0.85, 0.0]  # chin
    lm[10] = [0.5, 0.15, 0.0]  # forehead
    return FaceLandmarks(landmarks=lm, confidence=0.95, image_width=512, image_height=512)


def _make_profile_face() -> FaceLandmarks:
    """Create a face turned to the left (viewer's right)."""
    lm = np.zeros((478, 3), dtype=np.float32)
    lm[:, 0] = 0.5
    lm[:, 1] = 0.5
    # Left eye close to nose, right eye far (face turned left)
    lm[33] = [0.48, 0.4, 0.0]  # left eye (near nose)
    lm[263] = [0.7, 0.4, 0.0]  # right eye (far from nose)
    lm[1] = [0.45, 0.55, 0.0]  # nose tip shifted left
    lm[152] = [0.5, 0.85, 0.0]
    lm[10] = [0.5, 0.15, 0.0]
    return FaceLandmarks(landmarks=lm, confidence=0.95, image_width=512, image_height=512)


class TestFaceYaw:
    def test_frontal_face_near_zero(self):
        face = _make_frontal_face()
        yaw = face.face_yaw
        assert isinstance(yaw, float)
        assert abs(yaw) < 15.0

    def test_profile_face_nonzero(self):
        face = _make_profile_face()
        yaw = face.face_yaw
        assert abs(yaw) > 5.0

    def test_returns_float(self):
        face = _make_face()
        assert isinstance(face.face_yaw, float)


# ---------------------------------------------------------------------------
# FaceLandmarks: face_view
# ---------------------------------------------------------------------------


class TestFaceView:
    def test_frontal(self):
        face = _make_frontal_face()
        assert face.face_view == "frontal"

    def test_profile_not_frontal(self):
        face = _make_profile_face()
        assert face.face_view != "frontal"

    def test_valid_values(self):
        face = _make_face()
        assert face.face_view in {
            "frontal",
            "three_quarter_left",
            "three_quarter_right",
            "profile_left",
            "profile_right",
        }


# ---------------------------------------------------------------------------
# FaceLandmarks: visible_side
# ---------------------------------------------------------------------------


class TestVisibleSide:
    def test_frontal_both(self):
        face = _make_frontal_face()
        assert face.visible_side == "both"

    def test_profile_not_both(self):
        face = _make_profile_face()
        side = face.visible_side
        assert side in {"left", "right"}

    def test_returns_string(self):
        face = _make_face()
        assert isinstance(face.visible_side, str)


# ---------------------------------------------------------------------------
# get_teeth_mask
# ---------------------------------------------------------------------------


class TestGetTeethMask:
    def test_returns_correct_shape(self):
        from landmarkdiff.landmarks import get_teeth_mask

        face = _make_face()
        mask = get_teeth_mask(face, (256, 256))
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32

    def test_mask_has_some_ones(self):
        from landmarkdiff.landmarks import get_teeth_mask

        face = _make_face()
        mask = get_teeth_mask(face, (512, 512))
        assert mask.max() > 0, "Mask should have some non-zero region"

    def test_mask_bounded(self):
        from landmarkdiff.landmarks import get_teeth_mask

        face = _make_face()
        mask = get_teeth_mask(face, (256, 256))
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


# ---------------------------------------------------------------------------
# detect_glasses_region
# ---------------------------------------------------------------------------


class TestDetectGlassesRegion:
    def test_plain_image_no_glasses(self):
        from landmarkdiff.landmarks import detect_glasses_region

        face = _make_face()
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        result = detect_glasses_region(face, img)
        assert isinstance(result, bool)

    def test_edgy_image_may_detect_glasses(self):
        from landmarkdiff.landmarks import detect_glasses_region

        face = _make_face()
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        result = detect_glasses_region(face, img)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# get_accessory_mask
# ---------------------------------------------------------------------------


class TestGetAccessoryMask:
    def test_returns_correct_shape(self):
        from landmarkdiff.landmarks import get_accessory_mask

        face = _make_face()
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        mask = get_accessory_mask(face, img)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32

    def test_teeth_only(self):
        from landmarkdiff.landmarks import get_accessory_mask

        face = _make_face()
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        mask = get_accessory_mask(face, img, include_glasses=False, include_teeth=True)
        assert mask.max() > 0, "Should have teeth region"

    def test_no_accessories(self):
        from landmarkdiff.landmarks import get_accessory_mask

        face = _make_face()
        img = np.full((512, 512, 3), 128, dtype=np.uint8)
        mask = get_accessory_mask(face, img, include_glasses=False, include_teeth=False)
        assert mask.max() == 0.0, "No accessories should yield zero mask"


# ---------------------------------------------------------------------------
# select_largest_face
# ---------------------------------------------------------------------------


class TestSelectLargestFace:
    def test_empty_list_returns_none(self):
        from landmarkdiff.landmarks import select_largest_face

        assert select_largest_face([]) is None

    def test_single_face_returns_it(self):
        from landmarkdiff.landmarks import select_largest_face

        face = _make_face()
        assert select_largest_face([face]) is face

    def test_selects_largest(self):
        from landmarkdiff.landmarks import select_largest_face

        # Small face: landmarks in narrow range
        small_lm = np.zeros((478, 3), dtype=np.float32)
        small_lm[:, 0] = np.linspace(0.4, 0.5, 478)
        small_lm[:, 1] = np.linspace(0.4, 0.5, 478)
        small = FaceLandmarks(landmarks=small_lm, confidence=0.9, image_width=512, image_height=512)

        # Large face: landmarks in wide range
        large_lm = np.zeros((478, 3), dtype=np.float32)
        large_lm[:, 0] = np.linspace(0.1, 0.9, 478)
        large_lm[:, 1] = np.linspace(0.1, 0.9, 478)
        large = FaceLandmarks(landmarks=large_lm, confidence=0.9, image_width=512, image_height=512)

        result = select_largest_face([small, large])
        assert result is large
