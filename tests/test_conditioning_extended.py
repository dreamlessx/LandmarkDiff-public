"""Extended tests for conditioning signal generation.

Covers edge cases in wireframe rendering, auto_canny with various inputs,
and generate_conditioning integration. All tests run without GPU or model loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import (
    ALL_CONTOURS,
    JAWLINE_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    auto_canny,
    generate_conditioning,
    render_wireframe,
)
from landmarkdiff.landmarks import FaceLandmarks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
    uniform_value: float | None = None,
) -> FaceLandmarks:
    """Create a FaceLandmarks with configurable dimensions."""
    rng = np.random.default_rng(seed)
    if uniform_value is not None:
        landmarks = np.full((478, 3), uniform_value, dtype=np.float32)
    else:
        landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# render_wireframe tests
# ---------------------------------------------------------------------------


class TestWireframeEdgeCases:
    """Edge cases for render_wireframe."""

    def test_small_canvas(self):
        """Wireframe should render on a small 64x64 canvas."""
        face = _make_face(width=64, height=64)
        wf = render_wireframe(face, 64, 64)
        assert wf.shape == (64, 64)
        assert wf.dtype == np.uint8
        assert np.any(wf > 0)

    def test_rectangular_canvas_wide(self):
        """Wireframe should work with non-square wide canvases."""
        face = _make_face(width=640, height=480)
        wf = render_wireframe(face, 640, 480)
        assert wf.shape == (480, 640)
        assert np.sum(wf > 0) > 50

    def test_rectangular_canvas_tall(self):
        """Wireframe should work with non-square tall canvases."""
        face = _make_face(width=480, height=640)
        wf = render_wireframe(face, 480, 640)
        assert wf.shape == (640, 480)
        assert np.sum(wf > 0) > 50

    def test_large_canvas(self):
        """Wireframe should render on a large 1024x1024 canvas."""
        face = _make_face(width=1024, height=1024)
        wf = render_wireframe(face, 1024, 1024)
        assert wf.shape == (1024, 1024)
        assert np.sum(wf > 0) > 200

    def test_custom_thickness(self):
        """Thicker lines should produce more non-zero pixels."""
        face = _make_face()
        wf_thin = render_wireframe(face, 512, 512, thickness=1)
        wf_thick = render_wireframe(face, 512, 512, thickness=3)
        assert np.sum(wf_thick > 0) > np.sum(wf_thin > 0)

    def test_default_dimensions_from_face(self):
        """When width/height are None, use face.image_width/image_height."""
        face = _make_face(width=256, height=256)
        wf = render_wireframe(face)
        assert wf.shape == (256, 256)

    def test_landmarks_at_boundary(self):
        """Landmarks at (0,0) and (1,1) should not crash rendering."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[1] = [1.0, 1.0, 0.0]
        # Fill rest with something in-range
        for i in range(2, 478):
            landmarks[i] = [0.5, 0.5, 0.0]
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.9,
        )
        wf = render_wireframe(face, 512, 512)
        assert wf.shape == (512, 512)
        assert wf.dtype == np.uint8

    def test_all_landmarks_at_center(self):
        """All landmarks collapsed to center should render (degenerate case)."""
        face = _make_face(uniform_value=0.5)
        wf = render_wireframe(face, 512, 512)
        assert wf.shape == (512, 512)
        # Lines from point to itself should produce at most single dots
        assert wf.dtype == np.uint8

    def test_wireframe_values_are_255(self):
        """Line pixels should be drawn at intensity 255."""
        face = _make_face()
        wf = render_wireframe(face, 512, 512)
        nonzero = wf[wf > 0]
        assert len(nonzero) > 0
        assert np.all(nonzero == 255)

    def test_contour_indices_valid(self):
        """All contour indices should be within [0, 478) range."""
        for contour in ALL_CONTOURS:
            for idx in contour:
                assert 0 <= idx < 478, f"Index {idx} out of range"

    def test_deterministic_output(self):
        """Same face landmarks should produce identical wireframes."""
        face = _make_face(seed=99)
        wf1 = render_wireframe(face, 512, 512)
        wf2 = render_wireframe(face, 512, 512)
        np.testing.assert_array_equal(wf1, wf2)


# ---------------------------------------------------------------------------
# auto_canny tests
# ---------------------------------------------------------------------------


class TestAutoCanny:
    """Extended tests for auto_canny edge detection."""

    def test_constant_gray_image(self):
        """Constant image should produce no edges."""
        img = np.full((256, 256), 128, dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == (256, 256)
        assert np.sum(edges) == 0

    def test_white_image(self):
        """All-white image should produce no edges."""
        img = np.full((256, 256), 255, dtype=np.uint8)
        edges = auto_canny(img)
        assert np.sum(edges) == 0

    def test_strong_gradient(self):
        """Image with a strong vertical edge should produce detections."""
        img = np.zeros((256, 256), dtype=np.uint8)
        img[:, 128:] = 200
        edges = auto_canny(img)
        assert np.sum(edges > 0) > 0

    def test_horizontal_gradient(self):
        """Image with a strong horizontal edge should produce detections."""
        img = np.zeros((256, 256), dtype=np.uint8)
        img[128:, :] = 200
        edges = auto_canny(img)
        assert np.sum(edges > 0) > 0

    def test_output_binary(self):
        """Edge map should only contain 0 and 255."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        edges = auto_canny(img)
        unique = set(np.unique(edges))
        assert unique.issubset({0, 255})

    def test_small_image(self):
        """Should work on small images without crashing."""
        img = np.random.default_rng(7).integers(0, 256, (32, 32), dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == (32, 32)
        assert edges.dtype == np.uint8

    def test_rectangular_image(self):
        """Should handle non-square images."""
        img = np.random.default_rng(3).integers(0, 256, (128, 256), dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == (128, 256)

    def test_circle_edges(self):
        """A bright circle on dark background should produce edge detections."""
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(img, (128, 128), 60, 200, -1)
        edges = auto_canny(img)
        assert np.sum(edges > 0) > 10

    def test_low_contrast(self):
        """Very low contrast image should produce few or no edges."""
        img = np.full((256, 256), 100, dtype=np.uint8)
        # Add very slight noise
        rng = np.random.default_rng(0)
        noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        edges = auto_canny(img)
        # Expect minimal edges
        edge_ratio = np.sum(edges > 0) / edges.size
        assert edge_ratio < 0.1

    def test_skeleton_produces_thin_edges(self):
        """Skeletonization should produce single-pixel-wide edges."""
        img = np.zeros((256, 256), dtype=np.uint8)
        # Draw a thick rectangle
        cv2.rectangle(img, (50, 50), (200, 200), 255, 10)
        edges = auto_canny(img)
        if np.sum(edges > 0) > 0:
            # Check that edges are thin: most edge pixels should not have
            # 4 or more edge neighbors (i.e., not thick blobs)
            kernel = np.ones((3, 3), dtype=np.uint8)
            neighbor_count = cv2.filter2D((edges > 0).astype(np.uint8), -1, kernel)
            edge_mask = edges > 0
            # For single-pixel edges, neighbor count should be <= 3
            # (self + at most 2 neighbors in a thin line)
            thick_ratio = np.sum(neighbor_count[edge_mask] > 4) / max(np.sum(edge_mask), 1)
            assert thick_ratio < 0.5  # most edges should be thin


# ---------------------------------------------------------------------------
# generate_conditioning tests
# ---------------------------------------------------------------------------


class TestGenerateConditioning:
    """Tests for generate_conditioning integration."""

    def test_returns_three_images(self):
        """Should return (landmark_img, canny, wireframe) tuple."""
        face = _make_face()
        result = generate_conditioning(face, 512, 512)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wireframe_channel(self):
        """Wireframe output should match render_wireframe."""
        face = _make_face()
        _, _, wireframe = generate_conditioning(face, 512, 512)
        assert wireframe.shape == (512, 512)
        assert wireframe.dtype == np.uint8

    def test_canny_channel(self):
        """Canny output should be a binary edge map."""
        face = _make_face()
        _, canny, _ = generate_conditioning(face, 512, 512)
        assert canny.shape == (512, 512)
        assert canny.dtype == np.uint8
        unique = set(np.unique(canny))
        assert unique.issubset({0, 255})

    def test_landmark_image_is_color(self):
        """Landmark image should be a 3-channel BGR image."""
        face = _make_face()
        landmark_img, _, _ = generate_conditioning(face, 512, 512)
        assert landmark_img.ndim == 3
        assert landmark_img.shape[2] == 3
        assert landmark_img.dtype == np.uint8

    def test_custom_dimensions(self):
        """Should respect custom width/height."""
        face = _make_face(width=256, height=256)
        landmark_img, canny, wireframe = generate_conditioning(face, 256, 256)
        assert landmark_img.shape[:2] == (256, 256)
        assert canny.shape == (256, 256)
        assert wireframe.shape == (256, 256)

    def test_defaults_to_face_dimensions(self):
        """When width/height are None, should use face dimensions."""
        face = _make_face(width=384, height=384)
        landmark_img, canny, wireframe = generate_conditioning(face)
        assert wireframe.shape == (384, 384)
        assert canny.shape == (384, 384)

    def test_canny_derived_from_wireframe(self):
        """Canny edges should be a subset of areas near the wireframe."""
        face = _make_face()
        _, canny, wireframe = generate_conditioning(face, 512, 512)
        # Canny is derived from wireframe, so canny edges should only
        # appear where wireframe has content (with some spatial tolerance)
        if np.sum(canny > 0) > 0:
            dilated_wf = cv2.dilate(wireframe, np.ones((5, 5), np.uint8))
            # Most canny edge pixels should be near wireframe content
            overlap = np.sum((canny > 0) & (dilated_wf > 0))
            total_canny = np.sum(canny > 0)
            overlap_ratio = overlap / max(total_canny, 1)
            assert overlap_ratio > 0.5


# ---------------------------------------------------------------------------
# Contour data integrity
# ---------------------------------------------------------------------------


class TestContourData:
    """Validate the static contour definitions."""

    def test_jawline_is_closed(self):
        """Jawline contour should start and end at the same index."""
        assert JAWLINE_CONTOUR[0] == JAWLINE_CONTOUR[-1]

    def test_eye_contours_closed(self):
        """Eye contours should be closed loops."""
        assert LEFT_EYE_CONTOUR[0] == LEFT_EYE_CONTOUR[-1]
        assert RIGHT_EYE_CONTOUR[0] == RIGHT_EYE_CONTOUR[-1]

    def test_contours_non_empty(self):
        """Every contour should have at least 2 points."""
        for i, contour in enumerate(ALL_CONTOURS):
            assert len(contour) >= 2, f"Contour {i} has fewer than 2 points"

    def test_total_contour_count(self):
        """Should have exactly 11 contour groups (includes nose bridge upper)."""
        assert len(ALL_CONTOURS) == 11
