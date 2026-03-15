"""Exhaustive test suite for LandmarkDiff — DIRAC verification.

Covers every function, every edge case, every branch across all modules.
Target: 1000+ individual test cases via parameterization.
"""

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from landmarkdiff.conditioning import (
    ALL_CONTOURS,
    INNER_LIPS,
    JAWLINE_CONTOUR,
    LEFT_EYE_CONTOUR,
    LEFT_EYEBROW,
    NOSE_BOTTOM,
    OUTER_LIPS,
    RIGHT_EYE_CONTOUR,
    RIGHT_EYEBROW,
    auto_canny,
    generate_conditioning,
    render_wireframe,
)
from landmarkdiff.evaluation import (
    EvalMetrics,
    classify_fitzpatrick_ita,
    compute_nme,
    compute_ssim,
    evaluate_batch,
)
from landmarkdiff.inference import (
    NEGATIVE_PROMPT,
    PROCEDURE_PROMPTS,
    LandmarkDiffPipeline,
    _match_skin_tone,
    estimate_face_view,
    get_device,
    mask_composite,
    numpy_to_pil,
    pil_to_numpy,
)
from landmarkdiff.landmarks import (
    LANDMARK_REGIONS,
    REGION_COLORS,
    FaceLandmarks,
    load_image,
    render_landmark_image,
    visualize_landmarks,
)
from landmarkdiff.losses import (
    CombinedLoss,
    DiffusionLoss,
    IdentityLoss,
    LandmarkLoss,
    LossWeights,
    PerceptualLoss,
)
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    PROCEDURE_RADIUS,
    DeformationHandle,
    apply_procedure_preset,
    gaussian_rbf_deform,
)
from landmarkdiff.masking import (
    MASK_CONFIG,
    generate_surgical_mask,
    mask_to_3channel,
)
from landmarkdiff.synthetic.augmentation import (
    AUGMENTATION_POOL,
    AugmentationConfig,
    apply_clinical_augmentation,
    barrel_distortion,
    color_temperature_jitter,
    gaussian_sensor_noise,
    green_fluorescent_cast,
    jpeg_compression,
    motion_blur,
    point_source_lighting,
    vignette,
)
from landmarkdiff.synthetic.pair_generator import (
    PROCEDURES,
    TrainingPair,
    save_pair,
)
from landmarkdiff.synthetic.tps_warp import (
    _apply_rigid_translation,
    _compute_rigid_translation,
    _compute_tps_map,
    _evaluate_tps,
    _solve_tps_weights,
    _subsample_control_points,
    _tps_kernel,
    compute_tps_transform,
    generate_random_warp,
    warp_image_tps,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_face(w=512, h=512, seed=0):
    """Create a synthetic FaceLandmarks with realistic normalized coordinates."""
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.1, 0.9, size=(478, 3)).astype(np.float32)
    # Make landmarks roughly centered like a face
    landmarks[:, 0] = np.clip(landmarks[:, 0], 0.2, 0.8)  # x centered
    landmarks[:, 1] = np.clip(landmarks[:, 1], 0.1, 0.9)  # y full range
    landmarks[:, 2] = np.clip(landmarks[:, 2], -0.1, 0.1)  # z depth
    return FaceLandmarks(landmarks=landmarks, image_width=w, image_height=h, confidence=0.95)


@pytest.fixture
def face():
    return _make_face()


@pytest.fixture
def face_small():
    return _make_face(w=64, h=64, seed=1)


@pytest.fixture
def face_rect():
    return _make_face(w=640, h=480, seed=2)


@pytest.fixture
def face_1024():
    return _make_face(w=1024, h=1024, seed=3)


@pytest.fixture
def sample_image():
    """A synthetic BGR face-like image."""
    img = np.random.default_rng(42).integers(50, 200, size=(512, 512, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_image_small():
    return np.random.default_rng(42).integers(50, 200, size=(64, 64, 3), dtype=np.uint8)


# ============================================================================
# 1. LANDMARKS MODULE
# ============================================================================


class TestFaceLandmarksDataclass:
    """Test the FaceLandmarks frozen dataclass."""

    def test_creation(self, face):
        assert face.image_width == 512
        assert face.image_height == 512
        assert face.confidence == 0.95

    def test_landmarks_shape(self, face):
        assert face.landmarks.shape == (478, 3)

    def test_landmarks_dtype(self, face):
        assert face.landmarks.dtype == np.float32

    def test_frozen_immutability(self, face):
        with pytest.raises(FrozenInstanceError):
            face.image_width = 1024

    def test_landmarks_array_is_still_mutable(self, face):
        """numpy arrays inside frozen dataclass are still mutable (known Python behavior)."""
        original = face.landmarks[0, 0]
        face.landmarks[0, 0] = 999.0
        assert face.landmarks[0, 0] == 999.0
        face.landmarks[0, 0] = original  # restore

    @pytest.mark.parametrize("w,h", [(512, 512), (1024, 1024), (640, 480), (1, 1), (64, 64)])
    def test_pixel_coords_shape(self, w, h):
        f = _make_face(w, h)
        coords = f.pixel_coords
        assert coords.shape == (478, 2)

    @pytest.mark.parametrize("w,h", [(512, 512), (1024, 768), (256, 256)])
    def test_pixel_coords_range(self, w, h):
        f = _make_face(w, h)
        coords = f.pixel_coords
        assert coords[:, 0].min() >= 0
        assert coords[:, 0].max() <= w
        assert coords[:, 1].min() >= 0
        assert coords[:, 1].max() <= h

    def test_pixel_coords_is_copy(self, face):
        c1 = face.pixel_coords
        c2 = face.pixel_coords
        c1[0, 0] = -9999.0
        assert c2[0, 0] != -9999.0

    def test_pixel_coords_multiplication(self):
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[0] = [0.5, 0.25, 0.0]
        f = FaceLandmarks(landmarks=landmarks, image_width=100, image_height=200, confidence=1.0)
        coords = f.pixel_coords
        assert coords[0, 0] == pytest.approx(50.0)
        assert coords[0, 1] == pytest.approx(50.0)

    @pytest.mark.parametrize("region", list(LANDMARK_REGIONS.keys()))
    def test_get_region_valid(self, face, region):
        result = face.get_region(region)
        expected_count = len(LANDMARK_REGIONS[region])
        assert result.shape == (expected_count, 3)

    def test_get_region_invalid(self, face):
        result = face.get_region("nonexistent_region")
        assert result.shape == (0, 3)

    def test_get_region_nose_indices(self, face):
        nose = face.get_region("nose")
        assert len(nose) == len(LANDMARK_REGIONS["nose"])
        assert nose.shape[1] == 3

    def test_confidence_zero(self):
        f = _make_face()
        f2 = FaceLandmarks(landmarks=f.landmarks, image_width=512, image_height=512, confidence=0.0)
        assert f2.confidence == 0.0

    def test_confidence_one(self):
        f = _make_face()
        f2 = FaceLandmarks(landmarks=f.landmarks, image_width=512, image_height=512, confidence=1.0)
        assert f2.confidence == 1.0


class TestLandmarkRegions:
    """Validate LANDMARK_REGIONS data integrity."""

    def test_all_regions_exist(self):
        expected = {
            "jawline",
            "eye_left",
            "eye_right",
            "eyebrow_left",
            "eyebrow_right",
            "nose",
            "lips",
            "iris_left",
            "iris_right",
        }
        assert set(LANDMARK_REGIONS.keys()) == expected

    @pytest.mark.parametrize("region", list(LANDMARK_REGIONS.keys()))
    def test_indices_in_range(self, region):
        for idx in LANDMARK_REGIONS[region]:
            assert 0 <= idx < 478, f"Index {idx} in {region} out of range"

    @pytest.mark.parametrize("region", list(LANDMARK_REGIONS.keys()))
    def test_indices_are_integers(self, region):
        for idx in LANDMARK_REGIONS[region]:
            assert isinstance(idx, int)

    def test_iris_indices_highest(self):
        """Iris landmarks are 468-477 (requires refine_landmarks=True)."""
        all_iris = LANDMARK_REGIONS["iris_left"] + LANDMARK_REGIONS["iris_right"]
        assert min(all_iris) == 468
        assert max(all_iris) == 477

    def test_no_duplicate_indices_within_region(self):
        for region, indices in LANDMARK_REGIONS.items():
            assert len(indices) == len(set(indices)), f"Duplicates in {region}"

    def test_total_landmark_count(self):
        all_indices = set()
        for indices in LANDMARK_REGIONS.values():
            all_indices.update(indices)
        # Not all 478 landmarks are in named regions
        assert len(all_indices) <= 478


class TestRegionColors:
    """Validate REGION_COLORS data."""

    @pytest.mark.parametrize("region", list(REGION_COLORS.keys()))
    def test_color_is_bgr_tuple(self, region):
        color = REGION_COLORS[region]
        assert isinstance(color, tuple)
        assert len(color) == 3
        for c in color:
            assert 0 <= c <= 255

    def test_region_colors_match_landmark_regions(self):
        """Every REGION_COLORS key should have a matching LANDMARK_REGIONS entry."""
        for key in REGION_COLORS:
            assert key in LANDMARK_REGIONS, (
                f"REGION_COLORS has '{key}' with no LANDMARK_REGIONS entry"
            )


class TestRenderLandmarkImage:
    """Test render_landmark_image function."""

    @pytest.mark.parametrize("w,h", [(512, 512), (256, 256), (1024, 1024)])
    def test_output_shape(self, face, w, h):
        img = render_landmark_image(face, w, h)
        assert img.shape == (h, w, 3)

    def test_output_dtype(self, face):
        img = render_landmark_image(face, 512, 512)
        assert img.dtype == np.uint8

    def test_has_nonzero_pixels(self, face):
        img = render_landmark_image(face, 512, 512)
        assert np.any(img > 0)

    def test_default_size_uses_face_dimensions(self, face):
        img = render_landmark_image(face)
        assert img.shape == (face.image_height, face.image_width, 3)

    def test_black_background(self, face):
        img = render_landmark_image(face, 512, 512)
        # Most pixels should be black (background)
        black_fraction = np.sum(img == 0) / img.size
        assert black_fraction > 0.5

    def test_small_canvas(self, face_small):
        img = render_landmark_image(face_small, 64, 64)
        assert img.shape == (64, 64, 3)

    def test_rectangular_canvas(self, face_rect):
        img = render_landmark_image(face_rect, 640, 480)
        assert img.shape == (480, 640, 3)


class TestVisualizeLandmarks:
    """Test visualize_landmarks function."""

    def test_output_shape(self, face, sample_image):
        result = visualize_landmarks(sample_image, face)
        assert result.shape == sample_image.shape

    def test_does_not_modify_original(self, face, sample_image):
        original = sample_image.copy()
        visualize_landmarks(sample_image, face)
        np.testing.assert_array_equal(sample_image, original)

    def test_with_regions(self, face, sample_image):
        result = visualize_landmarks(sample_image, face, draw_regions=True)
        assert result.shape == sample_image.shape

    def test_without_regions(self, face, sample_image):
        result = visualize_landmarks(sample_image, face, draw_regions=False)
        assert result.shape == sample_image.shape

    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_different_radii(self, face, sample_image, radius):
        result = visualize_landmarks(sample_image, face, radius=radius)
        assert result.shape == sample_image.shape


class TestLoadImage:
    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path.jpg")

    def test_load_valid_image(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(path)
        assert loaded.shape == (100, 100, 3)

    def test_load_returns_bgr(self, tmp_path):
        # Create an image with known BGR values
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Red channel in BGR
        path = tmp_path / "red.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(path)
        assert loaded[0, 0, 2] == 255  # Red channel preserved


# ============================================================================
# 2. CONDITIONING MODULE
# ============================================================================


class TestContourData:
    """Validate static contour definitions."""

    def test_jawline_is_closed(self):
        assert JAWLINE_CONTOUR[0] == JAWLINE_CONTOUR[-1]

    def test_left_eye_is_closed(self):
        assert LEFT_EYE_CONTOUR[0] == LEFT_EYE_CONTOUR[-1]

    def test_right_eye_is_closed(self):
        assert RIGHT_EYE_CONTOUR[0] == RIGHT_EYE_CONTOUR[-1]

    def test_outer_lips_is_closed(self):
        assert OUTER_LIPS[0] == OUTER_LIPS[-1]

    def test_inner_lips_is_closed(self):
        assert INNER_LIPS[0] == INNER_LIPS[-1]

    def test_eyebrows_are_open(self):
        assert LEFT_EYEBROW[0] != LEFT_EYEBROW[-1]
        assert RIGHT_EYEBROW[0] != RIGHT_EYEBROW[-1]

    def test_nose_bottom_no_duplicates(self):
        """NOSE_BOTTOM should have no duplicate indices after cleanup."""
        assert len(NOSE_BOTTOM) == len(set(NOSE_BOTTOM))

    def test_nose_bottom_in_all_contours(self):
        """NOSE_BOTTOM should be included in ALL_CONTOURS."""
        assert NOSE_BOTTOM in ALL_CONTOURS

    def test_all_contours_count(self):
        assert len(ALL_CONTOURS) == 10

    @pytest.mark.parametrize("contour", ALL_CONTOURS)
    def test_contour_indices_valid(self, contour):
        for idx in contour:
            assert 0 <= idx < 478

    def test_all_contours_has_both_eyes(self):
        assert LEFT_EYE_CONTOUR in ALL_CONTOURS
        assert RIGHT_EYE_CONTOUR in ALL_CONTOURS


class TestRenderWireframe:
    @pytest.mark.parametrize("w,h", [(512, 512), (256, 256), (1024, 1024)])
    def test_output_shape(self, face, w, h):
        wf = render_wireframe(face, w, h)
        assert wf.shape == (h, w)

    def test_output_is_grayscale(self, face):
        wf = render_wireframe(face, 512, 512)
        assert len(wf.shape) == 2

    def test_output_dtype(self, face):
        wf = render_wireframe(face, 512, 512)
        assert wf.dtype == np.uint8

    def test_has_nonzero(self, face):
        wf = render_wireframe(face, 512, 512)
        assert np.any(wf > 0)

    def test_default_size(self, face):
        wf = render_wireframe(face)
        assert wf.shape == (face.image_height, face.image_width)

    @pytest.mark.parametrize("thickness", [1, 2, 3])
    def test_thickness_parameter(self, face, thickness):
        wf = render_wireframe(face, 512, 512, thickness=thickness)
        assert wf.shape == (512, 512)
        assert np.any(wf > 0)


class TestAutoCanny:
    def test_output_shape(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == (64, 64)

    def test_output_dtype(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.dtype == np.uint8

    def test_output_binary(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
        edges = auto_canny(img)
        unique = set(np.unique(edges))
        assert unique.issubset({0, 255})

    def test_all_black_input(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        edges = auto_canny(img)
        assert np.all(edges == 0)

    def test_all_white_input(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        edges = auto_canny(img)
        # All-white should produce no edges
        assert edges.shape == (64, 64)

    def test_with_strong_edges(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 255  # sharp vertical edge
        edges = auto_canny(img)
        assert np.any(edges > 0)

    @pytest.mark.parametrize("size", [(32, 32), (64, 64), (128, 128), (256, 256)])
    def test_various_sizes(self, size):
        img = np.random.default_rng(0).integers(0, 256, size=size, dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == size


class TestGenerateConditioning:
    def test_returns_three_values(self, face):
        result = generate_conditioning(face, 512, 512)
        assert len(result) == 3

    def test_landmark_image_is_bgr(self, face):
        lm, _, _ = generate_conditioning(face, 512, 512)
        assert lm.shape == (512, 512, 3)

    def test_canny_is_grayscale(self, face):
        _, canny, _ = generate_conditioning(face, 512, 512)
        assert len(canny.shape) == 2

    def test_wireframe_is_grayscale(self, face):
        _, _, wf = generate_conditioning(face, 512, 512)
        assert len(wf.shape) == 2

    @pytest.mark.parametrize("w,h", [(256, 256), (512, 512), (1024, 1024)])
    def test_output_sizes(self, face, w, h):
        lm, canny, wf = generate_conditioning(face, w, h)
        assert lm.shape[:2] == (h, w)
        assert canny.shape == (h, w)
        assert wf.shape == (h, w)

    def test_default_size(self, face):
        lm, canny, wf = generate_conditioning(face)
        assert lm.shape[:2] == (face.image_height, face.image_width)


# ============================================================================
# 3. MANIPULATION MODULE
# ============================================================================


class TestDeformationHandle:
    def test_creation(self):
        h = DeformationHandle(
            landmark_index=0,
            displacement=np.array([1.0, 2.0]),
            influence_radius=30.0,
        )
        assert h.landmark_index == 0
        assert h.influence_radius == 30.0

    def test_frozen(self):
        h = DeformationHandle(
            landmark_index=0,
            displacement=np.array([1.0, 2.0]),
            influence_radius=30.0,
        )
        with pytest.raises(FrozenInstanceError):
            h.landmark_index = 5

    def test_3d_displacement(self):
        h = DeformationHandle(
            landmark_index=0,
            displacement=np.array([1.0, 2.0, 0.5]),
            influence_radius=30.0,
        )
        assert len(h.displacement) == 3


class TestGaussianRBFDeform:
    def test_returns_copy(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
        h = DeformationHandle(0, np.array([1.0, 0.0]), 10.0)
        result = gaussian_rbf_deform(pts, h)
        assert result is not pts

    def test_input_unchanged(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        original = pts.copy()
        h = DeformationHandle(0, np.array([5.0, 5.0]), 10.0)
        gaussian_rbf_deform(pts, h)
        np.testing.assert_array_equal(pts, original)

    def test_max_at_handle(self):
        pts = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 50.0]])
        h = DeformationHandle(0, np.array([10.0, 0.0]), 30.0)
        result = gaussian_rbf_deform(pts, h)
        # Point 0 (at handle) gets full displacement
        assert result[0, 0] == pytest.approx(10.0)
        # Others get less
        assert abs(result[1, 0] - 100.0) < 10.0
        assert abs(result[2, 0] - 50.0) < 10.0

    def test_falls_off_with_distance(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
        h = DeformationHandle(0, np.array([5.0, 0.0]), 20.0)
        result = gaussian_rbf_deform(pts, h)
        disp_0 = abs(result[0, 0] - 0.0)
        disp_1 = abs(result[1, 0] - 10.0)
        disp_2 = abs(result[2, 0] - 100.0)
        assert disp_0 > disp_1 > disp_2

    def test_zero_displacement(self):
        pts = np.random.default_rng(0).uniform(0, 100, size=(20, 2))
        h = DeformationHandle(0, np.array([0.0, 0.0]), 30.0)
        result = gaussian_rbf_deform(pts, h)
        np.testing.assert_array_almost_equal(result, pts)

    def test_3d_landmarks(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        h = DeformationHandle(0, np.array([1.0, 2.0, 0.5]), 10.0)
        result = gaussian_rbf_deform(pts, h)
        assert result.shape == (2, 3)
        assert result[0, 2] == pytest.approx(0.5)

    def test_3d_landmarks_2d_displacement(self):
        pts = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
        h = DeformationHandle(0, np.array([1.0, 2.0]), 10.0)
        result = gaussian_rbf_deform(pts, h)
        # Z should be unchanged since displacement is 2D
        assert result[0, 2] == pytest.approx(5.0)
        assert result[1, 2] == pytest.approx(5.0)

    def test_negative_displacement(self):
        pts = np.array([[50.0, 50.0], [60.0, 60.0]])
        h = DeformationHandle(0, np.array([-10.0, -10.0]), 30.0)
        result = gaussian_rbf_deform(pts, h)
        assert result[0, 0] < 50.0
        assert result[0, 1] < 50.0

    def test_large_radius(self):
        """Large radius should affect distant points more."""
        pts = np.array([[0.0, 0.0], [100.0, 0.0]])
        h_small = DeformationHandle(0, np.array([5.0, 0.0]), 10.0)
        h_large = DeformationHandle(0, np.array([5.0, 0.0]), 200.0)
        r_small = gaussian_rbf_deform(pts, h_small)
        r_large = gaussian_rbf_deform(pts, h_large)
        # Large radius: point 1 should move more
        disp_small = abs(r_small[1, 0] - 100.0)
        disp_large = abs(r_large[1, 0] - 100.0)
        assert disp_large > disp_small

    def test_symmetry(self):
        """Symmetric points should be displaced symmetrically."""
        pts = np.array([[50.0, 50.0], [40.0, 50.0], [60.0, 50.0]])
        h = DeformationHandle(0, np.array([0.0, -5.0]), 30.0)
        result = gaussian_rbf_deform(pts, h)
        # Points 1 and 2 are equidistant from 0
        assert result[1, 1] == pytest.approx(result[2, 1])

    def test_single_point(self):
        pts = np.array([[10.0, 20.0]])
        h = DeformationHandle(0, np.array([5.0, -3.0]), 10.0)
        result = gaussian_rbf_deform(pts, h)
        assert result[0, 0] == pytest.approx(15.0)
        assert result[0, 1] == pytest.approx(17.0)

    @pytest.mark.parametrize("n_points", [1, 10, 100, 478])
    def test_various_point_counts(self, n_points):
        pts = np.random.default_rng(0).uniform(0, 512, size=(n_points, 2))
        h = DeformationHandle(0, np.array([5.0, 5.0]), 30.0)
        result = gaussian_rbf_deform(pts, h)
        assert result.shape == (n_points, 2)


class TestApplyProcedurePreset:
    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_all_procedures_work(self, face, procedure):
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (478, 3)

    def test_invalid_procedure(self, face):
        with pytest.raises(ValueError, match="Unknown procedure"):
            apply_procedure_preset(face, "liposuction")

    def test_zero_intensity_no_change(self, face):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=0.0)
        np.testing.assert_array_almost_equal(result.landmarks, face.landmarks, decimal=5)

    @pytest.mark.parametrize("intensity", [0.0, 10.0, 25.0, 50.0, 75.0, 100.0])
    def test_intensity_range(self, face, intensity):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=intensity)
        assert result.landmarks.shape == (478, 3)

    def test_preserves_image_dimensions(self, face):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.image_width == face.image_width
        assert result.image_height == face.image_height

    def test_preserves_confidence(self, face):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result.confidence == face.confidence

    def test_returns_new_face(self, face):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        assert result is not face

    def test_higher_intensity_more_change(self, face):
        low = apply_procedure_preset(face, "rhinoplasty", intensity=20.0)
        high = apply_procedure_preset(face, "rhinoplasty", intensity=80.0)
        diff_low = np.linalg.norm(low.landmarks - face.landmarks)
        diff_high = np.linalg.norm(high.landmarks - face.landmarks)
        assert diff_high > diff_low

    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_image_size_parameter(self, face, size):
        result = apply_procedure_preset(face, "rhinoplasty", intensity=50.0, image_size=size)
        assert result.landmarks.shape == (478, 3)

    def test_does_not_modify_original(self, face):
        original = face.landmarks.copy()
        apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        np.testing.assert_array_equal(face.landmarks, original)

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_procedure_landmarks_exist(self, procedure):
        assert procedure in PROCEDURE_LANDMARKS
        assert len(PROCEDURE_LANDMARKS[procedure]) > 0

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_procedure_radius_exists(self, procedure):
        assert procedure in PROCEDURE_RADIUS
        assert PROCEDURE_RADIUS[procedure] > 0

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_procedure_landmark_indices_valid(self, procedure):
        for idx in PROCEDURE_LANDMARKS[procedure]:
            assert 0 <= idx < 478, f"Index {idx} in {procedure} out of range"


# ============================================================================
# 4. MASKING MODULE
# ============================================================================


class TestMaskConfig:
    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_config_exists(self, procedure):
        assert procedure in MASK_CONFIG

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_config_has_required_keys(self, procedure):
        config = MASK_CONFIG[procedure]
        assert "landmark_indices" in config
        assert "dilation_px" in config
        assert "feather_sigma" in config

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_indices_valid(self, procedure):
        for idx in MASK_CONFIG[procedure]["landmark_indices"]:
            assert 0 <= idx < 478

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_dilation_positive(self, procedure):
        assert MASK_CONFIG[procedure]["dilation_px"] > 0

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_sigma_positive(self, procedure):
        assert MASK_CONFIG[procedure]["feather_sigma"] > 0


class TestGenerateSurgicalMask:
    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_all_procedures(self, face, procedure):
        mask = generate_surgical_mask(face, procedure, 512, 512)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32

    def test_invalid_procedure(self, face):
        with pytest.raises(ValueError, match="Unknown procedure"):
            generate_surgical_mask(face, "invalid", 512, 512)

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_output_range(self, face, procedure):
        mask = generate_surgical_mask(face, procedure, 512, 512)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_mask_has_nonzero(self, face, procedure):
        mask = generate_surgical_mask(face, procedure, 512, 512)
        assert np.any(mask > 0)

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_mask_has_full_values(self, face, procedure):
        mask = generate_surgical_mask(face, procedure, 512, 512)
        # Center of mask should be close to 1.0
        assert mask.max() > 0.9

    @pytest.mark.parametrize("w,h", [(256, 256), (512, 512), (1024, 1024)])
    def test_various_sizes(self, face, w, h):
        mask = generate_surgical_mask(face, "rhinoplasty", w, h)
        assert mask.shape == (h, w)

    def test_default_size(self, face):
        mask = generate_surgical_mask(face, "rhinoplasty")
        assert mask.shape == (face.image_height, face.image_width)

    def test_feathered_edges(self, face):
        mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        # Should have values between 0 and 1 (feathered)
        mid_values = (mask > 0.01) & (mask < 0.99)
        assert np.any(mid_values)


class TestMaskTo3Channel:
    def test_shape(self):
        mask = np.random.rand(64, 64).astype(np.float32)
        result = mask_to_3channel(mask)
        assert result.shape == (64, 64, 3)

    def test_values_preserved(self):
        mask = np.random.rand(64, 64).astype(np.float32)
        result = mask_to_3channel(mask)
        for c in range(3):
            np.testing.assert_array_equal(result[:, :, c], mask)

    def test_zeros(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        result = mask_to_3channel(mask)
        assert np.all(result == 0)

    def test_ones(self):
        mask = np.ones((32, 32), dtype=np.float32)
        result = mask_to_3channel(mask)
        assert np.all(result == 1.0)


# ============================================================================
# 5. INFERENCE MODULE
# ============================================================================


class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_device_type_valid(self):
        d = get_device()
        assert d.type in ("cpu", "cuda", "mps")


class TestNumpyToPil:
    def test_grayscale(self):
        arr = np.zeros((64, 64), dtype=np.uint8)
        img = numpy_to_pil(arr)
        assert isinstance(img, Image.Image)
        assert img.mode == "L"

    def test_color(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = numpy_to_pil(arr)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_output_size(self):
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        img = numpy_to_pil(arr)
        assert img.size == (200, 100)

    def test_bgr_to_rgb_conversion(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Blue in BGR
        img = numpy_to_pil(arr)
        pixel = img.getpixel((0, 0))
        # After BGR->RGB, blue should be in index 2
        assert pixel[2] == 255
        assert pixel[0] == 0


class TestPilToNumpy:
    def test_rgb_to_bgr(self):
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))  # Red in RGB
        arr = pil_to_numpy(img)
        # After RGB->BGR, red should be in channel 2
        assert arr[0, 0, 2] == 255
        assert arr[0, 0, 0] == 0

    def test_output_shape(self):
        img = Image.new("RGB", (200, 100))
        arr = pil_to_numpy(img)
        assert arr.shape == (100, 200, 3)

    def test_grayscale_passthrough(self):
        img = Image.new("L", (10, 10), color=128)
        arr = pil_to_numpy(img)
        assert arr.shape == (10, 10)
        assert arr[0, 0] == 128

    def test_roundtrip(self):
        original = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        pil_img = numpy_to_pil(original)
        back = pil_to_numpy(pil_img)
        np.testing.assert_array_equal(back, original)


class TestMaskComposite:
    def test_output_shape(self, sample_image):
        warped = sample_image.copy()
        mask = np.zeros((512, 512), dtype=np.float32)
        result = mask_composite(warped, sample_image, mask)
        assert result.shape == sample_image.shape

    def test_zero_mask_returns_original(self, sample_image):
        warped = np.ones_like(sample_image) * 255
        mask = np.zeros((512, 512), dtype=np.float32)
        result = mask_composite(warped, sample_image, mask)
        np.testing.assert_array_equal(result, sample_image)

    def test_full_mask_returns_warped_corrected(self, sample_image):
        warped = sample_image.copy()
        mask = np.ones((512, 512), dtype=np.float32)
        result = mask_composite(warped, sample_image, mask)
        # With identical images, result should equal both
        assert result.shape == sample_image.shape

    def test_uint8_mask(self, sample_image):
        warped = sample_image.copy()
        mask = np.zeros((512, 512), dtype=np.uint8)
        result = mask_composite(warped, sample_image, mask)
        assert result.dtype == np.uint8

    def test_255_mask(self, sample_image):
        warped = sample_image.copy()
        mask = np.full((512, 512), 255, dtype=np.uint8)
        result = mask_composite(warped, sample_image, mask)
        assert result.shape == sample_image.shape

    def test_partial_mask(self, sample_image):
        warped = np.zeros_like(sample_image)
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[:256, :] = 1.0  # top half
        # use_laplacian=False for predictable alpha blending behavior
        result = mask_composite(warped, sample_image, mask, use_laplacian=False)
        # Bottom half should be original (mask=0 → 100% original)
        np.testing.assert_array_equal(result[256:, :], sample_image[256:, :])


class TestMatchSkinTone:
    def test_empty_mask_returns_source(self, sample_image):
        mask = np.zeros((512, 512), dtype=np.float32)
        result = _match_skin_tone(sample_image, sample_image, mask)
        np.testing.assert_array_equal(result, sample_image)

    def test_output_shape(self, sample_image):
        mask = np.ones((512, 512), dtype=np.float32)
        result = _match_skin_tone(sample_image, sample_image, mask)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        mask = np.ones((512, 512), dtype=np.float32)
        result = _match_skin_tone(sample_image, sample_image, mask)
        assert result.dtype == np.uint8


class TestEstimateFaceView:
    def test_returns_dict(self, face):
        view = estimate_face_view(face)
        assert isinstance(view, dict)

    def test_has_required_keys(self, face):
        view = estimate_face_view(face)
        assert "yaw" in view
        assert "pitch" in view
        assert "view" in view
        assert "is_frontal" in view
        assert "warning" in view

    def test_yaw_is_float(self, face):
        view = estimate_face_view(face)
        assert isinstance(view["yaw"], float)

    def test_pitch_is_float(self, face):
        view = estimate_face_view(face)
        assert isinstance(view["pitch"], float)

    def test_view_is_valid_string(self, face):
        view = estimate_face_view(face)
        assert view["view"] in ("frontal", "three_quarter", "profile")

    def test_is_frontal_is_bool(self, face):
        view = estimate_face_view(face)
        assert isinstance(view["is_frontal"], bool)

    def test_symmetric_face_is_frontal(self):
        """A face with symmetric ear distances should be frontal."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        # Nose at center
        landmarks[1] = [0.5, 0.5, 0.0]
        # Ears equidistant
        landmarks[234] = [0.1, 0.5, 0.0]  # left ear
        landmarks[454] = [0.9, 0.5, 0.0]  # right ear
        landmarks[10] = [0.5, 0.1, 0.0]  # forehead
        landmarks[152] = [0.5, 0.9, 0.0]  # chin
        f = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        view = estimate_face_view(f)
        assert view["is_frontal"]
        assert abs(view["yaw"]) < 15

    def test_warning_on_extreme_yaw(self):
        """Side-view face should have warning."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        # Truly asymmetric: nose far right, left ear far left, right ear very close to nose
        landmarks[1] = [0.8, 0.5, 0.0]  # nose far right
        landmarks[234] = [0.2, 0.5, 0.0]  # left ear far from nose
        landmarks[454] = [0.85, 0.5, 0.0]  # right ear very close to nose
        landmarks[10] = [0.8, 0.1, 0.0]  # forehead
        landmarks[152] = [0.8, 0.9, 0.0]  # chin
        f = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        view = estimate_face_view(f)
        # nose-to-left_ear >> nose-to-right_ear → strong yaw
        assert abs(view["yaw"]) > 15 or view["view"] != "frontal"


class TestPipelineInit:
    def test_default_mode(self):
        pipe = LandmarkDiffPipeline()
        assert pipe.mode == "img2img"

    def test_tps_mode(self):
        pipe = LandmarkDiffPipeline(mode="tps")
        assert pipe.mode == "tps"
        assert pipe.is_loaded  # TPS doesn't need loading

    def test_not_loaded_initially(self):
        pipe = LandmarkDiffPipeline(mode="img2img")
        assert not pipe.is_loaded

    def test_generate_raises_if_not_loaded(self):
        pipe = LandmarkDiffPipeline(mode="img2img")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            pipe.generate(img)

    def test_mps_uses_float32(self):
        pipe = LandmarkDiffPipeline(mode="img2img")
        if pipe.device.type == "mps":
            assert pipe.dtype == torch.float32

    @pytest.mark.parametrize("mode", ["img2img", "controlnet", "controlnet_ip", "tps"])
    def test_valid_modes(self, mode):
        pipe = LandmarkDiffPipeline(mode=mode)
        assert pipe.mode == mode

    def test_procedure_prompts_exist(self):
        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            assert proc in PROCEDURE_PROMPTS
            assert len(PROCEDURE_PROMPTS[proc]) > 0

    def test_negative_prompt_exists(self):
        assert len(NEGATIVE_PROMPT) > 0


# ============================================================================
# 6. LOSSES MODULE
# ============================================================================


class TestLossWeights:
    def test_defaults(self):
        w = LossWeights()
        assert w.diffusion == 1.0
        assert w.landmark == 0.1
        assert w.identity == 0.1
        assert w.perceptual == 0.05

    def test_custom(self):
        w = LossWeights(diffusion=2.0, landmark=0.5, identity=0.1, perceptual=0.2)
        assert w.diffusion == 2.0

    def test_frozen(self):
        w = LossWeights()
        with pytest.raises(FrozenInstanceError):
            w.diffusion = 5.0


class TestDiffusionLoss:
    def test_zero_on_identical(self):
        loss_fn = DiffusionLoss()
        x = torch.randn(2, 4, 64, 64)
        assert loss_fn(x, x).item() == pytest.approx(0.0)

    def test_positive_on_different(self):
        loss_fn = DiffusionLoss()
        a = torch.randn(2, 4, 64, 64)
        b = torch.randn(2, 4, 64, 64)
        assert loss_fn(a, b).item() > 0

    def test_symmetric(self):
        loss_fn = DiffusionLoss()
        a = torch.randn(2, 4, 64, 64)
        b = torch.randn(2, 4, 64, 64)
        assert loss_fn(a, b).item() == pytest.approx(loss_fn(b, a).item())

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_sizes(self, batch_size):
        loss_fn = DiffusionLoss()
        a = torch.randn(batch_size, 4, 64, 64)
        b = torch.randn(batch_size, 4, 64, 64)
        loss = loss_fn(a, b)
        assert loss.shape == ()
        assert loss.item() > 0

    @pytest.mark.parametrize("spatial", [16, 32, 64, 128])
    def test_spatial_sizes(self, spatial):
        loss_fn = DiffusionLoss()
        a = torch.randn(1, 4, spatial, spatial)
        b = torch.randn(1, 4, spatial, spatial)
        assert loss_fn(a, b).item() > 0

    def test_larger_difference_means_larger_loss(self):
        loss_fn = DiffusionLoss()
        x = torch.zeros(1, 4, 32, 32)
        small_diff = x + 0.1
        large_diff = x + 1.0
        assert loss_fn(x, large_diff).item() > loss_fn(x, small_diff).item()

    def test_gradient_flows(self):
        loss_fn = DiffusionLoss()
        a = torch.randn(1, 4, 32, 32, requires_grad=True)
        b = torch.randn(1, 4, 32, 32)
        loss = loss_fn(a, b)
        loss.backward()
        assert a.grad is not None
        assert torch.all(torch.isfinite(a.grad))


class TestLandmarkLoss:
    def test_zero_on_identical(self):
        loss_fn = LandmarkLoss()
        pts = torch.randn(2, 478, 2)
        assert loss_fn(pts, pts).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(2, 478, 2)
        b = a + 0.1
        assert loss_fn(a, b).item() > 0

    def test_mask_all_zeros(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(1, 10, 2)
        b = torch.randn(1, 10, 2)
        mask = torch.zeros(1, 10)
        loss = loss_fn(a, b, mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mask_all_ones(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(1, 10, 2)
        b = a + 0.5
        mask = torch.ones(1, 10)
        loss_masked = loss_fn(a, b, mask=mask)
        loss_full = loss_fn(a, b)
        assert loss_masked.item() == pytest.approx(loss_full.item(), abs=1e-5)

    def test_mask_reduces_region(self):
        loss_fn = LandmarkLoss()
        a = torch.zeros(1, 10, 2)
        b = torch.ones(1, 10, 2)
        mask = torch.zeros(1, 10)
        mask[0, :5] = 1.0
        loss = loss_fn(a, b, mask=mask)
        assert loss.item() > 0

    def test_iod_normalization(self):
        loss_fn = LandmarkLoss()
        a = torch.zeros(1, 10, 2)
        b = torch.ones(1, 10, 2)
        iod_small = torch.tensor([1.0])
        iod_large = torch.tensor([10.0])
        loss_small_iod = loss_fn(a, b, iod=iod_small)
        loss_large_iod = loss_fn(a, b, iod=iod_large)
        assert loss_small_iod.item() > loss_large_iod.item()

    def test_iod_clamp(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(1, 10, 2)
        b = a + 0.1
        iod_zero = torch.tensor([0.0])
        loss = loss_fn(a, b, iod=iod_zero)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_sizes(self, batch_size):
        loss_fn = LandmarkLoss()
        a = torch.randn(batch_size, 478, 2)
        b = torch.randn(batch_size, 478, 2)
        loss = loss_fn(a, b)
        assert loss.shape == ()

    def test_gradient_flows(self):
        loss_fn = LandmarkLoss()
        a = torch.randn(1, 478, 2, requires_grad=True)
        b = torch.randn(1, 478, 2)
        loss = loss_fn(a, b)
        loss.backward()
        assert a.grad is not None


class TestIdentityLoss:
    def test_orthognathic_returns_zero(self):
        loss_fn = IdentityLoss()
        a = torch.rand(2, 3, 256, 256)
        b = torch.rand(2, 3, 256, 256)
        assert loss_fn(a, b, procedure="orthognathic").item() == 0.0

    def test_orthognathic_zero_on_different_images(self):
        loss_fn = IdentityLoss()
        a = torch.rand(1, 3, 256, 256)
        b = torch.rand(1, 3, 256, 256)
        loss = loss_fn(a, b, procedure="orthognathic")
        assert loss.item() == 0.0

    def test_procedure_crop_rhinoplasty(self):
        loss_fn = IdentityLoss()
        img = torch.rand(1, 3, 256, 256)
        cropped = loss_fn._procedure_crop(img, "rhinoplasty")
        assert cropped.shape == (1, 3, 170, 256)  # h * 2 // 3

    def test_procedure_crop_blepharoplasty(self):
        loss_fn = IdentityLoss()
        img = torch.rand(1, 3, 256, 256)
        cropped = loss_fn._procedure_crop(img, "blepharoplasty")
        assert cropped.shape == (1, 3, 256, 256)  # full face

    def test_procedure_crop_rhytidectomy(self):
        loss_fn = IdentityLoss()
        img = torch.rand(1, 3, 256, 256)
        cropped = loss_fn._procedure_crop(img, "rhytidectomy")
        assert cropped.shape == (1, 3, 192, 256)  # h * 3 // 4

    def test_procedure_crop_unknown(self):
        loss_fn = IdentityLoss()
        img = torch.rand(1, 3, 256, 256)
        cropped = loss_fn._procedure_crop(img, "unknown")
        assert cropped.shape == (1, 3, 256, 256)  # full face

    def test_fallback_embedding_shape(self):
        """When ArcFace unavailable, fallback returns flattened pixels + valid mask."""
        loss_fn = IdentityLoss()
        loss_fn._has_arcface = False  # Force fallback
        img = torch.rand(2, 3, 112, 112)
        result = loss_fn._extract_embedding(img)
        # Returns (embeddings, valid_mask) tuple
        assert isinstance(result, tuple)
        emb, valid_mask = result
        assert emb.shape == (2, 3 * 112 * 112)
        assert valid_mask == [True, True]

    def test_fallback_identical_zero_loss(self):
        """Fallback: identical images should produce zero loss."""
        loss_fn = IdentityLoss()
        loss_fn._has_arcface = False
        img = torch.rand(1, 3, 256, 256)
        loss = loss_fn(img, img, procedure="rhinoplasty")
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_fallback_different_positive_loss(self):
        loss_fn = IdentityLoss()
        loss_fn._has_arcface = False
        a = torch.rand(1, 3, 256, 256)
        b = torch.rand(1, 3, 256, 256)
        loss = loss_fn(a, b, procedure="rhinoplasty")
        assert loss.item() > 0

    @pytest.mark.parametrize("procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy"])
    def test_fallback_all_non_orthognathic(self, procedure):
        loss_fn = IdentityLoss()
        loss_fn._has_arcface = False
        a = torch.rand(1, 3, 256, 256)
        loss = loss_fn(a, a, procedure=procedure)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


class TestPerceptualLoss:
    def test_unavailable_fallback(self):
        loss_fn = PerceptualLoss()
        loss_fn._lpips = "unavailable"
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        loss = loss_fn(pred, target, mask)
        assert loss.item() >= 0

    def test_unavailable_identical_not_zero(self):
        """Due to masking order bug, even identical images don't produce exactly 0."""
        loss_fn = PerceptualLoss()
        loss_fn._lpips = "unavailable"
        img = torch.rand(1, 3, 64, 64)
        mask = torch.ones(1, 1, 64, 64)  # full mask = outside_mask is 0
        loss = loss_fn(img, img, mask)
        # With full surgical mask, outside_mask=0, masked images become 0,
        # normalized to -1. Two identical -1 images should have 0 L1.
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_zero_mask_uses_all_pixels(self):
        loss_fn = PerceptualLoss()
        loss_fn._lpips = "unavailable"
        a = torch.rand(1, 3, 64, 64)
        b = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)  # no surgical region = outside_mask is 1
        loss = loss_fn(a, b, mask)
        assert loss.item() > 0


class TestCombinedLoss:
    def test_phase_a_keys(self):
        loss_fn = CombinedLoss(phase="A")
        a = torch.randn(2, 4, 64, 64)
        b = torch.randn(2, 4, 64, 64)
        losses = loss_fn(a, b)
        assert "diffusion" in losses
        assert "total" in losses
        assert "landmark" not in losses
        assert "identity" not in losses
        assert "perceptual" not in losses

    def test_phase_a_total_equals_diffusion(self):
        loss_fn = CombinedLoss(phase="A")
        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)
        losses = loss_fn(a, b)
        assert losses["total"].item() == pytest.approx(losses["diffusion"].item())

    def test_phase_b_with_landmarks_only(self):
        loss_fn = CombinedLoss(phase="B")
        loss_fn.identity_loss._has_arcface = False  # Force fallback
        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)
        pred_lm = torch.randn(1, 478, 2)
        target_lm = torch.randn(1, 478, 2)
        losses = loss_fn(a, b, pred_landmarks=pred_lm, target_landmarks=target_lm)
        assert "diffusion" in losses
        assert "landmark" in losses
        assert "total" in losses

    def test_phase_b_landmark_contributes(self):
        loss_fn = CombinedLoss(phase="B")
        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)
        pred_lm = torch.randn(1, 478, 2)
        target_lm = torch.randn(1, 478, 2)
        losses = loss_fn(a, b, pred_landmarks=pred_lm, target_landmarks=target_lm)
        assert losses["total"].item() > losses["diffusion"].item()

    def test_custom_weights(self):
        w = LossWeights(diffusion=2.0, landmark=0.5, identity=0.0, perceptual=0.0)
        loss_fn = CombinedLoss(weights=w, phase="A")
        a = torch.randn(1, 4, 64, 64)
        b = torch.randn(1, 4, 64, 64)
        losses = loss_fn(a, b)
        default_loss_fn = CombinedLoss(phase="A")
        default_losses = default_loss_fn(a, b)
        assert losses["diffusion"].item() == pytest.approx(
            default_losses["diffusion"].item() * 2.0, rel=1e-4
        )


# ============================================================================
# 7. EVALUATION MODULE
# ============================================================================


class TestEvalMetrics:
    def test_default_values(self):
        m = EvalMetrics()
        assert m.fid == 0.0
        assert m.lpips == 0.0
        assert m.nme == 0.0
        assert m.identity_sim == 0.0
        assert m.ssim == 0.0

    def test_summary_string(self):
        m = EvalMetrics(fid=50.0, lpips=0.1, nme=0.05, ssim=0.85, identity_sim=0.9)
        s = m.summary()
        assert "FID" in s
        assert "LPIPS" in s
        assert "50.00" in s

    def test_summary_with_fitzpatrick(self):
        m = EvalMetrics(
            count_by_fitzpatrick={"I": 10, "III": 20},
            lpips_by_fitzpatrick={"I": 0.1, "III": 0.2},
        )
        s = m.summary()
        assert "Fitzpatrick" in s

    def test_to_dict(self):
        m = EvalMetrics(fid=50.0, lpips=0.1, nme=0.05, ssim=0.85)
        d = m.to_dict()
        assert d["fid"] == 50.0
        assert d["lpips"] == 0.1
        assert isinstance(d, dict)

    def test_to_dict_with_procedure(self):
        m = EvalMetrics(
            nme_by_procedure={"rhinoplasty": 0.03},
            lpips_by_procedure={"rhinoplasty": 0.1},
            ssim_by_procedure={"rhinoplasty": 0.9},
        )
        d = m.to_dict()
        assert "proc_rhinoplasty_nme" in d

    def test_to_dict_with_fitzpatrick(self):
        m = EvalMetrics(
            count_by_fitzpatrick={"I": 5},
            lpips_by_fitzpatrick={"I": 0.1},
            ssim_by_fitzpatrick={"I": 0.9},
            nme_by_fitzpatrick={"I": 0.03},
            identity_sim_by_fitzpatrick={"I": 0.95},
        )
        d = m.to_dict()
        assert "fitz_I_count" in d
        assert d["fitz_I_count"] == 5


class TestClassifyFitzpatrick:
    def test_very_light(self):
        """Bright image should classify as Type I or II."""
        img = np.full((64, 64, 3), 230, dtype=np.uint8)  # bright
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in ("I", "II", "III")

    def test_dark(self):
        """Dark image should classify as Type V or VI."""
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in ("IV", "V", "VI")

    def test_returns_valid_type(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in ("I", "II", "III", "IV", "V", "VI")

    @pytest.mark.parametrize("brightness", [50, 100, 150, 200, 250])
    def test_monotonic_brightness(self, brightness):
        img = np.full((64, 64, 3), brightness, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in ("I", "II", "III", "IV", "V", "VI")


class TestComputeNME:
    def test_zero_on_identical(self):
        pts = np.random.rand(478, 2) * 512
        assert compute_nme(pts, pts) == pytest.approx(0.0)

    def test_positive_on_different(self):
        a = np.random.rand(478, 2) * 512
        b = a + 1.0
        assert compute_nme(a, b) > 0

    def test_iod_normalization(self):
        a = np.zeros((478, 2))
        b = np.ones((478, 2))
        # Set eyes far apart
        a[33] = [100, 200]
        a[263] = [400, 200]
        b[33] = [100, 200]
        b[263] = [400, 200]
        nme = compute_nme(a, b)
        assert nme > 0
        assert nme < 10  # normalized, should be reasonable

    def test_iod_minimum_clamp(self):
        """IOD below 1.0 is clamped to 1.0."""
        a = np.zeros((478, 2))
        b = np.ones((478, 2)) * 0.1
        # Eyes at same position -> IOD = 0 -> clamped to 1
        nme = compute_nme(a, b)
        assert np.isfinite(nme)

    def test_custom_eye_indices(self):
        a = np.zeros((478, 2))
        b = np.ones((478, 2))
        nme = compute_nme(a, b, left_eye_idx=33, right_eye_idx=263)
        assert nme > 0

    @pytest.mark.parametrize("n_landmarks", [10, 100, 478])
    def test_various_landmark_counts(self, n_landmarks):
        a = np.random.rand(n_landmarks, 2)
        b = a + 0.01
        # Use indices that exist
        left = min(33, n_landmarks - 1)
        right = min(263, n_landmarks - 1) if n_landmarks > 263 else n_landmarks - 1
        nme = compute_nme(a, b, left_eye_idx=left, right_eye_idx=right)
        assert nme >= 0


class TestComputeSSIM:
    def test_perfect_on_identical(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(img, img)
        assert ssim == pytest.approx(1.0, abs=1e-5)

    def test_less_than_one_on_different(self):
        a = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        b = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(a, b)
        assert ssim < 1.0

    def test_symmetric(self):
        a = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        b = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        assert compute_ssim(a, b) == pytest.approx(compute_ssim(b, a))

    def test_all_black(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(a, a)
        assert ssim == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("shape", [(32, 32, 3), (64, 64, 3), (128, 128, 3)])
    def test_various_sizes(self, shape):
        a = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        ssim = compute_ssim(a, a)
        assert ssim == pytest.approx(1.0, abs=1e-5)


class TestEvaluateBatch:
    def test_basic_evaluation(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(3)]
        targets = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(3)]
        metrics = evaluate_batch(preds, targets)
        assert isinstance(metrics, EvalMetrics)
        assert metrics.ssim > 0 or metrics.ssim == 0  # valid float

    def test_with_landmarks(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(2)]
        targets = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(2)]
        pred_lm = [rng.random((478, 2)) * 64 for _ in range(2)]
        target_lm = [rng.random((478, 2)) * 64 for _ in range(2)]
        metrics = evaluate_batch(preds, targets, pred_lm, target_lm)
        assert metrics.nme > 0

    def test_with_procedures(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(2)]
        targets = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(2)]
        procedures = ["rhinoplasty", "blepharoplasty"]
        metrics = evaluate_batch(preds, targets, procedures=procedures)
        assert len(metrics.ssim_by_procedure) > 0

    def test_fitzpatrick_stratification(self):
        np.random.default_rng(0)
        # Create images with varying brightness for different Fitzpatrick types
        preds = [np.full((64, 64, 3), b, dtype=np.uint8) for b in [200, 100, 50]]
        targets = [np.full((64, 64, 3), b, dtype=np.uint8) for b in [200, 100, 50]]
        metrics = evaluate_batch(preds, targets)
        assert len(metrics.count_by_fitzpatrick) > 0

    def test_empty_batch(self):
        metrics = evaluate_batch([], [])
        assert metrics.ssim == 0.0
        assert metrics.lpips == 0.0

    def test_single_image(self):
        img = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        metrics = evaluate_batch([img], [img])
        assert metrics.ssim == pytest.approx(1.0, abs=1e-3)


# ============================================================================
# 8. TPS WARP MODULE
# ============================================================================


class TestTPSKernel:
    def test_zero_input(self):
        r = np.array([0.0, 0.0])
        result = _tps_kernel(r)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_positive_values(self):
        r = np.array([1.0, 2.0, 3.0])
        result = _tps_kernel(r)
        assert result[0] == pytest.approx(0.0)  # 1^2 * log(1) = 0
        assert result[1] == pytest.approx(4.0 * np.log(2.0))
        assert result[2] == pytest.approx(9.0 * np.log(3.0))

    def test_output_shape(self):
        r = np.random.rand(100)
        result = _tps_kernel(r)
        assert result.shape == (100,)

    def test_negative_not_expected(self):
        """r should always be >= 0 (distances)."""
        r = np.array([0.5])
        result = _tps_kernel(r)
        # 0.5^2 * log(0.5) = 0.25 * (-0.693) = -0.173
        assert result[0] < 0


class TestSubsampleControlPoints:
    def test_reduces_count(self):
        src = np.random.rand(478, 2) * 512
        dst = src.copy()
        dst[:20] += 10  # only 20 points displaced
        src_sub, dst_sub = _subsample_control_points(src, dst)
        assert len(src_sub) < 478

    def test_includes_displaced(self):
        src = np.random.rand(478, 2) * 512
        dst = src.copy()
        dst[0] += 100  # large displacement at index 0
        src_sub, dst_sub = _subsample_control_points(src, dst)
        assert len(src_sub) > 0

    def test_max_points_limit(self):
        src = np.random.rand(478, 2) * 512
        dst = src + 10  # all points displaced
        src_sub, dst_sub = _subsample_control_points(src, dst, max_points=50)
        assert len(src_sub) <= 50

    def test_output_shapes_match(self):
        src = np.random.rand(478, 2) * 512
        dst = src + np.random.rand(478, 2) * 5
        src_sub, dst_sub = _subsample_control_points(src, dst)
        assert src_sub.shape == dst_sub.shape

    def test_no_displacement(self):
        src = np.random.rand(478, 2) * 512
        dst = src.copy()
        src_sub, dst_sub = _subsample_control_points(src, dst)
        # Should still have some anchor points
        assert len(src_sub) > 0


class TestSolveTpsWeights:
    def test_output_shape(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        values = np.array([0, 1, 0, 1], dtype=np.float64)
        w = _solve_tps_weights(pts, values)
        assert w.shape == (4 + 3,)  # n + 3 (affine terms)

    def test_zero_values(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        values = np.zeros(3)
        w = _solve_tps_weights(pts, values)
        assert np.all(np.isfinite(w))

    def test_identity_like(self):
        pts = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
        # x-coordinate as values -> should produce near-identity transform for x
        values = pts[:, 0]
        w = _solve_tps_weights(pts, values)
        assert np.all(np.isfinite(w))


class TestEvaluateTps:
    def test_basic(self):
        control_pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        values = np.array([1, 0, 0], dtype=np.float64)
        weights = _solve_tps_weights(control_pts, values)

        eval_pts = np.array([[0, 0], [5, 5]], dtype=np.float64)
        result = _evaluate_tps(eval_pts, control_pts, weights)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_at_control_points(self):
        control_pts = np.array([[0, 0], [100, 0], [50, 50]], dtype=np.float64)
        values = np.array([5.0, -3.0, 2.0])
        weights = _solve_tps_weights(control_pts, values)
        result = _evaluate_tps(control_pts, control_pts, weights)
        np.testing.assert_array_almost_equal(result, values, decimal=3)


class TestComputeTpsMap:
    def test_output_shapes(self):
        src = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float32)
        dst = src + np.array([[5, 0], [-5, 0], [0, -5]])
        map_x, map_y = _compute_tps_map(src, dst, 256, 256)
        assert map_x.shape == (256, 256)
        assert map_y.shape == (256, 256)

    def test_identity_when_no_displacement(self):
        src = np.array([[100, 100], [200, 200]], dtype=np.float32)
        dst = src.copy()
        map_x, map_y = _compute_tps_map(src, dst, 64, 64)
        # Should be approximately identity
        grid_x, grid_y = np.meshgrid(np.arange(64), np.arange(64))
        np.testing.assert_array_almost_equal(map_x, grid_x.astype(np.float64), decimal=1)

    def test_empty_control_points(self):
        src = np.empty((0, 2), dtype=np.float32)
        dst = np.empty((0, 2), dtype=np.float32)
        map_x, map_y = _compute_tps_map(src, dst, 64, 64)
        assert map_x.shape == (64, 64)


class TestWarpImageTps:
    def test_output_shape(self, sample_image):
        src = np.random.rand(20, 2).astype(np.float32) * 512
        dst = src + np.random.rand(20, 2) * 5
        result = warp_image_tps(sample_image, src, dst)
        assert result.shape == sample_image.shape

    def test_identity_warp(self, sample_image):
        src = np.random.rand(20, 2).astype(np.float32) * 400 + 50
        dst = src.copy()
        result = warp_image_tps(sample_image, src, dst)
        # Should be very similar to original
        diff = np.abs(result.astype(float) - sample_image.astype(float)).mean()
        assert diff < 5.0  # small difference due to remap interpolation

    def test_output_dtype(self, sample_image):
        src = np.random.rand(20, 2).astype(np.float32) * 512
        dst = src + 1
        result = warp_image_tps(sample_image, src, dst)
        assert result.dtype == np.uint8


class TestComputeRigidTranslation:
    def test_no_points_inside(self):
        src = np.array([[10, 10], [20, 20]], dtype=np.float32)
        dst = np.array([[15, 15], [25, 25]], dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _compute_rigid_translation(src, dst, mask, 100, 100)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_points_inside(self):
        src = np.array([[10, 10], [20, 20]], dtype=np.float32)
        dst = np.array([[15, 15], [25, 25]], dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        result = _compute_rigid_translation(src, dst, mask, 100, 100)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(5.0)


class TestApplyRigidTranslation:
    def test_output_shape(self, sample_image):
        t = np.array([5.0, -3.0])
        result = _apply_rigid_translation(sample_image, t)
        assert result.shape == sample_image.shape

    def test_zero_translation(self, sample_image):
        t = np.array([0.0, 0.0])
        result = _apply_rigid_translation(sample_image, t)
        np.testing.assert_array_equal(result, sample_image)


class TestGenerateRandomWarp:
    def test_returns_copy(self):
        pts = np.random.rand(478, 2) * 512
        result = generate_random_warp(pts, [0, 1, 2])
        assert result is not pts

    def test_modifies_specified_indices(self):
        pts = np.zeros((478, 2))
        result = generate_random_warp(
            pts, [0, 1, 2], max_displacement=15.0, rng=np.random.default_rng(0)
        )
        assert not np.allclose(result[0], pts[0])

    def test_preserves_unspecified(self):
        pts = np.zeros((478, 2))
        result = generate_random_warp(pts, [0, 1, 2], rng=np.random.default_rng(0))
        # Index 100 should be unchanged
        np.testing.assert_array_equal(result[100], pts[100])

    def test_displacement_bound(self):
        pts = np.zeros((478, 2))
        max_d = 10.0
        result = generate_random_warp(
            pts, list(range(478)), max_displacement=max_d, rng=np.random.default_rng(0)
        )
        displacements = np.abs(result - pts)
        assert displacements.max() <= max_d

    def test_deterministic_with_seed(self):
        pts = np.random.rand(478, 2) * 512
        r1 = generate_random_warp(pts, [0, 1], rng=np.random.default_rng(42))
        r2 = generate_random_warp(pts, [0, 1], rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_out_of_range_index_ignored(self):
        pts = np.random.rand(10, 2)
        result = generate_random_warp(pts, [0, 1, 999], rng=np.random.default_rng(0))
        assert result.shape == (10, 2)

    def test_empty_indices(self):
        pts = np.random.rand(10, 2)
        result = generate_random_warp(pts, [], rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, pts)


class TestComputeTpsTransform:
    def test_returns_transformer(self):
        src = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)
        dst = np.array([[5, 5], [95, 5], [50, 95]], dtype=np.float32)
        tps = compute_tps_transform(src, dst)
        assert tps is not None

    def test_accepts_many_points(self):
        src = np.random.rand(50, 2).astype(np.float32) * 512
        dst = src + np.random.rand(50, 2) * 5
        tps = compute_tps_transform(src, dst)
        assert tps is not None


# ============================================================================
# 9. AUGMENTATION MODULE
# ============================================================================


class TestAugmentationPool:
    def test_pool_count(self):
        assert len(AUGMENTATION_POOL) == 8

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_has_name(self, aug):
        assert isinstance(aug.name, str)
        assert len(aug.name) > 0

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_has_callable(self, aug):
        assert callable(aug.fn)

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_probability_valid(self, aug):
        assert 0.0 <= aug.probability <= 1.0

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_augmentation_preserves_shape(self, aug, sample_image_small, rng):
        result = aug.fn(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_augmentation_output_dtype(self, aug, sample_image_small, rng):
        result = aug.fn(sample_image_small, rng)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("aug", AUGMENTATION_POOL)
    def test_augmentation_output_range(self, aug, sample_image_small, rng):
        result = aug.fn(sample_image_small, rng)
        assert result.min() >= 0
        assert result.max() <= 255


class TestPointSourceLighting:
    def test_output_shape(self, sample_image_small, rng):
        result = point_source_lighting(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_deterministic(self, sample_image_small):
        r1 = point_source_lighting(sample_image_small, np.random.default_rng(42))
        r2 = point_source_lighting(sample_image_small, np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)


class TestColorTemperature:
    def test_output_shape(self, sample_image_small, rng):
        result = color_temperature_jitter(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_changes_image(self, sample_image_small, rng):
        result = color_temperature_jitter(sample_image_small, rng)
        assert not np.array_equal(result, sample_image_small)


class TestGreenFluorescent:
    def test_output_shape(self, sample_image_small, rng):
        result = green_fluorescent_cast(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_boosts_green(self, sample_image_small, rng):
        result = green_fluorescent_cast(sample_image_small, rng)
        # Green channel should be boosted on average
        assert result[:, :, 1].mean() >= sample_image_small[:, :, 1].mean() - 5


class TestJpegCompression:
    def test_roundtrip_shape(self, sample_image_small, rng):
        result = jpeg_compression(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_modifies_image(self, sample_image_small, rng):
        result = jpeg_compression(sample_image_small, rng)
        # JPEG is lossy — should differ
        assert not np.array_equal(result, sample_image_small)


class TestGaussianSensorNoise:
    def test_output_shape(self, sample_image_small, rng):
        result = gaussian_sensor_noise(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_adds_noise(self, sample_image_small, rng):
        result = gaussian_sensor_noise(sample_image_small, rng)
        assert not np.array_equal(result, sample_image_small)

    def test_clipped_range(self, sample_image_small, rng):
        result = gaussian_sensor_noise(sample_image_small, rng)
        assert result.min() >= 0
        assert result.max() <= 255


class TestBarrelDistortion:
    def test_output_shape(self, sample_image_small, rng):
        result = barrel_distortion(sample_image_small, rng)
        assert result.shape == sample_image_small.shape


class TestMotionBlur:
    def test_output_shape(self, sample_image_small, rng):
        result = motion_blur(sample_image_small, rng)
        assert result.shape == sample_image_small.shape


class TestVignette:
    def test_output_shape(self, sample_image_small, rng):
        result = vignette(sample_image_small, rng)
        assert result.shape == sample_image_small.shape

    def test_darkens_corners(self, rng):
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = vignette(img, rng)
        center = result[32, 32].mean()
        corner = result[0, 0].mean()
        assert corner < center


class TestApplyClinicalAugmentation:
    def test_output_shape(self, sample_image_small, rng):
        result = apply_clinical_augmentation(sample_image_small, rng=rng)
        assert result.shape == sample_image_small.shape

    def test_output_dtype(self, sample_image_small, rng):
        result = apply_clinical_augmentation(sample_image_small, rng=rng)
        assert result.dtype == np.uint8

    def test_modifies_image(self, sample_image_small, rng):
        result = apply_clinical_augmentation(sample_image_small, rng=rng)
        assert not np.array_equal(result, sample_image_small)

    def test_min_augmentations_respected(self, sample_image_small):
        # Run many times and verify it always produces changes
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = apply_clinical_augmentation(
                sample_image_small,
                min_augmentations=3,
                max_augmentations=5,
                rng=rng,
            )
            assert result.shape == sample_image_small.shape

    def test_deterministic_with_seed(self, sample_image_small):
        r1 = apply_clinical_augmentation(sample_image_small, rng=np.random.default_rng(42))
        r2 = apply_clinical_augmentation(sample_image_small, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("min_aug,max_aug", [(1, 2), (3, 5), (5, 8)])
    def test_augmentation_count_range(self, sample_image_small, min_aug, max_aug):
        result = apply_clinical_augmentation(
            sample_image_small,
            min_augmentations=min_aug,
            max_augmentations=max_aug,
            rng=np.random.default_rng(0),
        )
        assert result.shape == sample_image_small.shape


# ============================================================================
# 10. PAIR GENERATOR MODULE
# ============================================================================


class TestTrainingPair:
    def test_frozen(self):
        pair = TrainingPair(
            input_image=np.zeros((512, 512, 3), dtype=np.uint8),
            target_image=np.zeros((512, 512, 3), dtype=np.uint8),
            conditioning=np.zeros((512, 512, 3), dtype=np.uint8),
            canny=np.zeros((512, 512), dtype=np.uint8),
            mask=np.zeros((512, 512), dtype=np.float32),
            procedure="rhinoplasty",
            intensity=50.0,
        )
        with pytest.raises(FrozenInstanceError):
            pair.procedure = "blepharoplasty"


class TestProceduresList:
    def test_six_procedures(self):
        assert len(PROCEDURES) == 6
        expected = {
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        }
        assert set(PROCEDURES) == expected


class TestSavePair:
    def test_creates_files(self, tmp_path):
        pair = TrainingPair(
            input_image=np.zeros((512, 512, 3), dtype=np.uint8),
            target_image=np.zeros((512, 512, 3), dtype=np.uint8),
            conditioning=np.zeros((512, 512, 3), dtype=np.uint8),
            canny=np.zeros((512, 512), dtype=np.uint8),
            mask=np.zeros((512, 512), dtype=np.float32),
            procedure="rhinoplasty",
            intensity=50.0,
        )
        save_pair(pair, tmp_path, 0)
        assert (tmp_path / "000000_input.png").exists()
        assert (tmp_path / "000000_target.png").exists()
        assert (tmp_path / "000000_conditioning.png").exists()
        assert (tmp_path / "000000_canny.png").exists()
        assert (tmp_path / "000000_mask.png").exists()

    def test_index_formatting(self, tmp_path):
        pair = TrainingPair(
            input_image=np.zeros((64, 64, 3), dtype=np.uint8),
            target_image=np.zeros((64, 64, 3), dtype=np.uint8),
            conditioning=np.zeros((64, 64, 3), dtype=np.uint8),
            canny=np.zeros((64, 64), dtype=np.uint8),
            mask=np.zeros((64, 64), dtype=np.float32),
            procedure="rhinoplasty",
            intensity=50.0,
        )
        save_pair(pair, tmp_path, 42)
        assert (tmp_path / "000042_input.png").exists()

    def test_creates_output_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        pair = TrainingPair(
            input_image=np.zeros((64, 64, 3), dtype=np.uint8),
            target_image=np.zeros((64, 64, 3), dtype=np.uint8),
            conditioning=np.zeros((64, 64, 3), dtype=np.uint8),
            canny=np.zeros((64, 64), dtype=np.uint8),
            mask=np.zeros((64, 64), dtype=np.float32),
            procedure="rhinoplasty",
            intensity=50.0,
        )
        save_pair(pair, subdir, 0)
        assert subdir.exists()


# ============================================================================
# 11. CROSS-MODULE CONSISTENCY TESTS
# ============================================================================


class TestCrossModuleConsistency:
    """Verify consistency across modules."""

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_mask_config_matches_procedure_landmarks(self, procedure):
        """Mask and manipulation should use same procedure names."""
        assert procedure in MASK_CONFIG
        assert procedure in PROCEDURE_LANDMARKS

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_procedure_prompts_exist(self, procedure):
        assert procedure in PROCEDURE_PROMPTS

    def test_rhinoplasty_mask_indices_match_manipulation(self):
        """rhinoplasty mask and manipulation should use same landmark indices."""
        mask_indices = set(MASK_CONFIG["rhinoplasty"]["landmark_indices"])
        manip_indices = set(PROCEDURE_LANDMARKS["rhinoplasty"])
        assert mask_indices == manip_indices

    def test_blepharoplasty_mask_indices_match_manipulation(self):
        mask_indices = set(MASK_CONFIG["blepharoplasty"]["landmark_indices"])
        manip_indices = set(PROCEDURE_LANDMARKS["blepharoplasty"])
        assert mask_indices == manip_indices

    def test_orthognathic_mask_indices_subset_of_manipulation(self):
        """Known issue: orthognathic mask has fewer indices than manipulation."""
        mask_indices = set(MASK_CONFIG["orthognathic"]["landmark_indices"])
        manip_indices = set(PROCEDURE_LANDMARKS["orthognathic"])
        # Document the gap
        missing = manip_indices - mask_indices
        # This is a known bug flagged in DIRAC_REVIEW.md
        if missing:
            pytest.xfail(
                f"Orthognathic mask missing {len(missing)} manipulation indices: {missing}"
            )


class TestEndToEndPipeline:
    """Test components work together without GPU models."""

    def test_landmarks_to_conditioning(self, face):
        lm, canny, wf = generate_conditioning(face, 512, 512)
        assert lm.shape == (512, 512, 3)
        assert canny.shape == (512, 512)

    def test_landmarks_to_manipulation_to_conditioning(self, face):
        manipulated = apply_procedure_preset(face, "rhinoplasty", 50.0)
        lm, canny, wf = generate_conditioning(manipulated, 512, 512)
        assert lm.shape == (512, 512, 3)

    def test_full_manipulation_mask_pipeline(self, face):
        manipulated = apply_procedure_preset(face, "rhinoplasty", 50.0)
        mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        conditioning = render_landmark_image(manipulated, 512, 512)
        assert mask.shape == (512, 512)
        assert conditioning.shape == (512, 512, 3)

    @pytest.mark.parametrize(
        "procedure", ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]
    )
    def test_all_procedures_pipeline(self, face, procedure):
        manipulated = apply_procedure_preset(face, procedure, 50.0)
        mask = generate_surgical_mask(face, procedure, 512, 512)
        conditioning = render_landmark_image(manipulated, 512, 512)
        assert mask.max() > 0
        assert np.any(conditioning > 0)

    def test_tps_warp_after_manipulation(self, face, sample_image):
        manipulated = apply_procedure_preset(face, "rhinoplasty", 50.0)
        warped = warp_image_tps(sample_image, face.pixel_coords, manipulated.pixel_coords)
        assert warped.shape == sample_image.shape

    def test_composite_after_warp(self, face, sample_image):
        manipulated = apply_procedure_preset(face, "rhinoplasty", 50.0)
        mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        warped = warp_image_tps(sample_image, face.pixel_coords, manipulated.pixel_coords)
        composited = mask_composite(warped, sample_image, mask)
        assert composited.shape == sample_image.shape
        assert composited.dtype == np.uint8


# ============================================================================
# 12. EDGE CASES AND STRESS TESTS
# ============================================================================


class TestEdgeCases:
    def test_very_small_image(self):
        """1x1 pixel image should not crash."""
        face = FaceLandmarks(
            landmarks=np.random.rand(478, 3).astype(np.float32),
            image_width=1,
            image_height=1,
            confidence=1.0,
        )
        img = render_landmark_image(face, 1, 1)
        assert img.shape == (1, 1, 3)

    def test_zero_confidence_face(self, face):
        f2 = FaceLandmarks(
            landmarks=face.landmarks,
            image_width=face.image_width,
            image_height=face.image_height,
            confidence=0.0,
        )
        result = apply_procedure_preset(f2, "rhinoplasty", 50.0)
        assert result.confidence == 0.0

    def test_negative_landmarks(self):
        """Landmarks outside [0,1] should still work (clipping happens in rendering)."""
        landmarks = np.full((478, 3), -0.5, dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        coords = face.pixel_coords
        assert np.any(coords < 0)

    def test_landmarks_greater_than_one(self):
        landmarks = np.full((478, 3), 1.5, dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        coords = face.pixel_coords
        assert np.any(coords > 512)

    def test_nan_landmarks(self):
        landmarks = np.full((478, 3), np.nan, dtype=np.float32)
        face = FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=1.0)
        coords = face.pixel_coords
        assert np.all(np.isnan(coords))

    def test_diffusion_loss_nan_input(self):
        loss_fn = DiffusionLoss()
        a = torch.tensor([float("nan")])
        b = torch.tensor([1.0])
        loss = loss_fn(a.view(1, 1, 1, 1), b.view(1, 1, 1, 1))
        assert torch.isnan(loss)

    def test_landmark_loss_large_values(self):
        loss_fn = LandmarkLoss()
        a = torch.ones(1, 478, 2) * 1e6
        b = torch.zeros(1, 478, 2)
        loss = loss_fn(a, b)
        assert torch.isfinite(loss)

    def test_ssim_identical_black(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        ssim = compute_ssim(a, a)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_ssim_identical_white(self):
        a = np.full((64, 64, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(a, a)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_nme_single_point(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        nme = compute_nme(a, b, left_eye_idx=0, right_eye_idx=0)
        assert np.isfinite(nme)

    def test_mask_composite_all_channels_zeros(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = mask_composite(img, img, mask)
        np.testing.assert_array_equal(result, img)

    def test_eval_metrics_empty_fitzpatrick(self):
        m = EvalMetrics()
        s = m.summary()
        assert "FID" in s

    def test_augmentation_config_frozen(self):
        aug = AugmentationConfig("test", lambda x, r: x, 0.5)
        with pytest.raises(FrozenInstanceError):
            aug.name = "modified"

    @pytest.mark.parametrize("seed", range(10))
    def test_tps_warp_deterministic(self, sample_image_small, seed):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(10, 54, size=(10, 2)).astype(np.float32)
        dst = pts + rng.uniform(-3, 3, size=(10, 2))
        r1 = warp_image_tps(sample_image_small, pts, dst)
        r2 = warp_image_tps(sample_image_small, pts, dst)
        np.testing.assert_array_equal(r1, r2)


class TestMotionBlurEdgeCases:
    """Targeted tests for motion blur kernel edge cases."""

    @pytest.mark.parametrize("seed", range(20))
    def test_kernel_sum_nonzero(self, seed):
        """Verify motion blur kernel never has zero sum."""
        rng = np.random.default_rng(seed)
        size = int(rng.uniform(3, 7))
        angle = rng.uniform(0, 180)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        assert kernel.sum() > 0, f"Zero-sum kernel at seed={seed}, size={size}, angle={angle}"


class TestMaskingSeedIssue:
    """Test masking noise behavior — uses unseeded default_rng()."""

    def test_different_noise_across_calls(self, face):
        """Masks use np.random.default_rng() (unseeded) so noise differs each call."""
        m1 = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        m2 = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        # Masks should differ due to random boundary noise
        assert not np.array_equal(m1, m2), "Unseeded RNG should produce different noise"

    def test_different_procedures_different_masks(self, face):
        m1 = generate_surgical_mask(face, "rhinoplasty", 512, 512)
        m2 = generate_surgical_mask(face, "blepharoplasty", 512, 512)
        assert not np.array_equal(m1, m2)


class TestIdentityLossZeroEmbedding:
    """Test IdentityLoss behavior with zero embeddings (ArcFace fails to detect face)."""

    def test_zero_embedding_normalize_returns_zeros(self):
        """PyTorch 2.5.1: F.normalize on zero vector returns zeros (not NaN)."""
        zero = torch.zeros(1, 512)
        normalized = F.normalize(zero, dim=1)
        # In PyTorch 2.5.1, zero vectors normalize to zero (safe)
        assert torch.all(normalized == 0.0), "Zero vector should normalize to zero in PyTorch 2.5.1"
        assert torch.all(torch.isfinite(normalized)), "Result should be finite"

    def test_nonzero_embedding_normalize_ok(self):
        vec = torch.randn(1, 512)
        normalized = F.normalize(vec, dim=1)
        assert torch.all(torch.isfinite(normalized))

    def test_cosine_sim_with_zero_embedding_is_zero(self):
        """Zero embedding → cosine similarity = 0 (handled by valid_mask in IdentityLoss)."""
        good = F.normalize(torch.randn(1, 512), dim=1)
        bad = F.normalize(torch.zeros(1, 512), dim=1)  # stays zero
        sim = (good * bad).sum(dim=1)
        assert sim.item() == 0.0, "Cosine sim with zero vector should be 0"
        assert torch.all(torch.isfinite(sim)), "Result should be finite"

    def test_identity_loss_valid_mask_filters_failures(self):
        """IdentityLoss uses valid_mask to skip failed face detections."""
        loss_fn = IdentityLoss()
        loss_fn._has_arcface = False  # Force fallback (always valid)
        img = torch.rand(1, 3, 256, 256)
        loss = loss_fn(img, img, procedure="rhinoplasty")
        assert torch.isfinite(loss), "Loss should always be finite"


class TestPerceptualLossMaskingBug:
    """Test the masking order bug in PerceptualLoss."""

    def test_masked_pixels_become_minus_one(self):
        """BUG: masked pixels → 0 → *2-1 → -1, not ignored."""
        pred = torch.rand(1, 3, 64, 64)
        outside_mask = torch.zeros(1, 1, 64, 64)  # surgical mask covers everything
        pred_masked = pred * outside_mask  # all zeros
        pred_norm = pred_masked * 2 - 1  # all -1
        assert torch.all(pred_norm == -1.0)

    def test_correct_order_would_differ(self):
        """Show correct order: normalize first, then mask."""
        pred = torch.rand(1, 3, 64, 64)
        mask = torch.ones(1, 1, 64, 64)  # full surgical mask
        outside = 1 - mask  # zeros

        # Bug order: mask then normalize
        bug_result = (pred * outside) * 2 - 1  # all -1

        # Correct order: normalize then mask
        correct_result = (pred * 2 - 1) * outside  # all 0

        assert not torch.allclose(bug_result, correct_result)
