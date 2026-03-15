"""Integration tests for LandmarkDiff pipeline (no GPU/torch dependency).

Tests the full pipeline from landmark extraction through conditioning,
manipulation, masking, TPS warping, and post-processing. All tests use
mock landmarks (since MediaPipe may not detect faces in synthetic images)
and avoid imports that require torch at module level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.landmarks import FaceLandmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

# Procedures supported by the masking module
MASKABLE_PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]

# All 6 procedures for manipulation tests
ALL_PROCEDURES = [
    "rhinoplasty",
    "blepharoplasty",
    "rhytidectomy",
    "orthognathic",
    "brow_lift",
    "mentoplasty",
]


@pytest.fixture
def mock_face():
    """Create a deterministic mock FaceLandmarks with anatomically plausible layout."""
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4  # x: 0.3-0.7
        landmarks[i, 1] = 0.2 + rng.random() * 0.6  # y: 0.2-0.8
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        confidence=0.95,
        image_width=512,
        image_height=512,
    )


@pytest.fixture
def synthetic_face_512():
    """Create a synthetic 512x512 face-like image (BGR)."""
    img = np.full((512, 512, 3), 180, dtype=np.uint8)
    img[:, :, 0] = 150
    img[:, :, 1] = 170
    img[:, :, 2] = 200
    cv2.ellipse(img, (256, 256), (140, 180), 0, 0, 360, (140, 160, 190), -1)
    cv2.circle(img, (200, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (312, 220), 15, (50, 50, 50), -1)
    pts = np.array([[256, 250], [240, 300], [272, 300]], np.int32)
    cv2.fillPoly(img, [pts], (130, 150, 180))
    cv2.ellipse(img, (256, 340), (40, 15), 0, 0, 360, (100, 120, 170), -1)
    noise = np.random.default_rng(42).integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class TestFullPipelineEndToEnd:
    """End-to-end pipeline tests: landmarks -> manipulation -> mask -> warp -> composite."""

    @pytest.mark.parametrize("procedure", MASKABLE_PROCEDURES)
    def test_full_tps_pipeline_per_procedure(self, mock_face, synthetic_face_512, procedure):
        """Full TPS pipeline produces valid 512x512 BGR output for each procedure."""
        manip = apply_procedure_preset(mock_face, procedure, 65.0, image_size=512)
        mask = generate_surgical_mask(mock_face, procedure, 512, 512)
        warped = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )

        # Simple alpha composite (avoids importing inference.mask_composite which needs torch)
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        composite = (
            warped.astype(np.float32) * mask_3ch
            + synthetic_face_512.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)

        assert composite.shape == (512, 512, 3)
        assert composite.dtype == np.uint8
        # Composite should differ from input
        assert not np.array_equal(composite, synthetic_face_512)

    def test_pipeline_preserves_unmasked_region(self, mock_face, synthetic_face_512):
        """Pixels outside the surgical mask should remain unchanged."""
        procedure = "rhinoplasty"
        manip = apply_procedure_preset(mock_face, procedure, 65.0, image_size=512)
        mask = generate_surgical_mask(mock_face, procedure, 512, 512)
        warped = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )

        mask_f = mask.astype(np.float32)
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        composite = (
            warped.astype(np.float32) * mask_3ch
            + synthetic_face_512.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)

        # Where mask is exactly 0, composite should equal original
        zero_mask = mask_f == 0.0
        if zero_mask.any():
            for ch in range(3):
                np.testing.assert_array_equal(
                    composite[:, :, ch][zero_mask],
                    synthetic_face_512[:, :, ch][zero_mask],
                )


class TestAllProceduresProduceDifferentOutputs:
    """Verify each procedure applies a distinct deformation pattern."""

    def test_different_procedures_different_displacements(self, mock_face):
        """Each of the 6 procedures produces a unique displacement field."""
        results = {}
        for proc in ALL_PROCEDURES:
            manip = apply_procedure_preset(mock_face, proc, 65.0, image_size=512)
            results[proc] = manip.pixel_coords.copy()

        # Pairwise comparison: each procedure should differ from every other
        procs = list(results.keys())
        for i in range(len(procs)):
            for j in range(i + 1, len(procs)):
                diff = np.abs(results[procs[i]] - results[procs[j]]).sum()
                assert diff > 0.0, f"{procs[i]} and {procs[j]} produced identical displacements"

    def test_all_procedures_move_landmarks(self, mock_face):
        """Each procedure actually displaces at least some landmarks."""
        for proc in ALL_PROCEDURES:
            manip = apply_procedure_preset(mock_face, proc, 65.0, image_size=512)
            total_displacement = np.abs(manip.pixel_coords - mock_face.pixel_coords).sum()
            assert total_displacement > 1.0, f"{proc} produced near-zero total displacement"


class TestIntensitySweep:
    """Verify intensity parameter produces monotonically increasing deformation."""

    @pytest.mark.parametrize("procedure", MASKABLE_PROCEDURES)
    def test_monotonic_displacement_with_intensity(self, mock_face, procedure):
        """Higher intensity should produce larger total displacement."""
        intensities = [10.0, 30.0, 50.0, 70.0, 90.0]
        displacements = []
        for intensity in intensities:
            manip = apply_procedure_preset(mock_face, procedure, intensity, image_size=512)
            disp = np.linalg.norm(manip.pixel_coords - mock_face.pixel_coords, axis=1).sum()
            displacements.append(disp)

        # Each step should be >= previous (monotonic non-decreasing)
        for i in range(1, len(displacements)):
            assert displacements[i] >= displacements[i - 1] - 1e-3, (
                f"{procedure} intensity sweep not monotonic at "
                f"intensity={intensities[i]}: {displacements[i]:.4f} < {displacements[i - 1]:.4f}"
            )

    def test_zero_intensity_no_displacement(self, mock_face):
        """Intensity 0 should produce no displacement."""
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 0.0, image_size=512)
        diff = np.abs(manip.pixel_coords - mock_face.pixel_coords).max()
        assert diff < 1e-5, f"Zero intensity produced displacement: {diff}"


class TestConditioningPipeline:
    """Test conditioning signal generation from landmarks."""

    def test_conditioning_shapes(self, mock_face):
        """Conditioning outputs have correct shape and dtype."""
        landmark_img, canny, wireframe = generate_conditioning(mock_face, 512, 512)
        assert landmark_img.shape == (512, 512, 3)
        assert canny.shape == (512, 512)
        assert wireframe.shape == (512, 512)
        assert landmark_img.dtype == np.uint8
        assert canny.dtype == np.uint8
        assert wireframe.dtype == np.uint8

    def test_conditioning_not_blank(self, mock_face):
        """Conditioning images should have non-zero content."""
        landmark_img, canny, wireframe = generate_conditioning(mock_face, 512, 512)
        assert landmark_img.sum() > 0, "Landmark image is blank"
        assert wireframe.sum() > 0, "Wireframe is blank"

    def test_manipulated_conditioning_differs(self, mock_face):
        """Conditioning from manipulated landmarks should differ from original."""
        _, _, wf_orig = generate_conditioning(mock_face, 512, 512)
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 80.0, image_size=512)
        _, _, wf_manip = generate_conditioning(manip, 512, 512)
        assert not np.array_equal(wf_orig, wf_manip)

    def test_render_landmark_image(self, mock_face):
        """render_landmark_image produces a valid colored mesh image."""
        img = render_landmark_image(mock_face, 512, 512)
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8
        assert img.sum() > 0


class TestMaskingPipeline:
    """Test surgical mask generation."""

    @pytest.mark.parametrize("procedure", MASKABLE_PROCEDURES)
    def test_mask_shape_and_range(self, mock_face, procedure):
        """Masks should be 512x512 float32 in [0, 1]."""
        mask = generate_surgical_mask(mock_face, procedure, 512, 512)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    @pytest.mark.parametrize("procedure", MASKABLE_PROCEDURES)
    def test_mask_not_empty(self, mock_face, procedure):
        """Each procedure mask should have non-zero area."""
        mask = generate_surgical_mask(mock_face, procedure, 512, 512)
        assert mask.sum() > 0, f"{procedure} mask is empty"

    def test_different_procedures_different_masks(self, mock_face):
        """Different procedures should produce different mask regions."""
        masks = {}
        for proc in MASKABLE_PROCEDURES:
            masks[proc] = generate_surgical_mask(mock_face, proc, 512, 512)

        for i, p1 in enumerate(MASKABLE_PROCEDURES):
            for p2 in MASKABLE_PROCEDURES[i + 1 :]:
                assert not np.array_equal(masks[p1], masks[p2]), (
                    f"{p1} and {p2} masks are identical"
                )

    def test_invalid_procedure_raises(self, mock_face):
        """Unknown procedure should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown procedure"):
            generate_surgical_mask(mock_face, "nonexistent_surgery", 512, 512)


class TestTPSWarp:
    """Test TPS warping module."""

    def test_warp_preserves_shape(self, synthetic_face_512):
        """Warped output has same shape as input."""
        src = np.random.default_rng(0).random((50, 2)).astype(np.float32) * 512
        dst = src + np.random.default_rng(1).standard_normal((50, 2)).astype(np.float32) * 3
        result = warp_image_tps(synthetic_face_512, src, dst)
        assert result.shape == synthetic_face_512.shape
        assert result.dtype == np.uint8

    def test_identity_warp(self, synthetic_face_512):
        """Warping with identical src and dst should preserve the image."""
        pts = np.random.default_rng(42).random((50, 2)).astype(np.float32) * 450 + 30
        result = warp_image_tps(synthetic_face_512, pts, pts)
        # Should be very close to original (small numerical error possible)
        diff = np.abs(result.astype(float) - synthetic_face_512.astype(float)).mean()
        assert diff < 5.0, f"Identity warp produced mean diff of {diff}"

    def test_warp_with_displacement(self, mock_face, synthetic_face_512):
        """Warping with manipulated landmarks produces a different image."""
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 80.0, image_size=512)
        warped = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )
        assert not np.array_equal(warped, synthetic_face_512)


class TestPostProcessPipeline:
    """Test post-processing pipeline components."""

    def test_laplacian_blend_roundtrip(self, synthetic_face_512):
        """Laplacian blend with full mask returns source."""
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        full_mask = np.ones((512, 512), dtype=np.float32)
        target = np.random.default_rng(99).integers(50, 200, (512, 512, 3), dtype=np.uint8)
        result = laplacian_pyramid_blend(synthetic_face_512, target, full_mask, levels=4)
        diff = np.abs(result.astype(float) - synthetic_face_512.astype(float)).mean()
        assert diff < 5.0

    def test_histogram_match_preserves_shape(self, synthetic_face_512):
        """Histogram matching output has correct shape."""
        from landmarkdiff.postprocess import histogram_match_skin

        target = np.random.default_rng(99).integers(50, 200, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.circle(mask, (256, 256), 100, 1.0, -1)
        result = histogram_match_skin(synthetic_face_512, target, mask)
        assert result.shape == synthetic_face_512.shape
        assert result.dtype == np.uint8

    def test_frequency_sharpen_preserves_shape(self, synthetic_face_512):
        """Sharpening output has correct shape and dtype."""
        from landmarkdiff.postprocess import frequency_aware_sharpen

        result = frequency_aware_sharpen(synthetic_face_512, strength=0.3)
        assert result.shape == synthetic_face_512.shape
        assert result.dtype == np.uint8

    def test_full_postprocess_pipeline(self, mock_face, synthetic_face_512):
        """Full post-processing pipeline runs without error."""
        from landmarkdiff.postprocess import full_postprocess

        mask = generate_surgical_mask(mock_face, "rhinoplasty", 512, 512)
        # Use synthetic image as both source and target for shape/type checks
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 65.0, image_size=512)
        warped = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )
        result = full_postprocess(
            warped,
            synthetic_face_512,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert "image" in result
        assert result["image"].shape == (512, 512, 3)
        assert result["image"].dtype == np.uint8


class TestFaceLandmarksObject:
    """Test FaceLandmarks dataclass behavior."""

    def test_pixel_coords_shape(self, mock_face):
        """pixel_coords should be (478, 2) float32."""
        assert mock_face.pixel_coords.shape == (478, 2)
        assert mock_face.pixel_coords.dtype == np.float32

    def test_pixel_coords_in_image_bounds(self, mock_face):
        """pixel_coords should be within image dimensions."""
        coords = mock_face.pixel_coords
        assert coords[:, 0].min() >= 0
        assert coords[:, 0].max() <= mock_face.image_width
        assert coords[:, 1].min() >= 0
        assert coords[:, 1].max() <= mock_face.image_height

    def test_landmarks_shape(self, mock_face):
        """Raw landmarks should be (478, 3)."""
        assert mock_face.landmarks.shape == (478, 3)

    def test_confidence(self, mock_face):
        """Confidence should be in [0, 1]."""
        assert 0.0 <= mock_face.confidence <= 1.0


class TestDeterminism:
    """Verify pipeline reproducibility with fixed seeds."""

    def test_manipulation_deterministic(self, mock_face):
        """Same inputs produce identical manipulation output."""
        r1 = apply_procedure_preset(mock_face, "rhinoplasty", 65.0, image_size=512)
        r2 = apply_procedure_preset(mock_face, "rhinoplasty", 65.0, image_size=512)
        np.testing.assert_array_equal(r1.pixel_coords, r2.pixel_coords)

    def test_conditioning_deterministic(self, mock_face):
        """Same landmarks produce identical conditioning."""
        lm1, _, wf1 = generate_conditioning(mock_face, 512, 512)
        lm2, _, wf2 = generate_conditioning(mock_face, 512, 512)
        np.testing.assert_array_equal(lm1, lm2)
        np.testing.assert_array_equal(wf1, wf2)

    def test_mask_deterministic(self, mock_face):
        """Same landmarks produce nearly identical masks (small float rounding allowed)."""
        m1 = generate_surgical_mask(mock_face, "rhinoplasty", 512, 512)
        m2 = generate_surgical_mask(mock_face, "rhinoplasty", 512, 512)
        np.testing.assert_allclose(m1, m2, atol=0.02)

    def test_tps_deterministic(self, mock_face, synthetic_face_512):
        """Same control points produce identical TPS warp."""
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 65.0, image_size=512)
        w1 = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )
        w2 = warp_image_tps(
            synthetic_face_512,
            mock_face.pixel_coords,
            manip.pixel_coords,
        )
        np.testing.assert_array_equal(w1, w2)


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_max_intensity(self, mock_face):
        """Intensity 100 should not produce NaN or Inf."""
        manip = apply_procedure_preset(mock_face, "rhinoplasty", 100.0, image_size=512)
        assert np.isfinite(manip.pixel_coords).all()

    def test_small_image_size(self, mock_face):
        """Pipeline should handle smaller image sizes (256)."""
        small_face = FaceLandmarks(
            landmarks=mock_face.landmarks.copy(),
            confidence=mock_face.confidence,
            image_width=256,
            image_height=256,
        )
        manip = apply_procedure_preset(small_face, "rhinoplasty", 65.0, image_size=256)
        assert manip.pixel_coords.shape == (478, 2)
        assert np.isfinite(manip.pixel_coords).all()

    def test_warp_small_image(self):
        """TPS warp works on a 64x64 image."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        src = np.random.default_rng(0).random((20, 2)).astype(np.float32) * 64
        dst = src + np.random.default_rng(1).standard_normal((20, 2)).astype(np.float32)
        result = warp_image_tps(img, src, dst)
        assert result.shape == (64, 64, 3)

    def test_mask_custom_resolution(self, mock_face):
        """Masks work at non-standard resolution."""
        mask = generate_surgical_mask(mock_face, "rhinoplasty", 256, 256)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
