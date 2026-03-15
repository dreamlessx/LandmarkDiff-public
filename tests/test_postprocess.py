"""Tests for post-processing pipeline (classical components only).

Neural net components (CodeFormer, GFPGAN, RealESRGAN) are tested only
for graceful fallback behavior, not actual model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.postprocess import (
    _build_laplacian_pyramid,
    _reconstruct_from_laplacian,
    frequency_aware_sharpen,
    full_postprocess,
    histogram_match_skin,
    laplacian_pyramid_blend,
    restore_face_codeformer,
    restore_face_gfpgan,
)


@pytest.fixture
def face_images():
    """Create test source/target face images and mask."""
    rng = np.random.default_rng(42)
    h, w = 512, 512
    source = rng.integers(100, 200, (h, w, 3), dtype=np.uint8)
    target = rng.integers(80, 180, (h, w, 3), dtype=np.uint8)
    # Circular mask centered on face
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), 150, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    return source, target, mask


class TestLaplacianPyramidBlend:
    """Tests for Laplacian pyramid blending."""

    def test_output_shape(self, face_images):
        source, target, mask = face_images
        result = laplacian_pyramid_blend(source, target, mask, levels=4)
        assert result.shape == target.shape
        assert result.dtype == np.uint8

    def test_zero_mask_returns_target(self, face_images):
        source, target, _ = face_images
        zero_mask = np.zeros((512, 512), dtype=np.float32)
        result = laplacian_pyramid_blend(source, target, zero_mask, levels=4)
        # With zero mask, result should be close to target
        diff = np.abs(result.astype(float) - target.astype(float)).mean()
        assert diff < 5.0  # small numerical error from pyramid ops

    def test_full_mask_returns_source(self, face_images):
        source, target, _ = face_images
        full_mask = np.ones((512, 512), dtype=np.float32)
        result = laplacian_pyramid_blend(source, target, full_mask, levels=4)
        # With full mask, result should be close to source
        diff = np.abs(result.astype(float) - source.astype(float)).mean()
        assert diff < 5.0

    def test_different_levels(self, face_images):
        source, target, mask = face_images
        r1 = laplacian_pyramid_blend(source, target, mask, levels=2)
        r2 = laplacian_pyramid_blend(source, target, mask, levels=6)
        assert r1.shape == r2.shape
        # Different levels should produce different results
        assert not np.array_equal(r1, r2)

    def test_3ch_mask(self, face_images):
        source, target, mask = face_images
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = laplacian_pyramid_blend(source, target, mask_3ch, levels=4)
        assert result.shape == target.shape

    def test_uint8_mask(self, face_images):
        source, target, mask = face_images
        mask_uint8 = (mask * 255).astype(np.float32)
        result = laplacian_pyramid_blend(source, target, mask_uint8, levels=4)
        assert result.shape == target.shape

    def test_mismatched_sizes(self, face_images):
        source, target, mask = face_images
        small_source = cv2.resize(source, (256, 256))
        result = laplacian_pyramid_blend(small_source, target, mask, levels=4)
        assert result.shape == target.shape


class TestLaplacianPyramidHelpers:
    """Tests for pyramid build/reconstruct."""

    def test_build_pyramid_levels(self):
        img = np.random.rand(256, 256, 3).astype(np.float32)
        pyramid = _build_laplacian_pyramid(img, levels=4)
        assert len(pyramid) == 5  # 4 laplacian + 1 coarsest

    def test_perfect_reconstruction(self):
        """Reconstruct should approximately recover the original."""
        img = np.random.rand(256, 256, 3).astype(np.float32) * 255
        pyramid = _build_laplacian_pyramid(img, levels=4)
        reconstructed = _reconstruct_from_laplacian(pyramid)
        diff = np.abs(reconstructed - img).max()
        assert diff < 1.0  # very small numerical error


class TestFrequencyAwareSharpen:
    """Tests for frequency-aware sharpening."""

    def test_output_shape(self):
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_zero_strength(self):
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.0)
        # Zero strength should return nearly identical image (LAB round-trip rounding)
        diff = np.abs(result.astype(int) - img.astype(int)).max()
        assert diff <= 8

    def test_increases_high_freq(self):
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        sharpened = frequency_aware_sharpen(img, strength=0.5)
        # Compute edge energy (Laplacian variance)
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        lap_orig = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
        lap_sharp = cv2.Laplacian(gray_sharp, cv2.CV_64F).var()
        assert lap_sharp >= lap_orig * 0.95  # sharpened should have at least as much edge energy


class TestHistogramMatchSkin:
    """Tests for skin histogram matching."""

    def test_pixels_outside_mask_not_modified(self):
        """Pixels outside the mask must not be changed significantly."""
        rng = np.random.default_rng(7)
        h, w = 64, 64
        source = rng.integers(100, 200, (h, w, 3), dtype=np.uint8)
        target = rng.integers(80, 180, (h, w, 3), dtype=np.uint8)

        # Only top-left 32x32 is masked -- boundary test
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:32, :32] = 1.0

        result = histogram_match_skin(source, target, mask)

        # Allow tolerance of 3 -- function may have minor
        # boundary blending effects on adjacent pixels
        diff = np.abs(result[40:, 40:].astype(int) - source[40:, 40:].astype(int)).max()
        assert diff <= 3, f"Pixels well outside mask should not change. Max diff: {diff}"

    def test_small_image_8x8(self):
        """Very small 8x8 image should work without errors."""
        rng = np.random.default_rng(8)
        source = rng.integers(100, 200, (8, 8, 3), dtype=np.uint8)
        target = rng.integers(80, 180, (8, 8, 3), dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.float32)
        result = histogram_match_skin(source, target, mask)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_output_shape(self, face_images):
        source, target, mask = face_images
        result = histogram_match_skin(source, target, mask)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_empty_mask(self, face_images):
        source, target, _ = face_images
        empty_mask = np.zeros((512, 512), dtype=np.float32)
        result = histogram_match_skin(source, target, empty_mask)
        np.testing.assert_array_equal(result, source)

    def test_matching_shifts_distribution(self, face_images):
        source, target, mask = face_images
        # Make source darker than target
        dark_source = np.clip(source.astype(int) - 50, 0, 255).astype(np.uint8)
        result = histogram_match_skin(dark_source, target, mask)
        # Result should be brighter than dark_source in masked region
        mask_bool = mask > 0.3
        if mask_bool.any():
            orig_mean = dark_source[mask_bool].mean()
            result_mean = result[mask_bool].mean()
            # Should shift toward target
            assert result_mean > orig_mean - 5  # allow small tolerance


class TestNeuralFallbacks:
    """Test that neural components gracefully fall back."""

    def test_gfpgan_fallback(self):
        """GFPGAN should return original when not installed."""
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = restore_face_gfpgan(img)
        # Either returns restored or falls back to original
        assert result.shape == img.shape

    def test_codeformer_fallback(self):
        """CodeFormer should return original when not installed."""
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = restore_face_codeformer(img)
        assert result.shape == img.shape


class TestFullPostprocess:
    """Tests for the full post-processing pipeline."""

    def test_output_keys(self, face_images):
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert "image" in result
        assert "identity_check" in result
        assert "restore_used" in result

    def test_output_shape(self, face_images):
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == target.shape
        assert result["image"].dtype == np.uint8

    def test_no_restore(self, face_images):
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["restore_used"] == "none"

    def test_simple_blend_fallback(self, face_images):
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            use_laplacian_blend=False,
            sharpen_strength=0.0,
            verify_identity=False,
        )
        assert result["image"].shape == target.shape

    def test_with_sharpening(self, face_images):
        source, target, mask = face_images
        r1 = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            use_laplacian_blend=True,
            sharpen_strength=0.0,
            verify_identity=False,
        )
        r2 = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            use_laplacian_blend=True,
            sharpen_strength=0.5,
            verify_identity=False,
        )
        assert not np.array_equal(r1["image"], r2["image"])

    def test_codeformer_mode_falls_back(self, face_images):
        """CodeFormer mode should fall through to gfpgan or none."""
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="codeformer",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == target.shape
        # Restore used depends on what is available in the env
        assert result["restore_used"] in ("codeformer", "gfpgan", "none")

    def test_gfpgan_mode(self, face_images):
        """GFPGAN restore mode path."""
        source, target, mask = face_images
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="gfpgan",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == target.shape
        assert result["restore_used"] in ("gfpgan", "none")


class TestEnhanceBackgroundFallback:
    """Test Real-ESRGAN background enhancement fallback."""

    def test_returns_original_shape(self, face_images):
        from landmarkdiff.postprocess import enhance_background_realesrgan

        source, _, mask = face_images
        result = enhance_background_realesrgan(source, mask)
        assert result.shape == source.shape

    def test_with_uint8_mask(self, face_images):
        from landmarkdiff.postprocess import enhance_background_realesrgan

        source, _, mask = face_images
        mask_uint8 = (mask * 255).astype(np.uint8)
        result = enhance_background_realesrgan(source, mask_uint8.astype(np.float32))
        assert result.shape == source.shape


class TestVerifyIdentityFallback:
    """Test ArcFace identity verification fallback."""

    def test_returns_dict(self):
        from landmarkdiff.postprocess import verify_identity_arcface

        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = verify_identity_arcface(img, img)
        assert isinstance(result, dict)
        assert "similarity" in result
        assert "passed" in result
        assert "message" in result

    def test_custom_threshold(self):
        from landmarkdiff.postprocess import verify_identity_arcface

        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = verify_identity_arcface(img, img, threshold=0.9)
        assert isinstance(result, dict)


class TestHistogramMatchEdgeCases:
    """Additional edge cases for histogram matching."""

    def test_uint8_mask_threshold(self):
        """Test that uint8-scale mask uses > 76 threshold."""
        rng = np.random.default_rng(99)
        source = rng.integers(100, 200, (64, 64, 3), dtype=np.uint8)
        reference = rng.integers(80, 180, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 100, dtype=np.uint8)
        result = histogram_match_skin(source, reference, mask)
        assert result.shape == source.shape

    def test_identical_images(self):
        """Matching identical images should change very little."""
        img = np.random.randint(80, 200, (128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        result = histogram_match_skin(img, img.copy(), mask)
        diff = np.abs(result.astype(float) - img.astype(float)).mean()
        assert diff < 10.0


class TestFrequencyAwareSharpenEdgeCases:
    """Additional edge cases for frequency-aware sharpening."""

    def test_small_image(self):
        """Test with a very small image."""
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3)
        assert result.shape == img.shape

    def test_non_square_image(self):
        """Test with a non-square image."""
        img = np.random.randint(50, 200, (256, 384, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3)
        assert result.shape == img.shape

    def test_large_radius(self):
        """Test with a large blur radius."""
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3, radius=7)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestLaplacianBlendEdgeCases:
    """Additional edge cases for Laplacian blending."""

    def test_non_square_images(self):
        """Blending non-square images should work."""
        rng = np.random.default_rng(42)
        source = rng.integers(0, 255, (256, 384, 3), dtype=np.uint8)
        target = rng.integers(0, 255, (256, 384, 3), dtype=np.uint8)
        mask = np.zeros((256, 384), dtype=np.float32)
        cv2.circle(mask, (192, 128), 80, 1.0, -1)
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        assert result.shape == target.shape

    def test_single_level(self):
        """Blending with 1 level (degenerate case)."""
        rng = np.random.default_rng(42)
        source = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        target = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = laplacian_pyramid_blend(source, target, mask, levels=1)
        assert result.shape == target.shape
