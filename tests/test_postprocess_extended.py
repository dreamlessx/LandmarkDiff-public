"""Extended tests for postprocess module -- classical compositing functions."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Laplacian pyramid blending
# ---------------------------------------------------------------------------


class TestLaplacianPyramidBlend:
    """Tests for laplacian_pyramid_blend."""

    def test_basic_blend(self):
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        source = np.full((64, 64, 3), 200, dtype=np.uint8)
        target = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_full_mask(self):
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        source = np.full((64, 64, 3), 200, dtype=np.uint8)
        target = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        # With full mask, should be close to source
        assert result.mean() > 150

    def test_zero_mask(self):
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        source = np.full((64, 64, 3), 200, dtype=np.uint8)
        target = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        # With zero mask, should be close to target
        assert result.mean() < 100

    def test_uint8_mask(self):
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        source = np.full((64, 64, 3), 200, dtype=np.uint8)
        target = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        assert result.shape == (64, 64, 3)

    def test_different_sizes(self):
        from landmarkdiff.postprocess import laplacian_pyramid_blend

        source = np.full((128, 128, 3), 200, dtype=np.uint8)
        target = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = laplacian_pyramid_blend(source, target, mask, levels=3)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Frequency-aware sharpening
# ---------------------------------------------------------------------------


class TestFrequencyAwareSharpen:
    """Tests for frequency_aware_sharpen."""

    def test_basic_sharpen(self):
        from landmarkdiff.postprocess import frequency_aware_sharpen

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_no_sharpen(self):
        from landmarkdiff.postprocess import frequency_aware_sharpen

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.0)
        assert result.shape == (64, 64, 3)

    def test_high_sharpen(self):
        from landmarkdiff.postprocess import frequency_aware_sharpen

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.8)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Texture-aware blending
# ---------------------------------------------------------------------------


class TestTextureAwareBlend:
    """Tests for texture_aware_blend."""

    def test_basic_blend(self):
        from landmarkdiff.postprocess import texture_aware_blend

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = texture_aware_blend(original, generated, mask)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_full_original_texture(self):
        from landmarkdiff.postprocess import texture_aware_blend

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = texture_aware_blend(original, generated, mask, texture_weight=1.0)
        assert result.shape == (64, 64, 3)

    def test_no_original_texture(self):
        from landmarkdiff.postprocess import texture_aware_blend

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = texture_aware_blend(original, generated, mask, texture_weight=0.0)
        assert result.shape == (64, 64, 3)

    def test_different_gen_size(self):
        from landmarkdiff.postprocess import texture_aware_blend

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        generated = np.full((128, 128, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = texture_aware_blend(original, generated, mask)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Histogram matching
# ---------------------------------------------------------------------------


class TestHistogramMatchSkin:
    """Tests for histogram_match_skin."""

    def test_basic_match(self):
        from landmarkdiff.postprocess import histogram_match_skin

        source = np.full((64, 64, 3), 120, dtype=np.uint8)
        reference = np.full((64, 64, 3), 160, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = histogram_match_skin(source, reference, mask)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_no_mask(self):
        from landmarkdiff.postprocess import histogram_match_skin

        source = np.full((64, 64, 3), 120, dtype=np.uint8)
        reference = np.full((64, 64, 3), 160, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = histogram_match_skin(source, reference, mask)
        # No mask → should return source unchanged
        assert result.shape == (64, 64, 3)

    def test_3ch_mask(self):
        from landmarkdiff.postprocess import histogram_match_skin

        source = np.full((64, 64, 3), 120, dtype=np.uint8)
        reference = np.full((64, 64, 3), 160, dtype=np.uint8)
        mask = np.ones((64, 64, 3), dtype=np.float32)
        result = histogram_match_skin(source, reference, mask)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Full postprocess pipeline (mocked neural, classical only)
# ---------------------------------------------------------------------------


class TestFullPostprocess:
    """Tests for full_postprocess with restore_mode='none'."""

    def test_no_restore(self):
        from landmarkdiff.postprocess import full_postprocess

        generated = np.full((64, 64, 3), 150, dtype=np.uint8)
        original = np.full((64, 64, 3), 120, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = full_postprocess(
            generated,
            original,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == (64, 64, 3)
        assert result["restore_used"] == "none"

    def test_simple_alpha_blend(self):
        from landmarkdiff.postprocess import full_postprocess

        generated = np.full((64, 64, 3), 200, dtype=np.uint8)
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = full_postprocess(
            generated,
            original,
            mask,
            restore_mode="none",
            use_realesrgan=False,
            use_laplacian_blend=False,
            sharpen_strength=0.0,
            verify_identity=False,
        )
        assert result["image"].shape == (64, 64, 3)
