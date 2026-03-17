"""Post-processing tests covering ImportError fallbacks and classical pipeline paths.

Covers: restore_face_gfpgan/codeformer/realesrgan ImportError fallbacks,
CodeFormer fidelity clamping, verify_identity_arcface InsightFace fallback,
_has_cuda fallback, full_postprocess classical-only pipeline, simple alpha
blend, histogram_match_skin edge cases.

NOTE: gfpgan, codeformer, realesrgan, and basicsr all fail to import naturally
(version incompatibility with torchvision or not installed). We call the
functions directly and let the ImportError cascade trigger the fallback paths
without any sys.modules mocking, which avoids corrupting torch's internal state.
Only insightface is mocked (it IS importable in this env).
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import numpy as np

from landmarkdiff.postprocess import (
    _has_cuda,
    enhance_background_realesrgan,
    frequency_aware_sharpen,
    full_postprocess,
    histogram_match_skin,
    laplacian_pyramid_blend,
    restore_face_codeformer,
    restore_face_gfpgan,
    texture_aware_blend,
    verify_identity_arcface,
)


def _make_images(h=64, w=64):
    """Create a pair of test images and a mask."""
    rng = np.random.default_rng(42)
    source = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    target = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return source, target, mask


# ---------------------------------------------------------------------------
# restore_face_gfpgan — ImportError fallback (lines 250-252)
# gfpgan is installed but fails due to basicsr/torchvision incompatibility
# ---------------------------------------------------------------------------


class TestRestoreFaceGFPGAN:
    def test_returns_original_when_gfpgan_unavailable(self):
        """gfpgan import fails naturally; returns the input image."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = restore_face_gfpgan(img)
        assert result is img


# ---------------------------------------------------------------------------
# restore_face_codeformer — ImportError fallback + fidelity clamping
# codeformer is not installed at all
# ---------------------------------------------------------------------------


class TestRestoreFaceCodeFormer:
    def test_returns_original_when_codeformer_unavailable(self):
        """codeformer is not installed; returns input image unchanged."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = restore_face_codeformer(img)
        assert result is img

    def test_fidelity_clamping_warning(self, capsys):
        """Low fidelity triggers warning and clamps to minimum.

        The fidelity clamping and warning happen BEFORE any imports.
        In the full test suite, other tests may modify logging config so
        the warning goes to stderr rather than caplog; we check both.
        """
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with contextlib.suppress(Exception):
            restore_face_codeformer(img, fidelity=0.01)
        captured = capsys.readouterr()
        assert "clamping" in captured.err.lower() or "fidelity" in captured.err.lower()


# ---------------------------------------------------------------------------
# enhance_background_realesrgan — ImportError fallback (lines 403-408)
# realesrgan/basicsr fail naturally due to torchvision incompatibility
# ---------------------------------------------------------------------------


class TestEnhanceBackgroundRealESRGAN:
    def test_returns_original_when_realesrgan_unavailable(self):
        """realesrgan import fails naturally; returns the input image."""
        source, _, mask = _make_images()
        result = enhance_background_realesrgan(source, mask)
        assert result is source


# ---------------------------------------------------------------------------
# verify_identity_arcface — InsightFace fallback (lines 482-489)
# insightface IS installed, so we must mock it
# ---------------------------------------------------------------------------


class TestVerifyIdentityArcface:
    def test_returns_skip_when_insightface_unavailable(self):
        """When insightface is not installed, returns skip result."""
        source, target, _ = _make_images()
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            result = verify_identity_arcface(source, target)
        assert result["similarity"] == -1.0
        assert result["passed"] is True
        msg = result["message"].lower()
        assert "skipped" in msg or "not installed" in msg


# ---------------------------------------------------------------------------
# _has_cuda — works normally (torch IS installed)
# ---------------------------------------------------------------------------


class TestHasCuda:
    def test_returns_bool(self):
        """_has_cuda should return a boolean."""
        result = _has_cuda()
        assert isinstance(result, bool)

    def test_returns_false_when_torch_unavailable(self):
        """When torch is not installed, _has_cuda returns False."""
        with patch.dict("sys.modules", {"torch": None}):
            assert _has_cuda() is False


# ---------------------------------------------------------------------------
# histogram_match_skin — edge cases (line 586: empty mask)
# ---------------------------------------------------------------------------


class TestHistogramMatchSkin:
    def test_empty_mask_returns_source(self):
        """If mask has no active pixels, source is returned unchanged."""
        source, reference, _ = _make_images()
        empty_mask = np.zeros((64, 64), dtype=np.float32)
        result = histogram_match_skin(source, reference, empty_mask)
        np.testing.assert_array_equal(result, source)

    def test_3d_mask(self):
        """3-channel mask should work (line 564-565)."""
        source, reference, mask = _make_images()
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = histogram_match_skin(source, reference, mask_3ch)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_uint8_mask(self):
        """Uint8 mask (0-255 range) should work."""
        source, reference, _ = _make_images()
        mask_u8 = np.zeros((64, 64), dtype=np.uint8)
        mask_u8[16:48, 16:48] = 255
        result = histogram_match_skin(source, reference, mask_u8)
        assert result.shape == source.shape


# ---------------------------------------------------------------------------
# texture_aware_blend — mask normalization (line 204: mask > 1.0)
# ---------------------------------------------------------------------------


class TestTextureAwareBlend:
    def test_uint8_mask_normalized(self):
        """texture_aware_blend with 0-255 mask should normalize it."""
        source, target, _ = _make_images()
        mask_255 = np.zeros((64, 64), dtype=np.float32)
        mask_255[16:48, 16:48] = 255.0
        result = texture_aware_blend(source, target, mask_255)
        assert result.shape == source.shape
        assert result.dtype == np.uint8

    def test_different_sizes(self):
        """texture_aware_blend resizes generated to match original."""
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        generated = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = texture_aware_blend(original, generated, mask)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# laplacian_pyramid_blend — mask resize branch (line 97)
# ---------------------------------------------------------------------------


class TestLaplacianPyramidBlend:
    def test_non_divisible_dimensions(self):
        """Images with non-power-of-2 dimensions should still work."""
        source = np.random.default_rng(0).integers(0, 255, (65, 67, 3), dtype=np.uint8)
        target = np.random.default_rng(1).integers(0, 255, (65, 67, 3), dtype=np.uint8)
        mask = np.zeros((65, 67), dtype=np.float32)
        mask[10:55, 10:57] = 1.0
        result = laplacian_pyramid_blend(source, target, mask)
        assert result.shape == (65, 67, 3)
        assert result.dtype == np.uint8

    def test_uint8_mask_blend(self):
        """Blend with uint8 mask (0-255 range)."""
        source, target, _ = _make_images()
        mask_u8 = np.zeros((64, 64), dtype=np.uint8)
        mask_u8[16:48, 16:48] = 255
        result = laplacian_pyramid_blend(source, target, mask_u8.astype(np.float32))
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# frequency_aware_sharpen — basic smoke test
# ---------------------------------------------------------------------------


class TestFrequencyAwareSharpen:
    def test_output_shape(self):
        img = np.random.default_rng(0).integers(50, 200, (64, 64, 3), dtype=np.uint8)
        result = frequency_aware_sharpen(img, strength=0.3, radius=3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# full_postprocess — classical-only pipeline (all neural nets unavailable)
# No sys.modules mocking needed — all neural imports fail naturally
# except insightface which we mock.
# ---------------------------------------------------------------------------


class TestFullPostprocess:
    def test_classical_pipeline_codeformer(self):
        """full_postprocess with codeformer mode — falls back through classical.

        Mock restore_face_codeformer and restore_face_gfpgan at function level
        to avoid torch state corruption issues in the full test suite.
        """
        source, target, mask = _make_images()
        with (
            patch("landmarkdiff.postprocess.restore_face_codeformer", return_value=source),
            patch("landmarkdiff.postprocess.restore_face_gfpgan", return_value=source),
            patch("landmarkdiff.postprocess.enhance_background_realesrgan", return_value=source),
            patch.dict("sys.modules", {"insightface": None, "insightface.app": None}),
        ):
            result = full_postprocess(
                source,
                target,
                mask,
                restore_mode="codeformer",
                verify_identity=True,
            )
        assert "image" in result
        assert result["image"].shape == source.shape
        assert result["identity_check"]["passed"] is True

    def test_simple_alpha_blend(self):
        """full_postprocess with use_laplacian_blend=False uses simple alpha."""
        source, target, mask = _make_images()
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            use_laplacian_blend=False,
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == source.shape
        assert result["restore_used"] == "none"

    def test_gfpgan_mode(self):
        """full_postprocess with restore_mode='gfpgan' falls back cleanly.

        Mock gfpgan to None because when codeformer has already loaded basicsr,
        gfpgan's import triggers a duplicate ARCH_REGISTRY error (AssertionError
        not ImportError). The mock ensures a clean ImportError fallback.
        """
        source, target, mask = _make_images()
        with patch.dict("sys.modules", {"gfpgan": None}):
            result = full_postprocess(
                source,
                target,
                mask,
                restore_mode="gfpgan",
                use_realesrgan=False,
                verify_identity=False,
            )
        assert result["image"].shape == source.shape

    def test_no_sharpen(self):
        """full_postprocess with sharpen_strength=0 skips sharpening."""
        source, target, mask = _make_images()
        result = full_postprocess(
            source,
            target,
            mask,
            restore_mode="none",
            sharpen_strength=0.0,
            use_realesrgan=False,
            verify_identity=False,
        )
        assert result["image"].shape == source.shape
