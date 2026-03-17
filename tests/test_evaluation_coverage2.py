"""Additional evaluation tests targeting remaining uncovered lines.

Covers: compute_lpips ImportError fallback, compute_fid ImportError,
classify_fitzpatrick_ita with cv2=None, compute_identity_similarity fallback,
evaluate_batch Fitzpatrick exception branch.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from landmarkdiff.evaluation import (
    compute_fid,
    compute_identity_similarity,
    compute_lpips,
    evaluate_batch,
)

# ---------------------------------------------------------------------------
# compute_lpips — ImportError fallback (lines 229-230)
# ---------------------------------------------------------------------------


class TestComputeLpipsImportError:
    def test_returns_nan_when_lpips_unavailable(self):
        """When lpips/torch is not installed, returns NaN."""
        with patch.dict("sys.modules", {"lpips": None}):
            a = np.zeros((64, 64, 3), dtype=np.uint8)
            b = np.zeros((64, 64, 3), dtype=np.uint8)
            result = compute_lpips(a, b)
            assert np.isnan(result)


# ---------------------------------------------------------------------------
# compute_fid — ImportError (lines 258-263)
# ---------------------------------------------------------------------------


class TestComputeFidImportError:
    def test_raises_import_error_when_torch_fidelity_unavailable(self):
        """When torch-fidelity is not installed, raises ImportError."""
        import pytest

        with (
            patch.dict("sys.modules", {"torch_fidelity": None}),
            pytest.raises(ImportError, match="torch-fidelity"),
        ):
            compute_fid("/tmp/real", "/tmp/fake")


# ---------------------------------------------------------------------------
# compute_identity_similarity — fallback to SSIM (lines 305-312)
# ---------------------------------------------------------------------------


class TestComputeIdentitySimilarityFallback:
    def test_falls_back_to_ssim_when_insightface_unavailable(self):
        """Without insightface, falls back to SSIM-based proxy."""
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            a = np.random.default_rng(0).integers(50, 200, (64, 64, 3), dtype=np.uint8)
            sim = compute_identity_similarity(a, a)
            assert isinstance(sim, float)
            # SSIM of identical images should be near 1.0
            assert sim > 0.8

    def test_different_images_lower_similarity(self):
        """Different images should have lower fallback similarity."""
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            a = np.zeros((64, 64, 3), dtype=np.uint8)
            b = np.full((64, 64, 3), 255, dtype=np.uint8)
            sim = compute_identity_similarity(a, b)
            assert isinstance(sim, float)
            assert sim < 0.5


# ---------------------------------------------------------------------------
# classify_fitzpatrick_ita with cv2=None — line 110
# ---------------------------------------------------------------------------


class TestClassifyFitzpatrickNoCv2:
    def test_raises_import_error(self):
        """When cv2 is None, classify_fitzpatrick_ita raises ImportError."""
        import pytest

        with patch("landmarkdiff.evaluation.cv2", None):
            from landmarkdiff.evaluation import classify_fitzpatrick_ita

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            with pytest.raises(ImportError, match="opencv"):
                classify_fitzpatrick_ita(img)


# ---------------------------------------------------------------------------
# evaluate_batch — Fitzpatrick exception branch (lines 364-365)
# ---------------------------------------------------------------------------


class TestEvaluateBatchFitzpatrickException:
    def test_fitzpatrick_exception_handled(self):
        """If classify_fitzpatrick_ita raises, evaluate_batch continues."""
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
        targets = [p.copy() for p in preds]

        # Mock both classify_fitzpatrick_ita (to raise) and compute_lpips
        # (to avoid torch reimport issues in some environments)
        with (
            patch(
                "landmarkdiff.evaluation.classify_fitzpatrick_ita",
                side_effect=ValueError("bad image"),
            ),
            patch(
                "landmarkdiff.evaluation.compute_lpips",
                return_value=0.1,
            ),
        ):
            metrics = evaluate_batch(preds, targets)
        # Should complete without error even though classification failed
        assert metrics.ssim > 0
        # No Fitzpatrick groups populated
        assert len(metrics.count_by_fitzpatrick) == 0
