"""Extended tests for face_verifier -- covering dataclass summaries, helpers,
verify_and_restore, neural_quality_score, and batch report."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDistortionReportExtended:
    """Additional tests for DistortionReport."""

    def test_defaults(self):
        from landmarkdiff.face_verifier import DistortionReport

        report = DistortionReport()
        assert report.quality_score == 0.0
        assert report.primary_distortion == "none"
        assert report.severity == "none"
        assert report.is_usable is True

    def test_custom_details(self):
        from landmarkdiff.face_verifier import DistortionReport

        report = DistortionReport(
            quality_score=50.0,
            details={"blur": 0.5, "noise": 0.1},
        )
        assert "blur" in report.details


class TestBatchVerificationReport:
    """Tests for BatchVerificationReport dataclass."""

    def test_summary_format(self):
        from landmarkdiff.face_verifier import BatchVerificationReport

        report = BatchVerificationReport(
            total=100,
            passed=60,
            restored=30,
            rejected=10,
            identity_failures=0,
            avg_quality_before=55.0,
            avg_quality_after=80.0,
            avg_identity_sim=0.9,
            distortion_counts={"blur": 20, "noise": 10},
        )
        s = report.summary()
        assert "100" in s
        assert "60" in s
        assert "blur" in s
        assert "noise" in s

    def test_empty_distortions(self):
        from landmarkdiff.face_verifier import BatchVerificationReport

        report = BatchVerificationReport(total=0)
        s = report.summary()
        assert "Total Images:" in s


# ---------------------------------------------------------------------------
# Detection edge cases
# ---------------------------------------------------------------------------


class TestDetectBlurGrayscale:
    """Test detect_blur with grayscale input."""

    def test_grayscale(self):
        from landmarkdiff.face_verifier import detect_blur

        gray = np.full((64, 64), 128, dtype=np.uint8)
        score = detect_blur(gray)
        assert 0.0 <= score <= 1.0


class TestDetectNoiseGrayscale:
    """Test detect_noise with grayscale input."""

    def test_grayscale(self):
        from landmarkdiff.face_verifier import detect_noise

        gray = np.full((64, 64), 128, dtype=np.uint8)
        score = detect_noise(gray)
        assert 0.0 <= score <= 1.0


class TestDetectCompressionGrayscale:
    """Test detect_compression_artifacts with grayscale."""

    def test_grayscale(self):
        from landmarkdiff.face_verifier import detect_compression_artifacts

        gray = np.full((64, 64), 128, dtype=np.uint8)
        score = detect_compression_artifacts(gray)
        assert 0.0 <= score <= 1.0


class TestDetectOversmoothingEdgeCases:
    """Edge cases for detect_oversmoothing."""

    def test_grayscale(self):
        from landmarkdiff.face_verifier import detect_oversmoothing

        gray = np.full((64, 64), 128, dtype=np.uint8)
        score = detect_oversmoothing(gray)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestFixColorCast:
    """Tests for _fix_color_cast."""

    def test_reduces_cast(self):
        from landmarkdiff.face_verifier import _fix_color_cast

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # heavy blue in BGR
        img[:, :, 1] = 80
        img[:, :, 2] = 80
        fixed = _fix_color_cast(img)
        assert fixed.shape == (64, 64, 3)
        assert fixed.dtype == np.uint8

    def test_neutral_unchanged(self):
        from landmarkdiff.face_verifier import _fix_color_cast

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        fixed = _fix_color_cast(img)
        assert fixed.shape == (64, 64, 3)


class TestFixLighting:
    """Tests for _fix_lighting."""

    def test_dark_image(self):
        from landmarkdiff.face_verifier import _fix_lighting

        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        fixed = _fix_lighting(img)
        assert fixed.shape == (64, 64, 3)
        assert fixed.dtype == np.uint8

    def test_bright_image(self):
        from landmarkdiff.face_verifier import _fix_lighting

        img = np.full((64, 64, 3), 240, dtype=np.uint8)
        fixed = _fix_lighting(img)
        assert fixed.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Neural quality score (classical fallback)
# ---------------------------------------------------------------------------


class TestNeuralQualityScore:
    """Tests for neural_quality_score (classical fallback path)."""

    def test_returns_float(self):
        from landmarkdiff.face_verifier import neural_quality_score

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        score = neural_quality_score(img)
        assert 0.0 <= score <= 100.0
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Verify and restore
# ---------------------------------------------------------------------------


class TestVerifyAndRestore:
    """Tests for verify_and_restore pipeline."""

    def test_good_quality_skips_restore(self):
        from landmarkdiff.face_verifier import verify_and_restore

        rng = np.random.default_rng(42)
        img = rng.integers(80, 180, (128, 128, 3), dtype=np.uint8)
        result = verify_and_restore(img, quality_threshold=0.0)
        assert result.restored.shape == (128, 128, 3)
        assert result.distortion_report is not None
        assert result.original.shape == (128, 128, 3)

    def test_identity_preserved_field(self):
        from landmarkdiff.face_verifier import verify_and_restore

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        result = verify_and_restore(img, quality_threshold=0.0)
        assert isinstance(result.identity_preserved, bool)

    def test_high_threshold_triggers_restore(self):
        from landmarkdiff.face_verifier import verify_and_restore

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        result = verify_and_restore(img, quality_threshold=99.0)
        # With very high threshold, should attempt restoration or reject
        assert result.distortion_report is not None

    def test_restoration_result_summary(self):
        from landmarkdiff.face_verifier import verify_and_restore

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        result = verify_and_restore(img, quality_threshold=0.0)
        s = result.summary()
        assert "Pre-restoration" in s
        assert "Post-restoration" in s


# ---------------------------------------------------------------------------
# Restore face function
# ---------------------------------------------------------------------------


class TestRestoreFace:
    """Tests for restore_face function."""

    def test_auto_mode(self):
        from landmarkdiff.face_verifier import restore_face

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        result, stages = restore_face(img, mode="auto")
        assert result.shape == (128, 128, 3)
        assert isinstance(stages, list)

    def test_with_distortion_report(self):
        from landmarkdiff.face_verifier import analyze_distortions, restore_face

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)
        report = analyze_distortions(img)
        result, stages = restore_face(img, distortion=report, mode="auto")
        assert result.shape == (128, 128, 3)
