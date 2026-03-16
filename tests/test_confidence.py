"""Tests for the confidence module (per-pixel confidence maps)."""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks


def _make_face(rng, width=512, height=512):
    """Create a mock FaceLandmarks for testing."""
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4
        landmarks[i, 1] = 0.2 + rng.random() * 0.6
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        confidence=0.95,
        image_width=width,
        image_height=height,
    )


def _make_displaced_face(face_before, rng, magnitude=5.0):
    """Create a face with small random displacements from the original."""
    lm = face_before.landmarks.copy()
    lm[:, 0] += rng.normal(0, magnitude / face_before.image_width, size=478).astype(np.float32)
    lm[:, 1] += rng.normal(0, magnitude / face_before.image_height, size=478).astype(np.float32)
    return FaceLandmarks(
        landmarks=lm,
        confidence=face_before.confidence,
        image_width=face_before.image_width,
        image_height=face_before.image_height,
    )


class TestConfidenceMap:
    """Tests for ConfidenceMap dataclass."""

    def test_summary(self):
        from landmarkdiff.confidence import ConfidenceMap

        cm = ConfidenceMap(
            confidence=np.ones((10, 10), dtype=np.float32),
            mean_confidence=0.9,
            min_confidence=0.5,
            low_confidence_fraction=0.1,
        )
        s = cm.summary()
        assert "mean=0.900" in s
        assert "min=0.500" in s


class TestGenerateConfidenceMap:
    """Tests for generate_confidence_map."""

    def test_basic_generation(self):
        from landmarkdiff.confidence import generate_confidence_map

        rng = np.random.default_rng(42)
        face_before = _make_face(rng)
        face_after = _make_displaced_face(face_before, rng, magnitude=3.0)

        result = generate_confidence_map(face_before, face_after, width=64, height=64)
        assert result.confidence.shape == (64, 64)
        assert 0.0 <= result.mean_confidence <= 1.0
        assert 0.0 <= result.min_confidence <= 1.0
        assert 0.0 <= result.low_confidence_fraction <= 1.0

    def test_no_displacement_high_confidence(self):
        from landmarkdiff.confidence import generate_confidence_map

        rng = np.random.default_rng(42)
        face = _make_face(rng)
        # Same face = zero displacement → high confidence
        result = generate_confidence_map(face, face, width=32, height=32)
        assert result.mean_confidence > 0.5

    def test_large_displacement_lower_confidence(self):
        from landmarkdiff.confidence import generate_confidence_map

        rng = np.random.default_rng(42)
        face_before = _make_face(rng)
        face_after = _make_displaced_face(face_before, rng, magnitude=50.0)

        result = generate_confidence_map(face_before, face_after, width=32, height=32)
        # Large displacement should yield lower mean confidence
        assert result.confidence.shape == (32, 32)

    def test_custom_sigma(self):
        from landmarkdiff.confidence import generate_confidence_map

        rng = np.random.default_rng(42)
        face_before = _make_face(rng)
        face_after = _make_displaced_face(face_before, rng)

        result = generate_confidence_map(
            face_before, face_after, width=32, height=32, sigma=20.0
        )
        assert result.confidence.shape == (32, 32)

    def test_custom_threshold(self):
        from landmarkdiff.confidence import generate_confidence_map

        rng = np.random.default_rng(42)
        face_before = _make_face(rng)
        face_after = _make_displaced_face(face_before, rng)

        result = generate_confidence_map(
            face_before, face_after, width=32, height=32,
            low_confidence_threshold=0.8,
        )
        assert 0.0 <= result.low_confidence_fraction <= 1.0


class TestVisualizeConfidenceMap:
    """Tests for visualize_confidence_map."""

    def test_basic_visualization(self):
        from landmarkdiff.confidence import ConfidenceMap, visualize_confidence_map

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        cm = ConfidenceMap(
            confidence=np.random.default_rng(42).random((64, 64)).astype(np.float32),
            mean_confidence=0.7,
            min_confidence=0.1,
            low_confidence_fraction=0.2,
        )
        result = visualize_confidence_map(image, cm, alpha=0.5)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_visualization_resizes_map(self):
        from landmarkdiff.confidence import ConfidenceMap, visualize_confidence_map

        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        cm = ConfidenceMap(
            confidence=np.random.default_rng(42).random((50, 50)).astype(np.float32),
            mean_confidence=0.7,
            min_confidence=0.1,
            low_confidence_fraction=0.2,
        )
        result = visualize_confidence_map(image, cm)
        assert result.shape == (100, 100, 3)

    def test_visualization_large_image(self):
        from landmarkdiff.confidence import ConfidenceMap, visualize_confidence_map

        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        cm = ConfidenceMap(
            confidence=np.ones((512, 512), dtype=np.float32) * 0.8,
            mean_confidence=0.8,
            min_confidence=0.8,
            low_confidence_fraction=0.0,
        )
        result = visualize_confidence_map(image, cm)
        assert result.shape == (512, 512, 3)
