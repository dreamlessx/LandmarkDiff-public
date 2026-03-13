"""Tests for conditioning signal generation."""

import numpy as np

from landmarkdiff.conditioning import auto_canny, render_wireframe
from landmarkdiff.landmarks import FaceLandmarks


def _make_fake_face() -> FaceLandmarks:
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(landmarks=landmarks, image_width=512, image_height=512, confidence=0.9)


class TestWireframe:
    def test_wireframe_shape(self):
        face = _make_fake_face()
        wf = render_wireframe(face, 512, 512)
        assert wf.shape == (512, 512)
        assert wf.dtype == np.uint8

    def test_wireframe_has_lines(self):
        face = _make_fake_face()
        wf = render_wireframe(face, 512, 512)
        assert np.sum(wf > 0) > 100  # should have substantial line content


class TestAutoCanny:
    def test_canny_output_binary(self):
        img = np.random.default_rng(0).integers(0, 256, (512, 512), dtype=np.uint8)
        edges = auto_canny(img)
        unique = np.unique(edges)
        assert all(v in (0, 255) for v in unique)

    def test_canny_on_blank(self):
        blank = np.zeros((512, 512), dtype=np.uint8)
        edges = auto_canny(blank)
        assert np.sum(edges) == 0
