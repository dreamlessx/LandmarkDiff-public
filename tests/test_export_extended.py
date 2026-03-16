"""Extended tests for export module (GIF, mesh export)."""

from __future__ import annotations

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


def _make_face(seed=42):
    """Create a mock FaceLandmarks."""
    rng = np.random.default_rng(seed)
    lm = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        lm[i, 0] = 0.3 + rng.random() * 0.4
        lm[i, 1] = 0.2 + rng.random() * 0.6
        lm[i, 2] = rng.random() * 0.1
    return FaceLandmarks(landmarks=lm, confidence=0.95, image_width=512, image_height=512)


class TestExportMeshObj:
    """Tests for export_mesh_obj."""

    def test_creates_obj_file(self, tmp_path):
        from landmarkdiff.export import export_mesh_obj

        face = _make_face()
        output = tmp_path / "test.obj"
        export_mesh_obj(face, str(output))
        assert output.exists()
        content = output.read_text()
        assert "v " in content  # vertex lines


class TestExportMeshPly:
    """Tests for export_mesh_ply."""

    def test_creates_ply_file(self, tmp_path):
        from landmarkdiff.export import export_mesh_ply

        face = _make_face()
        output = tmp_path / "test.ply"
        export_mesh_ply(face, str(output))
        assert output.exists()
        content = output.read_text()
        assert "ply" in content.lower()


class TestGenerateProgressiveFrames:
    """Tests for generate_progressive_frames."""

    def test_generates_frames(self):
        from landmarkdiff.export import generate_progressive_frames

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        prediction = np.full((64, 64, 3), 200, dtype=np.uint8)
        frames = generate_progressive_frames(original, prediction, n_frames=5)
        assert len(frames) == 5
        for f in frames:
            assert f.shape == (64, 64, 3)

    def test_first_frame_is_original(self):
        from landmarkdiff.export import generate_progressive_frames

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 255, dtype=np.uint8)
        frames = generate_progressive_frames(original, prediction, n_frames=3)
        # First frame should be close to original
        assert frames[0].mean() < 50

    def test_last_frame_is_prediction(self):
        from landmarkdiff.export import generate_progressive_frames

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 200, dtype=np.uint8)
        frames = generate_progressive_frames(original, prediction, n_frames=3)
        # Last frame should be close to prediction
        assert frames[-1].mean() > 150


class TestExportBeforeAfterGif:
    """Tests for export_before_after_gif."""

    def test_creates_gif(self, tmp_path):
        from landmarkdiff.export import export_before_after_gif

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        prediction = np.full((64, 64, 3), 200, dtype=np.uint8)
        output = tmp_path / "test.gif"
        export_before_after_gif(original, prediction, str(output))
        assert output.exists()
        assert output.stat().st_size > 0


class TestExportProgressiveGif:
    """Tests for export_progressive_gif."""

    def test_creates_gif(self, tmp_path):
        from landmarkdiff.export import export_progressive_gif

        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        prediction = np.full((64, 64, 3), 200, dtype=np.uint8)
        output = tmp_path / "test_progressive.gif"
        export_progressive_gif(original, prediction, str(output), n_frames=5)
        assert output.exists()
        assert output.stat().st_size > 0
