"""Tests for 3D mesh export (OBJ and PLY formats)."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from landmarkdiff.export import (
    _get_tessellation_triangles,
    export_mesh_obj,
    export_mesh_ply,
)
from landmarkdiff.landmarks import FaceLandmarks


@pytest.fixture
def mock_face():
    rng = np.random.default_rng(42)
    landmarks = rng.uniform(0.2, 0.8, (478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=512,
        image_height=512,
        confidence=0.95,
    )


class TestTessellation:
    def test_triangle_count(self):
        triangles = _get_tessellation_triangles()
        assert len(triangles) == 852

    def test_triangles_have_three_vertices(self):
        triangles = _get_tessellation_triangles()
        for tri in triangles:
            assert len(tri) == 3

    def test_vertex_indices_in_range(self):
        triangles = _get_tessellation_triangles()
        for tri in triangles:
            for idx in tri:
                assert 0 <= idx < 478


class TestOBJExport:
    def test_creates_file(self, mock_face, tmp_path):
        out = export_mesh_obj(mock_face, tmp_path / "face.obj")
        assert out.exists()
        assert out.suffix == ".obj"

    def test_vertex_count(self, mock_face, tmp_path):
        out = export_mesh_obj(mock_face, tmp_path / "face.obj")
        text = out.read_text()
        v_lines = [ln for ln in text.splitlines() if ln.startswith("v ")]
        assert len(v_lines) == 478

    def test_face_count(self, mock_face, tmp_path):
        out = export_mesh_obj(mock_face, tmp_path / "face.obj")
        text = out.read_text()
        f_lines = [ln for ln in text.splitlines() if ln.startswith("f ")]
        assert len(f_lines) == 852

    def test_faces_are_1_indexed(self, mock_face, tmp_path):
        out = export_mesh_obj(mock_face, tmp_path / "face.obj")
        text = out.read_text()
        for line in text.splitlines():
            if line.startswith("f "):
                indices = [int(x) for x in line.split()[1:]]
                for idx in indices:
                    assert idx >= 1
                    assert idx <= 478

    def test_scale_factor(self, mock_face, tmp_path):
        out = export_mesh_obj(mock_face, tmp_path / "face.obj", scale=200.0)
        text = out.read_text()
        for line in text.splitlines():
            if line.startswith("v "):
                coords = [float(x) for x in line.split()[1:]]
                # With 0.2-0.8 range * 200 scale, x and z should be 40-160
                assert coords[0] >= 0
                assert coords[0] <= 200
                break

    def test_y_flipped(self, mock_face, tmp_path):
        """MediaPipe y is top-down; OBJ y should be bottom-up."""
        out = export_mesh_obj(mock_face, tmp_path / "face.obj", scale=1.0)
        text = out.read_text()
        first_v = next(ln for ln in text.splitlines() if ln.startswith("v "))
        _, x, y, z = first_v.split()
        raw_y = float(mock_face.landmarks[0, 1])
        exported_y = float(y)
        assert abs(exported_y - (1.0 - raw_y)) < 1e-4


class TestPLYExport:
    def test_creates_ascii_file(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply")
        assert out.exists()
        text = out.read_text()
        assert text.startswith("ply\n")
        assert "format ascii 1.0" in text

    def test_vertex_count_in_header(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply")
        text = out.read_text()
        assert "element vertex 478" in text

    def test_face_count_in_header(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply")
        text = out.read_text()
        assert "element face 852" in text

    def test_ascii_data_lines(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply")
        text = out.read_text()
        lines = text.split("end_header\n", 1)[1].strip().splitlines()
        # 478 vertex lines + 852 face lines
        assert len(lines) == 478 + 852

    def test_binary_creates_file(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply", binary=True)
        assert out.exists()
        data = out.read_bytes()
        assert data.startswith(b"ply\n")
        assert b"binary_little_endian" in data

    def test_binary_file_smaller_than_ascii(self, mock_face, tmp_path):
        ascii_out = export_mesh_ply(mock_face, tmp_path / "ascii.ply", binary=False)
        bin_out = export_mesh_ply(mock_face, tmp_path / "binary.ply", binary=True)
        assert bin_out.stat().st_size < ascii_out.stat().st_size

    def test_binary_vertex_data_parseable(self, mock_face, tmp_path):
        out = export_mesh_ply(mock_face, tmp_path / "face.ply", binary=True)
        data = out.read_bytes()
        header_end = data.index(b"end_header\n") + len(b"end_header\n")
        vertex_data = data[header_end : header_end + 12]  # first vertex: 3 floats
        x, y, z = struct.unpack("<fff", vertex_data)
        assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
