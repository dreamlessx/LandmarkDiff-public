"""Tests for landmarkdiff.data_version."""

from __future__ import annotations

import json

import pytest

from landmarkdiff.data_version import DataManifest, FileEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_dataset(tmp_path, n_files=5, proc="rhinoplasty"):
    """Create a fake dataset directory with images."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        f = tmp_path / f"{proc}_{i:04d}_input.png"
        f.write_bytes(b"\x89PNG" + bytes(range(256)) * 4)
    return tmp_path


# ---------------------------------------------------------------------------
# FileEntry
# ---------------------------------------------------------------------------


class TestFileEntry:
    def test_from_path(self, tmp_path):
        f = tmp_path / "rhinoplasty_0001_input.png"
        f.write_bytes(b"test data here")
        entry = FileEntry.from_path(f, base_dir=tmp_path)
        assert entry.path == "rhinoplasty_0001_input.png"
        assert entry.size_bytes == 14
        assert len(entry.checksum) == 32  # md5 hex
        assert entry.procedure == "rhinoplasty"

    def test_procedure_inference(self, tmp_path):
        for proc in ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic", "brow_lift", "mentoplasty"]:
            f = tmp_path / f"{proc}_test.png"
            f.write_bytes(b"x")
            entry = FileEntry.from_path(f, base_dir=tmp_path)
            assert entry.procedure == proc

    def test_unknown_procedure(self, tmp_path):
        f = tmp_path / "generic_image.png"
        f.write_bytes(b"x")
        entry = FileEntry.from_path(f, base_dir=tmp_path)
        assert entry.procedure == ""


# ---------------------------------------------------------------------------
# DataManifest creation
# ---------------------------------------------------------------------------


class TestManifestCreation:
    def test_from_directory(self, tmp_path):
        _create_dataset(tmp_path, n_files=3)
        manifest = DataManifest.from_directory(tmp_path)
        assert manifest.total_files == 3

    def test_total_size(self, tmp_path):
        _create_dataset(tmp_path, n_files=2)
        manifest = DataManifest.from_directory(tmp_path)
        assert manifest.total_size_bytes > 0
        assert manifest.total_size_mb > 0

    def test_checksum_deterministic(self, tmp_path):
        _create_dataset(tmp_path)
        m1 = DataManifest.from_directory(tmp_path)
        m2 = DataManifest.from_directory(tmp_path)
        assert m1.checksum == m2.checksum

    def test_by_procedure(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "rhinoplasty_001_input.png").write_bytes(b"a")
        (d / "rhinoplasty_002_input.png").write_bytes(b"b")
        (d / "blepharoplasty_001_input.png").write_bytes(b"c")
        manifest = DataManifest.from_directory(d)
        procs = manifest.by_procedure
        assert procs["rhinoplasty"] == 2
        assert procs["blepharoplasty"] == 1

    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            DataManifest.from_directory("/nonexistent/path")

    def test_custom_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"x")
        manifest = DataManifest.from_directory(tmp_path, extensions={".csv"})
        assert manifest.total_files == 1
        assert manifest.files[0].path == "data.csv"

    def test_include_patterns(self, tmp_path):
        (tmp_path / "a_input.png").write_bytes(b"a")
        (tmp_path / "a_target.png").write_bytes(b"b")
        (tmp_path / "a_mask.png").write_bytes(b"c")
        manifest = DataManifest.from_directory(
            tmp_path,
            include_patterns=["*_input.png"],
        )
        assert manifest.total_files == 1


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        _create_dataset(tmp_path / "data", n_files=3)
        manifest = DataManifest.from_directory(tmp_path / "data")
        manifest.save(tmp_path / "manifest.json")

        loaded = DataManifest.load(tmp_path / "manifest.json")
        assert loaded.total_files == 3
        assert loaded.checksum == manifest.checksum

    def test_save_creates_dirs(self, tmp_path):
        manifest = DataManifest(files=[])
        path = manifest.save(tmp_path / "sub" / "dir" / "manifest.json")
        assert path.exists()

    def test_json_structure(self, tmp_path):
        _create_dataset(tmp_path / "data", n_files=2)
        manifest = DataManifest.from_directory(tmp_path / "data")
        manifest.save(tmp_path / "m.json")

        with open(tmp_path / "m.json") as f:
            data = json.load(f)

        assert "version" in data
        assert "checksum" in data
        assert "total_files" in data
        assert "files" in data
        assert len(data["files"]) == 2


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerify:
    def test_verify_intact(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=3)
        manifest = DataManifest.from_directory(d)
        ok, issues = manifest.verify(d)
        assert ok is True
        assert issues == []

    def test_verify_missing_file(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=3)
        manifest = DataManifest.from_directory(d)

        # Remove a file
        files = list(d.glob("*.png"))
        files[0].unlink()

        ok, issues = manifest.verify(d)
        assert ok is False
        assert len(issues) == 1
        assert "Missing" in issues[0]

    def test_verify_size_mismatch(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=2)
        manifest = DataManifest.from_directory(d)

        # Modify a file
        files = list(d.glob("*.png"))
        files[0].write_bytes(b"short")

        ok, issues = manifest.verify(d)
        assert ok is False
        assert any("Size mismatch" in i or "Checksum mismatch" in i for i in issues)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_no_changes(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=2)
        m1 = DataManifest.from_directory(d)
        m2 = DataManifest.from_directory(d)
        diff = m1.diff(m2)
        assert diff["added"] == []
        assert diff["removed"] == []
        assert diff["modified"] == []

    def test_added_file(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=2)
        m1 = DataManifest.from_directory(d)

        (d / "extra_file.png").write_bytes(b"new")
        m2 = DataManifest.from_directory(d)

        diff = m1.diff(m2)
        assert "extra_file.png" in diff["added"]
        assert diff["removed"] == []

    def test_removed_file(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=3)
        m1 = DataManifest.from_directory(d)

        files = list(d.glob("*.png"))
        removed_name = files[0].name
        files[0].unlink()
        m2 = DataManifest.from_directory(d)

        diff = m1.diff(m2)
        assert removed_name in diff["removed"]

    def test_modified_file(self, tmp_path):
        d = tmp_path / "data"
        _create_dataset(d, n_files=2)
        m1 = DataManifest.from_directory(d)

        files = list(d.glob("*.png"))
        files[0].write_bytes(b"completely different content!!" * 100)
        m2 = DataManifest.from_directory(d)

        diff = m1.diff(m2)
        assert files[0].name in diff["modified"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_text(self, tmp_path):
        _create_dataset(tmp_path, n_files=2)
        manifest = DataManifest.from_directory(tmp_path)
        text = manifest.summary()
        assert "Files: 2" in text
        assert "Checksum:" in text
        assert "rhinoplasty" in text

    def test_empty_manifest_summary(self):
        manifest = DataManifest()
        text = manifest.summary()
        assert "Files: 0" in text
