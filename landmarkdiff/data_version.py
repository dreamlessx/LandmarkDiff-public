"""Dataset versioning and provenance tracking.

Tracks dataset composition, checksums, and lineage for reproducible training.
Creates manifest files that record exactly which data was used for each
training run.

Usage:
    from landmarkdiff.data_version import DataManifest

    manifest = DataManifest.from_directory("data/training")
    manifest.save("data/training/manifest.json")

    # Later, verify data hasn't changed
    manifest2 = DataManifest.from_directory("data/training")
    assert manifest.checksum == manifest2.checksum
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class FileEntry:
    """Metadata for a single dataset file."""

    path: str
    size_bytes: int
    checksum: str  # md5 of first 64KB (fast approximate)
    procedure: str = ""

    @staticmethod
    def from_path(filepath: Path, base_dir: Path | None = None) -> FileEntry:
        """Create entry from a file path."""
        rel = str(filepath.relative_to(base_dir)) if base_dir else str(filepath)
        size = filepath.stat().st_size

        # Fast checksum: first 64KB
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            h.update(f.read(65536))

        # Infer procedure from filename
        proc = ""
        for p in [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ]:
            if p in filepath.name or p in str(filepath.parent):
                proc = p
                break

        return FileEntry(path=rel, size_bytes=size, checksum=h.hexdigest(), procedure=proc)


@dataclass
class DataManifest:
    """Dataset manifest for versioning and reproducibility.

    Attributes:
        version: Manifest format version.
        created_at: Creation timestamp.
        root_dir: Root directory of the dataset.
        files: List of file entries.
        metadata: Additional dataset metadata.
    """

    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    root_dir: str = ""
    files: list[FileEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def checksum(self) -> str:
        """Compute aggregate checksum from all file checksums."""
        h = hashlib.md5()
        for f in sorted(self.files, key=lambda x: x.path):
            h.update(f"{f.path}:{f.checksum}:{f.size_bytes}".encode())
        return h.hexdigest()

    @property
    def by_procedure(self) -> dict[str, int]:
        """Count files by procedure."""
        counts: dict[str, int] = {}
        for f in self.files:
            key = f.procedure or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def from_directory(
        directory: str | Path,
        extensions: set[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> DataManifest:
        """Create manifest from a directory of dataset files.

        Args:
            directory: Path to dataset directory.
            extensions: File extensions to include (default: image types).
            include_patterns: Glob patterns to include.

        Returns:
            DataManifest with entries for all matching files.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if extensions is None:
            extensions = {".png", ".jpg", ".jpeg", ".webp", ".npy", ".npz"}

        files: list[FileEntry] = []
        if include_patterns:
            for pattern in include_patterns:
                for fp in sorted(directory.glob(pattern)):
                    if fp.is_file():
                        files.append(FileEntry.from_path(fp, base_dir=directory))
        else:
            for fp in sorted(directory.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in extensions:
                    files.append(FileEntry.from_path(fp, base_dir=directory))

        manifest = DataManifest(
            root_dir=str(directory),
            files=files,
            metadata={
                "extensions": sorted(extensions),
                "host": _get_hostname(),
            },
        )
        return manifest

    def save(self, path: str | Path) -> Path:
        """Save manifest to JSON.

        Args:
            path: Output file path.

        Returns:
            Path to saved manifest.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.version,
            "created_at": self.created_at,
            "root_dir": self.root_dir,
            "checksum": self.checksum,
            "total_files": self.total_files,
            "total_size_mb": round(self.total_size_mb, 2),
            "by_procedure": self.by_procedure,
            "metadata": self.metadata,
            "files": [
                {
                    "path": f.path,
                    "size_bytes": f.size_bytes,
                    "checksum": f.checksum,
                    "procedure": f.procedure,
                }
                for f in self.files
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    @staticmethod
    def load(path: str | Path) -> DataManifest:
        """Load manifest from JSON.

        Args:
            path: Path to manifest file.

        Returns:
            DataManifest instance.
        """
        with open(path) as f:
            data = json.load(f)

        files = [
            FileEntry(
                path=fe["path"],
                size_bytes=fe["size_bytes"],
                checksum=fe["checksum"],
                procedure=fe.get("procedure", ""),
            )
            for fe in data.get("files", [])
        ]

        return DataManifest(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            root_dir=data.get("root_dir", ""),
            files=files,
            metadata=data.get("metadata", {}),
        )

    def verify(self, directory: str | Path | None = None) -> tuple[bool, list[str]]:
        """Verify dataset matches this manifest.

        Args:
            directory: Directory to verify (default: original root_dir).

        Returns:
            (all_match, list_of_issues)
        """
        directory = Path(directory or self.root_dir)
        issues: list[str] = []

        for entry in self.files:
            fp = directory / entry.path
            if not fp.exists():
                issues.append(f"Missing: {entry.path}")
                continue

            actual_size = fp.stat().st_size
            if actual_size != entry.size_bytes:
                issues.append(
                    f"Size mismatch: {entry.path} (expected {entry.size_bytes}, got {actual_size})"
                )

            # Check checksum
            h = hashlib.md5()
            with open(fp, "rb") as f:
                h.update(f.read(65536))
            if h.hexdigest() != entry.checksum:
                issues.append(f"Checksum mismatch: {entry.path}")

        return len(issues) == 0, issues

    def diff(self, other: DataManifest) -> dict[str, list[str]]:
        """Compare two manifests.

        Returns:
            Dict with 'added', 'removed', 'modified' file lists.
        """
        self_files = {f.path: f for f in self.files}
        other_files = {f.path: f for f in other.files}

        self_paths = set(self_files.keys())
        other_paths = set(other_files.keys())

        added = sorted(other_paths - self_paths)
        removed = sorted(self_paths - other_paths)
        modified = sorted(
            p for p in self_paths & other_paths if self_files[p].checksum != other_files[p].checksum
        )

        return {"added": added, "removed": removed, "modified": modified}

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Dataset Manifest v{self.version}",
            f"  Root: {self.root_dir}",
            f"  Files: {self.total_files}",
            f"  Size: {self.total_size_mb:.1f} MB",
            f"  Checksum: {self.checksum}",
        ]
        procs = self.by_procedure
        if procs:
            lines.append("  By procedure:")
            for proc, count in sorted(procs.items()):
                lines.append(f"    {proc}: {count}")
        return "\n".join(lines)


def _get_hostname() -> str:
    """Get hostname safely."""
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return "unknown"
