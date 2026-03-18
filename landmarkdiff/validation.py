"""Landmark file validation utilities for JSON and CSV formats."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


MEDIAPIPE_LANDMARK_COUNT = 478
DLIB_LANDMARK_COUNT = 68

VALID_COUNTS = {MEDIAPIPE_LANDMARK_COUNT, DLIB_LANDMARK_COUNT}


@dataclass
class ValidationResult:
    """Structured result from validate_landmarks()."""

    valid: bool
    landmark_count: Optional[int] = None
    dimensions: Optional[int] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"[{status}] Landmark file validation"]
        if self.landmark_count is not None:
            lines.append(f"  Count: {self.landmark_count}")
        if self.dimensions is not None:
            lines.append(f"  Dimensions: {self.dimensions}D")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_landmarks(
    path: str | Path,
    expected_count: Optional[int] = MEDIAPIPE_LANDMARK_COUNT,
    min_confidence: float = 0.0,
    pixel_coords: bool = False,
    image_size: int = 512,
) -> ValidationResult:
    """Validate a landmark file (JSON or CSV) before processing.

    Args:
        path: Path to the landmark file (.json or .csv).
        expected_count: Expected number of landmarks. Defaults to 478
            (MediaPipe). Pass None to accept any count.
        min_confidence: Minimum confidence score threshold (0-1).
            Only checked if confidence values are present.
        pixel_coords: If True, coordinates are expected in pixel space
            [0, image_size] rather than normalized [0, 1].
        image_size: Image size used when pixel_coords=True.

    Returns:
        ValidationResult with valid flag, counts, errors, and warnings.
    """
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    # Check file exists and is readable
    if not path.exists():
        return ValidationResult(
            valid=False,
            errors=[f"File not found: {path}"],
        )
    if not path.is_file():
        return ValidationResult(
            valid=False,
            errors=[f"Path is not a file: {path}"],
        )

    suffix = path.suffix.lower()
    if suffix not in (".json", ".csv"):
        return ValidationResult(
            valid=False,
            errors=[f"Unsupported format '{suffix}'. Expected .json or .csv"],
        )

    # Parse the file
    try:
        if suffix == ".json":
            coords, confidences = _parse_json(path)
        else:
            coords, confidences = _parse_csv(path)
    except ValueError as exc:
        return ValidationResult(valid=False, errors=[str(exc)])

    landmark_count = len(coords)
    dimensions = len(coords[0]) if coords else 0

    # Check landmark count
    if expected_count is not None and landmark_count != expected_count:
        errors.append(
            f"Expected {expected_count} landmarks, got {landmark_count}. "
            f"Valid counts are: {sorted(VALID_COUNTS)}"
        )
    elif landmark_count not in VALID_COUNTS and expected_count is None:
        warnings.append(
            f"Landmark count {landmark_count} is not a standard count "
            f"{sorted(VALID_COUNTS)}"
        )

    # Check dimensions (2D or 3D)
    if dimensions not in (2, 3):
        errors.append(
            f"Each landmark must have 2 or 3 coordinates, got {dimensions}"
        )

    # Check coordinate bounds
    coord_errors, coord_warnings = _validate_coords(
        coords, pixel_coords=pixel_coords, image_size=image_size
    )
    errors.extend(coord_errors)
    warnings.extend(coord_warnings)

    # Check confidence scores
    if confidences:
        low_conf = [
            i for i, c in enumerate(confidences) if c < min_confidence
        ]
        if low_conf:
            warnings.append(
                f"{len(low_conf)} landmark(s) have confidence below "
                f"{min_confidence}: indices {low_conf[:5]}"
                + (" ..." if len(low_conf) > 5 else "")
            )

    return ValidationResult(
        valid=len(errors) == 0,
        landmark_count=landmark_count,
        dimensions=dimensions,
        errors=errors,
        warnings=warnings,
    )


def _parse_json(path: Path) -> tuple[list[list[float]], list[float]]:
    """Parse JSON landmark file.

    Expected schema:
        {"landmarks": [[x, y, z], ...], "confidence": [...]}
    Or a bare list:
        [[x, y, z], ...]
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, list):
        coords = data
        confidences: list[float] = []
    elif isinstance(data, dict):
        if "landmarks" not in data:
            raise ValueError(
                "JSON must have a 'landmarks' key or be a bare list of coordinates"
            )
        coords = data["landmarks"]
        confidences = data.get("confidence", [])
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data).__name__}")

    if not isinstance(coords, list) or not coords:
        raise ValueError("'landmarks' must be a non-empty list")

    # Validate each landmark is a list/tuple of numbers
    parsed: list[list[float]] = []
    for i, lm in enumerate(coords):
        if not isinstance(lm, (list, tuple)) or len(lm) < 2:
            raise ValueError(
                f"Landmark {i} must be a list of at least 2 numbers, got {lm!r}"
            )
        try:
            parsed.append([float(v) for v in lm])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Landmark {i} contains non-numeric value: {exc}"
            ) from exc

    return parsed, [float(c) for c in confidences]


def _parse_csv(path: Path) -> tuple[list[list[float]], list[float]]:
    """Parse CSV landmark file.

    Expected columns: x,y or x,y,z (optionally with a header row).
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if not rows:
        raise ValueError("CSV file is empty")

    # Skip header if first row contains non-numeric values
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1

    parsed: list[list[float]] = []
    for i, row in enumerate(rows[start:], start=start):
        if not row:
            continue
        if len(row) < 2:
            raise ValueError(
                f"Row {i} has fewer than 2 columns: {row!r}"
            )
        try:
            parsed.append([float(v) for v in row[:3]])
        except ValueError as exc:
            raise ValueError(
                f"Row {i} contains non-numeric value: {exc}"
            ) from exc

    if not parsed:
        raise ValueError("CSV contains no data rows")

    return parsed, []


def _validate_coords(
    coords: list[list[float]],
    pixel_coords: bool,
    image_size: int,
) -> tuple[list[str], list[str]]:
    """Check coordinate bounds and NaN/inf values."""
    errors: list[str] = []
    warnings: list[str] = []

    nan_indices: list[int] = []
    inf_indices: list[int] = []
    out_of_bounds: list[int] = []

    upper = float(image_size) if pixel_coords else 1.0
    bound_label = f"[0, {image_size}]" if pixel_coords else "[0, 1]"

    for i, lm in enumerate(coords):
        for v in lm:
            if math.isnan(v):
                nan_indices.append(i)
                break
            if math.isinf(v):
                inf_indices.append(i)
                break
        else:
            if any(v < 0.0 or v > upper for v in lm):
                out_of_bounds.append(i)

    if nan_indices:
        errors.append(
            f"{len(nan_indices)} landmark(s) contain NaN values: "
            f"indices {nan_indices[:5]}"
            + (" ..." if len(nan_indices) > 5 else "")
        )
    if inf_indices:
        errors.append(
            f"{len(inf_indices)} landmark(s) contain infinite values: "
            f"indices {inf_indices[:5]}"
            + (" ..." if len(inf_indices) > 5 else "")
        )
    if out_of_bounds:
        warnings.append(
            f"{len(out_of_bounds)} landmark(s) have coordinates outside "
            f"{bound_label}: indices {out_of_bounds[:5]}"
            + (" ..." if len(out_of_bounds) > 5 else "")
        )

    return errors, warnings