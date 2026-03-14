"""Shared test fixtures and configuration for LandmarkDiff tests."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_face_image():
    """Create a synthetic face-like test image (512x512 BGR uint8)."""
    img = np.full((512, 512, 3), 180, dtype=np.uint8)
    img[:, :, 0] = 150  # B
    img[:, :, 1] = 170  # G
    img[:, :, 2] = 200  # R
    cv2.ellipse(img, (256, 256), (140, 180), 0, 0, 360, (140, 160, 190), -1)
    cv2.circle(img, (200, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (312, 220), 15, (50, 50, 50), -1)
    pts = np.array([[256, 250], [240, 300], [272, 300]], np.int32)
    cv2.fillPoly(img, [pts], (130, 150, 180))
    cv2.ellipse(img, (256, 340), (40, 15), 0, 0, 360, (100, 120, 170), -1)
    noise = np.random.default_rng(42).integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def mock_landmarks_478():
    """Create mock 478-point face landmarks (normalized coords)."""
    rng = np.random.default_rng(42)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4  # x: 0.3-0.7
        landmarks[i, 1] = 0.2 + rng.random() * 0.6  # y: 0.2-0.8
        landmarks[i, 2] = rng.random() * 0.1  # z
    return landmarks


@pytest.fixture
def mock_face_landmarks(mock_landmarks_478):
    """Create a FaceLandmarks object for testing."""
    from landmarkdiff.landmarks import FaceLandmarks
    return FaceLandmarks(
        landmarks=mock_landmarks_478,
        confidence=0.95,
        image_width=512,
        image_height=512,
    )


@pytest.fixture
def sample_training_dir(tmp_path):
    """Create a temp directory with sample training pairs for 3 procedures."""
    procedures = ["rhinoplasty", "blepharoplasty", "rhytidectomy"]
    for i in range(6):
        proc = procedures[i % 3]
        prefix = f"{proc}_{i:06d}"
        for suffix in ["input", "target", "conditioning"]:
            img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"{prefix}_{suffix}.png"), img)
        mask = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{prefix}_mask.png"), mask)
    return tmp_path
