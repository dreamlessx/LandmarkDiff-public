"""Tests for scripts/process_hda_database.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from process_hda_database import discover_pairs, process_pair, HDA_PROCEDURE_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hda_db(tmp_path):
    """Create a mock HDA database structure."""
    for category in ["Nose", "Eyelid", "Facelift", "FacialBones", "Eyebrow"]:
        cat_dir = tmp_path / category
        cat_dir.mkdir()
        # Create 3 pairs per category
        for i in range(1, 4):
            pair_id = f"{i:03d}"
            # Before image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(cat_dir / f"{pair_id}_b.jpg"), img)
            # After image
            img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(cat_dir / f"{pair_id}_a.jpg"), img2)
    return tmp_path


@pytest.fixture
def single_pair(tmp_path):
    """Create a single pair for process_pair tests."""
    cat_dir = tmp_path / "Nose"
    cat_dir.mkdir()
    before = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    after = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(cat_dir / "001_b.jpg"), before)
    cv2.imwrite(str(cat_dir / "001_a.jpg"), after)
    return {
        "id": "001",
        "before_path": cat_dir / "001_b.jpg",
        "after_path": cat_dir / "001_a.jpg",
        "hda_category": "Nose",
        "procedure": "rhinoplasty",
    }


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------

class TestDiscoverPairs:
    def test_finds_all_pairs(self, hda_db):
        pairs = discover_pairs(hda_db)
        # 5 categories × 3 pairs each = 15
        assert len(pairs) == 15

    def test_pair_fields(self, hda_db):
        pairs = discover_pairs(hda_db)
        pair = pairs[0]
        assert "id" in pair
        assert "before_path" in pair
        assert "after_path" in pair
        assert "hda_category" in pair
        assert "procedure" in pair

    def test_procedure_mapping(self, hda_db):
        pairs = discover_pairs(hda_db)
        procedures = {p["procedure"] for p in pairs}
        # Nose→rhinoplasty, Eyelid→blepharoplasty, Facelift→rhytidectomy,
        # FacialBones→orthognathic, Eyebrow→blepharoplasty
        assert "rhinoplasty" in procedures
        assert "blepharoplasty" in procedures
        assert "rhytidectomy" in procedures
        assert "orthognathic" in procedures

    def test_category_counts(self, hda_db):
        pairs = discover_pairs(hda_db)
        categories = {}
        for p in pairs:
            cat = p["hda_category"]
            categories[cat] = categories.get(cat, 0) + 1
        assert categories["Nose"] == 3
        assert categories["Eyelid"] == 3
        assert categories["Facelift"] == 3

    def test_skips_unknown_category(self, hda_db):
        # Create a directory that's not a known category
        (hda_db / "Unknown").mkdir()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(hda_db / "Unknown" / "001_b.jpg"), img)
        cv2.imwrite(str(hda_db / "Unknown" / "001_a.jpg"), img)
        pairs = discover_pairs(hda_db)
        # Should still be 15 (unknown skipped)
        assert len(pairs) == 15

    def test_skips_missing_after(self, hda_db):
        # Remove an after file
        after = hda_db / "Nose" / "001_a.jpg"
        after.unlink()
        pairs = discover_pairs(hda_db)
        assert len(pairs) == 14

    def test_empty_directory(self, tmp_path):
        pairs = discover_pairs(tmp_path)
        assert pairs == []

    def test_paths_are_correct(self, hda_db):
        pairs = discover_pairs(hda_db)
        for p in pairs:
            assert p["before_path"].exists()
            assert p["after_path"].exists()
            assert str(p["before_path"]).endswith("_b.jpg")
            assert str(p["after_path"]).endswith("_a.jpg")


# ---------------------------------------------------------------------------
# process_pair
# ---------------------------------------------------------------------------

class TestProcessPair:
    @patch("process_hda_database.extract_landmarks")
    def test_returns_none_no_before_face(self, mock_extract, single_pair, tmp_path):
        mock_extract.return_value = None
        output = tmp_path / "out"
        output.mkdir()
        result = process_pair(single_pair, output)
        assert result is None

    @patch("process_hda_database.extract_landmarks")
    def test_returns_none_no_after_face(self, mock_extract, single_pair, tmp_path):
        from landmarkdiff.landmarks import FaceLandmarks
        mock_lm = FaceLandmarks(
            landmarks=np.random.rand(478, 3).astype(np.float32),
            image_width=400,
            image_height=300,
            confidence=1.0,
        )
        # First call (before) succeeds, second (after) fails
        mock_extract.side_effect = [mock_lm, None]
        output = tmp_path / "out"
        output.mkdir()
        result = process_pair(single_pair, output)
        assert result is None

    @patch("process_hda_database.extract_landmarks")
    @patch("process_hda_database.generate_conditioning")
    @patch("process_hda_database.generate_surgical_mask")
    def test_successful_processing(self, mock_mask, mock_cond, mock_extract, single_pair, tmp_path):
        from landmarkdiff.landmarks import FaceLandmarks
        lm = np.random.rand(478, 3).astype(np.float32) * 0.5 + 0.25
        face = FaceLandmarks(landmarks=lm, image_width=400, image_height=300, confidence=1.0)
        mock_extract.return_value = face

        # Mock conditioning output
        mock_cond.return_value = (
            np.zeros((512, 512, 3), dtype=np.uint8),
            np.zeros((512, 512), dtype=np.uint8),
            np.zeros((512, 512), dtype=np.uint8),
        )
        mock_mask.return_value = np.ones((512, 512), dtype=np.float32) * 0.5

        output = tmp_path / "out"
        output.mkdir()
        result = process_pair(single_pair, output, resolution=512)

        assert result is not None
        assert result["procedure"] == "rhinoplasty"
        assert result["hda_category"] == "Nose"
        assert result["source"] == "HDA_PlasticSurgery_CVPRW2020"
        assert "quality_score" in result
        assert "mean_displacement" in result

        # Check output files exist
        prefix = result["prefix"]
        assert (output / f"{prefix}_input.png").exists()
        assert (output / f"{prefix}_target.png").exists()
        assert (output / f"{prefix}_conditioning.png").exists()
        assert (output / f"{prefix}_mask.png").exists()

    @patch("process_hda_database.extract_landmarks")
    def test_low_quality_skipped(self, mock_extract, single_pair, tmp_path):
        from landmarkdiff.landmarks import FaceLandmarks
        # Create landmarks with very different positions to get low quality
        lm_before = np.random.rand(478, 3).astype(np.float32) * 0.3 + 0.2
        lm_after = np.random.rand(478, 3).astype(np.float32) * 0.3 + 0.5  # Very different
        face_before = FaceLandmarks(landmarks=lm_before, image_width=400, image_height=300, confidence=1.0)
        face_after = FaceLandmarks(landmarks=lm_after, image_width=400, image_height=300, confidence=1.0)
        mock_extract.side_effect = [face_before, face_after]

        output = tmp_path / "out"
        output.mkdir()
        result = process_pair(single_pair, output, min_quality=0.99)
        assert result is None


# ---------------------------------------------------------------------------
# HDA_PROCEDURE_MAP
# ---------------------------------------------------------------------------

class TestProcedureMap:
    def test_all_categories_mapped(self):
        expected = {"Nose", "Eyelid", "Facelift", "FacialBones", "Eyebrow"}
        assert set(HDA_PROCEDURE_MAP.keys()) == expected

    def test_nose_maps_to_rhinoplasty(self):
        assert HDA_PROCEDURE_MAP["Nose"] == "rhinoplasty"

    def test_eyelid_maps_to_blepharoplasty(self):
        assert HDA_PROCEDURE_MAP["Eyelid"] == "blepharoplasty"

    def test_facelift_maps_to_rhytidectomy(self):
        assert HDA_PROCEDURE_MAP["Facelift"] == "rhytidectomy"

    def test_facialbones_maps_to_orthognathic(self):
        assert HDA_PROCEDURE_MAP["FacialBones"] == "orthognathic"

    def test_eyebrow_maps_to_blepharoplasty(self):
        assert HDA_PROCEDURE_MAP["Eyebrow"] == "blepharoplasty"
