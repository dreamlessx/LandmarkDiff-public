"""Extended tests for validation module: procedure map and selection logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from landmarkdiff.validation import ValidationCallback  # noqa: E402


class _BareDataset:
    """Minimal dataset stub with only __len__, no procedure metadata."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length


# ---------------------------------------------------------------------------
# _build_procedure_map
# ---------------------------------------------------------------------------


class TestBuildProcedureMap:
    """Tests for _build_procedure_map."""

    def test_with_sample_procedures(self, tmp_path):
        """Dataset with _sample_procedures attribute."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=4)
        pair0 = MagicMock()
        pair0.stem = "rhino_001_input"
        pair1 = MagicMock()
        pair1.stem = "rhino_002_input"
        pair2 = MagicMock()
        pair2.stem = "bleph_001_input"
        pair3 = MagicMock()
        pair3.stem = "bleph_002_input"
        ds.pairs = [pair0, pair1, pair2, pair3]
        ds._sample_procedures = {
            "rhino_001": "rhinoplasty",
            "rhino_002": "rhinoplasty",
            "bleph_001": "blepharoplasty",
            "bleph_002": "blepharoplasty",
        }

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        proc_map = cb._procedure_indices

        assert "rhinoplasty" in proc_map
        assert "blepharoplasty" in proc_map
        assert len(proc_map["rhinoplasty"]) == 2
        assert len(proc_map["blepharoplasty"]) == 2

    def test_with_get_procedure(self, tmp_path):
        """Dataset with get_procedure method."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=3)
        ds._sample_procedures = None
        ds.get_procedure = MagicMock(side_effect=["rhinoplasty", "blepharoplasty", "rhinoplasty"])
        # hasattr check: _sample_procedures is None → falsy
        type(ds)._sample_procedures = None

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        proc_map = cb._procedure_indices

        assert "rhinoplasty" in proc_map
        assert "blepharoplasty" in proc_map

    def test_empty_dataset(self, tmp_path):
        """Dataset with no procedure metadata."""
        ds = _BareDataset(3)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        proc_map = cb._procedure_indices
        assert proc_map == {}

    def test_all_unknown_preserved(self, tmp_path):
        """When only 'unknown' procedures exist, they are kept."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=2)
        ds._sample_procedures = None
        ds.get_procedure = MagicMock(return_value="unknown")

        # Remove _sample_procedures so it's falsy
        type(ds)._sample_procedures = None

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        proc_map = cb._procedure_indices
        assert "unknown" in proc_map


# ---------------------------------------------------------------------------
# _select_per_procedure_indices
# ---------------------------------------------------------------------------


class TestSelectPerProcedureIndices:
    """Tests for _select_per_procedure_indices."""

    def test_empty_map_fallback(self, tmp_path):
        """Falls back to sequential indices when no procedure map."""
        ds = _BareDataset(10)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val", num_samples=4)
        # Ensure empty procedure indices
        cb._procedure_indices = {}

        selected = cb._select_per_procedure_indices()
        assert len(selected) == 4
        assert all(proc == "unknown" for _, proc in selected)
        assert [idx for idx, _ in selected] == [0, 1, 2, 3]

    def test_per_procedure_selection(self, tmp_path):
        """Selects samples_per_procedure from each procedure."""
        ds = _BareDataset(10)

        cb = ValidationCallback(
            val_dataset=ds,
            output_dir=tmp_path / "val",
            num_samples=8,
            samples_per_procedure=2,
        )
        cb._procedure_indices = {
            "rhinoplasty": [0, 1, 2, 3],
            "blepharoplasty": [4, 5, 6],
        }

        selected = cb._select_per_procedure_indices()
        # 2 per procedure × 2 procedures = 4
        assert len(selected) == 4
        procs = [proc for _, proc in selected]
        assert procs.count("rhinoplasty") == 2
        assert procs.count("blepharoplasty") == 2

    def test_fewer_samples_than_requested(self, tmp_path):
        """Procedure with fewer samples than samples_per_procedure."""
        ds = _BareDataset(5)

        cb = ValidationCallback(
            val_dataset=ds,
            output_dir=tmp_path / "val",
            num_samples=8,
            samples_per_procedure=3,
        )
        cb._procedure_indices = {
            "rhinoplasty": [0],  # Only 1 available
        }

        selected = cb._select_per_procedure_indices()
        assert len(selected) == 1
        assert selected[0] == (0, "rhinoplasty")


# ---------------------------------------------------------------------------
# plot_history
# ---------------------------------------------------------------------------


class TestPlotHistory:
    """Tests for plot_history."""

    def test_plot_with_data(self, tmp_path):
        ds = _BareDataset(4)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        cb.history = [
            {"step": 100, "ssim_mean": 0.5, "lpips_mean": 0.5},
            {"step": 200, "ssim_mean": 0.6, "lpips_mean": 0.4},
            {"step": 300, "ssim_mean": 0.7, "lpips_mean": 0.3},
        ]

        out_path = str(tmp_path / "val" / "curves.png")
        cb.plot_history(output_path=out_path)
        assert Path(out_path).exists()

    def test_plot_default_path(self, tmp_path):
        ds = _BareDataset(4)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        cb.history = [
            {"step": 100, "ssim_mean": 0.5, "lpips_mean": 0.5},
        ]

        cb.plot_history()
        assert (tmp_path / "val" / "validation_curves.png").exists()

    def test_plot_empty_is_noop(self, tmp_path):
        ds = _BareDataset(4)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        cb.history = []
        cb.plot_history()  # Should return immediately, no error


# ---------------------------------------------------------------------------
# ValidationCallback properties
# ---------------------------------------------------------------------------


class TestValidationCallbackProperties:
    """Tests for ValidationCallback init properties."""

    def test_samples_per_procedure(self, tmp_path):
        ds = _BareDataset(10)

        cb = ValidationCallback(
            val_dataset=ds,
            output_dir=tmp_path / "val",
            samples_per_procedure=5,
        )
        assert cb.samples_per_procedure == 5

    def test_default_values(self, tmp_path):
        ds = _BareDataset(10)

        cb = ValidationCallback(val_dataset=ds, output_dir=tmp_path / "val")
        assert cb.num_inference_steps == 25
        assert cb.guidance_scale == 7.5
        assert cb.samples_per_procedure == 2
        assert cb.output_dir == tmp_path / "val"
