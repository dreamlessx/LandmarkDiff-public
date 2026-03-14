"""Tests for ValidationCallback (non-GPU components only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.validation import ValidationCallback


class TestValidationCallbackInit:
    """Tests for ValidationCallback initialization."""

    def test_creates_output_dir(self, tmp_path):
        out_dir = tmp_path / "val_output"
        # Simple mock dataset
        dataset = [{"conditioning": None, "target": None}] * 4
        cb = ValidationCallback(
            val_dataset=dataset,
            output_dir=out_dir,
            num_samples=4,
        )
        assert out_dir.exists()
        assert cb.num_samples == 4
        assert cb.num_inference_steps == 25
        assert cb.guidance_scale == 7.5
        assert len(cb.history) == 0

    def test_num_samples_capped(self, tmp_path):
        dataset = [None] * 2  # only 2 samples
        cb = ValidationCallback(
            val_dataset=dataset,
            output_dir=tmp_path / "val",
            num_samples=10,
        )
        assert cb.num_samples == 2  # capped at dataset size

    def test_custom_params(self, tmp_path):
        dataset = [None] * 10
        cb = ValidationCallback(
            val_dataset=dataset,
            output_dir=tmp_path / "val",
            num_samples=8,
            num_inference_steps=50,
            guidance_scale=12.0,
        )
        assert cb.num_inference_steps == 50
        assert cb.guidance_scale == 12.0


class TestValidationHistory:
    """Tests for validation history tracking."""

    def test_empty_history(self, tmp_path):
        dataset = [None] * 4
        cb = ValidationCallback(val_dataset=dataset, output_dir=tmp_path / "val")
        assert len(cb.history) == 0

    def test_plot_history_empty(self, tmp_path):
        dataset = [None] * 4
        cb = ValidationCallback(val_dataset=dataset, output_dir=tmp_path / "val")
        # Should not raise even with empty history
        cb.plot_history()

    def test_plot_history_with_data(self, tmp_path):
        dataset = [None] * 4
        cb = ValidationCallback(val_dataset=dataset, output_dir=tmp_path / "val")
        # Manually add history entries
        cb.history = [
            {"step": 1000, "ssim_mean": 0.6, "lpips_mean": 0.4},
            {"step": 2000, "ssim_mean": 0.7, "lpips_mean": 0.3},
            {"step": 3000, "ssim_mean": 0.75, "lpips_mean": 0.25},
        ]
        output_path = str(tmp_path / "val" / "curves.png")
        cb.plot_history(output_path=output_path)
        assert Path(output_path).exists()
