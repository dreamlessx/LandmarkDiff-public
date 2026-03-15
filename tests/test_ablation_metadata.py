"""Tests for ablation experiments, metadata generation, and table population."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Fixtures ───


@pytest.fixture
def mock_test_dir(tmp_path):
    """Create a mock test directory with input/target pairs."""
    for proc in ["rhinoplasty", "blepharoplasty"]:
        for i in range(3):
            prefix = f"{proc}_{i:04d}"
            for suffix in ["input", "target", "conditioning"]:
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{prefix}_{suffix}.png"), img)
    return tmp_path


@pytest.fixture
def mock_benchmark_json(tmp_path):
    """Create a mock benchmark results file."""
    data = {
        "methods": {
            "TPS Baseline": {
                "aggregate": {
                    "ssim_mean": 0.95,
                    "lpips_mean": 0.02,
                    "nme_mean": 0.005,
                    "n": 50,
                },
                "per_procedure": {
                    "rhinoplasty": {"ssim": 0.96, "lpips": 0.018, "nme": 0.004, "n": 25},
                    "blepharoplasty": {"ssim": 0.94, "lpips": 0.022, "nme": 0.006, "n": 25},
                },
                "per_fitzpatrick": {
                    "III": {"ssim": 0.95, "lpips": 0.02, "nme": 0.005, "n": 30},
                },
            },
            "Phase B (Ours)": {
                "aggregate": {
                    "ssim_mean": 0.97,
                    "lpips_mean": 0.015,
                    "nme_mean": 0.003,
                    "n": 50,
                },
                "per_procedure": {
                    "rhinoplasty": {"ssim": 0.98, "lpips": 0.012, "nme": 0.002, "n": 25},
                },
                "per_fitzpatrick": {},
            },
        },
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def mock_ablation_json(tmp_path):
    """Create a mock ablation results file."""
    data = {
        "Full TPS": {
            "aggregate": {"ssim_mean": 0.95, "lpips_mean": 0.02, "nme_mean": 0.005},
            "n": 50,
        },
        "No Mask": {
            "aggregate": {"ssim_mean": 0.70, "lpips_mean": 0.15, "nme_mean": 0.02},
            "n": 50,
        },
    }
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(data))
    return path


# ─── Metadata Generation Tests ───


class TestMetadataGeneration:
    def test_infer_procedure_from_filename(self):
        from scripts.generate_metadata import infer_procedure

        assert infer_procedure("rhinoplasty_0001") == "rhinoplasty"
        assert infer_procedure("blepharoplasty_face_0042") == "blepharoplasty"
        assert infer_procedure("rhytidectomy_v3_0100") == "rhytidectomy"
        assert infer_procedure("orthognathic_pair_0001") == "orthognathic"
        assert infer_procedure("unknown_pair_0001") == "unknown"

    def test_infer_wave_from_filename(self):
        from scripts.generate_metadata import infer_wave

        assert infer_wave("pair_v3_rhinoplasty") == "wave3_realistic"
        assert infer_wave("pair_v2_bleph") == "wave2_displacement"
        assert infer_wave("pair_v1_rhytid") == "wave1_basic"

    def test_estimate_displacement_intensity(self):
        from scripts.generate_metadata import estimate_displacement_intensity

        # Identical images → low intensity
        img = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
        intensity = estimate_displacement_intensity(img, img.copy())
        assert intensity < 0.1

        # Very different images → high intensity
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img2 = np.full((128, 128, 3), 255, dtype=np.uint8)
        intensity = estimate_displacement_intensity(img1, img2)
        assert intensity > 0.5

    def test_generate_metadata_creates_json(self, mock_test_dir):
        from scripts.generate_metadata import generate_metadata

        out_path = str(mock_test_dir / "metadata.json")
        meta = generate_metadata(str(mock_test_dir), output_path=out_path)

        assert Path(out_path).exists()
        assert meta["total_pairs"] == 6  # 2 procedures × 3 images each
        assert "pairs" in meta
        assert len(meta["pairs"]) == 6

    def test_generate_metadata_infers_procedures(self, mock_test_dir):
        from scripts.generate_metadata import generate_metadata

        meta = generate_metadata(str(mock_test_dir))

        procs = set()
        for info in meta["pairs"].values():
            procs.add(info["procedure"])
        assert "rhinoplasty" in procs
        assert "blepharoplasty" in procs


# ─── Ablation Tests ───


class TestAblationExperiments:
    def test_load_test_images(self, mock_test_dir):
        from scripts.run_ablation_experiments import load_test_images

        images = load_test_images(str(mock_test_dir))
        assert len(images) == 6
        assert all("input_path" in img for img in images)
        assert all("procedure" in img for img in images)

    def test_load_test_images_max_samples(self, mock_test_dir):
        from scripts.run_ablation_experiments import load_test_images

        images = load_test_images(str(mock_test_dir), max_samples=3)
        assert len(images) == 3

    def test_generate_ablation_latex(self, tmp_path):
        from scripts.run_ablation_experiments import generate_ablation_latex

        results = {
            "Full TPS": {
                "aggregate": {
                    "ssim_mean": 0.95,
                    "lpips_mean": 0.02,
                    "nme_mean": 0.005,
                }
            },
            "No Mask": {
                "aggregate": {
                    "ssim_mean": 0.70,
                    "lpips_mean": 0.15,
                    "nme_mean": 0.02,
                }
            },
        }
        out = tmp_path / "table.tex"
        latex = generate_ablation_latex(results, out)
        assert "\\begin{table}" in latex
        assert "0.9500" in latex
        assert "0.7000" in latex
        assert out.exists()


# ─── Table Population Tests ───


class TestPopulateTables:
    def test_fmt_metric_ssim(self):
        from scripts.populate_paper_tables import fmt_metric

        assert fmt_metric(0.95123, "ssim") == "0.951"
        assert fmt_metric(0.99, "ssim") == "0.990"

    def test_fmt_metric_lpips(self):
        from scripts.populate_paper_tables import fmt_metric

        assert fmt_metric(0.0234, "lpips") == "0.0234"

    def test_fmt_metric_nan(self):
        from scripts.populate_paper_tables import fmt_metric

        assert fmt_metric(float("nan"), "ssim") == "-- "

    def test_load_json_missing(self, tmp_path):
        from scripts.populate_paper_tables import load_json

        assert load_json(str(tmp_path / "nonexistent.json")) is None

    def test_load_json_valid(self, mock_benchmark_json):
        from scripts.populate_paper_tables import load_json

        data = load_json(str(mock_benchmark_json))
        assert data is not None
        assert "methods" in data


# ─── DDP Curriculum Fix Tests ───


class TestCurriculumDDP:
    def test_curriculum_weights_computed(self):
        """Curriculum weights should be computed regardless of DDP mode."""
        from landmarkdiff.curriculum import ProcedureCurriculum

        curriculum = ProcedureCurriculum(total_steps=10000)

        # Step 0: easy procs have higher weights
        w_bleph = curriculum.get_weight(0, "blepharoplasty")
        w_ortho = curriculum.get_weight(0, "orthognathic")
        assert w_bleph > w_ortho, "Easy procs should have higher weight at step 0"

        # Final step: all weights = 1.0
        w_bleph_end = curriculum.get_weight(10000, "blepharoplasty")
        w_ortho_end = curriculum.get_weight(10000, "orthognathic")
        assert w_bleph_end == 1.0
        assert w_ortho_end == 1.0

    def test_loss_weighting_normalization(self):
        """Loss weights should normalize so mean ≈ 1."""
        import torch

        weights = torch.tensor([0.4, 0.6, 0.8, 1.0])
        normalized = weights / weights.mean().clamp(min=0.1)
        # Mean of normalized weights should be close to 1
        assert abs(normalized.mean().item() - 1.0) < 1e-5

    def test_curriculum_weights_broadcast_shape(self):
        """Curriculum weights tensor should be 1D with length = dataset size."""
        import torch

        n_samples = 100
        weights = torch.ones(n_samples)
        # Simulate curriculum update
        from landmarkdiff.curriculum import ProcedureCurriculum

        curriculum = ProcedureCurriculum(total_steps=1000)
        for i in range(n_samples):
            proc = [
                "rhinoplasty",
                "blepharoplasty",
                "rhytidectomy",
                "orthognathic",
                "brow_lift",
                "mentoplasty",
            ][i % 6]
            weights[i] = curriculum.get_weight(500, proc)
        assert weights.shape == (n_samples,)
        assert weights.min() > 0  # No zero weights
