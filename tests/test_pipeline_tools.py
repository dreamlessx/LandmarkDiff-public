"""Tests for pipeline tools: launcher, monitor, post-training pipeline.

Tests the training workflow automation tools.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Training Launcher Tests ──────────────────────────────────────


class TestTrainingLauncher:
    """Tests for scripts/launch_training.py."""

    def test_import(self):
        from scripts.launch_training import (
            SLURM_DEFAULTS,
            generate_slurm_script,
        )

        assert generate_slurm_script is not None
        assert "partition" in SLURM_DEFAULTS

    def test_generate_slurm_script_single_gpu(self):
        from scripts.launch_training import generate_slurm_script

        config = {
            "experiment_name": "test_exp",
            "training": {"phase": "A"},
            "output_dir": "checkpoints_test",
        }
        script = generate_slurm_script("configs/test.yaml", config, n_gpus=1)

        assert "#!/bin/bash" in script
        assert "#SBATCH" in script
        assert "--gres=gpu:" in script
        assert "python scripts/train_controlnet.py" in script
        assert "torchrun" not in script
        assert "configs/test.yaml" in script

    def test_generate_slurm_script_multi_gpu(self):
        from scripts.launch_training import generate_slurm_script

        config = {
            "experiment_name": "test_multi",
            "training": {"phase": "A"},
            "output_dir": "checkpoints_test",
        }
        script = generate_slurm_script("configs/test.yaml", config, n_gpus=4)

        assert "torchrun" in script
        assert "nproc_per_node" in script
        assert "gpu:" in script

    def test_generate_slurm_script_custom_time(self):
        from scripts.launch_training import generate_slurm_script

        config = {
            "experiment_name": "short_run",
            "training": {"phase": "A"},
            "output_dir": "checkpoints_test",
        }
        script = generate_slurm_script("test.yaml", config, n_gpus=1, time_limit="24:00:00")

        assert "24:00:00" in script

    def test_generate_slurm_script_phase_b(self):
        from scripts.launch_training import generate_slurm_script

        config = {
            "experiment_name": "phaseB_prod",
            "training": {"phase": "B"},
            "output_dir": "checkpoints_phaseB",
        }
        script = generate_slurm_script("configs/phaseB.yaml", config, n_gpus=1)

        assert "phaseB_prod" in script

    def test_check_existing_jobs_no_slurm(self):
        """Should handle missing squeue gracefully."""
        from scripts.launch_training import check_existing_jobs

        # Will return empty list if squeue not available
        jobs = check_existing_jobs("nonexistent_prefix")
        assert isinstance(jobs, list)

    def test_load_config(self, tmp_path):
        """Load a YAML config file."""
        from scripts.launch_training import load_config

        config_content = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: A
              learning_rate: 1e-5
            data:
              train_dir: data/test
        """)
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))
        assert config["experiment_name"] == "test"
        assert config["training"]["phase"] == "A"


# ── Training Monitor Tests ───────────────────────────────────────


class TestTrainingMonitor:
    """Tests for scripts/monitor_training.py."""

    def test_import(self):
        from scripts.monitor_training import (
            parse_training_log,
        )

        assert parse_training_log is not None

    def test_parse_training_log(self, tmp_path):
        """Parse a synthetic training log."""
        from scripts.monitor_training import parse_training_log

        log_content = "\n".join(
            [
                "=== Phase A Training ===",
                "Start: 2024-12-15 10:00:00",
                "GPU: NVIDIA RTX A6000",
                "Dataset: 30987 pairs | Batch: 4 | Accum: 4",
                "Step 100/50000 | Loss: 0.123456 | LR: 9.80e-06 | GradNorm: 1.23 | 3.5 it/s | ETA: 4.0h",
                "Step 200/50000 | Loss: 0.098765 | LR: 9.60e-06 | GradNorm: 0.98 | 3.6 it/s | ETA: 3.8h",
                "Step 300/50000 | Loss: 0.087654 | LR: 9.40e-06 | GradNorm: 0.85 | 3.4 it/s | ETA: 4.1h",
                "Checkpoint saved: checkpoints_phaseA/checkpoint-300",
            ]
        )
        log_path = tmp_path / "slurm-test.out"
        log_path.write_text(log_content)

        data = parse_training_log(str(log_path))

        assert len(data["steps"]) == 3
        assert data["steps"] == [100, 200, 300]
        assert data["total_steps"] == 50000
        assert len(data["losses"]) == 3
        assert data["losses"][0] == pytest.approx(0.123456)
        assert len(data["checkpoints"]) == 1
        assert "checkpoint-300" in data["checkpoints"][0]

    def test_parse_empty_log(self, tmp_path):
        """Handle empty or non-training log."""
        from scripts.monitor_training import parse_training_log

        log_path = tmp_path / "empty.out"
        log_path.write_text("Just some output\nNo training data here\n")

        data = parse_training_log(str(log_path))
        assert data["steps"] == []
        assert data["total_steps"] == 0

    def test_detect_convergence_decreasing(self, tmp_path):
        """Detect decreasing loss trend."""
        from scripts.monitor_training import detect_convergence, parse_training_log

        lines = ["=== Training ==="]
        for i in range(50):
            step = (i + 1) * 100
            loss = 0.5 * np.exp(-i / 20) + 0.01
            lines.append(
                f"Step {step}/50000 | Loss: {loss:.6f} | LR: 1.00e-05 | "
                f"GradNorm: 1.00 | 3.0 it/s | ETA: 4.0h"
            )

        log_path = tmp_path / "converging.out"
        log_path.write_text("\n".join(lines))

        data = parse_training_log(str(log_path))
        analysis = detect_convergence(data)
        assert "DECREASING" in analysis or "progressing" in analysis

    def test_detect_convergence_too_few(self, tmp_path):
        """Handle too few data points."""
        from scripts.monitor_training import detect_convergence, parse_training_log

        log_content = (
            "Step 100/50000 | Loss: 0.5 | LR: 1e-05 | GradNorm: 1.0 | 3.0 it/s | ETA: 4.0h\n"
            "Step 200/50000 | Loss: 0.4 | LR: 1e-05 | GradNorm: 1.0 | 3.0 it/s | ETA: 4.0h\n"
        )
        log_path = tmp_path / "short.out"
        log_path.write_text(log_content)

        data = parse_training_log(str(log_path))
        analysis = detect_convergence(data)
        assert "Too few" in analysis

    def test_export_metrics(self, tmp_path):
        """Export metrics to JSON."""
        from scripts.monitor_training import export_metrics, parse_training_log

        log_content = "\n".join(
            [
                "Step 100/10000 | Loss: 0.5 | LR: 1e-05 | GradNorm: 1.0 | 3.0 it/s | ETA: 1.0h",
                "Step 200/10000 | Loss: 0.4 | LR: 9e-06 | GradNorm: 0.9 | 3.1 it/s | ETA: 0.9h",
            ]
        )
        log_path = tmp_path / "export_test.out"
        log_path.write_text(log_content)

        data = parse_training_log(str(log_path))
        export_path = tmp_path / "metrics.json"
        export_metrics(data, str(export_path))

        assert export_path.exists()
        with open(export_path) as f:
            exported = json.load(f)
        assert exported["steps"] == [100, 200]
        assert "summary" in exported
        assert exported["summary"]["current_step"] == 200

    def test_find_log_by_job_id(self, tmp_path):
        """Find log file by SLURM job ID."""
        from scripts.monitor_training import find_log_by_job_id

        # This will return None since no matching log exists in PROJECT_ROOT
        result = find_log_by_job_id(99999999)
        assert result is None  # Expected: no matching file


# ── Post-Training Pipeline Tests ─────────────────────────────────


class TestPostTrainingPipeline:
    """Tests for scripts/post_training_pipeline.py."""

    def test_import(self):
        from scripts.post_training_pipeline import (
            PipelineStep,
        )

        assert PipelineStep is not None

    def test_pipeline_step_success(self):
        from scripts.post_training_pipeline import PipelineStep

        step = PipelineStep("test", "A test step")
        assert step.status == "pending"

        result = step.run(lambda: {"value": 42})
        assert step.status == "completed"
        assert result["value"] == 42
        assert step.elapsed > 0

    def test_pipeline_step_failure(self):
        from scripts.post_training_pipeline import PipelineStep

        step = PipelineStep("failing", "Should fail")

        def bad_func():
            raise ValueError("Intentional error")

        step.run(bad_func)
        assert step.status == "failed"
        assert "Intentional error" in step.error

    def test_pipeline_step_to_dict(self):
        from scripts.post_training_pipeline import PipelineStep

        step = PipelineStep("test", "desc")
        step.run(lambda: {"x": 1})
        d = step.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "completed"
        assert "elapsed_s" in d
        assert d["result"] == {"x": 1}

    def test_step_generate_summary(self, tmp_path):
        from scripts.post_training_pipeline import PipelineStep, step_generate_summary

        steps = [
            PipelineStep("Analyze", "Analysis step"),
            PipelineStep("Score", "Scoring step"),
        ]
        steps[0].status = "completed"
        steps[0].result = {"n_checkpoints": 3}
        steps[0].elapsed = 1.5
        steps[1].status = "failed"
        steps[1].error = "No checkpoints"
        steps[1].elapsed = 0.1

        result = step_generate_summary(tmp_path, steps)
        assert "report_path" in result

        report_path = Path(result["report_path"])
        assert report_path.exists()
        content = report_path.read_text()
        assert "Post-Training Pipeline Report" in content
        assert "[PASS] Analyze" in content
        assert "[FAIL] Score" in content

        # Check JSON output too
        json_path = tmp_path / "pipeline_results.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["steps"]) == 2


# ── ControlNet Evaluation Tests ──────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestControlNetEvaluation:
    """Tests for evaluate_controlnet in scripts/run_evaluation.py."""

    def test_import(self):
        from scripts.run_evaluation import evaluate_controlnet

        assert evaluate_controlnet is not None

    def test_fallback_no_gpu(self, tmp_path, monkeypatch):
        """Falls back to TPS proxy when no GPU available."""
        # Force torch.cuda.is_available() to return False
        import torch

        from scripts.run_evaluation import evaluate_controlnet

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Create minimal fake samples (need actual images for TPS fallback)
        # Since TPS baseline needs real image data, we test the fallback logic
        # by checking it returns proxy results with empty samples
        results = evaluate_controlnet([], tmp_path, "fake_checkpoint")
        assert results == []

    def test_fallback_missing_checkpoint(self, tmp_path, monkeypatch):
        """Falls back to TPS proxy when checkpoint doesn't exist."""
        import torch

        from scripts.run_evaluation import evaluate_controlnet

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        results = evaluate_controlnet([], tmp_path, "/nonexistent/checkpoint")
        assert results == []

    def test_method_label_proxy(self):
        """Verify proxy label detection in run_evaluation."""
        from scripts.run_evaluation import aggregate_metrics

        fake_results = [
            {
                "method": "controlnet_proxy_tps",
                "prefix": "001",
                "procedure": "rhinoplasty",
                "fitzpatrick": "III",
                "ssim": 0.85,
                "lpips": 0.12,
                "nme": 0.03,
                "identity": 0.9,
            },
        ]
        agg = aggregate_metrics(fake_results)
        assert agg["n"] == 1
        assert agg["ssim_mean"] == 0.85


# ── Training Dashboard Tests ─────────────────────────────────────


class TestTrainingDashboard:
    """Tests for scripts/training_dashboard.py."""

    def test_import(self):
        from scripts.training_dashboard import (
            generate_dashboard_html,
        )

        assert generate_dashboard_html is not None

    def test_parse_log_data(self, tmp_path):
        from scripts.training_dashboard import parse_log_data

        log_content = "\n".join(
            [
                "=== Phase A Training ===",
                "GPU: NVIDIA RTX A6000",
                "Dataset: 30987 pairs | Batch: 4 | Accum: 4",
                "Step 100/50000 | Loss: 0.15 | LR: 9.80e-06 | GradNorm: 1.2 | 3.5 it/s | ETA: 4.0h",
                "Step 200/50000 | Loss: 0.12 | LR: 9.60e-06 | GradNorm: 0.9 | 3.6 it/s | ETA: 3.8h",
                "Checkpoint saved: checkpoints/checkpoint-200",
            ]
        )
        log_path = tmp_path / "slurm-test.out"
        log_path.write_text(log_content)

        data = parse_log_data(str(log_path))
        assert len(data["steps"]) == 2
        assert len(data["losses"]) == 2
        assert data["total_steps"] == 50000
        assert len(data["checkpoints"]) == 1
        assert "generated" in data

    def test_generate_html(self, tmp_path):
        from scripts.training_dashboard import generate_dashboard_html

        data = {
            "steps": [100, 200, 300],
            "losses": [0.15, 0.12, 0.10],
            "learning_rates": [1e-5, 9.5e-6, 9e-6],
            "grad_norms": [1.2, 0.9, 0.8],
            "speeds": [3.5, 3.6, 3.4],
            "etas": ["4.0h", "3.8h", "3.5h"],
            "total_steps": 50000,
            "checkpoints": ["checkpoint-200"],
            "convergence": "DECREASING: loss trending downward",
            "header": {"phase": "Phase A", "gpu": "A6000", "dataset": "30987 pairs"},
            "log_path": "slurm-test-12345.out",
            "generated": "2026-03-13 12:00:00",
        }

        html = generate_dashboard_html(data)
        assert "<!DOCTYPE html>" in html
        assert "Training Dashboard" in html
        assert "lossChart" in html
        assert "checkpoint-200" in html
        assert "DECREASING" in html

    def test_generate_dashboard_file(self, tmp_path):
        from scripts.training_dashboard import generate_dashboard

        log_content = "\n".join(
            [
                "Step 100/1000 | Loss: 0.5 | LR: 1e-05 | GradNorm: 1.0 | 3.0 it/s | ETA: 1.0h",
                "Step 200/1000 | Loss: 0.4 | LR: 9e-06 | GradNorm: 0.9 | 3.1 it/s | ETA: 0.9h",
            ]
        )
        log_path = tmp_path / "slurm-test.out"
        log_path.write_text(log_content)

        output_path = tmp_path / "dashboard.html"
        result = generate_dashboard(str(log_path), str(output_path))

        assert result == str(output_path)
        assert output_path.exists()
        html = output_path.read_text()
        assert "<canvas" in html


# ── Dry-Run Validator Tests ──────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDryRunValidator:
    """Tests for scripts/dry_run_training.py."""

    def test_import(self):
        from scripts.dry_run_training import (
            DryRunResult,
        )

        assert DryRunResult is not None

    def test_dry_run_result_tracking(self):
        from scripts.dry_run_training import DryRunResult

        result = DryRunResult()
        result.check("Test 1", True, "ok")
        result.check("Test 2", False, "failed")
        result.check("Test 3", True, "ok")

        assert result.n_passed == 2
        assert result.n_failed == 1
        assert not result.all_passed

    def test_dry_run_result_all_pass(self):
        from scripts.dry_run_training import DryRunResult

        result = DryRunResult()
        result.check("A", True)
        result.check("B", True)

        assert result.all_passed
        assert "2/2" in result.summary()

    def test_create_synthetic_dataset(self, tmp_path):
        from scripts.dry_run_training import create_synthetic_dataset

        data_dir = tmp_path / "synth"
        create_synthetic_dataset(data_dir, n_pairs=5)

        # Check files exist
        inputs = sorted(data_dir.glob("*_input.png"))
        targets = sorted(data_dir.glob("*_target.png"))
        conds = sorted(data_dir.glob("*_conditioning.png"))

        assert len(inputs) == 5
        assert len(targets) == 5
        assert len(conds) == 5

        # Check metadata
        meta_path = data_dir / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert len(meta["pairs"]) == 5

    def test_synthetic_dataset_loads_in_trainer(self, tmp_path):
        """Verify synthetic dataset works with SyntheticPairDataset."""
        from scripts.dry_run_training import create_synthetic_dataset
        from scripts.train_controlnet import SyntheticPairDataset

        data_dir = tmp_path / "synth"
        create_synthetic_dataset(data_dir, n_pairs=3)

        dataset = SyntheticPairDataset(
            str(data_dir),
            resolution=512,
            clinical_augment=False,
            geometric_augment=False,
        )
        assert len(dataset) == 3

        sample = dataset[0]
        assert sample["input"].shape == (3, 512, 512)
        assert sample["target"].shape == (3, 512, 512)
        assert sample["conditioning"].shape == (3, 512, 512)


# ── Fast Split Tests (additional) ────────────────────────────────


class TestFastSplit:
    """Tests for scripts/fast_split.py."""

    def test_import(self):
        from scripts.fast_split import fast_split

        assert fast_split is not None

    def test_fast_split_basic(self, tmp_path):
        """Basic split with synthetic data."""
        from scripts.fast_split import fast_split

        # Create fake training data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(100):
            (data_dir / f"{i:06d}_input.png").write_text("fake")
            (data_dir / f"{i:06d}_target.png").write_text("fake")
            (data_dir / f"{i:06d}_conditioning.png").write_text("fake")

        output_dir = tmp_path / "splits"
        info = fast_split(data_dir, output_dir, val_frac=0.1, test_frac=0.1)

        assert "counts" in info
        counts = info["counts"]
        assert counts["train"] > 0
        assert counts["val"] > 0
        assert counts["test"] > 0
        assert counts["train"] + counts["val"] + counts["test"] == 100

        # Verify no overlap
        train_set = set(info["train_prefixes"])
        val_set = set(info["val_prefixes"])
        test_set = set(info["test_prefixes"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_fast_split_with_metadata(self, tmp_path):
        """Split with metadata for stratification."""
        from scripts.fast_split import fast_split

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        metadata = {"pairs": {}}
        procedures = ["rhinoplasty", "blepharoplasty"]

        for i in range(40):
            prefix = f"{i:06d}"
            (data_dir / f"{prefix}_input.png").write_text("fake")
            (data_dir / f"{prefix}_target.png").write_text("fake")
            proc = procedures[i % 2]
            metadata["pairs"][prefix] = {"procedure": proc, "source": "synthetic"}

        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        output_dir = tmp_path / "splits"
        info = fast_split(data_dir, output_dir, val_frac=0.1, test_frac=0.1)

        # Should have entries for both procedures in per_procedure
        per_proc = info["per_procedure"]
        assert "train" in per_proc
        # Both procedures should appear in train
        train_procs = per_proc["train"]
        assert len(train_procs) >= 2

    def test_fast_split_empty_dir(self, tmp_path):
        """Handle empty data directory."""
        from scripts.fast_split import fast_split

        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        output_dir = tmp_path / "splits"

        info = fast_split(data_dir, output_dir)
        assert info == {}

    def test_verify_splits(self, tmp_path):
        """Verify split integrity."""
        from scripts.fast_split import fast_split, verify_splits

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(20):
            (data_dir / f"{i:06d}_input.png").write_text("fake")
            (data_dir / f"{i:06d}_target.png").write_text("fake")

        output_dir = tmp_path / "splits"
        fast_split(data_dir, output_dir, val_frac=0.2, test_frac=0.2)

        # verify_splits should complete without error
        verify_splits(output_dir)


# ── Comparison Grid Generator Tests ─────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestComparisonGrid:
    """Tests for scripts/compare_outputs.py."""

    def test_import(self):
        from scripts.compare_outputs import (
            add_label,
            generate_comparison_grid,
        )

        assert add_label is not None
        assert generate_comparison_grid is not None

    def test_add_label_bottom(self):
        """Add label at bottom of image."""
        from scripts.compare_outputs import add_label

        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        labeled = add_label(img, "Test Label", position="bottom")

        assert labeled.shape == img.shape
        # Original should not be modified
        assert np.all(img == 128)
        # Bottom region should have overlay (not all 128)
        assert not np.all(labeled[-20:, :, :] == 128)

    def test_add_label_top(self):
        """Add label at top of image."""
        from scripts.compare_outputs import add_label

        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        labeled = add_label(img, "Top Label", position="top")

        assert labeled.shape == img.shape
        # Top region should have overlay
        assert not np.all(labeled[:20, :, :] == 128)

    def test_compute_diff_heatmap(self):
        """Compute difference heatmap between two images."""
        from scripts.compare_outputs import compute_diff_heatmap

        a = np.full((256, 256, 3), 100, dtype=np.uint8)
        b = np.full((256, 256, 3), 150, dtype=np.uint8)
        heatmap = compute_diff_heatmap(a, b, amplify=3.0)

        assert heatmap.shape == (256, 256, 3)
        assert heatmap.dtype == np.uint8
        # Should not be uniform (JET colormap maps to colors)
        assert heatmap.std() > 0

    def test_compute_diff_heatmap_identical(self):
        """Identical images should produce minimal diff."""
        from scripts.compare_outputs import compute_diff_heatmap

        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        heatmap = compute_diff_heatmap(img, img)

        # JET colormap at 0 is dark blue [128, 0, 0] in BGR
        # All pixels should be the same since diff is uniform zero
        assert heatmap.std(axis=(0, 1)).max() < 1

    def test_generate_latex_figure(self):
        """Generate LaTeX figure code."""
        from scripts.compare_outputs import generate_latex_figure

        columns = ["Input", "TPS Baseline", "ControlNet"]
        procedures = ["rhinoplasty", "blepharoplasty"]

        latex = generate_latex_figure(columns, procedures)

        assert "\\begin{figure*}" in latex
        assert "\\end{figure*}" in latex
        assert "2 procedures" in latex
        assert "Input" in latex
        assert "\\label{fig:qualitative}" in latex

    def test_load_test_images_empty(self, tmp_path):
        """Handle empty test directory."""
        from scripts.compare_outputs import load_test_images

        by_proc = load_test_images(tmp_path, max_per_proc=2)
        assert by_proc == {}

    def test_load_test_images_with_data(self, tmp_path):
        """Load test images grouped by procedure."""
        import cv2

        from scripts.compare_outputs import load_test_images

        # Create fake test images
        for proc in ["rhinoplasty", "blepharoplasty"]:
            for i in range(3):
                prefix = f"{proc}_{i:03d}"
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"{prefix}_input.png"), img)
                cv2.imwrite(str(tmp_path / f"{prefix}_target.png"), img)

        by_proc = load_test_images(tmp_path, max_per_proc=2)

        assert "rhinoplasty" in by_proc
        assert "blepharoplasty" in by_proc
        assert len(by_proc["rhinoplasty"]) == 2  # max_per_proc=2
        assert len(by_proc["blepharoplasty"]) == 2

        sample = by_proc["rhinoplasty"][0]
        assert "input" in sample
        assert "target" in sample
        assert "procedure" in sample
        assert sample["input"].shape == (512, 512, 3)


# ── Training Resilience Tests ───────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTrainingResilience:
    """Tests for scripts/training_resilience.py."""

    def test_import(self):
        from scripts.training_resilience import (
            GradientWatchdog,
            SlurmSignalHandler,
        )

        assert SlurmSignalHandler is not None
        assert GradientWatchdog is not None

    def test_slurm_handler_register_unregister(self):
        """Signal handler can be registered and unregistered."""
        from scripts.training_resilience import SlurmSignalHandler

        handler = SlurmSignalHandler(save_fn=lambda: None)
        assert not handler.should_exit
        assert handler.signal_received is None

        handler.register()
        handler.unregister()

    def test_slurm_handler_should_exit(self):
        """should_exit flag works."""
        from scripts.training_resilience import SlurmSignalHandler

        handler = SlurmSignalHandler()
        assert not handler.should_exit
        handler.should_exit = True
        assert handler.should_exit

    def test_oom_handler_basic(self):
        """OOM handler tracks batch size reduction."""
        from scripts.training_resilience import OOMHandler

        oom = OOMHandler(initial_batch_size=8, min_batch_size=1)
        assert oom.current_batch_size == 8

        # Simulate OOM
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        recovered = oom.handle_oom(error)

        assert recovered
        assert oom.current_batch_size == 4
        assert oom.oom_count == 1

    def test_oom_handler_non_oom_error(self):
        """Non-OOM errors are not handled."""
        from scripts.training_resilience import OOMHandler

        oom = OOMHandler(initial_batch_size=4)
        error = RuntimeError("Some other error")
        recovered = oom.handle_oom(error)

        assert not recovered
        assert oom.current_batch_size == 4

    def test_oom_handler_min_batch_size(self):
        """OOM handler respects minimum batch size."""
        from scripts.training_resilience import OOMHandler

        oom = OOMHandler(initial_batch_size=2, min_batch_size=1, max_oom_retries=5)
        error = RuntimeError("CUDA out of memory")

        # First OOM: 2 → 1
        assert oom.handle_oom(error)
        assert oom.current_batch_size == 1

        # Second OOM: can't go below 1
        assert not oom.handle_oom(error)

    def test_oom_handler_summary(self):
        """OOM handler produces summary."""
        from scripts.training_resilience import OOMHandler

        oom = OOMHandler(initial_batch_size=8)
        assert "No OOM" in oom.summary()

        oom.handle_oom(RuntimeError("CUDA out of memory"))
        summary = oom.summary()
        assert "1 OOM" in summary

    def test_validate_checkpoint_missing(self, tmp_path):
        """Validate missing checkpoint."""
        from scripts.training_resilience import validate_checkpoint

        result = validate_checkpoint(tmp_path / "nonexistent")
        assert not result.valid
        assert len(result.errors) > 0

    def test_validate_checkpoint_good(self, tmp_path):
        """Validate a valid checkpoint."""
        import torch

        from scripts.training_resilience import validate_checkpoint

        # Create a fake checkpoint
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()

        state = {
            "controlnet": {"layer.weight": torch.randn(10, 10)},
            "ema_controlnet": {"layer.weight": torch.randn(10, 10)},
            "optimizer": {"state": {0: {"step": 100}}},
            "scheduler": {"last_epoch": 100},
            "global_step": 100,
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        result = validate_checkpoint(ckpt_dir)
        assert result.valid
        assert result.step == 100
        assert result.size_mb > 0

    def test_validate_checkpoint_nan(self, tmp_path):
        """Detect NaN in checkpoint weights."""
        import torch

        from scripts.training_resilience import validate_checkpoint

        ckpt_dir = tmp_path / "checkpoint-bad"
        ckpt_dir.mkdir()

        bad_tensor = torch.randn(10, 10)
        bad_tensor[0, 0] = float("nan")

        state = {
            "controlnet": {"layer.weight": bad_tensor},
            "ema_controlnet": {"layer.weight": torch.randn(10, 10)},
            "optimizer": {"state": {}},
            "global_step": 50,
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        result = validate_checkpoint(ckpt_dir)
        assert not result.valid
        assert any("NaN" in e for e in result.errors)

    def test_validate_checkpoint_shape_mismatch(self, tmp_path):
        """Detect shape mismatch between ControlNet and EMA."""
        import torch

        from scripts.training_resilience import validate_checkpoint

        ckpt_dir = tmp_path / "checkpoint-mismatch"
        ckpt_dir.mkdir()

        state = {
            "controlnet": {"layer.weight": torch.randn(10, 10)},
            "ema_controlnet": {"layer.weight": torch.randn(20, 20)},
            "optimizer": {"state": {}},
            "global_step": 50,
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        result = validate_checkpoint(ckpt_dir)
        assert not result.valid
        assert any("Shape" in e or "mismatch" in e for e in result.errors)

    def test_gradient_watchdog_ok(self):
        """Gradient watchdog passes on healthy gradients."""
        import torch

        from scripts.training_resilience import GradientWatchdog

        watchdog = GradientWatchdog()

        # Create a simple model with gradients
        param = torch.nn.Parameter(torch.randn(10, 10))
        loss = (param**2).sum()
        loss.backward()

        action = watchdog.check([param], loss.item(), step=1)
        assert action == "ok"

    def test_gradient_watchdog_nan_loss(self):
        """Gradient watchdog catches NaN loss."""
        import torch

        from scripts.training_resilience import GradientWatchdog

        watchdog = GradientWatchdog()
        param = torch.nn.Parameter(torch.randn(5))

        action = watchdog.check([param], float("nan"), step=1)
        assert action == "skip"

    def test_gradient_watchdog_summary(self):
        """Gradient watchdog produces summary."""
        from scripts.training_resilience import GradientWatchdog

        watchdog = GradientWatchdog()
        summary = watchdog.summary()
        assert "NaN events: 0" in summary

    def test_create_emergency_save_fn(self, tmp_path):
        """Emergency save function creates checkpoint."""
        from unittest.mock import MagicMock

        import torch

        from scripts.training_resilience import create_emergency_save_fn

        # Create minimal model state dicts
        model = torch.nn.Linear(10, 10)
        # EMA needs save_pretrained (diffusers method) — use a mock
        ema = MagicMock()
        ema.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        step_ref = [42]
        save_fn = create_emergency_save_fn(tmp_path, model, ema, opt, sched, step_ref)

        save_fn()

        ckpt_dir = tmp_path / "emergency-checkpoint-42"
        assert ckpt_dir.exists()
        assert (ckpt_dir / "training_state.pt").exists()

        state = torch.load(ckpt_dir / "training_state.pt", weights_only=True)
        assert state["global_step"] == 42
        assert state["emergency"] is True
        ema.save_pretrained.assert_called_once()

    def test_training_integration_import(self):
        """Verify resilience is importable from train_controlnet."""
        # This tests that the import in train_controlnet.py works
        from scripts.train_controlnet import HAS_RESILIENCE

        assert HAS_RESILIENCE


# ── Config Schema Validator Tests ───────────────────────────────


class TestConfigValidator:
    """Tests for scripts/validate_config.py."""

    def test_import(self):
        from scripts.validate_config import (
            validate_config,
        )

        assert validate_config is not None

    def test_valid_config(self, tmp_path):
        """Valid config passes validation."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test_exp
            training:
              phase: "A"
              learning_rate: 1.0e-5
              batch_size: 4
              max_train_steps: 10000
            data:
              image_size: 512
            output_dir: checkpoints_test
        """)
        config_path = tmp_path / "valid.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert result.valid
        assert len(result.errors) == 0

    def test_unknown_top_level_key(self, tmp_path):
        """Unknown top-level key produces warning."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            trainng:
              phase: "A"
        """)
        config_path = tmp_path / "typo.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert any("Unknown top-level key: 'trainng'" in w for w in result.warnings)

    def test_unknown_section_key(self, tmp_path):
        """Unknown key in section produces warning with typo suggestion."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: "A"
              learnig_rate: 1.0e-5
        """)
        config_path = tmp_path / "section_typo.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert any("learnig_rate" in w for w in result.warnings)
        # Should suggest learning_rate
        assert any("learning_rate" in w for w in result.warnings)

    def test_value_out_of_range(self, tmp_path):
        """Out-of-range value produces error."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: "A"
              learning_rate: 5.0
              batch_size: 4
        """)
        config_path = tmp_path / "bad_lr.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert not result.valid
        assert any("learning_rate" in e and "out of range" in e for e in result.errors)

    def test_invalid_choice(self, tmp_path):
        """Invalid choice value produces error."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: "C"
        """)
        config_path = tmp_path / "bad_phase.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert not result.valid
        assert any("phase" in e and "not valid" in e for e in result.errors)

    def test_phase_b_no_checkpoint_warning(self, tmp_path):
        """Phase B without checkpoint produces warning."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: "B"
              learning_rate: 5.0e-6
        """)
        config_path = tmp_path / "phaseb.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert any("Phase B" in w for w in result.warnings)

    def test_strict_mode(self, tmp_path):
        """Strict mode promotes warnings to errors."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            trainng:
              phase: "A"
        """)
        config_path = tmp_path / "strict.yaml"
        config_path.write_text(config)

        result = validate_config(config_path, strict=True)
        assert not result.valid
        assert len(result.warnings) == 0
        assert len(result.errors) > 0

    def test_missing_file(self):
        """Missing file produces error."""
        from scripts.validate_config import validate_config

        result = validate_config("/nonexistent/config.yaml")
        assert not result.valid
        assert any("not found" in e for e in result.errors)

    def test_empty_config(self, tmp_path):
        """Empty config is handled gracefully."""
        from scripts.validate_config import validate_config

        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        result = validate_config(config_path)
        # Empty YAML parses as None — caught as error
        assert not result.valid or len(result.warnings) > 0

    def test_validate_all_configs(self, tmp_path):
        """Validate multiple configs in a directory."""
        from scripts.validate_config import validate_all_configs

        (tmp_path / "good.yaml").write_text(
            textwrap.dedent("""\
            experiment_name: good
            training:
              phase: "A"
        """)
        )
        (tmp_path / "bad.yaml").write_text(
            textwrap.dedent("""\
            experiment_name: bad
            training:
              phase: "X"
        """)
        )

        results = validate_all_configs(tmp_path)
        assert len(results) == 2
        assert sum(1 for r in results if r.valid) == 1

    def test_edit_distance_typo_detection(self):
        """Edit distance finds closest match."""
        from scripts.validate_config import _closest_match

        valid = {"learning_rate", "batch_size", "max_train_steps"}
        assert _closest_match("learnig_rate", valid) == "learning_rate"
        assert _closest_match("batch_szie", valid) == "batch_size"
        assert _closest_match("xyz_totally_wrong", valid) is None

    def test_save_checkpoint_exceeds_total(self, tmp_path):
        """Warning when checkpoint interval exceeds total steps."""
        from scripts.validate_config import validate_config

        config = textwrap.dedent("""\
            experiment_name: test
            training:
              phase: "A"
              max_train_steps: 1000
              save_every_n_steps: 5000
        """)
        config_path = tmp_path / "no_ckpt.yaml"
        config_path.write_text(config)

        result = validate_config(config_path)
        assert any("save_every_n_steps" in w for w in result.warnings)

    def test_config_validation_summary(self):
        """ConfigValidation summary output."""
        from scripts.validate_config import ConfigValidation

        result = ConfigValidation(path="test.yaml")
        result.errors.append("bad key")
        result.warnings.append("suspicious value")

        summary = result.summary()
        assert "FAIL" in summary
        assert "bad key" in summary
        assert "suspicious value" in summary

    def test_valid_production_config(self):
        """Validate the actual production config."""
        from scripts.validate_config import validate_config

        config_path = Path("configs/phaseA_production.yaml")
        if config_path.exists():
            result = validate_config(config_path)
            # Production config should have no errors
            assert result.valid, f"Production config invalid: {result.errors}"


# ── Experiment Lineage Tests ────────────────────────────────────


class TestExperimentLineage:
    """Tests for scripts/experiment_lineage.py."""

    def test_import(self):
        from scripts.experiment_lineage import (
            LineageDB,
        )

        assert LineageDB is not None

    def test_config_record_from_file(self, tmp_path):
        """Create config record from YAML file."""
        from scripts.experiment_lineage import ConfigRecord

        config = tmp_path / "test.yaml"
        config.write_text(
            textwrap.dedent("""\
            experiment_name: test_exp
            training:
              phase: "A"
        """)
        )

        record = ConfigRecord.from_file(config)
        assert record.hash  # non-empty hash
        assert record.experiment_name == "test_exp"
        assert record.phase == "A"
        assert len(record.hash) == 16

    def test_config_record_hash_changes(self, tmp_path):
        """Config hash changes when content changes."""
        from scripts.experiment_lineage import ConfigRecord

        config = tmp_path / "test.yaml"
        config.write_text("experiment_name: v1\ntraining:\n  phase: A\n")
        hash1 = ConfigRecord.from_file(config).hash

        config.write_text("experiment_name: v2\ntraining:\n  phase: A\n")
        hash2 = ConfigRecord.from_file(config).hash

        assert hash1 != hash2

    def test_lineage_db_save_load(self, tmp_path):
        """Save and load lineage database."""
        from scripts.experiment_lineage import LineageDB

        db = LineageDB()
        db.configs["abc123"] = {"path": "test.yaml", "hash": "abc123"}
        db.training_runs.append({"id": "run1", "config_hash": "abc123"})

        db_path = tmp_path / "lineage.json"
        db.save(db_path)

        loaded = LineageDB.load(db_path)
        assert "abc123" in loaded.configs
        assert len(loaded.training_runs) == 1

    def test_record_training(self, tmp_path):
        """Record a training run with config."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: test\ntraining:\n  phase: A\n")

        db = LineageDB()
        record = db.record_training(
            str(config),
            "checkpoints/final",
            steps=50000,
            final_loss=0.12,
        )

        assert record.id.startswith("train_")
        assert record.steps == 50000
        assert len(db.training_runs) == 1
        assert len(db.configs) == 1

    def test_record_evaluation(self, tmp_path):
        """Record an evaluation linked to checkpoint."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: test\ntraining:\n  phase: A\n")

        db = LineageDB()
        db.record_training(str(config), "checkpoints/final")

        eval_rec = db.record_evaluation(
            "checkpoints/final",
            "results/eval.json",
            metrics={"ssim_mean": 0.95, "lpips_mean": 0.05},
            n_samples=100,
        )

        assert eval_rec.id.startswith("eval_")
        assert eval_rec.config_hash  # should be linked
        assert len(db.evaluations) == 1

    def test_check_stale_no_change(self, tmp_path):
        """No stale results when config hasn't changed."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: test\ntraining:\n  phase: A\n")

        db = LineageDB()
        db.record_training(str(config), "ckpt/final")
        db.record_evaluation("ckpt/final", "results.json")

        stale = db.check_stale()
        assert len(stale) == 0

    def test_check_stale_config_changed(self, tmp_path):
        """Detect stale results when config changes."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: v1\ntraining:\n  phase: A\n")

        db = LineageDB()
        db.record_training(str(config), "ckpt/final")
        db.record_evaluation("ckpt/final", "results.json")

        # Now change the config
        config.write_text("experiment_name: v2\ntraining:\n  phase: A\n  learning_rate: 5e-6\n")

        stale = db.check_stale()
        assert len(stale) >= 1
        assert "changed" in stale[0]["reason"]

    def test_paper_link(self, tmp_path):
        """Link paper table to evaluation."""
        from scripts.experiment_lineage import LineageDB

        db = LineageDB()
        db.link_paper_table("Table2", "eval_123")

        assert db.paper_links["Table2"] == "eval_123"

    def test_generate_report(self, tmp_path):
        """Generate lineage report."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: test\ntraining:\n  phase: A\n")

        db = LineageDB()
        db.record_training(str(config), "ckpt/final", steps=10000, final_loss=0.1)
        db.record_evaluation("ckpt/final", "results.json", metrics={"ssim_mean": 0.95})

        report = db.generate_report()
        assert "LINEAGE REPORT" in report
        assert "Training Runs: 1" in report
        assert "Evaluations: 1" in report
        assert "stale" in report.lower()

    def test_get_latest_training(self, tmp_path):
        """Get most recent training run."""
        from scripts.experiment_lineage import LineageDB

        config = tmp_path / "train.yaml"
        config.write_text("experiment_name: test\ntraining:\n  phase: A\n")

        db = LineageDB()
        db.record_training(str(config), "ckpt1", steps=1000)
        db.record_training(str(config), "ckpt2", steps=2000)

        latest = db.get_latest_training()
        assert latest["checkpoint_path"] == "ckpt2"

    def test_empty_db(self):
        """Empty lineage DB works gracefully."""
        from scripts.experiment_lineage import LineageDB

        db = LineageDB()
        assert db.check_stale() == []
        assert db.get_latest_training() is None
        assert db.get_latest_eval() is None

        report = db.generate_report()
        assert "Configs: 0" in report


# ── Pipeline Integration Tests ────────────────────────────────


class TestPipelineIntegration:
    """Tests for validate_config + lineage integration into pipeline."""

    def test_post_training_has_lineage_import(self):
        """post_training_pipeline.py imports lineage module."""
        import scripts.post_training_pipeline as ptp

        assert hasattr(ptp, "HAS_LINEAGE")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_train_controlnet_has_lineage_import(self):
        """train_controlnet.py imports lineage module."""
        import scripts.train_controlnet as tc

        assert hasattr(tc, "HAS_LINEAGE")

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_train_controlnet_has_resilience_import(self):
        """train_controlnet.py imports resilience module."""
        import scripts.train_controlnet as tc

        assert hasattr(tc, "HAS_RESILIENCE")

    def test_orchestrator_has_config_validation(self):
        """pipeline_orchestrator.sh includes config validation stage."""
        orch_path = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_orchestrator.sh"
        if orch_path.exists():
            content = orch_path.read_text()
            assert "validate_config.py" in content
            assert "Stage 0: Config Validation" in content

    def test_orchestrator_has_lineage_recording(self):
        """pipeline_orchestrator.sh records training and eval in lineage."""
        orch_path = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_orchestrator.sh"
        if orch_path.exists():
            content = orch_path.read_text()
            assert "experiment_lineage.py record-training" in content
            assert "experiment_lineage.py record-eval" in content
            assert "experiment_lineage.py check-stale" in content

    def test_orchestrator_has_lineage_report(self):
        """pipeline_orchestrator.sh generates lineage report at end."""
        orch_path = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_orchestrator.sh"
        if orch_path.exists():
            content = orch_path.read_text()
            assert "experiment_lineage.py report" in content
            assert "lineage_report.txt" in content

    def test_lineage_full_pipeline_flow(self, tmp_path):
        """End-to-end: config → training → eval → stale check."""
        from scripts.experiment_lineage import LineageDB

        # Create config file
        config = tmp_path / "config.yaml"
        config.write_text(
            "experiment_name: e2e_test\ntraining:\n  phase: A\n  learning_rate: 1e-5\n"
        )

        db_path = tmp_path / "lineage.json"
        db = LineageDB()

        # Record training
        train_rec = db.record_training(
            str(config),
            str(tmp_path / "checkpoints/final"),
            steps=50000,
            final_loss=0.08,
            slurm_job_id="12345",
        )
        assert train_rec.slurm_job_id == "12345"

        # Record evaluation
        eval_rec = db.record_evaluation(
            str(tmp_path / "checkpoints/final"),
            str(tmp_path / "results/eval.json"),
            metrics={"ssim_mean": 0.92, "lpips_mean": 0.07},
            n_samples=200,
        )
        assert eval_rec.config_hash == train_rec.config_hash

        # Link paper table
        db.link_paper_table("Table1_main_results", eval_rec.id)

        # Save and reload
        db.save(db_path)
        db2 = LineageDB.load(db_path)
        assert len(db2.training_runs) == 1
        assert len(db2.evaluations) == 1
        assert "Table1_main_results" in db2.paper_links

        # No stale results yet
        assert len(db2.check_stale()) == 0

        # Change config → should become stale
        config.write_text(
            "experiment_name: e2e_test_v2\ntraining:\n  phase: A\n  learning_rate: 5e-6\n"
        )
        stale = db2.check_stale()
        assert len(stale) >= 1  # eval is stale
        # Paper link should also be flagged
        stale_reasons = [s["reason"] for s in stale]
        assert any("Table1" in r for r in stale_reasons)

    def test_validate_config_in_preflight(self):
        """preflight_training.py integrates config schema validation."""
        preflight_path = (
            Path(__file__).resolve().parent.parent / "scripts" / "preflight_training.py"
        )
        if preflight_path.exists():
            content = preflight_path.read_text()
            assert "validate_config" in content
            assert "schema_validate" in content


# ── Displacement Analysis Tests ──────────────────────────────


class TestDisplacementAnalysis:
    """Tests for scripts/displacement_analysis.py."""

    def test_import(self):
        from scripts.displacement_analysis import (
            generate_displacement_report,
        )

        assert generate_displacement_report is not None

    def test_ablation_template_generation(self, tmp_path):
        """Generate ablation results template with expected structure."""
        from scripts.displacement_analysis import generate_ablation_template

        out = tmp_path / "ablation.json"
        result = generate_ablation_template(out)

        assert out.exists()
        assert "diffusion_only" in result
        assert "diff_identity" in result
        assert "diff_perceptual" in result
        assert "full" in result

        # Each config should have all 5 metrics
        for key in ["diffusion_only", "diff_identity", "diff_perceptual", "full"]:
            assert "ssim" in result[key]
            assert "lpips" in result[key]
            assert "nme" in result[key]
            assert "identity_sim" in result[key]
            assert "fid" in result[key]

    def test_ablation_template_loads_as_json(self, tmp_path):
        """Template is valid JSON that generate_paper_tables can consume."""
        from scripts.displacement_analysis import generate_ablation_template

        out = tmp_path / "ablation.json"
        generate_ablation_template(out)

        with open(out) as f:
            data = json.load(f)

        assert isinstance(data["full"], dict)
        assert data["full"]["ssim"] == 0.0

    def test_update_ablation_results(self, tmp_path):
        """Update ablation results with real metrics."""
        from scripts.displacement_analysis import (
            generate_ablation_template,
            update_ablation_results,
        )

        out = tmp_path / "ablation.json"
        generate_ablation_template(out)

        # Update with real metrics
        updated = update_ablation_results(
            out,
            "full",
            {
                "ssim": 0.9234,
                "lpips": 0.0456,
                "nme": 0.0123,
            },
        )

        assert updated["full"]["ssim"] == 0.9234
        assert updated["full"]["lpips"] == 0.0456
        # Unreplaced metrics should remain
        assert updated["full"]["fid"] == 0.0

        # Verify persisted to disk
        with open(out) as f:
            disk_data = json.load(f)
        assert disk_data["full"]["ssim"] == 0.9234

    def test_update_ablation_invalid_config(self, tmp_path):
        """Reject invalid config keys."""
        from scripts.displacement_analysis import update_ablation_results

        out = tmp_path / "ablation.json"
        with open(out, "w") as f:
            json.dump({"full": {"ssim": 0.0}}, f)

        try:
            update_ablation_results(out, "invalid_config", {"ssim": 0.5})
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "invalid_config" in str(e)

    def test_displacement_report_missing_dir(self, tmp_path):
        """Handle missing pairs directory gracefully."""
        from scripts.displacement_analysis import generate_displacement_report

        report = generate_displacement_report(
            tmp_path / "nonexistent",
            tmp_path / "report.json",
        )
        assert report["total_pairs"] == 0
        assert "error" in report

    def test_region_statistics_missing_model(self, tmp_path):
        """Handle missing model file gracefully."""
        from scripts.displacement_analysis import compute_region_statistics

        stats = compute_region_statistics(tmp_path / "nonexistent.npz")
        assert "error" in stats

    def test_full_analysis_no_data(self, tmp_path):
        """Full analysis with no data generates ablation template."""
        from scripts.displacement_analysis import generate_full_analysis

        summary = generate_full_analysis(
            pairs_dir=None,
            model_path=tmp_path / "nonexistent.npz",
            output_dir=tmp_path / "results",
        )

        assert "ablation_template" in summary["analyses"]
        assert (tmp_path / "results" / "ablation_results.json").exists()

    def test_orchestrator_displacement_analysis(self):
        """pipeline_orchestrator.sh includes displacement analysis steps."""
        orch_path = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_orchestrator.sh"
        if orch_path.exists():
            content = orch_path.read_text()
            assert "displacement_analysis.py" in content
            assert "ablation-template" in content

    def test_ablation_table2_compatibility(self, tmp_path):
        """Ablation template works with generate_paper_tables Table 2."""
        from scripts.displacement_analysis import (
            generate_ablation_template,
            update_ablation_results,
        )
        from scripts.generate_paper_tables import generate_table2_ablation

        ablation_path = tmp_path / "ablation.json"
        generate_ablation_template(ablation_path)

        # Fill in sample values
        update_ablation_results(
            ablation_path,
            "diffusion_only",
            {
                "ssim": 0.72,
                "lpips": 0.15,
                "nme": 0.023,
                "identity_sim": 0.85,
                "fid": 45.2,
            },
        )
        update_ablation_results(
            ablation_path,
            "full",
            {
                "ssim": 0.89,
                "lpips": 0.06,
                "nme": 0.012,
                "identity_sim": 0.94,
                "fid": 28.1,
            },
        )

        # Generate table
        table = generate_table2_ablation(str(ablation_path))
        assert "\\begin{table}" in table
        assert "ablation" in table.lower()
        assert "\\textbf" in table  # best values should be bolded
