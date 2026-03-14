"""Local experiment tracker for training reproducibility.

Tracks all training runs with their configs, metrics, and results.
Each experiment gets a unique ID and timestamp.

Usage::

    tracker = ExperimentTracker("experiments/")

    # Start a new experiment
    exp_id = tracker.start(
        name="phaseA_v2",
        config={
            "phase": "A", "lr": 1e-5, "batch": 4,
            "steps": 100000, "data": "training_combined",
        },
    )

    # Log metrics during training
    tracker.log_metric(exp_id, step=1000, loss=0.045, ssim=0.82)

    # Record final results
    tracker.finish(exp_id, results={"fid": 42.3, "ssim": 0.87})

    # List all experiments
    tracker.list_experiments()

    # Compare experiments
    tracker.compare(["exp_001", "exp_002"])
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """Simple file-based experiment tracker."""

    def __init__(self, experiments_dir: str = "experiments"):
        self.dir = Path(experiments_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"experiments": {}, "counter": 0}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def start(
        self,
        name: str,
        config: dict,
        tags: list[str] | None = None,
    ) -> str:
        """Start a new experiment. Returns experiment ID."""
        self._index["counter"] += 1
        exp_id = f"exp_{self._index['counter']:03d}"

        exp = {
            "id": exp_id,
            "name": name,
            "config": config,
            "tags": tags or [],
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "results": {},
            "metrics_file": f"{exp_id}_metrics.jsonl",
        }

        self._index["experiments"][exp_id] = exp
        self._save_index()

        # Create metrics log file
        metrics_path = self.dir / exp["metrics_file"]
        metrics_path.touch()

        print(f"Experiment started: {exp_id} ({name})")
        return exp_id

    def log_metric(self, exp_id: str, step: int | None = None, **metrics) -> None:
        """Log metrics for a training step."""
        exp = self._index["experiments"].get(exp_id)
        if not exp:
            return

        entry = {
            "timestamp": time.time(),
            "step": step,
            **metrics,
        }

        metrics_path = self.dir / exp["metrics_file"]
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finish(
        self,
        exp_id: str,
        results: dict | None = None,
        status: str = "completed",
    ) -> None:
        """Mark experiment as finished."""
        exp = self._index["experiments"].get(exp_id)
        if not exp:
            return

        exp["status"] = status
        exp["finished_at"] = datetime.now().isoformat()
        if results:
            exp["results"] = results

        self._save_index()
        print(f"Experiment {exp_id} {status}")

    def get_metrics(self, exp_id: str) -> list[dict]:
        """Load all logged metrics for an experiment."""
        exp = self._index["experiments"].get(exp_id)
        if not exp:
            return []

        metrics_path = self.dir / exp["metrics_file"]
        if not metrics_path.exists():
            return []

        entries = []
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def list_experiments(self) -> list[dict]:
        """List all experiments with summary info."""
        experiments = []
        for exp_id, exp in sorted(self._index["experiments"].items()):
            summary = {
                "id": exp_id,
                "name": exp["name"],
                "status": exp["status"],
                "started": exp["started_at"][:19],
                "tags": exp.get("tags", []),
            }
            if exp["results"]:
                for key in ["fid", "ssim", "lpips", "nme"]:
                    if key in exp["results"]:
                        summary[key] = exp["results"][key]
            experiments.append(summary)
        return experiments

    def compare(self, exp_ids: list[str]) -> dict:
        """Compare multiple experiments by their results."""
        comparison = {}
        for exp_id in exp_ids:
            exp = self._index["experiments"].get(exp_id)
            if exp:
                comparison[exp_id] = {
                    "name": exp["name"],
                    "config": exp["config"],
                    "results": exp["results"],
                }
        return comparison

    def print_summary(self) -> None:
        """Print a summary table of all experiments."""
        experiments = self.list_experiments()
        if not experiments:
            print("No experiments found.")
            return

        # Header
        print(f"{'ID':<10} {'Name':<20} {'Status':<12} {'FID':>6} {'SSIM':>6} {'LPIPS':>6}")
        print("-" * 70)

        for exp in experiments:
            fid = f"{exp.get('fid', '')}" if "fid" in exp else "--"
            ssim = f"{exp.get('ssim', ''):.4f}" if "ssim" in exp else "--"
            lpips = f"{exp.get('lpips', ''):.4f}" if "lpips" in exp else "--"
            print(
                f"{exp['id']:<10} {exp['name']:<20}"
                f" {exp['status']:<12} {fid:>6} {ssim:>6} {lpips:>6}"
            )

    def get_best(self, metric: str = "fid", lower_is_better: bool = True) -> str | None:
        """Get the experiment ID with the best value for a given metric."""
        best_id = None
        best_val = float("inf") if lower_is_better else float("-inf")

        for exp_id, exp in self._index["experiments"].items():
            if exp["status"] != "completed":
                continue
            val = exp["results"].get(metric)
            if val is None:
                continue
            if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
                best_val = val
                best_id = exp_id

        return best_id
