#!/usr/bin/env python3
"""Experiment lineage tracker: traces config → training → eval → paper.

Provides a unified record of the full experiment lifecycle, tracking
which config produced which checkpoint, which evaluation used which
checkpoint, and whether results are stale (config changed since eval).

Key features:
1. Records experiment runs with config hash + timestamps
2. Links checkpoints to training configs
3. Links evaluation results to checkpoints
4. Detects stale results (config changed → invalidates downstream)
5. Generates lineage reports and DAG visualization

Usage:
    # Record a training run
    python scripts/experiment_lineage.py record-training \
        --config configs/phaseA_production.yaml \
        --checkpoint checkpoints_phaseA/final

    # Record an evaluation
    python scripts/experiment_lineage.py record-eval \
        --checkpoint checkpoints_phaseA/final \
        --results results/evaluation.json

    # Check for stale results
    python scripts/experiment_lineage.py check-stale

    # Show lineage report
    python scripts/experiment_lineage.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LINEAGE_FILE = PROJECT_ROOT / ".lineage.json"


# ── Data Model ───────────────────────────────────────────────────


@dataclass
class ConfigRecord:
    """A snapshot of a config file."""

    path: str
    hash: str  # SHA-256 of file contents
    timestamp: str
    experiment_name: str = ""
    phase: str = ""

    @staticmethod
    def from_file(config_path: str | Path) -> ConfigRecord:
        """Create a ConfigRecord from a config file."""
        path = Path(config_path)
        content = path.read_text()
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Parse YAML for metadata
        experiment_name = ""
        phase = ""
        try:
            import yaml

            data = yaml.safe_load(content)
            if isinstance(data, dict):
                experiment_name = data.get("experiment_name", "")
                phase = data.get("training", {}).get("phase", "")
        except Exception:
            pass

        return ConfigRecord(
            path=str(path),
            hash=file_hash,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name=experiment_name,
            phase=phase,
        )


@dataclass
class TrainingRecord:
    """A training run."""

    id: str  # unique run ID
    config_hash: str  # links to ConfigRecord
    config_path: str
    checkpoint_path: str
    start_time: str
    end_time: str = ""
    steps: int = 0
    final_loss: float = 0.0
    slurm_job_id: str = ""
    status: str = "running"  # running, completed, failed


@dataclass
class EvalRecord:
    """An evaluation run."""

    id: str
    checkpoint_path: str
    config_hash: str  # config that trained this checkpoint
    results_path: str
    timestamp: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    status: str = "completed"


@dataclass
class LineageDB:
    """The full lineage database."""

    configs: dict[str, dict] = field(default_factory=dict)  # hash → ConfigRecord
    training_runs: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    paper_links: dict[str, str] = field(default_factory=dict)  # table_name → eval_id

    @staticmethod
    def load(path: str | Path | None = None) -> LineageDB:
        """Load lineage database from disk."""
        if path is None:
            path = LINEAGE_FILE
        path = Path(path)

        if not path.exists():
            return LineageDB()

        with open(path) as f:
            data = json.load(f)

        return LineageDB(
            configs=data.get("configs", {}),
            training_runs=data.get("training_runs", []),
            evaluations=data.get("evaluations", []),
            paper_links=data.get("paper_links", {}),
        )

    def save(self, path: str | Path | None = None) -> None:
        """Save lineage database to disk."""
        if path is None:
            path = LINEAGE_FILE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def record_config(self, config_path: str | Path) -> ConfigRecord:
        """Record a config file snapshot."""
        record = ConfigRecord.from_file(config_path)
        self.configs[record.hash] = asdict(record)
        return record

    def record_training(
        self,
        config_path: str,
        checkpoint_path: str,
        steps: int = 0,
        final_loss: float = 0.0,
        slurm_job_id: str = "",
        status: str = "completed",
    ) -> TrainingRecord:
        """Record a training run."""
        config_record = self.record_config(config_path)

        run_id = f"train_{config_record.hash}_{int(time.time())}"
        record = TrainingRecord(
            id=run_id,
            config_hash=config_record.hash,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            steps=steps,
            final_loss=final_loss,
            slurm_job_id=slurm_job_id,
            status=status,
        )
        self.training_runs.append(asdict(record))
        return record

    def record_evaluation(
        self,
        checkpoint_path: str,
        results_path: str,
        metrics: dict[str, float] | None = None,
        n_samples: int = 0,
    ) -> EvalRecord:
        """Record an evaluation run."""
        # Find which config produced this checkpoint
        config_hash = ""
        for run in self.training_runs:
            if run["checkpoint_path"] == checkpoint_path:
                config_hash = run["config_hash"]
                break

        eval_id = f"eval_{int(time.time())}"
        record = EvalRecord(
            id=eval_id,
            checkpoint_path=checkpoint_path,
            config_hash=config_hash,
            results_path=results_path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metrics=metrics or {},
            n_samples=n_samples,
        )
        self.evaluations.append(asdict(record))
        return record

    def link_paper_table(self, table_name: str, eval_id: str) -> None:
        """Link a paper table to an evaluation."""
        self.paper_links[table_name] = eval_id

    def check_stale(self) -> list[dict]:
        """Check for stale results where config changed since evaluation.

        Returns list of stale items with details.
        """
        stale = []

        for eval_rec in self.evaluations:
            config_hash = eval_rec.get("config_hash", "")
            if not config_hash:
                continue

            # Check if config still exists and hash matches
            config_info = self.configs.get(config_hash, {})
            config_path = config_info.get("path", "")

            if config_path and Path(config_path).exists():
                current_hash = ConfigRecord.from_file(config_path).hash
                if current_hash != config_hash:
                    stale.append(
                        {
                            "eval_id": eval_rec["id"],
                            "checkpoint": eval_rec["checkpoint_path"],
                            "config": config_path,
                            "old_hash": config_hash,
                            "new_hash": current_hash,
                            "eval_time": eval_rec["timestamp"],
                            "reason": "Config file changed since evaluation",
                        }
                    )

        # Check paper links to stale evals
        stale_eval_ids = {s["eval_id"] for s in stale}
        for table_name, eval_id in self.paper_links.items():
            if eval_id in stale_eval_ids:
                stale.append(
                    {
                        "table": table_name,
                        "eval_id": eval_id,
                        "reason": f"Paper table '{table_name}' uses stale evaluation",
                    }
                )

        return stale

    def generate_report(self) -> str:
        """Generate a human-readable lineage report."""
        lines = [
            "=" * 60,
            "EXPERIMENT LINEAGE REPORT",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ]

        # Configs
        lines.append(f"\nConfigs: {len(self.configs)}")
        for h, cfg in self.configs.items():
            lines.append(
                f"  [{h}] {cfg.get('experiment_name', '?')} "
                f"(Phase {cfg.get('phase', '?')}): {cfg.get('path', '?')}"
            )

        # Training runs
        lines.append(f"\nTraining Runs: {len(self.training_runs)}")
        for run in self.training_runs:
            lines.append(
                f"  [{run['id'][:20]}...] config={run['config_hash']} "
                f"→ {run['checkpoint_path']} "
                f"({run['steps']} steps, loss={run['final_loss']:.4f}) "
                f"[{run['status']}]"
            )

        # Evaluations
        lines.append(f"\nEvaluations: {len(self.evaluations)}")
        for ev in self.evaluations:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ev.get("metrics", {}).items())
            lines.append(
                f"  [{ev['id'][:20]}...] checkpoint={ev['checkpoint_path']} "
                f"→ {ev['results_path']} ({ev.get('n_samples', 0)} samples)"
            )
            if metrics_str:
                lines.append(f"    Metrics: {metrics_str}")

        # Paper links
        if self.paper_links:
            lines.append(f"\nPaper Links: {len(self.paper_links)}")
            for table, eval_id in self.paper_links.items():
                lines.append(f"  {table} → {eval_id}")

        # Stale check
        stale = self.check_stale()
        if stale:
            lines.append(f"\nSTALE RESULTS: {len(stale)}")
            for s in stale:
                lines.append(f"  WARNING: {s['reason']}")
                if "config" in s:
                    lines.append(f"    Config: {s['config']}")
                if "eval_id" in s:
                    lines.append(f"    Eval: {s['eval_id']}")
        else:
            lines.append("\nNo stale results detected.")

        # Lineage chains
        lines.append("\nLineage Chains:")
        for run in self.training_runs:
            chain = [
                f"config[{run['config_hash']}]",
                f"→ train[{run['id'][:12]}]",
                f"→ ckpt[{Path(run['checkpoint_path']).name}]",
            ]

            # Find evals for this checkpoint
            for ev in self.evaluations:
                if ev["checkpoint_path"] == run["checkpoint_path"]:
                    chain.append(f"→ eval[{ev['id'][:12]}]")

                    # Find paper links for this eval
                    for table, eid in self.paper_links.items():
                        if eid == ev["id"]:
                            chain.append(f"→ paper[{table}]")

            lines.append(f"  {' '.join(chain)}")

        lines.append("")
        return "\n".join(lines)

    def get_latest_training(self, phase: str | None = None) -> dict | None:
        """Get the most recent training run, optionally filtered by phase."""
        for run in reversed(self.training_runs):
            if phase:
                config = self.configs.get(run["config_hash"], {})
                if config.get("phase") != phase:
                    continue
            return run
        return None

    def get_latest_eval(self, checkpoint: str | None = None) -> dict | None:
        """Get the most recent evaluation, optionally for a specific checkpoint."""
        for ev in reversed(self.evaluations):
            if checkpoint and ev["checkpoint_path"] != checkpoint:
                continue
            return ev
        return None


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment lineage tracker")
    sub = parser.add_subparsers(dest="command")

    # record-training
    rt = sub.add_parser("record-training", help="Record a training run")
    rt.add_argument("--config", required=True, help="Config file path")
    rt.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    rt.add_argument("--steps", type=int, default=0)
    rt.add_argument("--loss", type=float, default=0.0)
    rt.add_argument("--job-id", default="")

    # record-eval
    re = sub.add_parser("record-eval", help="Record an evaluation")
    re.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    re.add_argument("--results", required=True, help="Results JSON path")
    re.add_argument("--n-samples", type=int, default=0)

    # check-stale
    sub.add_parser("check-stale", help="Check for stale results")

    # report
    sub.add_parser("report", help="Generate lineage report")

    # link-paper
    lp = sub.add_parser("link-paper", help="Link paper table to evaluation")
    lp.add_argument("--table", required=True, help="Table name (e.g., Table2)")
    lp.add_argument("--eval-id", required=True, help="Evaluation ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    db = LineageDB.load()

    if args.command == "record-training":
        record = db.record_training(
            args.config,
            args.checkpoint,
            steps=args.steps,
            final_loss=args.loss,
            slurm_job_id=args.job_id,
        )
        db.save()
        print(f"Recorded training run: {record.id}")

    elif args.command == "record-eval":
        # Load metrics from results file if available
        metrics = {}
        if Path(args.results).exists():
            try:
                with open(args.results) as f:
                    data = json.load(f)
                # Try common metric locations
                for key in ["ssim_mean", "lpips_mean", "nme_mean", "fid"]:
                    if key in data:
                        metrics[key] = data[key]
                    elif "aggregated" in data and key in data["aggregated"]:
                        metrics[key] = data["aggregated"][key]
            except Exception:
                pass

        record = db.record_evaluation(
            args.checkpoint,
            args.results,
            metrics=metrics,
            n_samples=args.n_samples,
        )
        db.save()
        print(f"Recorded evaluation: {record.id}")

    elif args.command == "check-stale":
        stale = db.check_stale()
        if stale:
            print(f"Found {len(stale)} stale result(s):")
            for s in stale:
                print(f"  WARNING: {s['reason']}")
            sys.exit(1)
        else:
            print("No stale results.")

    elif args.command == "report":
        print(db.generate_report())

    elif args.command == "link-paper":
        db.link_paper_table(args.table, args.eval_id)
        db.save()
        print(f"Linked {args.table} → {args.eval_id}")


if __name__ == "__main__":
    main()
