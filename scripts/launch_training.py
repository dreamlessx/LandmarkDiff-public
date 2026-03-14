#!/usr/bin/env python3
"""Training launcher with integrated preflight checks and SLURM submission.

Runs all preflight checks before submitting the training job.
Generates a customized SLURM script if needed, or uses existing ones.

Usage:
    # Launch Phase A with production config (preflight + submit)
    python scripts/launch_training.py --config configs/phaseA_production.yaml

    # Dry run — check everything but don't submit
    python scripts/launch_training.py --config configs/phaseA_production.yaml --dry-run

    # Multi-GPU Phase A
    python scripts/launch_training.py --config configs/phaseA_production.yaml --gpus 2

    # Phase B (auto-detects Phase A checkpoint)
    python scripts/launch_training.py --config configs/phaseB_production.yaml

    # Monitor after launch
    python scripts/launch_training.py --config configs/phaseA_production.yaml --monitor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# SLURM defaults
SLURM_DEFAULTS = {
    "partition": "batch_gpu",
    "account": os.environ.get("SLURM_ACCOUNT", "default_gpu"),
    "gpu_type": "nvidia_rtx_a6000",
    "mem_per_gpu": "48G",
    "cpus_per_gpu": 8,
    "time_limit": "48:00:00",
    "conda_env": "landmarkdiff",
    "work_dir": str(Path(__file__).resolve().parent.parent),
}


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_preflight(config_path: str) -> tuple[bool, list[str]]:
    """Run preflight checks and return (passed, messages)."""
    from scripts.preflight_training import (
        check_config,
        check_dataset,
        check_metadata,
        check_splits,
    )

    config = load_config(config_path)
    checks = [
        check_config(config, config_path),
        check_dataset(config),
        check_metadata(config),
        check_splits(config),
    ]

    messages = []
    all_passed = True
    warnings = 0

    for check in checks:
        if not check.passed:
            all_passed = False
            messages.append(f"FAIL: {check.name} — {check.message}")
        elif check.warning:
            warnings += 1
            messages.append(f"WARN: {check.name} — {check.message}")
        else:
            messages.append(f"PASS: {check.name}")

    return all_passed, messages, warnings


def generate_slurm_script(
    config_path: str,
    config: dict,
    n_gpus: int = 1,
    time_limit: str | None = None,
    job_name: str | None = None,
) -> str:
    """Generate a SLURM submission script."""
    phase = config.get("training", {}).get("phase", "A")
    exp_name = config.get("experiment_name", f"phase{phase}")

    if job_name is None:
        job_name = f"ldiff_{exp_name}"

    if time_limit is None:
        time_limit = SLURM_DEFAULTS["time_limit"]

    mem = f"{48 * n_gpus}G"
    cpus = SLURM_DEFAULTS["cpus_per_gpu"] * n_gpus

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={SLURM_DEFAULTS['partition']}",
        f"#SBATCH --account={SLURM_DEFAULTS['account']}",
        f"#SBATCH --gres=gpu:{SLURM_DEFAULTS['gpu_type']}:{n_gpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=slurm-{exp_name}-%j.out",
        "",
        "set -euo pipefail",
        "",
        f'WORK_DIR="{SLURM_DEFAULTS["work_dir"]}"',
        'cd "$WORK_DIR"',
        "",
        "source $CONDA_PREFIX/etc/profile.d/conda.sh",
        f"conda activate {SLURM_DEFAULTS['conda_env']}",
        "",
        f'echo "=== LandmarkDiff Training: {exp_name} ==="',
        'echo "Start: $(date)"',
        'echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"',
        f'echo "Config: {config_path}"',
        "",
    ]

    # Training command
    if n_gpus > 1:
        lines.append(f"NGPUS={n_gpus}")
        lines.append(
            f'torchrun --nproc_per_node=$NGPUS scripts/train_controlnet.py --config "{config_path}"'
        )
    else:
        lines.append(f'python scripts/train_controlnet.py --config "{config_path}"')

    lines.extend(
        [
            "",
            'echo ""',
            'echo "Training complete: $(date)"',
            "",
            "# Post-training analysis (non-blocking)",
            f'OUTPUT_DIR="{config.get("output_dir", "checkpoints")}"',
            'if [ -d "$OUTPUT_DIR" ]; then',
            '    python scripts/analyze_training_run.py --run_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/analysis_report.md" 2>&1 || echo "Analysis skipped"',
            "fi",
        ]
    )

    return "\n".join(lines) + "\n"


def submit_slurm(script_path: str, dry_run: bool = False) -> int | None:
    """Submit a SLURM script and return job ID."""
    if dry_run:
        logger.info("[DRY RUN] Would submit: sbatch %s", script_path)
        return None

    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("sbatch failed: %s", result.stderr.strip())
        return None

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        return int(match.group(1))

    logger.error("Unexpected sbatch output: %s", result.stdout.strip())
    return None


def check_existing_jobs(job_name_prefix: str) -> list[dict]:
    """Check for existing SLURM jobs with similar names."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--format=%i %j %T %M", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                jid, name, state, elapsed = parts[0], parts[1], parts[2], parts[3]
                if job_name_prefix in name:
                    jobs.append({"id": jid, "name": name, "state": state, "elapsed": elapsed})
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Launch training with preflight checks")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--time", default=None, help="SLURM time limit (e.g. 48:00:00)")
    parser.add_argument("--dry-run", action="store_true", help="Check everything but don't submit")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run training dry-run validation (mini training loop on CPU)",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    parser.add_argument("--force", action="store_true", help="Submit even if warnings exist")
    parser.add_argument("--monitor", action="store_true", help="Launch monitor after submission")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)
    phase = config.get("training", {}).get("phase", "A")
    exp_name = config.get("experiment_name", f"phase{phase}")

    logger.info("=" * 60)
    logger.info("TRAINING LAUNCHER: %s", exp_name)
    logger.info("=" * 60)
    logger.info("Config: %s", config_path)
    logger.info("Phase: %s | GPUs: %d", phase, args.gpus)

    # Check for existing jobs
    existing = check_existing_jobs("ldiff_")
    if existing:
        logger.info("Active training jobs:")
        for j in existing:
            logger.info("  %s %s [%s] %s", j["id"], j["name"], j["state"], j["elapsed"])

    # Preflight checks
    if not args.skip_preflight:
        logger.info("Running preflight checks...")
        passed, messages, n_warnings = run_preflight(config_path)

        for msg in messages:
            logger.info("  %s", msg)

        if not passed:
            logger.error("PREFLIGHT FAILED -- fix issues before submitting.")
            logger.error("Use --skip-preflight to bypass (not recommended).")
            sys.exit(1)

        if n_warnings > 0 and not args.force:
            logger.warning("%d warning(s). Use --force to submit anyway.", n_warnings)
            # Dependency/GPU warnings are expected on login nodes
            # Auto-force if all warnings are dep/GPU related
            dep_gpu_only = all(
                "Dependencies" in m or "GPU" in m for m in messages if m.startswith("WARN:")
            )
            if dep_gpu_only:
                logger.info("  (Auto-forcing: only dependency/GPU warnings from login node)")
            else:
                sys.exit(1)

        logger.info("Preflight PASSED")
    else:
        logger.info("Preflight checks SKIPPED")

    # Generate SLURM script
    script_content = generate_slurm_script(
        config_path,
        config,
        args.gpus,
        args.time,
    )

    script_path = PROJECT_ROOT / f"slurm_launch_{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    logger.info("SLURM script: %s", script_path)

    if args.dry_run or args.validate:
        logger.info("[DRY RUN] Script contents:")
        logger.info("-" * 40)
        logger.info("%s", script_content)
        logger.info("-" * 40)

        if args.validate:
            logger.info("Running training dry-run validation...")
            from scripts.dry_run_training import run_dry_run

            dr_result = run_dry_run(config_path, n_steps=3)
            if not dr_result.all_passed:
                logger.error("Dry-run validation FAILED. Fix issues before submitting.")
                sys.exit(1)

        logger.info("Use without --dry-run to submit.")
        return

    # Submit
    logger.info("Submitting to SLURM...")
    job_id = submit_slurm(str(script_path))

    if job_id:
        logger.info("Job submitted: %d", job_id)
        logger.info("Monitor: squeue -j %d", job_id)
        logger.info("Logs: tail -f slurm-%s-%d.out", exp_name, job_id)

        # Save launch info
        launch_info = {
            "job_id": job_id,
            "config": config_path,
            "phase": phase,
            "experiment": exp_name,
            "gpus": args.gpus,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": str(script_path),
        }
        info_path = PROJECT_ROOT / "launch_info.json"
        with open(info_path, "w") as f:
            json.dump(launch_info, f, indent=2)

        if args.monitor:
            logger.info("Starting monitor...")
            os.execvp(
                "python",
                [
                    "python",
                    "scripts/monitor_training.py",
                    "--job_id",
                    str(job_id),
                ],
            )
    else:
        logger.error("Submission failed.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
