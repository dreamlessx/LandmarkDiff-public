#!/usr/bin/env python3
"""Launch a training experiment from a YAML config file.

Generates a SLURM job script and optionally submits it, or runs locally.
Handles Phase A/B configuration, checkpointing, and WandB integration.

Usage:
    # Generate SLURM script without submitting
    python scripts/launch_experiment.py --config configs/phaseA_default.yaml --dry-run

    # Submit to SLURM
    python scripts/launch_experiment.py --config configs/phaseA_default.yaml --submit

    # Run locally (no SLURM)
    python scripts/launch_experiment.py --config configs/phaseA_default.yaml --local
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.config import ExperimentConfig

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=${{SLURM_ACCOUNT:-batch}}
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --constraint=a6000
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

echo "============================================"
echo "Experiment: {experiment_name}"
echo "Config: {config_path}"
echo "Phase: {phase}"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

# Environment setup
source "${{CONDA_PREFIX:-$HOME/miniconda3}}"/etc/profile.d/conda.sh
conda activate ${{LANDMARKDIFF_ENV:-landmarkdiff}}

cd {project_root}

# Copy config to output dir for reproducibility
mkdir -p {output_dir}
cp {config_path} {output_dir}/config.yaml

# Run training
python scripts/train_controlnet.py \\
    --data_dir {data_dir} \\
    --output_dir {output_dir} \\
    --resolution {resolution} \\
    --train_batch_size {batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --learning_rate {learning_rate} \\
    --lr_scheduler {lr_scheduler} \\
    --max_train_steps {max_train_steps} \\
    --mixed_precision {mixed_precision} \\
    --seed {seed} \\
    --phase {phase} \\
    --ema_decay {ema_decay} \\
    --save_every {save_every} \\
    --validate_every {validate_every} \\
    --num_validation_samples {num_val_samples} \\
    {extra_args}

echo "============================================"
echo "Training complete: $(date)"
echo "Output: {output_dir}"
echo "============================================"
"""


def generate_slurm_script(config: ExperimentConfig, config_path: str) -> str:
    """Generate a SLURM job script from config."""
    project_root = Path(__file__).resolve().parent.parent

    # Build extra args for Phase B
    extra_args = []
    if config.training.phase == "B":
        if config.training.use_differentiable_arcface:
            extra_args.append("--use_differentiable_arcface")
        if config.training.arcface_weights_path:
            extra_args.append(f"--arcface_weights {config.training.arcface_weights_path}")
        if config.training.resume_from_checkpoint:
            extra_args.append(f"--resume_from {config.training.resume_from_checkpoint}")

    if config.wandb.enabled:
        extra_args.append("--wandb")
        if config.wandb.run_name:
            extra_args.append(f"--wandb_run_name {config.wandb.run_name}")

    # Resource estimation
    mem = 48 if config.training.phase == "A" else 64  # Phase B needs more for ArcFace
    cpus = min(config.data.num_workers + 2, 8)
    time_hours = max(4, config.training.max_train_steps // 2000)
    time_limit = f"{min(time_hours, 72):02d}:00:00"

    return SLURM_TEMPLATE.format(
        job_name=f"ld_{config.experiment_name[:20]}",
        experiment_name=config.experiment_name,
        config_path=config_path,
        phase=config.training.phase,
        num_gpus=1,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        output_dir=config.output_dir,
        project_root=str(project_root),
        data_dir=config.data.train_dir,
        resolution=config.data.image_size,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler=config.training.lr_scheduler,
        max_train_steps=config.training.max_train_steps,
        mixed_precision=config.training.mixed_precision,
        seed=config.training.seed,
        ema_decay=config.model.ema_decay,
        save_every=config.training.save_every_n_steps,
        validate_every=config.training.validate_every_n_steps,
        num_val_samples=config.training.num_validation_samples,
        extra_args=" \\\n    ".join(extra_args) if extra_args else "",
    )


def main():
    parser = argparse.ArgumentParser(description="Launch experiment from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--submit", action="store_true", help="Submit to SLURM")
    parser.add_argument("--dry-run", action="store_true", help="Print script without submitting")
    parser.add_argument("--local", action="store_true", help="Run locally (no SLURM)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    config = ExperimentConfig.from_yaml(config_path)
    print(f"Experiment: {config.experiment_name}")
    print(f"Phase: {config.training.phase}")
    print(f"LR: {config.training.learning_rate}")
    print(f"Steps: {config.training.max_train_steps}")
    print(f"Output: {config.output_dir}")

    if args.local:
        # Run directly without SLURM
        print("\nRunning locally...")
        script = generate_slurm_script(config, str(config_path))
        # Extract the python command from the script
        lines = script.split("\n")
        cmd_lines = []
        capturing = False
        for line in lines:
            if "python scripts/train_controlnet.py" in line:
                capturing = True
            if capturing:
                cmd_lines.append(line.rstrip(" \\"))
                if not line.rstrip().endswith("\\"):
                    break
        cmd = " ".join(cmd_lines)
        print(f"Command: {cmd}")
        return

    # Generate SLURM script
    script = generate_slurm_script(config, str(config_path))

    # Save script
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = Path(config.output_dir) / f"train_{timestamp}.slurm"
    script_path.write_text(script)
    print(f"\nSLURM script: {script_path}")

    if args.dry_run:
        print("\n--- SLURM Script ---")
        print(script)
        print("--- End Script ---")
        print("\nDry run; not submitted. Use --submit to submit.")
        return

    if args.submit:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"ERROR: {result.stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()
