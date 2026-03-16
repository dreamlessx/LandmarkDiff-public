#!/usr/bin/env python3
"""Launch a WandB hyperparameter sweep on SLURM.

Generates agent SLURM scripts for parallel sweep execution.
Each agent runs independently, pulling configurations from the sweep server.

Usage:
    # Create sweep and launch 4 agents on SLURM
    python scripts/launch_sweep.py --sweep configs/wandb_sweep_phaseA.yaml --agents 4

    # Generate scripts only (dry run)
    python scripts/launch_sweep.py --sweep configs/wandb_sweep_phaseA.yaml --agents 4 --dry-run

    # Use existing sweep ID
    python scripts/launch_sweep.py --sweep-id abc123 --agents 4
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

SLURM_AGENT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=sweep_{sweep_name}_agent{agent_idx}
#SBATCH --account=${{SLURM_ACCOUNT:-batch}}
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}
#SBATCH --output={output_dir}/agent{agent_idx}_%j.out
#SBATCH --error={output_dir}/agent{agent_idx}_%j.err

echo "============================================"
echo "Sweep: {sweep_id}"
echo "Agent: {agent_idx}/{total_agents}"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

# Environment
source "${{CONDA_PREFIX:-$HOME/miniconda3}}"/etc/profile.d/conda.sh
conda activate ${{LANDMARKDIFF_ENV:-landmarkdiff}}

cd {project_root}

# WandB offline mode for HPC
export WANDB_MODE=offline
export WANDB_DIR={output_dir}/wandb

# Run sweep agent
wandb agent {sweep_id}

echo "Agent {agent_idx} complete: $(date)"
"""


def main():
    parser = argparse.ArgumentParser(description="Launch WandB sweep on SLURM")
    parser.add_argument("--sweep", default=None, help="Path to sweep YAML config")
    parser.add_argument("--sweep-id", default=None, help="Existing sweep ID (skip creation)")
    parser.add_argument("--agents", type=int, default=4, help="Number of parallel agents")
    parser.add_argument("--mem", type=int, default=48, help="Memory per agent (GB)")
    parser.add_argument("--time", default="24:00:00", help="Time limit per agent")
    parser.add_argument("--output-dir", default=None, help="Output directory for scripts and logs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate scripts without submitting"
    )
    parser.add_argument("--submit", action="store_true", help="Submit to SLURM after generation")
    args = parser.parse_args()

    if args.sweep is None and args.sweep_id is None:
        parser.error("Provide --sweep (config YAML) or --sweep-id (existing sweep)")

    project_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine sweep name
    if args.sweep:
        sweep_name = Path(args.sweep).stem.replace("wandb_sweep_", "")
    else:
        sweep_name = args.sweep_id[:8]

    output_dir = Path(args.output_dir or f"outputs/sweep_{sweep_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sweep if needed
    sweep_id = args.sweep_id
    if sweep_id is None:
        print(f"Creating sweep from {args.sweep}...")
        print("NOTE: WandB sweep creation requires network access.")
        print(
            "  On your SLURM cluster, run this on a login node or use"
            " --sweep-id with a pre-created sweep."
        )
        print(f"  To create manually: wandb sweep {args.sweep}")
        sweep_id = f"PLACEHOLDER_{sweep_name}"

    print(f"Sweep ID: {sweep_id}")
    print(f"Agents: {args.agents}")
    print(f"Output: {output_dir}")

    # Generate agent scripts
    scripts = []
    for i in range(args.agents):
        script = SLURM_AGENT_TEMPLATE.format(
            sweep_name=sweep_name,
            sweep_id=sweep_id,
            agent_idx=i,
            total_agents=args.agents,
            project_root=str(project_root),
            output_dir=str(output_dir),
            mem=args.mem,
            time_limit=args.time,
        )
        script_path = output_dir / f"agent_{i}.slurm"
        script_path.write_text(script)
        scripts.append(script_path)
        print(f"  Agent {i}: {script_path}")

    # Master submission script
    master = "#!/bin/bash\n"
    master += f"# Submit all {args.agents} sweep agents\n"
    master += f"# Sweep: {sweep_id}\n\n"
    for i, sp in enumerate(scripts):
        master += f'JOB{i}=$(sbatch --parsable "{sp}")\n'
        master += f'echo "Agent {i}: $JOB{i}"\n'
    master += f'\necho "All {args.agents} agents submitted."'

    master_path = output_dir / "submit_all.sh"
    master_path.write_text(master)
    print(f"\nMaster script: {master_path}")

    if args.dry_run:
        print("\nDry run; scripts generated but not submitted.")
        print(f"To submit: bash {master_path}")
        return

    if args.submit:
        for sp in scripts:
            result = subprocess.run(["sbatch", str(sp)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {result.stdout.strip()}")
            else:
                print(f"  ERROR: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
