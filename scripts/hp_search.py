"""Hyperparameter search via SLURM job array.

Generates a grid of hyperparameter configs and submits them as a SLURM
job array. Each job runs a short training experiment, and results are
compared using the experiment tracker.

Usage:
    # Generate configs and submit
    python scripts/hp_search.py --submit

    # Generate configs only (dry run)
    python scripts/hp_search.py --dry-run

    # Analyze results after jobs complete
    python scripts/hp_search.py --analyze --experiments_dir outputs/hp_search/experiments
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default search space
DEFAULT_SEARCH_SPACE = {
    "learning_rate": [5e-6, 1e-5, 2e-5],
    "ema_decay": [0.999, 0.9999],
    "gradient_accumulation_steps": [2, 4],
}


def generate_configs(
    search_space: dict[str, list],
    base_config: dict | None = None,
    output_dir: str = "outputs/hp_search",
) -> list[dict]:
    """Generate all combinations of hyperparameters."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(itertools.product(*values)):
        config = base_config.copy() if base_config else {}
        for k, v in zip(keys, combo, strict=True):
            config[k] = v
        config["trial_id"] = i
        config["output_dir"] = str(out / f"trial_{i:03d}")
        configs.append(config)

    # Save all configs
    with open(out / "hp_search_configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Generated {len(configs)} trial configs")
    return configs


def generate_slurm_script(
    configs: list[dict],
    output_dir: str = "outputs/hp_search",
    steps_per_trial: int = 5000,
    data_dir: str = "data/training_combined",
) -> str:
    """Generate a SLURM job array script."""
    n = len(configs)
    out = Path(output_dir)

    script = f"""#!/bin/bash
#SBATCH --job-name=surgery_hp_search
#SBATCH --output={out}/slurm-%A_%a.out
#SBATCH --error={out}/slurm-%A_%a.err
#SBATCH --partition=batch_gpu
#SBATCH --account=${{SLURM_ACCOUNT:-batch}}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-{n - 1}

source "${{CONDA_PREFIX:-$HOME/miniconda3}}"/etc/profile.d/conda.sh
conda activate ${{LANDMARKDIFF_ENV:-landmarkdiff}}

cd {PROJECT_ROOT}

# Read config for this trial
TRIAL_ID=$SLURM_ARRAY_TASK_ID
CONFIG_FILE="{out}/hp_search_configs.json"

# Extract hyperparameters using Python
read LR EMA ACCUM TRIAL_DIR <<< $(python -c "
import json
with open('$CONFIG_FILE') as f:
    configs = json.load(f)
c = configs[$TRIAL_ID]
print(c['learning_rate'], c['ema_decay'], c['gradient_accumulation_steps'], c['output_dir'])
")

echo "Trial $TRIAL_ID: LR=$LR, EMA=$EMA, ACCUM=$ACCUM"
echo "Output: $TRIAL_DIR"

python scripts/train_controlnet.py \\
    --data_dir {data_dir} \\
    --output_dir "$TRIAL_DIR" \\
    --learning_rate $LR \\
    --ema_decay $EMA \\
    --gradient_accumulation_steps $ACCUM \\
    --num_train_steps {steps_per_trial} \\
    --checkpoint_every {steps_per_trial} \\
    --log_every 50 \\
    --sample_every {steps_per_trial // 5} \\
    --phase A \\
    --seed 42

echo "Trial $TRIAL_ID complete"
"""

    script_path = out / "hp_search.sh"
    with open(script_path, "w") as f:
        f.write(script)
    script_path.chmod(0o755)

    return str(script_path)


def analyze_results(experiments_dir: str) -> None:
    """Analyze HP search results from experiment tracker."""
    from landmarkdiff.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(experiments_dir)
    exps = tracker.list_experiments()

    if not exps:
        print("No experiments found. Jobs may not have completed yet.")
        return

    print(f"\n{'=' * 70}")
    print(f"Hyperparameter Search Results ({len(exps)} trials)")
    print(f"{'=' * 70}")

    # Print table
    print(f"\n{'Trial':<10} {'Status':<12} {'LR':>10} {'FID':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 60)

    for exp in exps:
        fid = f"{exp.get('fid', ''):.1f}" if "fid" in exp else "--"
        ssim = f"{exp.get('ssim', ''):.4f}" if "ssim" in exp else "--"
        lpips = f"{exp.get('lpips', ''):.4f}" if "lpips" in exp else "--"
        print(f"{exp['id']:<10} {exp['status']:<12} {'':>10} {fid:>8} {ssim:>8} {lpips:>8}")

    # Find best
    best_fid = tracker.get_best("fid", lower_is_better=True)
    best_ssim = tracker.get_best("ssim", lower_is_better=False)
    if best_fid:
        print(f"\nBest FID:  {best_fid}")
    if best_ssim:
        print(f"Best SSIM: {best_ssim}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--output_dir", default="outputs/hp_search")
    parser.add_argument("--data_dir", default="data/training_combined")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps per trial")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM job array")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate configs without submitting"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze results from completed trials"
    )
    parser.add_argument(
        "--experiments_dir", default=None, help="Path to experiments directory for analysis"
    )
    args = parser.parse_args()

    if args.analyze:
        exp_dir = args.experiments_dir or f"{args.output_dir}/experiments"
        analyze_results(exp_dir)
        return

    configs = generate_configs(DEFAULT_SEARCH_SPACE, output_dir=args.output_dir)
    script_path = generate_slurm_script(
        configs,
        args.output_dir,
        args.steps,
        args.data_dir,
    )

    print(f"SLURM script: {script_path}")
    print(f"Trials: {len(configs)}")

    if args.submit:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"Submit failed: {result.stderr}")
    elif args.dry_run:
        print("\nDry run; not submitting. Review configs:")
        for c in configs[:3]:
            print(
                f"  Trial {c['trial_id']}: lr={c['learning_rate']}, "
                f"ema={c['ema_decay']}, accum={c['gradient_accumulation_steps']}"
            )
        if len(configs) > 3:
            print(f"  ... and {len(configs) - 3} more")
    else:
        print("Use --submit to submit or --dry-run to preview")


if __name__ == "__main__":
    main()
