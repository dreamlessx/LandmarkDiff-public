"""Monitor training progress from SLURM log files.

Parses training logs to extract loss curves, learning rates, and timing info.
Generates training curve plots and progress summaries.

Usage:
    # Monitor latest SLURM log
    python scripts/monitor_training.py

    # Monitor specific log file
    python scripts/monitor_training.py --log slurm-phaseA-12345.out

    # Monitor by SLURM job ID
    python scripts/monitor_training.py --job_id 12345

    # Live follow mode (like tail -f)
    python scripts/monitor_training.py --follow

    # Auto-refresh every 60 seconds
    python scripts/monitor_training.py --watch 60

    # Generate plot only
    python scripts/monitor_training.py --plot results/training_curve.png

    # Export metrics to JSON
    python scripts/monitor_training.py --export metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_training_log(log_path: str) -> dict:
    """Parse a training log file and extract metrics.

    Returns dict with lists of step, loss, lr, grad_norm, speed, eta.
    """
    steps = []
    losses = []
    lrs = []
    grad_norms = []
    speeds = []
    etas = []

    # Phase B loss components

    pattern = re.compile(
        r"Step\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+"
        r"LR:\s+([\d.e+-]+)\s+\|\s+GradNorm:\s+([\d.]+)\s+\|\s+"
        r"([\d.]+)\s+it/s\s+\|\s+ETA:\s+([\d.]+)h"
    )
    re.compile(r"val/ssim.*?([\d.]+).*?val/lpips.*?([\d.]+)")
    checkpoint_pattern = re.compile(r"Checkpoint saved:\s+(.+)")

    checkpoints = []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
                grad_norms.append(float(m.group(5)))
                speeds.append(float(m.group(6)))
                etas.append(float(m.group(7)))

            # Checkpoint saves
            cm = checkpoint_pattern.search(line)
            if cm:
                checkpoints.append(cm.group(1).strip())

    return {
        "steps": steps,
        "total_steps": int(m.group(2))
        if steps and (m := pattern.search(open(log_path).read()))
        else 0,
        "losses": losses,
        "lrs": lrs,
        "grad_norms": grad_norms,
        "speeds": speeds,
        "etas": etas,
        "checkpoints": checkpoints,
        "log_path": log_path,
    }


def print_summary(data: dict) -> None:
    """Print a concise training progress summary."""
    if not data["steps"]:
        print("No training data found in log.")
        return

    current = data["steps"][-1]
    total = data["total_steps"]
    progress = current / total * 100 if total > 0 else 0

    print(f"{'=' * 60}")
    print(f"Training Progress: {current:,}/{total:,} ({progress:.1f}%)")
    print(f"{'=' * 60}")
    print(f"  Log: {data['log_path']}")
    print(f"  Current loss:    {data['losses'][-1]:.6f}")
    print(
        f"  Min loss:        {min(data['losses']):.6f} (step {data['steps'][data['losses'].index(min(data['losses']))]})"
    )
    print(f"  Learning rate:   {data['lrs'][-1]:.2e}")
    print(f"  Grad norm:       {data['grad_norms'][-1]:.2f}")
    print(f"  Speed:           {data['speeds'][-1]:.1f} it/s")
    print(f"  ETA:             {data['etas'][-1]:.1f}h")

    # Loss trend (last 10 log points)
    if len(data["losses"]) >= 10:
        recent = data["losses"][-10:]
        older = data["losses"][-20:-10] if len(data["losses"]) >= 20 else data["losses"][:10]
        trend = np.mean(recent) - np.mean(older)
        direction = "improving" if trend < 0 else "increasing" if trend > 0 else "stable"
        print(f"  Loss trend:      {direction} ({trend:+.6f})")

    # Smoothed loss curve (window of 5)
    if len(data["losses"]) >= 5:
        window = min(5, len(data["losses"]))
        smoothed = np.convolve(data["losses"], np.ones(window) / window, mode="valid")
        print(f"  Smoothed loss:   {smoothed[-1]:.6f}")

    if data["checkpoints"]:
        print(f"  Last checkpoint: {data['checkpoints'][-1]}")

    print()


def plot_training(data: dict, output_path: str) -> None:
    """Generate training curve plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    if not data["steps"]:
        print("No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LandmarkDiff Training Progress", fontsize=14, fontweight="bold")

    steps = data["steps"]

    # Loss curve with smoothing
    ax = axes[0, 0]
    ax.plot(steps, data["losses"], alpha=0.3, color="blue", linewidth=0.5)
    if len(data["losses"]) >= 10:
        window = min(50, len(data["losses"]) // 5)
        if window >= 2:
            smoothed = np.convolve(data["losses"], np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1 :]
            ax.plot(
                smooth_steps, smoothed, color="blue", linewidth=2, label=f"Smoothed (w={window})"
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[0, 1]
    ax.plot(steps, data["lrs"], color="orange", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 0]
    ax.plot(steps, data["grad_norms"], alpha=0.5, color="green", linewidth=0.5)
    if len(data["grad_norms"]) >= 10:
        window = min(50, len(data["grad_norms"]) // 5)
        if window >= 2:
            smoothed = np.convolve(data["grad_norms"], np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1 :]
            ax.plot(smooth_steps, smoothed, color="green", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    # Speed / ETA
    ax = axes[1, 1]
    ax.plot(steps, data["speeds"], color="red", linewidth=1, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/sec")
    ax.set_title("Training Speed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to {output_path}")


def get_slurm_job_status(prefix: str = "surgery_") -> list[dict]:
    """Query SLURM for active training jobs."""

    try:
        result = subprocess.run(
            ["squeue", "-u", "$USER", "--format=%i %j %t %M %N %P %b", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**__import__("os").environ},
        )
        if result.returncode != 0:
            return []
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                job = {
                    "job_id": parts[0],
                    "name": parts[1],
                    "state": parts[2],
                    "time": parts[3],
                    "node": parts[4] if len(parts) > 4 else "",
                    "partition": parts[5] if len(parts) > 5 else "",
                    "gres": parts[6] if len(parts) > 6 else "",
                }
                if prefix and not job["name"].startswith(prefix):
                    continue
                jobs.append(job)
        return jobs
    except Exception:
        return []


def print_slurm_status() -> None:
    """Print status of SLURM training jobs."""
    jobs = get_slurm_job_status()
    if not jobs:
        print("  No active SLURM training jobs")
        return
    print(f"  Active SLURM jobs ({len(jobs)}):")
    for j in jobs:
        state_map = {"R": "RUNNING", "PD": "PENDING", "CG": "COMPLETING"}
        state = state_map.get(j["state"], j["state"])
        print(f"    {j['job_id']} | {j['name']:<25} | {state:<10} | {j['time']} | {j['node']}")


def find_latest_log() -> str | None:
    """Find the most recent SLURM training log."""
    work_dir = PROJECT_ROOT
    candidates = sorted(work_dir.glob("slurm-phase*-*.out"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(work_dir.glob("slurm-pipeline-*.out"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(work_dir.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def find_log_by_job_id(job_id: int) -> str | None:
    """Find SLURM log file for a specific job ID."""
    work_dir = PROJECT_ROOT
    for pattern in [f"slurm-*-{job_id}.out", f"slurm-{job_id}.out", f"slurm_{job_id}.out"]:
        matches = sorted(work_dir.glob(pattern))
        if matches:
            return str(matches[-1])

    # Check launch_info.json for experiment name
    info_path = work_dir / "launch_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if info.get("job_id") == job_id:
            exp = info.get("experiment", "")
            for pattern in [f"slurm-{exp}-{job_id}.out"]:
                matches = sorted(work_dir.glob(pattern))
                if matches:
                    return str(matches[-1])
    return None


def follow_log(log_path: str, refresh: float = 2.0) -> None:
    """Live-follow a log file (like tail -f) with parsed progress display."""
    path = Path(log_path)
    if not path.exists():
        print(f"Waiting for log file: {log_path}")
        while not path.exists():
            time.sleep(1)

    data = parse_training_log(log_path)
    print_summary(data)

    # Seek to end and follow
    try:
        with open(log_path) as f:
            f.seek(0, 2)
            last_display = time.time()
            while True:
                line = f.readline()
                if line:
                    # Check if this is a step line
                    if "Step" in line and "Loss:" in line:
                        data = parse_training_log(log_path)
                        if time.time() - last_display > refresh:
                            print("\033[2J\033[H", end="")  # clear screen
                            print_summary(data)
                            last_display = time.time()
                    elif "Training complete" in line or "All done" in line:
                        data = parse_training_log(log_path)
                        print_summary(data)
                        print("Training completed!")
                        break
                    elif "ERROR" in line or "CUDA out of memory" in line:
                        print(f"\nERROR DETECTED: {line.strip()}")
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        data = parse_training_log(log_path)
        print_summary(data)


def export_metrics(data: dict, output_path: str) -> None:
    """Export parsed training metrics to JSON."""
    export = {
        "steps": data["steps"],
        "total_steps": data["total_steps"],
        "losses": data["losses"],
        "learning_rates": data["lrs"],
        "grad_norms": data["grad_norms"],
        "speeds": data["speeds"],
        "etas": data["etas"],
        "checkpoints": data["checkpoints"],
        "log_path": data["log_path"],
    }

    if data["steps"]:
        export["summary"] = {
            "current_step": data["steps"][-1],
            "progress_pct": round(100.0 * data["steps"][-1] / data["total_steps"], 1)
            if data["total_steps"]
            else 0,
            "current_loss": data["losses"][-1],
            "min_loss": min(data["losses"]),
            "current_lr": data["lrs"][-1],
            "avg_speed": round(sum(data["speeds"]) / len(data["speeds"]), 2),
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Metrics exported to {output_path}")


def detect_convergence(data: dict) -> str:
    """Quick convergence analysis from parsed log data."""
    if len(data["losses"]) < 20:
        return "Too few data points for analysis"

    losses = data["losses"]
    lines = []

    # Check for NaN
    nan_count = sum(1 for val in losses if val != val or val > 1e6)
    if nan_count:
        lines.append(f"WARNING: {nan_count} NaN/extreme loss values detected")

    # Loss trend
    recent = losses[-20:]
    older = losses[-40:-20] if len(losses) >= 40 else losses[: len(losses) // 2]
    if older:
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        change_pct = (recent_avg - older_avg) / max(abs(older_avg), 1e-8) * 100
        if change_pct < -2:
            lines.append(f"Loss: DECREASING ({change_pct:.1f}% change); training is progressing")
        elif change_pct > 5:
            lines.append(f"Loss: INCREASING ({change_pct:+.1f}% change); possible divergence!")
        else:
            lines.append(f"Loss: PLATEAU ({change_pct:+.1f}% change); may be converging")

    # Gradient norms
    if data["grad_norms"]:
        max_gn = max(data["grad_norms"][-50:])
        if max_gn > 100:
            lines.append(f"WARNING: High gradient norms (max={max_gn:.1f}); possible instability")
        elif max_gn < 0.01:
            lines.append(
                f"WARNING: Very small gradient norms (max={max_gn:.4f}); possible vanishing gradients"
            )

    # ETA
    if data["etas"]:
        lines.append(f"ETA: {data['etas'][-1]:.1f}h remaining")

    return "\n  ".join(lines) if lines else "Training appears healthy"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log", default=None, help="Path to SLURM log file")
    parser.add_argument("--job_id", type=int, default=None, help="SLURM job ID")
    parser.add_argument("--plot", default=None, help="Save training curve plot")
    parser.add_argument(
        "--watch", type=int, default=0, help="Auto-refresh interval in seconds (0 = one-shot)"
    )
    parser.add_argument("--follow", action="store_true", help="Live follow mode (like tail -f)")
    parser.add_argument("--export", default=None, help="Export metrics to JSON")
    parser.add_argument("--slurm", action="store_true", help="Show SLURM job status")
    parser.add_argument("--convergence", action="store_true", help="Run convergence analysis")
    args = parser.parse_args()

    if args.slurm:
        print_slurm_status()
        if not args.log and not args.job_id:
            sys.exit(0)

    # Find log file
    log_path = args.log
    if not log_path and args.job_id:
        log_path = find_log_by_job_id(args.job_id)
        if log_path:
            print(f"Found log for job {args.job_id}: {log_path}")
    if not log_path:
        log_path = find_latest_log()

    if not log_path or not Path(log_path).exists():
        print("No training log found. Specify with --log <path> or --job_id <id>")
        sys.exit(1)

    # Follow mode
    if args.follow:
        follow_log(log_path)
        sys.exit(0)

    while True:
        data = parse_training_log(log_path)
        print_summary(data)

        if args.convergence:
            print("  Convergence Analysis:")
            print(f"  {detect_convergence(data)}")
            print()

        if args.slurm:
            print_slurm_status()

        if args.plot:
            plot_training(data, args.plot)

        if args.export:
            export_metrics(data, args.export)

        if args.watch <= 0:
            break
        print(f"Refreshing in {args.watch}s... (Ctrl+C to stop)\n")
        time.sleep(args.watch)
