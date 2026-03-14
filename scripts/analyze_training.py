#!/usr/bin/env python3
"""Parse SLURM training logs and generate matplotlib plots.

Extracts training metrics (loss, LR, grad norm, speed) and validation
metrics (SSIM, LPIPS, per-procedure breakdowns) from one or more SLURM
log files.  Generates PNG plots and detects anomalies (loss spikes,
NaN values, training stalls).

Usage:
    python scripts/analyze_training.py LOG_FILE [LOG_FILE2 ...] --output-dir plots/
    python scripts/analyze_training.py --latest  # auto-find latest slurm-phaseA-*.out
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── Regex patterns matching actual SLURM log format ──────────────

_STEP_RE = re.compile(
    r"Step\s+(\d+)/(\d+)\s*\|\s*Loss:\s*([\d.eE+-]+)\s*\|\s*"
    r"LR:\s*([\d.eE+-]+)\s*\|\s*GradNorm:\s*([\d.eE+-]+)\s*\|\s*"
    r"([\d.]+)\s*it/s\s*\|\s*ETA:\s*([\d.]+)h"
)
_VAL_RE = re.compile(
    r"Validation\s*@\s*step\s+(\d+):\s*"
    r"SSIM=([\d.]+)\+/-([\d.]+)\s+"
    r"LPIPS=([\d.]+)\+/-([\d.]+)\s*"
    r"\(([\d.]+)s\)"
)
_PROC_RE = re.compile(r"Per-procedure:\s*(.+)")
_PROC_ITEM_RE = re.compile(r"(\w+):\s*SSIM=([\d.]+)")


# ── Data containers ──────────────────────────────────────────────


@dataclass
class TrainRecord:
    step: int
    total_steps: int
    loss: float
    lr: float
    grad_norm: float
    speed: float
    eta_hours: float


@dataclass
class ValRecord:
    step: int
    ssim_mean: float
    ssim_std: float
    lpips_mean: float
    lpips_std: float
    duration_s: float
    per_procedure: dict[str, float] = field(default_factory=dict)


@dataclass
class RunData:
    """Parsed data from a single log file."""

    path: Path
    train: list[TrainRecord] = field(default_factory=list)
    val: list[ValRecord] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.path.stem


# ── Parsing ──────────────────────────────────────────────────────


def parse_log(path: Path) -> RunData:
    """Parse a single SLURM log file."""
    run = RunData(path=path)
    pending_val: ValRecord | None = None

    with open(path) as fh:
        for line in fh:
            # Training step
            m = _STEP_RE.search(line)
            if m:
                run.train.append(
                    TrainRecord(
                        step=int(m.group(1)),
                        total_steps=int(m.group(2)),
                        loss=float(m.group(3)),
                        lr=float(m.group(4)),
                        grad_norm=float(m.group(5)),
                        speed=float(m.group(6)),
                        eta_hours=float(m.group(7)),
                    )
                )
                continue

            # Validation header
            m = _VAL_RE.search(line)
            if m:
                if pending_val is not None:
                    run.val.append(pending_val)
                pending_val = ValRecord(
                    step=int(m.group(1)),
                    ssim_mean=float(m.group(2)),
                    ssim_std=float(m.group(3)),
                    lpips_mean=float(m.group(4)),
                    lpips_std=float(m.group(5)),
                    duration_s=float(m.group(6)),
                )
                continue

            # Per-procedure line (always follows validation header)
            m = _PROC_RE.search(line)
            if m and pending_val is not None:
                for pm in _PROC_ITEM_RE.finditer(m.group(1)):
                    pending_val.per_procedure[pm.group(1)] = float(pm.group(2))
                run.val.append(pending_val)
                pending_val = None
                continue

    # Flush any trailing validation without per-procedure
    if pending_val is not None:
        run.val.append(pending_val)

    logger.info(
        "Parsed %s: %d train steps, %d val checkpoints",
        path.name,
        len(run.train),
        len(run.val),
    )
    return run


def find_latest_logs() -> list[Path]:
    """Find the most recent slurm-phaseA-*.out in ROOT."""
    candidates = sorted(ROOT.glob("slurm-phaseA-*.out"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.warning("No slurm-phaseA-*.out files found in %s", ROOT)
        return []
    latest = candidates[-1]
    logger.info("Auto-selected latest log: %s", latest.name)
    return [latest]


# ── Anomaly detection ────────────────────────────────────────────


@dataclass
class Anomaly:
    kind: str
    step: int | None
    message: str


def detect_anomalies(run: RunData, window: int = 50) -> list[Anomaly]:
    """Detect loss spikes, NaN values, and training stalls."""
    anomalies: list[Anomaly] = []
    if len(run.train) < window:
        return anomalies

    losses = np.array([r.loss for r in run.train])
    steps = np.array([r.step for r in run.train])
    speeds = np.array([r.speed for r in run.train])

    # NaN / Inf
    nan_mask = ~np.isfinite(losses)
    if np.any(nan_mask):
        nan_steps = steps[nan_mask].tolist()
        anomalies.append(
            Anomaly(
                "nan",
                nan_steps[0],
                f"NaN/Inf loss at {len(nan_steps)} step(s): {nan_steps[:5]}",
            )
        )

    # Loss spikes: >3 std from rolling mean
    finite = np.where(np.isfinite(losses), losses, np.nan)
    for i in range(window, len(finite)):
        local = finite[i - window : i]
        local_clean = local[np.isfinite(local)]
        if len(local_clean) < 10:
            continue
        mu = np.mean(local_clean)
        sigma = np.std(local_clean)
        if sigma > 0 and np.isfinite(finite[i]) and (finite[i] - mu) > 3 * sigma:
            anomalies.append(
                Anomaly(
                    "spike",
                    int(steps[i]),
                    f"Loss spike at step {steps[i]}: {finite[i]:.6f} "
                    f"(rolling mean {mu:.6f}, 3*std threshold {mu + 3 * sigma:.6f})",
                )
            )

    # Stall: speed drops to <10% of median for >=5 consecutive steps
    if len(speeds) > 20:
        median_speed = np.median(speeds)
        if median_speed > 0:
            slow_mask = speeds < 0.1 * median_speed
            run_len = 0
            for i, is_slow in enumerate(slow_mask):
                if is_slow:
                    run_len += 1
                    if run_len == 5:
                        anomalies.append(
                            Anomaly(
                                "stall",
                                int(steps[i]),
                                f"Training stall near step {steps[i]}: "
                                f"speed dropped below {0.1 * median_speed:.2f} it/s "
                                f"(median {median_speed:.2f})",
                            )
                        )
                else:
                    run_len = 0

    return anomalies


# ── Plotting ─────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_loss(runs: list[RunData], out_dir: Path) -> None:
    """Loss curve over steps, one line per run."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for run in runs:
        if not run.train:
            continue
        steps = [r.step for r in run.train]
        losses = [r.loss for r in run.train]
        ax.plot(steps, losses, label=run.label, alpha=0.7, linewidth=0.8)
        # Rolling mean
        if len(losses) > 20:
            k = max(10, len(losses) // 50)
            kernel = np.ones(k) / k
            smoothed = np.convolve(losses, kernel, mode="valid")
            ax.plot(
                steps[k - 1 :],
                smoothed,
                linewidth=1.5,
                label=f"{run.label} (smooth)",
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "loss_curve")


def plot_lr(runs: list[RunData], out_dir: Path) -> None:
    """Learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        if not run.train:
            continue
        steps = [r.step for r in run.train]
        lrs = [r.lr for r in run.train]
        ax.plot(steps, lrs, label=run.label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "lr_schedule")


def plot_grad_norm(runs: list[RunData], out_dir: Path) -> None:
    """Gradient norm over steps."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        if not run.train:
            continue
        steps = [r.step for r in run.train]
        gnorms = [r.grad_norm for r in run.train]
        ax.plot(steps, gnorms, label=run.label, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "grad_norm")


def plot_val_metrics(runs: list[RunData], out_dir: Path) -> None:
    """Validation SSIM and LPIPS over steps (dual y-axis)."""
    has_val = any(run.val for run in runs)
    if not has_val:
        logger.info("No validation data; skipping val metrics plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    colors_ssim = plt.cm.tab10.colors  # type: ignore[attr-defined]
    colors_lpips = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for i, run in enumerate(runs):
        if not run.val:
            continue
        steps = [v.step for v in run.val]
        ssim = [v.ssim_mean for v in run.val]
        lpips = [v.lpips_mean for v in run.val]
        ssim_err = [v.ssim_std for v in run.val]
        lpips_err = [v.lpips_std for v in run.val]

        c1 = colors_ssim[i % len(colors_ssim)]
        c2 = colors_lpips[i % len(colors_lpips)]

        ax1.errorbar(
            steps,
            ssim,
            yerr=ssim_err,
            fmt="o-",
            color=c1,
            capsize=3,
            label=f"SSIM ({run.label})",
        )
        ax2.errorbar(
            steps,
            lpips,
            yerr=lpips_err,
            fmt="s--",
            color=c2,
            capsize=3,
            label=f"LPIPS ({run.label})",
        )

    ax1.set_xlabel("Step")
    ax1.set_ylabel("SSIM (higher = better)")
    ax2.set_ylabel("LPIPS (lower = better)")
    ax1.set_title("Validation Metrics")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.grid(True, alpha=0.3)
    _save(fig, out_dir, "val_metrics")


def plot_procedure_ssim(runs: list[RunData], out_dir: Path) -> None:
    """Per-procedure SSIM bar chart from the latest validation."""
    # Collect latest per-procedure data from each run
    bars: dict[str, dict[str, float]] = {}
    for run in runs:
        if not run.val:
            continue
        latest = run.val[-1]
        if latest.per_procedure:
            bars[run.label] = latest.per_procedure

    if not bars:
        logger.info("No per-procedure data; skipping bar chart")
        return

    all_procs = sorted({p for d in bars.values() for p in d})
    n_runs = len(bars)
    x = np.arange(len(all_procs))
    width = 0.7 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, proc_data) in enumerate(bars.items()):
        vals = [proc_data.get(p, 0.0) for p in all_procs]
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(all_procs, rotation=30, ha="right")
    ax.set_ylabel("SSIM")
    ax.set_title("Per-Procedure SSIM (latest validation)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "procedure_ssim")


# ── Summary report ───────────────────────────────────────────────


def log_summary(runs: list[RunData]) -> None:
    """Log a text summary of all runs and anomalies."""
    for run in runs:
        logger.info("--- %s ---", run.label)
        if run.train:
            losses = [r.loss for r in run.train]
            logger.info(
                "  Steps: %d-%d (%d records)",
                run.train[0].step,
                run.train[-1].step,
                len(run.train),
            )
            finite_losses = [l for l in losses if math.isfinite(l)]
            if finite_losses:
                logger.info(
                    "  Loss: min=%.6f  max=%.6f  final=%.6f",
                    min(finite_losses),
                    max(finite_losses),
                    finite_losses[-1],
                )
            logger.info(
                "  Speed: %.1f it/s (median)",
                float(np.median([r.speed for r in run.train])),
            )
        if run.val:
            latest = run.val[-1]
            logger.info(
                "  Latest val (step %d): SSIM=%.4f+/-%.4f  LPIPS=%.4f+/-%.4f",
                latest.step,
                latest.ssim_mean,
                latest.ssim_std,
                latest.lpips_mean,
                latest.lpips_std,
            )
            if latest.per_procedure:
                parts = [f"{k}={v:.3f}" for k, v in sorted(latest.per_procedure.items())]
                logger.info("  Per-procedure: %s", "  ".join(parts))

        anomalies = detect_anomalies(run)
        if anomalies:
            logger.info("  Anomalies (%d):", len(anomalies))
            for a in anomalies:
                logger.warning("    [%s] %s", a.kind.upper(), a.message)
        else:
            logger.info("  No anomalies detected")


# ── CLI ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze SLURM training logs and generate plots.",
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="One or more SLURM log files to parse",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Auto-find the latest slurm-phaseA-*.out in project root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "plots",
        help="Directory for output PNGs (default: plots/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve log files
    log_paths: list[Path] = list(args.logs) if args.logs else []
    if args.latest or not log_paths:
        if not log_paths:
            log_paths = find_latest_logs()
        else:
            log_paths.extend(find_latest_logs())

    if not log_paths:
        parser.error("No log files specified and none found with --latest")

    # Parse all runs
    runs: list[RunData] = []
    for p in log_paths:
        if not p.exists():
            logger.error("File not found: %s", p)
            continue
        runs.append(parse_log(p))

    if not runs:
        logger.error("No valid log files parsed")
        return

    # Summary
    log_summary(runs)

    # Generate plots
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(runs, args.output_dir)
    plot_lr(runs, args.output_dir)
    plot_grad_norm(runs, args.output_dir)
    plot_val_metrics(runs, args.output_dir)
    plot_procedure_ssim(runs, args.output_dir)

    logger.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
