#!/usr/bin/env python3
"""Comprehensive post-training analysis for LandmarkDiff.

Analyzes training checkpoints, detects convergence issues, recommends
phase transitions, and generates training reports.

Features:
  1. Loss curve analysis with convergence detection
  2. Checkpoint quality comparison on validation set
  3. Phase A → B transition readiness check
  4. Gradient health diagnostics
  5. Training speed and efficiency metrics
  6. Markdown report generation

Usage:
    # Analyze Phase A training
    python scripts/analyze_training_run.py --checkpoint_dir checkpoints_phaseA

    # Compare checkpoints on val set
    python scripts/analyze_training_run.py --checkpoint_dir checkpoints_phaseA --eval_val

    # Check Phase A → B readiness
    python scripts/analyze_training_run.py --checkpoint_dir checkpoints_phaseA --phase_check

    # Full analysis with report
    python scripts/analyze_training_run.py --checkpoint_dir checkpoints_phaseA --report results/phaseA_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class TrainingMetrics:
    """Parsed training metrics from logs."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    val_ssim: list[tuple[int, float]] = field(default_factory=list)
    val_lpips: list[tuple[int, float]] = field(default_factory=list)
    val_nme: list[tuple[int, float]] = field(default_factory=list)
    wall_times: list[float] = field(default_factory=list)
    # Phase B specific
    diff_losses: list[float] = field(default_factory=list)
    id_losses: list[float] = field(default_factory=list)
    perc_losses: list[float] = field(default_factory=list)
    lm_losses: list[float] = field(default_factory=list)


def parse_slurm_log(log_path: str) -> TrainingMetrics:
    """Parse SLURM training log for metrics."""
    metrics = TrainingMetrics()

    step_pattern = re.compile(r"step\s+(\d+).*?loss[=:]\s*([\d.e-]+)", re.IGNORECASE)
    lr_pattern = re.compile(r"lr[=:]\s*([\d.e-]+)", re.IGNORECASE)
    grad_pattern = re.compile(r"grad_norm[=:]\s*([\d.e-]+)", re.IGNORECASE)
    val_pattern = re.compile(r"val.*?(ssim|lpips|nme)[=:]\s*([\d.e-]+)", re.IGNORECASE)
    re.compile(r"([\d.]+)\s*(?:it/s|steps/s|s/step)", re.IGNORECASE)

    # Phase B loss components
    diff_pattern = re.compile(r"diff(?:usion)?_loss[=:]\s*([\d.e-]+)", re.IGNORECASE)
    id_pattern = re.compile(r"id(?:entity)?_loss[=:]\s*([\d.e-]+)", re.IGNORECASE)
    perc_pattern = re.compile(r"perc(?:eptual)?_loss[=:]\s*([\d.e-]+)", re.IGNORECASE)
    lm_pattern = re.compile(r"(?:landmark|lm|nme)_loss[=:]\s*([\d.e-]+)", re.IGNORECASE)

    with open(log_path) as f:
        for line in f:
            # Training step
            m = step_pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                metrics.steps.append(step)
                metrics.losses.append(loss)

                # LR
                lr_m = lr_pattern.search(line)
                if lr_m:
                    metrics.lrs.append(float(lr_m.group(1)))

                # Gradient norm
                grad_m = grad_pattern.search(line)
                if grad_m:
                    metrics.grad_norms.append(float(grad_m.group(1)))

                # Phase B components
                dm = diff_pattern.search(line)
                if dm:
                    metrics.diff_losses.append(float(dm.group(1)))
                im = id_pattern.search(line)
                if im:
                    metrics.id_losses.append(float(im.group(1)))
                pm = perc_pattern.search(line)
                if pm:
                    metrics.perc_losses.append(float(pm.group(1)))
                lm_m = lm_pattern.search(line)
                if lm_m:
                    metrics.lm_losses.append(float(lm_m.group(1)))

            # Validation metrics
            val_m = val_pattern.search(line)
            if val_m:
                metric_name = val_m.group(1).lower()
                metric_val = float(val_m.group(2))
                # Find nearest step
                step = metrics.steps[-1] if metrics.steps else 0
                if metric_name == "ssim":
                    metrics.val_ssim.append((step, metric_val))
                elif metric_name == "lpips":
                    metrics.val_lpips.append((step, metric_val))
                elif metric_name == "nme":
                    metrics.val_nme.append((step, metric_val))

    return metrics


def parse_jsonl_metrics(metrics_path: str) -> TrainingMetrics:
    """Parse experiment tracker JSONL metrics."""
    metrics = TrainingMetrics()

    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            step = entry.get("step", entry.get("global_step", 0))
            metrics.steps.append(step)

            if "loss" in entry:
                metrics.losses.append(entry["loss"])
            if "lr" in entry or "learning_rate" in entry:
                metrics.lrs.append(entry.get("lr", entry.get("learning_rate", 0)))
            if "grad_norm" in entry:
                metrics.grad_norms.append(entry["grad_norm"])
            if "diffusion_loss" in entry:
                metrics.diff_losses.append(entry["diffusion_loss"])
            if "identity_loss" in entry:
                metrics.id_losses.append(entry["identity_loss"])
            if "perceptual_loss" in entry:
                metrics.perc_losses.append(entry["perceptual_loss"])
            if "landmark_loss" in entry:
                metrics.lm_losses.append(entry["landmark_loss"])

            # Validation
            if "val_ssim" in entry:
                metrics.val_ssim.append((step, entry["val_ssim"]))
            if "val_lpips" in entry:
                metrics.val_lpips.append((step, entry["val_lpips"]))

    return metrics


def find_log_files(checkpoint_dir: str) -> list[str]:
    """Find training log files in or near checkpoint directory."""
    ckpt_path = Path(checkpoint_dir)
    logs = []

    # Look for SLURM logs
    for pattern in ["slurm-*.out", "*.log", "training.log"]:
        logs.extend(glob.glob(str(ckpt_path / pattern)))
        logs.extend(glob.glob(str(ckpt_path.parent / pattern)))

    # Look for JSONL metrics
    for pattern in ["*_metrics.jsonl", "metrics.jsonl"]:
        logs.extend(glob.glob(str(ckpt_path / "experiments" / pattern)))
        logs.extend(glob.glob(str(ckpt_path / pattern)))

    return sorted(set(logs))


def find_checkpoints(checkpoint_dir: str) -> list[dict]:
    """Find all checkpoints with their metadata."""
    ckpt_path = Path(checkpoint_dir)
    checkpoints = []

    # Look for checkpoint directories (diffusers format)
    for d in sorted(ckpt_path.glob("checkpoint-*")):
        if d.is_dir():
            step = int(d.name.split("-")[1])
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            checkpoints.append(
                {
                    "path": str(d),
                    "step": step,
                    "size_mb": round(size_mb, 1),
                    "is_ema": False,
                }
            )

    # Look for EMA checkpoints
    for d in sorted(ckpt_path.glob("checkpoint-*-ema")):
        if d.is_dir():
            step = int(d.name.split("-")[1])
            checkpoints.append(
                {
                    "path": str(d),
                    "step": step,
                    "size_mb": round(
                        sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1
                    ),
                    "is_ema": True,
                }
            )

    # PyTorch .pt files
    for f in sorted(ckpt_path.glob("*.pt")):
        step_m = re.search(r"(\d+)", f.stem)
        step = int(step_m.group(1)) if step_m else 0
        checkpoints.append(
            {
                "path": str(f),
                "step": step,
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "is_ema": "ema" in f.stem.lower(),
            }
        )

    return sorted(checkpoints, key=lambda x: x["step"])


# ── Convergence Analysis ──────────────────────────────────────────


def detect_convergence_issues(metrics: TrainingMetrics) -> list[dict]:
    """Detect common training issues from loss curves."""
    issues = []
    losses = np.array(metrics.losses) if metrics.losses else np.array([])

    if len(losses) < 100:
        issues.append(
            {
                "type": "insufficient_data",
                "severity": "info",
                "message": f"Only {len(losses)} loss values — too few for reliable analysis",
            }
        )
        return issues

    # 1. Divergence: loss increasing over last 20% of training
    last_20_pct = losses[int(len(losses) * 0.8) :]
    if len(last_20_pct) > 10:
        slope = np.polyfit(range(len(last_20_pct)), last_20_pct, 1)[0]
        if slope > 0.001:
            issues.append(
                {
                    "type": "divergence",
                    "severity": "critical",
                    "message": f"Loss increasing in final 20% (slope={slope:.4f}). Training may be diverging.",
                    "recommendation": "Reduce learning rate or revert to earlier checkpoint.",
                }
            )

    # 2. Loss plateau: flat loss for >1000 steps
    window = min(1000, len(losses) // 5)
    if window > 50:
        rolling_std = np.array(
            [np.std(losses[max(0, i - window) : i]) for i in range(window, len(losses))]
        )
        if len(rolling_std) > 0 and np.min(rolling_std) < 1e-6:
            plateau_start = np.argmin(rolling_std)
            issues.append(
                {
                    "type": "plateau",
                    "severity": "warning",
                    "message": f"Loss plateaued at step ~{metrics.steps[plateau_start + window]} (std < 1e-6 over {window} steps).",
                    "recommendation": "Consider reducing LR, adding warmup, or transitioning to Phase B.",
                }
            )

    # 3. Loss spikes: sudden increases > 10x median
    median_loss = np.median(losses)
    spike_mask = losses > 10 * median_loss
    n_spikes = int(np.sum(spike_mask))
    if n_spikes > 3:
        spike_steps = [metrics.steps[i] for i in range(len(losses)) if spike_mask[i]]
        issues.append(
            {
                "type": "loss_spikes",
                "severity": "warning",
                "message": f"{n_spikes} loss spikes detected (>10x median={median_loss:.4f}). Steps: {spike_steps[:5]}",
                "recommendation": "Enable gradient clipping or reduce batch size.",
            }
        )

    # 4. Gradient health
    if metrics.grad_norms:
        gnorms = np.array(metrics.grad_norms)
        if np.any(gnorms > 100):
            issues.append(
                {
                    "type": "gradient_explosion",
                    "severity": "critical",
                    "message": f"Gradient norms exceed 100 ({np.sum(gnorms > 100)} times). Max: {np.max(gnorms):.1f}",
                    "recommendation": "Enable gradient clipping (max_grad_norm=1.0).",
                }
            )
        if np.mean(gnorms[-100:] if len(gnorms) > 100 else gnorms) < 1e-6:
            issues.append(
                {
                    "type": "vanishing_gradients",
                    "severity": "critical",
                    "message": "Gradients near zero — model may not be learning.",
                    "recommendation": "Check loss function, increase learning rate, or check frozen parameters.",
                }
            )

    # 5. NaN/Inf detection
    if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
        nan_count = int(np.sum(np.isnan(losses) | np.isinf(losses)))
        issues.append(
            {
                "type": "nan_loss",
                "severity": "critical",
                "message": f"{nan_count} NaN/Inf loss values detected.",
                "recommendation": "Check data pipeline, reduce LR, or disable mixed precision.",
            }
        )

    # 6. Good convergence (no issues)
    if not issues:
        final_loss = np.mean(losses[-100:])
        improvement = (np.mean(losses[:100]) - final_loss) / np.mean(losses[:100]) * 100
        issues.append(
            {
                "type": "healthy",
                "severity": "info",
                "message": f"Training appears healthy. Loss improved {improvement:.1f}% from {np.mean(losses[:100]):.4f} to {final_loss:.4f}.",
            }
        )

    return issues


def check_phase_transition(metrics: TrainingMetrics, min_steps: int = 20000) -> dict:
    """Check if Phase A training is ready for Phase B transition."""
    result = {
        "ready": False,
        "reasons": [],
        "recommendations": [],
    }

    if not metrics.losses:
        result["reasons"].append("No training metrics found")
        return result

    total_steps = metrics.steps[-1] if metrics.steps else 0
    losses = np.array(metrics.losses)

    # Check minimum steps
    if total_steps < min_steps:
        result["reasons"].append(f"Only {total_steps} steps completed (minimum: {min_steps})")
        result["recommendations"].append(
            f"Continue training for {min_steps - total_steps} more steps"
        )
    else:
        result["reasons"].append(f"Sufficient steps: {total_steps} >= {min_steps}")

    # Check convergence: loss should be decreasing slowly
    if len(losses) > 500:
        last_10_pct = losses[int(len(losses) * 0.9) :]
        prev_10_pct = losses[int(len(losses) * 0.8) : int(len(losses) * 0.9)]
        improvement = (np.mean(prev_10_pct) - np.mean(last_10_pct)) / np.mean(prev_10_pct)

        if improvement < 0.01:
            result["reasons"].append(
                f"Loss converged (< 1% improvement in final 10%: {improvement * 100:.2f}%)"
            )
        else:
            result["reasons"].append(
                f"Loss still improving ({improvement * 100:.2f}% in final 10%)"
            )
            result["recommendations"].append("Continue Phase A — model is still learning")

    # Check validation metrics if available
    if metrics.val_ssim:
        best_ssim = max(v for _, v in metrics.val_ssim)
        latest_ssim = metrics.val_ssim[-1][1]
        result["reasons"].append(f"Best val SSIM: {best_ssim:.4f}, Latest: {latest_ssim:.4f}")

    # Check loss stability
    if len(losses) > 200:
        recent_std = np.std(losses[-200:])
        result["reasons"].append(f"Recent loss std: {recent_std:.6f}")
        if recent_std > 0.1:
            result["recommendations"].append("Loss still unstable — not ready for Phase B")

    # Decision
    issues = detect_convergence_issues(metrics)
    critical = [i for i in issues if i["severity"] == "critical"]
    if critical:
        result["reasons"].append(f"{len(critical)} critical issue(s) detected")
        result["recommendations"].append("Fix critical issues before Phase B")
    elif total_steps >= min_steps and (len(losses) < 500 or improvement < 0.02):
        result["ready"] = True
        result["recommendations"].append(
            "Phase A converged. Start Phase B with: python scripts/train_controlnet.py --config configs/phaseB_production.yaml"
        )

    return result


# ── Report Generation ─────────────────────────────────────────────


def generate_report(
    checkpoint_dir: str,
    metrics: TrainingMetrics,
    checkpoints: list[dict],
    issues: list[dict],
    phase_check: dict | None = None,
) -> str:
    """Generate a Markdown training analysis report."""
    lines = [
        "# Training Run Analysis",
        f"\nCheckpoint directory: `{checkpoint_dir}`",
        f"Total steps: {metrics.steps[-1] if metrics.steps else 0}",
        f"Data points: {len(metrics.losses)}",
    ]

    # Loss summary
    if metrics.losses:
        losses = np.array(metrics.losses)
        lines.append("\n## Loss Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Initial (first 100) | {np.mean(losses[:100]):.6f} |")
        lines.append(f"| Final (last 100) | {np.mean(losses[-100:]):.6f} |")
        lines.append(
            f"| Best | {np.min(losses):.6f} (step {metrics.steps[int(np.argmin(losses))]}) |"
        )
        lines.append(
            f"| Improvement | {(np.mean(losses[:100]) - np.mean(losses[-100:])) / np.mean(losses[:100]) * 100:.1f}% |"
        )

        if metrics.grad_norms:
            gnorms = np.array(metrics.grad_norms)
            lines.append(f"| Grad norm (mean) | {np.mean(gnorms):.4f} |")
            lines.append(f"| Grad norm (max) | {np.max(gnorms):.4f} |")

    # Phase B components
    if metrics.diff_losses:
        lines.append("\n## Phase B Loss Components")
        lines.append("| Component | Mean | Final |")
        lines.append("|-----------|------|-------|")
        for name, vals in [
            ("Diffusion", metrics.diff_losses),
            ("Identity", metrics.id_losses),
            ("Perceptual", metrics.perc_losses),
            ("Landmark", metrics.lm_losses),
        ]:
            if vals:
                lines.append(f"| {name} | {np.mean(vals):.6f} | {np.mean(vals[-10:]):.6f} |")

    # Validation
    if metrics.val_ssim:
        lines.append("\n## Validation Metrics")
        lines.append("| Step | SSIM | LPIPS | NME |")
        lines.append("|------|------|-------|-----|")
        for i, (step, ssim) in enumerate(metrics.val_ssim):
            lpips = metrics.val_lpips[i][1] if i < len(metrics.val_lpips) else "-"
            nme = metrics.val_nme[i][1] if i < len(metrics.val_nme) else "-"
            lpips_s = f"{lpips:.4f}" if isinstance(lpips, float) else lpips
            nme_s = f"{nme:.4f}" if isinstance(nme, float) else nme
            lines.append(f"| {step} | {ssim:.4f} | {lpips_s} | {nme_s} |")

    # Checkpoints
    if checkpoints:
        lines.append("\n## Checkpoints")
        lines.append("| Step | Size (MB) | EMA | Path |")
        lines.append("|------|-----------|-----|------|")
        for ckpt in checkpoints:
            ema = "yes" if ckpt["is_ema"] else "no"
            lines.append(f"| {ckpt['step']} | {ckpt['size_mb']} | {ema} | `{ckpt['path']}` |")

    # Issues
    lines.append("\n## Diagnostics")
    for issue in issues:
        icon = {"critical": "X", "warning": "!", "info": "i"}[issue["severity"]]
        lines.append(f"- [{icon}] **{issue['type']}**: {issue['message']}")
        if "recommendation" in issue:
            lines.append(f"  - Recommendation: {issue['recommendation']}")

    # Phase transition
    if phase_check:
        lines.append("\n## Phase Transition Check")
        status = "READY" if phase_check["ready"] else "NOT READY"
        lines.append(f"\nStatus: **{status}**\n")
        for reason in phase_check["reasons"]:
            lines.append(f"- {reason}")
        if phase_check["recommendations"]:
            lines.append("\nRecommendations:")
            for rec in phase_check["recommendations"]:
                lines.append(f"- {rec}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze training run")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory to analyze")
    parser.add_argument(
        "--log", default=None, help="Specific log file to parse (auto-detected if omitted)"
    )
    parser.add_argument(
        "--phase_check", action="store_true", help="Check Phase A → B transition readiness"
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=20000,
        help="Minimum steps for phase transition (default: 20000)",
    )
    parser.add_argument("--report", default=None, help="Output report path (Markdown)")
    parser.add_argument(
        "--eval_val", action="store_true", help="Evaluate checkpoints on validation set"
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    print(f"Analyzing: {ckpt_dir}")

    # Find checkpoints
    checkpoints = find_checkpoints(ckpt_dir)
    print(f"Found {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        ema_tag = " (EMA)" if ckpt["is_ema"] else ""
        print(f"  step {ckpt['step']:>6}{ema_tag}: {ckpt['size_mb']} MB")

    # Parse metrics
    metrics = TrainingMetrics()
    log_files = [args.log] if args.log else find_log_files(ckpt_dir)

    for log_file in log_files:
        print(f"\nParsing: {log_file}")
        if log_file.endswith(".jsonl"):
            m = parse_jsonl_metrics(log_file)
        else:
            m = parse_slurm_log(log_file)

        # Merge
        metrics.steps.extend(m.steps)
        metrics.losses.extend(m.losses)
        metrics.lrs.extend(m.lrs)
        metrics.grad_norms.extend(m.grad_norms)
        metrics.val_ssim.extend(m.val_ssim)
        metrics.val_lpips.extend(m.val_lpips)
        metrics.val_nme.extend(m.val_nme)
        metrics.diff_losses.extend(m.diff_losses)
        metrics.id_losses.extend(m.id_losses)
        metrics.perc_losses.extend(m.perc_losses)
        metrics.lm_losses.extend(m.lm_losses)

    print(f"\nTotal data points: {len(metrics.losses)}")

    # Convergence analysis
    issues = detect_convergence_issues(metrics)
    print(f"\n{'=' * 60}")
    print("DIAGNOSTICS")
    print(f"{'=' * 60}")
    for issue in issues:
        severity_colors = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
        print(f"  [{severity_colors[issue['severity']]}] {issue['type']}: {issue['message']}")
        if "recommendation" in issue:
            print(f"    → {issue['recommendation']}")

    # Phase transition check
    phase_check = None
    if args.phase_check:
        print(f"\n{'=' * 60}")
        print("PHASE TRANSITION CHECK")
        print(f"{'=' * 60}")
        phase_check = check_phase_transition(metrics, args.min_steps)
        status = "READY" if phase_check["ready"] else "NOT READY"
        print(f"  Status: {status}")
        for reason in phase_check["reasons"]:
            print(f"  - {reason}")
        for rec in phase_check["recommendations"]:
            print(f"  → {rec}")

    # Generate report
    if args.report:
        report = generate_report(ckpt_dir, metrics, checkpoints, issues, phase_check)
        out_path = Path(args.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
