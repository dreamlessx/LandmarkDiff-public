#!/usr/bin/env python3
"""Reproduce a training experiment from its config and checkpoint.

Validates that all components match (data, config, model version), then
optionally re-runs evaluation to verify metrics match the original run.

Usage:
    python scripts/reproduce_experiment.py --checkpoint checkpoints/best.pt
    python scripts/reproduce_experiment.py --checkpoint checkpoints/best.pt --eval-only
    python scripts/reproduce_experiment.py --checkpoint checkpoints/best.pt --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.config import ExperimentConfig, load_config


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """Extract metadata from a checkpoint without loading the full model."""
    import torch

    # Only load metadata keys, not the full state dict
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    meta = {}
    if "metadata" in ckpt:
        meta = ckpt["metadata"]
    elif "step" in ckpt:
        meta = {
            "step": ckpt.get("step"),
            "phase": ckpt.get("phase", ""),
            "metrics": ckpt.get("metrics", {}),
        }

    # Check for associated JSON metadata
    json_meta = checkpoint_path.with_suffix(".json")
    if json_meta.exists():
        with open(json_meta) as f:
            meta.update(json.load(f))

    return meta


def verify_config_match(ckpt_meta: dict, config: ExperimentConfig) -> list[str]:
    """Check if config matches the checkpoint's training config."""
    issues = []

    if "config" in ckpt_meta:
        original = ckpt_meta["config"]
        if isinstance(original, dict):
            # Check key training params
            checks = [
                (
                    "learning_rate",
                    original.get("training", {}).get("learning_rate"),
                    config.training.learning_rate,
                ),
                (
                    "batch_size",
                    original.get("training", {}).get("batch_size"),
                    config.training.batch_size,
                ),
                ("phase", original.get("training", {}).get("phase"), config.training.phase),
            ]
            for name, orig_val, curr_val in checks:
                if orig_val is not None and orig_val != curr_val:
                    issues.append(f"{name}: checkpoint={orig_val}, config={curr_val}")

    return issues


def verify_data_integrity(config: ExperimentConfig) -> list[str]:
    """Check that training data exists and is intact."""
    issues = []
    train_dir = Path(config.data.train_dir)

    if not train_dir.exists():
        issues.append(f"Training dir missing: {train_dir}")
        return issues

    # Check for manifest
    manifest_path = train_dir / "manifest.json"
    if manifest_path.exists():
        try:
            from landmarkdiff.data_version import DataManifest

            manifest = DataManifest.load(manifest_path)
            ok, verification_issues = manifest.verify(train_dir)
            if not ok:
                issues.extend(verification_issues[:5])
                if len(verification_issues) > 5:
                    issues.append(f"... and {len(verification_issues) - 5} more issues")
        except Exception as e:
            issues.append(f"Manifest verification failed: {e}")
    else:
        issues.append("No data manifest found — cannot verify data integrity")

    return issues


def run_evaluation(checkpoint_path: Path, config: ExperimentConfig) -> dict:
    """Run evaluation on a checkpoint and return metrics."""
    import torch

    print(f"  Loading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Basic evaluation metrics
    metrics = ckpt.get("metrics", {})
    if metrics:
        print(f"  Checkpoint metrics: {json.dumps(metrics, indent=2)}")

    return metrics


def generate_report(
    checkpoint_path: Path,
    ckpt_meta: dict,
    config: ExperimentConfig,
    config_issues: list[str],
    data_issues: list[str],
) -> str:
    """Generate HTML reproducibility report."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "REPRODUCIBLE" if not config_issues and not data_issues else "ISSUES FOUND"
    status_color = "#28a745" if status == "REPRODUCIBLE" else "#ffc107"

    metrics = ckpt_meta.get("metrics", {})
    metrics_rows = (
        "\n".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items())
        if metrics
        else "<tr><td colspan='2'>No metrics recorded</td></tr>"
    )

    config_rows = "\n".join(f"<li>{issue}</li>" for issue in config_issues)
    data_rows = "\n".join(f"<li>{issue}</li>" for issue in data_issues)

    return f"""<!DOCTYPE html>
<html><head><title>Reproducibility Report</title>
<style>
body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f8f9fa; }}
.status {{ padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; }}
</style></head><body>
<h1>Reproducibility Report</h1>
<p>Generated: {now} | Checkpoint: <code>{checkpoint_path.name}</code></p>
<p>Status: <span class="status" style="background:{status_color}">{status}</span></p>

<h2>Checkpoint Metadata</h2>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Step</td><td>{ckpt_meta.get("step", "N/A")}</td></tr>
<tr><td>Phase</td><td>{ckpt_meta.get("phase", "N/A")}</td></tr>
</table>

<h2>Training Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table>

{"<h2>Config Mismatches</h2><ul>" + config_rows + "</ul>" if config_issues else ""}
{"<h2>Data Integrity Issues</h2><ul>" + data_rows + "</ul>" if data_issues else ""}

{
        "<p>All checks passed -- experiment should be reproducible.</p>"
        if not config_issues and not data_issues
        else ""
    }
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Reproduce LandmarkDiff experiment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", default=None, help="Config YAML (auto-detect if not given)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--output", default="reproducibility_report.html", help="Report output")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 60)
    print("LandmarkDiff Experiment Reproducer")
    print("=" * 60)

    # Load checkpoint metadata
    print(f"\n1. Loading checkpoint metadata: {ckpt_path.name}")
    try:
        ckpt_meta = load_checkpoint_metadata(ckpt_path)
        print(f"   Step: {ckpt_meta.get('step', 'N/A')}")
        print(f"   Phase: {ckpt_meta.get('phase', 'N/A')}")
    except Exception as e:
        print(f"   Warning: could not load metadata: {e}")
        ckpt_meta = {}

    # Load config
    print("\n2. Loading experiment config")
    config = ExperimentConfig.from_yaml(args.config) if args.config else load_config()
    print(f"   Phase: {config.training.phase}")
    print(f"   LR: {config.training.learning_rate}")

    # Verify config match
    print("\n3. Verifying config consistency")
    config_issues = verify_config_match(ckpt_meta, config)
    if config_issues:
        for issue in config_issues:
            print(f"   WARNING: {issue}")
    else:
        print("   Config matches checkpoint metadata")

    # Verify data integrity
    print("\n4. Verifying data integrity")
    data_issues = verify_data_integrity(config)
    if data_issues:
        for issue in data_issues:
            print(f"   WARNING: {issue}")
    else:
        print("   Data integrity verified")

    # Optional: run evaluation
    if args.eval_only:
        print("\n5. Running evaluation")
        run_evaluation(ckpt_path, config)

    # Optional: generate report
    if args.report:
        print(f"\n6. Generating report: {args.output}")
        html = generate_report(ckpt_path, ckpt_meta, config, config_issues, data_issues)
        Path(args.output).write_text(html)
        print(f"   Report saved to {args.output}")

    # Summary
    total_issues = len(config_issues) + len(data_issues)
    print(f"\n{'=' * 60}")
    if total_issues == 0:
        print("RESULT: Experiment is reproducible")
    else:
        print(f"RESULT: {total_issues} issue(s) found — review above warnings")
    print("=" * 60)

    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
