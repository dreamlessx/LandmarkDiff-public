"""Generate a comprehensive project status report.

Summarizes dataset state, training progress, model performance,
and next steps. Useful for quick project overview.

Usage:
    python scripts/status_report.py
    python scripts/status_report.py --json  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK_DIR = Path(os.environ.get("LANDMARKDIFF_ROOT", str(PROJECT_ROOT)))


def count_files(directory: Path, pattern: str = "*_input.png") -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_dir_size_mb(directory: Path) -> float:
    """Get approximate directory size in MB using du."""
    if not directory.exists():
        return 0.0
    import subprocess

    try:
        result = subprocess.run(
            ["du", "-sm", str(directory)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.split()[0])
    except Exception:
        pass
    return 0.0


def dataset_status() -> dict:
    """Gather dataset status."""
    data_dir = WORK_DIR / "data"
    status = {}

    # Wave 1: Original synthetic
    w1_dir = data_dir / "synthetic_surgery_pairs"
    status["wave1"] = {
        "name": "Synthetic (RBF preset)",
        "pairs": count_files(w1_dir),
        "size_mb": round(get_dir_size_mb(w1_dir), 1),
        "exists": w1_dir.exists(),
    }

    # Wave 2: Scaled synthetic
    w2_dir = data_dir / "synthetic_surgery_pairs_v2"
    status["wave2"] = {
        "name": "Synthetic (scaled RBF)",
        "pairs": count_files(w2_dir),
        "size_mb": round(get_dir_size_mb(w2_dir), 1),
        "exists": w2_dir.exists(),
    }

    # Wave 3: Realistic displacement
    w3_dir = data_dir / "synthetic_surgery_pairs_v3"
    status["wave3"] = {
        "name": "Realistic (displacement model)",
        "pairs": count_files(w3_dir),
        "size_mb": round(get_dir_size_mb(w3_dir), 1),
        "exists": w3_dir.exists(),
    }

    # Real pairs
    real_dir = data_dir / "real_surgery_pairs" / "pairs"
    status["real"] = {
        "name": "Real surgery pairs",
        "pairs": count_files(real_dir),
        "size_mb": round(
            get_dir_size_mb(real_dir if real_dir.exists() else data_dir / "real_surgery_pairs"), 1
        ),
        "exists": real_dir.exists(),
    }

    # Combined training set
    combined_dir = data_dir / "training_combined"
    status["combined"] = {
        "name": "Combined training set",
        "pairs": count_files(combined_dir),
        "size_mb": round(get_dir_size_mb(combined_dir), 1),
        "exists": combined_dir.exists(),
    }

    # Test set
    test_dir = data_dir / "test_pairs"
    status["test"] = {
        "name": "Test set",
        "pairs": count_files(test_dir),
        "exists": test_dir.exists(),
    }

    # Displacement model
    dm_path = data_dir / "displacement_model.npz"
    status["displacement_model"] = {
        "exists": dm_path.exists(),
        "size_mb": round(dm_path.stat().st_size / 1e6, 1) if dm_path.exists() else 0,
    }

    # CelebA source
    celeba_dir = data_dir / "celeba_hq_extracted"
    status["celeba_source"] = {
        "images": len(list(celeba_dir.glob("*.jpg")) + list(celeba_dir.glob("*.png")))
        if celeba_dir.exists()
        else 0,
        "exists": celeba_dir.exists(),
    }

    return status


def training_status() -> dict:
    """Gather training checkpoint status."""
    status = {}

    for phase_name, ckpt_dir_name in [
        ("phaseA", "checkpoints_phaseA"),
        ("phaseB", "checkpoints_phaseB"),
    ]:
        ckpt_dir = WORK_DIR / ckpt_dir_name
        phase_info = {"exists": ckpt_dir.exists(), "checkpoints": []}

        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
            for ckpt in ckpts:
                state_file = ckpt / "training_state.pt"
                ema_dir = ckpt / "controlnet_ema"
                phase_info["checkpoints"].append(
                    {
                        "name": ckpt.name,
                        "has_state": state_file.exists(),
                        "has_ema": ema_dir.exists(),
                    }
                )

            final = ckpt_dir / "final"
            phase_info["final_exists"] = final.exists()
            if final.exists():
                phase_info["final_has_ema"] = (final / "controlnet_ema").exists()

            # Check experiment tracker
            exp_dir = ckpt_dir / "experiments"
            if exp_dir.exists():
                try:
                    from landmarkdiff.experiment_tracker import ExperimentTracker

                    tracker = ExperimentTracker(str(exp_dir))
                    exps = tracker.list_experiments()
                    phase_info["experiments"] = len(exps)
                    if exps:
                        phase_info["latest_experiment"] = exps[-1]
                except Exception:
                    pass

        status[phase_name] = phase_info

    return status


def model_cache_status() -> dict:
    """Check HuggingFace model cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return {
        "sd15_cached": len(list(cache_dir.glob("models--runwayml--stable-diffusion-v1-5*"))) > 0,
        "controlnet_cached": len(
            list(cache_dir.glob("models--CrucibleAI--ControlNetMediaPipeFace*"))
        )
        > 0,
        "cache_size_gb": round(get_dir_size_mb(cache_dir) / 1000, 1) if cache_dir.exists() else 0,
    }


def slurm_status() -> list[dict]:
    """Get active SLURM jobs."""
    import subprocess

    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--format=%i %j %t %M", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        jobs = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[1].startswith("surgery_"):
                jobs.append(
                    {
                        "id": parts[0],
                        "name": parts[1],
                        "state": parts[2],
                        "time": parts[3],
                    }
                )
        return jobs
    except Exception:
        return []


def code_stats() -> dict:
    """Count lines of code and tests."""
    src_lines = 0
    for f in (WORK_DIR / "landmarkdiff").rglob("*.py"):
        try:
            src_lines += sum(1 for _ in open(f))
        except Exception:
            pass

    test_lines = 0
    test_files = 0
    for f in (WORK_DIR / "tests").rglob("*.py"):
        try:
            test_lines += sum(1 for _ in open(f))
            test_files += 1
        except Exception:
            pass

    script_count = len(list((WORK_DIR / "scripts").glob("*.py")))

    return {
        "source_lines": src_lines,
        "test_lines": test_lines,
        "test_files": test_files,
        "script_count": script_count,
    }


def print_report(as_json: bool = False) -> None:
    """Print comprehensive status report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "datasets": dataset_status(),
        "training": training_status(),
        "model_cache": model_cache_status(),
        "slurm_jobs": slurm_status(),
        "code": code_stats(),
    }

    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return

    print("=" * 70)
    print(f"  LandmarkDiff Project Status: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Datasets
    print("\n--- Datasets ---")
    ds = report["datasets"]
    total_pairs = 0
    for key in ["wave1", "wave2", "wave3", "real"]:
        d = ds[key]
        icon = "OK" if d["exists"] and d["pairs"] > 0 else "--"
        print(f"  [{icon}] {d['name']}: {d['pairs']:,} pairs ({d['size_mb']:.0f} MB)")
        total_pairs += d["pairs"]

    combined = ds["combined"]
    print(
        f"  [{'OK' if combined['exists'] else '--'}] {combined['name']}: "
        f"{combined['pairs']:,} pairs ({combined['size_mb']:.0f} MB)"
    )
    print(f"  [{'OK' if ds['test']['exists'] else '--'}] Test set: {ds['test']['pairs']:,} pairs")
    print(
        f"  [{'OK' if ds['displacement_model']['exists'] else '--'}] Displacement model: "
        f"{ds['displacement_model']['size_mb']:.1f} MB"
    )
    print(
        f"  [{'OK' if ds['celeba_source']['exists'] else '--'}] CelebA source: "
        f"{ds['celeba_source']['images']:,} images"
    )

    # Training
    print("\n--- Training ---")
    for phase_key in ["phaseA", "phaseB"]:
        phase = report["training"][phase_key]
        label = "Phase A" if phase_key == "phaseA" else "Phase B"
        if not phase["exists"]:
            print(f"  [{label}] Not started")
        else:
            n_ckpts = len(phase["checkpoints"])
            has_final = phase.get("final_exists", False)
            status_str = "COMPLETE" if has_final else f"IN PROGRESS ({n_ckpts} checkpoints)"
            print(f"  [{label}] {status_str}")
            if phase.get("latest_experiment"):
                exp = phase["latest_experiment"]
                print(f"          Experiment: {exp.get('name', '?')} ({exp.get('status', '?')})")

    # Model cache
    print("\n--- Model Cache ---")
    mc = report["model_cache"]
    print(f"  [{'OK' if mc['sd15_cached'] else 'MISS'}] SD1.5 weights")
    print(f"  [{'OK' if mc['controlnet_cached'] else 'MISS'}] ControlNet weights")
    print(f"  Cache size: {mc['cache_size_gb']:.1f} GB")

    # SLURM
    print("\n--- SLURM Jobs ---")
    jobs = report["slurm_jobs"]
    if jobs:
        for j in jobs:
            print(f"  {j['id']} | {j['name']} | {j['state']} | {j['time']}")
    else:
        print("  No active surgery_ jobs")

    # Code
    print("\n--- Codebase ---")
    code = report["code"]
    print(f"  Source: {code['source_lines']:,} lines")
    print(f"  Tests: {code['test_lines']:,} lines ({code['test_files']} files)")
    print(f"  Scripts: {code['script_count']} files")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project status report")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    print_report(as_json=args.json)
