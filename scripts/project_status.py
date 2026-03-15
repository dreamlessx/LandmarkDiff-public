"""Project status dashboard for LandmarkDiff.

Shows a comprehensive overview of the project state including:
- Codebase statistics (modules, tests, scripts, LOC)
- Test suite health
- Data pipeline progress
- Training status
- Git history summary

Usage:
    python scripts/project_status.py
    python scripts/project_status.py --json  # machine-readable output
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def count_lines(pattern: str, base_dir: Path = ROOT) -> int:
    """Count total lines of Python code matching pattern."""
    total = 0
    for f in base_dir.rglob(pattern):
        if ".git" in str(f) or "__pycache__" in str(f):
            continue
        try:
            total += sum(1 for _ in open(f))
        except Exception:
            pass
    return total


def get_codebase_stats() -> dict:
    """Get codebase statistics."""
    modules_dir = ROOT / "landmarkdiff"
    tests_dir = ROOT / "tests"
    scripts_dir = ROOT / "scripts"
    configs_dir = ROOT / "configs"

    module_files = list(modules_dir.glob("*.py"))
    test_files = list(tests_dir.glob("test_*.py"))
    script_files = list(scripts_dir.glob("*.py"))
    config_files = list(configs_dir.glob("*.yaml"))

    # Count synthetic submodule
    synthetic_files = list((modules_dir / "synthetic").glob("*.py"))

    return {
        "modules": len(module_files) + len(synthetic_files),
        "test_files": len(test_files),
        "scripts": len(script_files),
        "configs": len(config_files),
        "module_loc": count_lines("*.py", modules_dir),
        "test_loc": count_lines("*.py", tests_dir),
        "script_loc": count_lines("*.py", scripts_dir),
    }


def get_data_stats() -> dict:
    """Get training data statistics."""
    data_dir = ROOT / "data"
    stats = {}

    for name in [
        "celeba_hq_extracted",
        "synthetic_surgery_pairs",
        "synthetic_surgery_pairs_v2",
        "synthetic_surgery_pairs_v3",
        "training_combined",
    ]:
        path = data_dir / name
        if path.exists() and path.is_dir():
            n_pairs = len(list(path.glob("*_input.png")))
            stats[name] = n_pairs
        else:
            stats[name] = 0

    # Displacement model
    dm_path = data_dir / "displacement_model.npz"
    stats["displacement_model"] = dm_path.exists()

    return stats


def get_git_stats() -> dict:
    """Get git repository statistics."""
    try:
        # Total commits
        result = subprocess.run(
            ["git", "log", "--oneline"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        total_commits = len(result.stdout.strip().split("\n"))

        # Last commit
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H %s"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        last_commit = result.stdout.strip()

        # Branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        branch = result.stdout.strip()

        return {
            "total_commits": total_commits,
            "last_commit": last_commit,
            "branch": branch,
        }
    except Exception:
        return {"error": "git not available"}


def get_version() -> str:
    """Get package version."""
    try:
        import landmarkdiff

        return landmarkdiff.__version__
    except Exception:
        return "unknown"


def format_dashboard(
    version: str,
    code_stats: dict,
    data_stats: dict,
    git_stats: dict,
) -> str:
    """Format the status dashboard."""
    total_loc = code_stats["module_loc"] + code_stats["test_loc"] + code_stats["script_loc"]
    total_data = sum(v for k, v in data_stats.items() if isinstance(v, int))

    lines = [
        "=" * 60,
        f"  LandmarkDiff v{version} — Project Status",
        "=" * 60,
        "",
        "--- Codebase ---",
        f"  Modules:      {code_stats['modules']:>5}   ({code_stats['module_loc']:,} LOC)",
        f"  Test files:   {code_stats['test_files']:>5}   ({code_stats['test_loc']:,} LOC)",
        f"  Scripts:      {code_stats['scripts']:>5}   ({code_stats['script_loc']:,} LOC)",
        f"  Configs:      {code_stats['configs']:>5}",
        f"  Total LOC:    {total_loc:>5,}",
        "",
        "--- Training Data ---",
    ]

    for name, count in data_stats.items():
        if name == "displacement_model":
            status = "found" if count else "missing"
            lines.append(f"  {name:<30s}: {status}")
        else:
            lines.append(f"  {name:<30s}: {count:>6,} pairs")

    lines.append(f"  {'Total':.<30s}: {total_data:>6,} pairs")

    lines.extend(
        [
            "",
            "--- Git ---",
            f"  Branch:       {git_stats.get('branch', '?')}",
            f"  Commits:      {git_stats.get('total_commits', '?')}",
            f"  Last:         {git_stats.get('last_commit', '?')[:60]}",
        ]
    )

    lines.extend(
        [
            "",
            "--- Pipeline Stages ---",
            "  1. Landmark Extraction    [MediaPipe 478]",
            "  2. Landmark Manipulation  [RBF displacement]",
            "  3. Conditioning Generation [3-ch mesh render]",
            "  4. Surgical Mask          [per-procedure]",
            "  5. ControlNet Inference   [SD1.5 + ControlNet]",
            "  6. Post-processing        [CodeFormer + blend]",
            "  7. Safety Validation      [identity + bounds]",
            "",
            "--- Supported Procedures ---",
            "  rhinoplasty | blepharoplasty | rhytidectomy"
            " | orthognathic | brow_lift | mentoplasty",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LandmarkDiff project status")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    version = get_version()
    code_stats = get_codebase_stats()
    data_stats = get_data_stats()
    git_stats = get_git_stats()

    if args.json:
        output = {
            "version": version,
            "codebase": code_stats,
            "data": data_stats,
            "git": git_stats,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_dashboard(version, code_stats, data_stats, git_stats))


if __name__ == "__main__":
    main()
