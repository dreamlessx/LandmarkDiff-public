#!/usr/bin/env python3
"""Local CI validation script for LandmarkDiff.

Runs all quality checks that should pass before pushing code:
1. Import validation (all modules importable)
2. Unit tests (pytest)
3. Code structure checks (no circular imports, no missing __all__)
4. Config validation (default configs are valid)

Usage:
    python scripts/ci_check.py          # run all checks
    python scripts/ci_check.py --quick  # skip slow tests
    python scripts/ci_check.py --tests  # only run tests
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class CIRunner:
    """Run CI checks and report results."""

    def __init__(self):
        self.results: list[dict] = []
        self.start_time = time.time()

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "passed": passed, "detail": detail})
        sym = "PASS" if passed else "FAIL"
        msg = f"  [{sym}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        return passed

    def run_stage(self, name: str, func, *args, **kwargs) -> bool:
        """Run a check stage with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.check(name, False, str(e))
            return False

    def summary(self) -> int:
        elapsed = time.time() - self.start_time
        total = len(self.results)
        passed = sum(r["passed"] for r in self.results)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"CI Results: {passed}/{total} passed, {failed} failed ({elapsed:.1f}s)")
        if failed == 0:
            print("STATUS: ALL CHECKS PASSED")
        else:
            print("STATUS: FAILED")
            for r in self.results:
                if not r["passed"]:
                    print(f"  FAIL: {r['name']} -- {r['detail']}")
        print("=" * 60)
        return 0 if failed == 0 else 1


def check_imports(ci: CIRunner) -> None:
    """Verify all modules are importable."""
    print("\n--- Import Validation ---")

    modules = [
        "landmarkdiff",
        "landmarkdiff.api_client",
        "landmarkdiff.arcface_torch",
        "landmarkdiff.audit",
        "landmarkdiff.augmentation",
        "landmarkdiff.benchmark",
        "landmarkdiff.clinical",
        "landmarkdiff.conditioning",
        "landmarkdiff.config",
        "landmarkdiff.curriculum",
        "landmarkdiff.data",
        "landmarkdiff.data_version",
        "landmarkdiff.displacement_model",
        "landmarkdiff.ensemble",
        "landmarkdiff.evaluation",
        "landmarkdiff.experiment_tracker",
        "landmarkdiff.face_verifier",
        "landmarkdiff.fid",
        "landmarkdiff.hyperparam",
        "landmarkdiff.inference",
        "landmarkdiff.landmarks",
        "landmarkdiff.log",
        "landmarkdiff.losses",
        "landmarkdiff.manipulation",
        "landmarkdiff.masking",
        "landmarkdiff.postprocess",
        "landmarkdiff.checkpoint_manager",
        "landmarkdiff.metrics_agg",
        "landmarkdiff.metrics_viz",
        "landmarkdiff.model_registry",
        "landmarkdiff.safety",
        "landmarkdiff.validation",
        "landmarkdiff.synthetic",
        "landmarkdiff.synthetic.augmentation",
        "landmarkdiff.synthetic.pair_generator",
        "landmarkdiff.synthetic.tps_warp",
    ]

    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            ci.check(f"import_{mod_name.split('.')[-1]}", True)
        except Exception as e:
            ci.check(f"import_{mod_name.split('.')[-1]}", False, str(e)[:80])


def check_version(ci: CIRunner) -> None:
    """Verify version consistency."""
    print("\n--- Version Consistency ---")

    import landmarkdiff

    pkg_version = landmarkdiff.__version__

    # Check pyproject.toml
    import tomllib

    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        toml_version = data.get("project", {}).get("version", "unknown")
        ci.check(
            "version_pyproject",
            pkg_version == toml_version,
            f"pkg={pkg_version}, pyproject={toml_version}",
        )

    # Check model card
    model_card = ROOT / "model_card.yaml"
    if model_card.exists():
        import yaml

        with open(model_card) as f:
            card = yaml.safe_load(f)
        card_version = card.get("version", "unknown")
        ci.check(
            "version_model_card",
            pkg_version == card_version,
            f"pkg={pkg_version}, card={card_version}",
        )


def check_exports(ci: CIRunner) -> None:
    """Verify __all__ exports are correct."""
    print("\n--- Export Validation ---")

    import landmarkdiff

    all_exports = landmarkdiff.__all__

    ci.check("exports_defined", len(all_exports) > 0, f"{len(all_exports)} modules exported")

    # Each exported name should be importable
    for name in all_exports:
        try:
            importlib.import_module(f"landmarkdiff.{name}")
            ci.check(f"export_{name}", True)
        except Exception as e:
            ci.check(f"export_{name}", False, str(e)[:60])


def check_config(ci: CIRunner) -> None:
    """Verify configuration system works."""
    print("\n--- Configuration ---")

    from landmarkdiff.config import load_config, validate_config

    cfg = load_config()
    ci.check("config_loads", True)

    warnings = validate_config(cfg)
    ci.check("config_valid", isinstance(warnings, list), f"{len(warnings)} warnings")

    # Check YAML configs
    import yaml

    config_dir = ROOT / "configs"
    if config_dir.exists():
        for cfg_file in config_dir.glob("*.yaml"):
            try:
                with open(cfg_file) as f:
                    data = yaml.safe_load(f)
                ci.check(f"yaml_{cfg_file.stem}", data is not None)
            except Exception as e:
                ci.check(f"yaml_{cfg_file.stem}", False, str(e)[:60])


def check_tests(ci: CIRunner, quick: bool = False) -> None:
    """Run pytest."""
    print("\n--- Test Suite ---")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests"),
        "-x",  # Stop on first failure
        "--tb=short",
        "-q",
    ]
    if quick:
        cmd.extend(["-k", "not (test_phase_b or test_basic_evaluation or TestArcFace)"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    # Parse output for pass/fail count
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout else ""
    ci.check("tests_pass", result.returncode == 0, last_line[:80])


def check_no_secrets(ci: CIRunner) -> None:
    """Check that no secrets are in tracked files."""
    print("\n--- Security ---")

    # Check for common secret patterns in Python files
    secret_patterns = [
        "password",
        "api_key",
        "secret_key",
        "PRIVATE_KEY",
    ]

    found_secrets = []
    for py_file in (ROOT / "landmarkdiff").rglob("*.py"):
        content = py_file.read_text()
        for pattern in secret_patterns:
            if f'{pattern} = "' in content or f"{pattern} = '" in content:
                found_secrets.append(f"{py_file.name}:{pattern}")

    ci.check(
        "no_hardcoded_secrets",
        len(found_secrets) == 0,
        f"found: {found_secrets}" if found_secrets else "clean",
    )


def main():
    parser = argparse.ArgumentParser(description="LandmarkDiff CI Checks")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--tests", action="store_true", help="Only run tests")
    parser.add_argument("--no-tests", action="store_true", help="Skip tests")
    args = parser.parse_args()

    ci = CIRunner()
    print("=" * 60)
    print("LandmarkDiff CI Validation")
    print("=" * 60)

    if args.tests:
        check_tests(ci, quick=args.quick)
    else:
        check_imports(ci)
        check_version(ci)
        check_exports(ci)
        check_config(ci)
        check_no_secrets(ci)
        if not args.no_tests:
            check_tests(ci, quick=args.quick)

    sys.exit(ci.summary())


if __name__ == "__main__":
    main()
