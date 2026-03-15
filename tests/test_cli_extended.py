"""Extended CLI tests covering __main__.py entry point, argument parsing, and error handling.

Tests the module-level CLI (python -m landmarkdiff) and the unified CLI
(landmarkdiff.cli.main) for argument validation, subcommand structure,
and graceful error handling.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from landmarkdiff import __version__

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestMainEntryPoint:
    """Tests for python -m landmarkdiff (__main__.py)."""

    def test_help_flag(self):
        """--help prints usage info and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "landmarkdiff" in result.stdout
        assert "infer" in result.stdout

    def test_version_flag(self):
        """--version prints version string and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_no_args_prints_help(self):
        """No arguments prints help (no crash)."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "landmarkdiff" in result.stdout

    def test_infer_subcommand_help(self):
        """infer --help shows procedure choices."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "infer", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "rhinoplasty" in result.stdout
        assert "blepharoplasty" in result.stdout

    def test_infer_missing_image_errors(self):
        """infer without image argument produces error exit."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "infer"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_landmarks_subcommand_help(self):
        """landmarks --help shows options."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "landmarks", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "image" in result.stdout

    def test_infer_nonexistent_image(self, tmp_path):
        """infer with nonexistent image file should fail gracefully."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "landmarkdiff",
                "infer",
                str(tmp_path / "does_not_exist.png"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_infer_all_mode_choices(self):
        """Parser accepts all valid mode choices."""
        for mode in ["tps", "controlnet", "img2img", "controlnet_ip"]:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "landmarkdiff",
                    "infer",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0
            assert mode in result.stdout

    def test_infer_all_procedure_choices(self):
        """Parser lists all 6 procedures."""
        result = subprocess.run(
            [sys.executable, "-m", "landmarkdiff", "infer", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for proc in [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ]:
            assert proc in result.stdout, f"Missing procedure {proc} in help output"


class TestUnifiedCLIExtended:
    """Extended tests for landmarkdiff.cli.main."""

    def test_version_contains_semver(self, capsys):
        from landmarkdiff.cli import main

        main(["version"])
        out = capsys.readouterr().out
        parts = out.strip().split("v")[-1].split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit(), f"Non-numeric semver part: {part}"

    def test_config_default_has_required_keys(self, capsys):
        from landmarkdiff.cli import main

        main(["config"])
        out = capsys.readouterr().out
        # All experiment configs should have these
        for key in ["experiment_name", "training"]:
            assert key in out

    def test_config_validate_empty_config(self, tmp_path, capsys):
        """Validating a minimal config file should not crash."""
        from landmarkdiff.cli import main

        cfg = tmp_path / "empty.yaml"
        cfg.write_text("experiment_name: test\nversion: '0.3.0'\n")
        main(["config", "--file", str(cfg), "--validate"])
        out = capsys.readouterr().out
        assert "valid" in out.lower() or "warning" in out.lower()

    def test_infer_missing_required_arg(self):
        """infer without image should exit with non-zero."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["infer"])
        assert exc.value.code != 0

    def test_ensemble_missing_required_arg(self):
        """ensemble without image should exit with non-zero."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["ensemble"])
        assert exc.value.code != 0

    def test_evaluate_missing_test_dir(self):
        """evaluate without --test-dir should exit with non-zero."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["evaluate"])
        assert exc.value.code != 0

    def test_validate_missing_positional(self):
        """validate without positional args should exit with non-zero."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["validate"])
        assert exc.value.code != 0

    def test_unknown_subcommand(self):
        """Unknown subcommand should exit 1."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["totally_fake_command"])
        assert exc.value.code != 0

    def test_no_subcommand_exits(self):
        """No subcommand should exit 1."""
        from landmarkdiff.cli import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1


class TestMainModuleDirectImport:
    """Tests for importing and calling __main__.main directly."""

    def test_main_callable(self):
        """__main__.main is a callable function."""
        from landmarkdiff.__main__ import main

        assert callable(main)

    def test_main_version_output(self, capsys):
        """__main__.main --version prints version."""
        from landmarkdiff.__main__ import main

        with patch("sys.argv", ["landmarkdiff", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert __version__ in out

    def test_main_no_args_shows_help(self, capsys):
        """__main__.main with no args shows help without crashing."""
        from landmarkdiff.__main__ import main

        with patch("sys.argv", ["landmarkdiff"]):
            main()
        out = capsys.readouterr().out
        assert "landmarkdiff" in out

    def test_infer_invalid_procedure(self):
        """Invalid procedure choice should cause argparse error."""
        from landmarkdiff.__main__ import main

        with patch(
            "sys.argv",
            ["landmarkdiff", "infer", "test.png", "--procedure", "invalid_proc"],
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code != 0

    def test_infer_invalid_mode(self):
        """Invalid mode choice should cause argparse error."""
        from landmarkdiff.__main__ import main

        with patch(
            "sys.argv",
            ["landmarkdiff", "infer", "test.png", "--mode", "invalid_mode"],
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code != 0
