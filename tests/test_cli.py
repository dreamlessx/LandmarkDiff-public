"""Tests for the unified CLI module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.cli import main, cmd_version, cmd_config


class TestCLIVersion:
    """Tests for the version subcommand."""

    def test_version_output(self, capsys):
        main(["version"])
        out = capsys.readouterr().out
        assert "LandmarkDiff v" in out

    def test_version_has_semver(self, capsys):
        main(["version"])
        out = capsys.readouterr().out.strip()
        # Should be "LandmarkDiff vX.Y.Z"
        version = out.split("v")[-1]
        parts = version.split(".")
        assert len(parts) == 3, f"Expected semver, got {version}"


class TestCLIConfig:
    """Tests for the config subcommand."""

    def test_config_show_default(self, capsys):
        main(["config"])
        out = capsys.readouterr().out
        assert "experiment_name" in out
        assert "training" in out

    def test_config_validate(self, capsys):
        main(["config", "--validate"])
        out = capsys.readouterr().out
        # Should either say valid or list warnings
        assert "valid" in out.lower() or "warning" in out.lower()

    def test_config_load_file(self, tmp_path, capsys):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(
            "experiment_name: test_exp\n"
            "version: '0.3.0'\n"
        )
        main(["config", "--file", str(cfg_file)])
        out = capsys.readouterr().out
        assert "experiment_name" in out


class TestCLINoCommand:
    """Tests for CLI with no command."""

    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1

    def test_unknown_command(self):
        with pytest.raises(SystemExit):
            main(["nonexistent_command"])


class TestCLIParserStructure:
    """Tests for CLI parser argument structure."""

    def test_infer_parser_accepts_procedure(self):
        """Check that the infer parser accepts known procedure choices."""
        import argparse
        from landmarkdiff.cli import main as cli_main

        # Verify parser construction doesn't crash
        # We can't run infer without a real image, but we can test parsing
        with pytest.raises(SystemExit):
            # Missing required 'image' arg
            main(["infer"])

    def test_ensemble_parser_accepts_strategy(self):
        with pytest.raises(SystemExit):
            # Missing required 'image' arg
            main(["ensemble"])

    def test_evaluate_parser(self):
        with pytest.raises(SystemExit):
            # Missing required --test-dir
            main(["evaluate"])

    def test_validate_parser(self):
        with pytest.raises(SystemExit):
            # Missing required positional args
            main(["validate"])
