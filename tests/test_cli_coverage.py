"""CLI smoke tests covering uncovered cmd_* and main() paths.

Covers: cmd_version, cmd_config (show + validate), main() no-args exit,
main(["version"]), main(["config", "--show"]).
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from landmarkdiff.cli import cmd_config, cmd_version, main

# ---------------------------------------------------------------------------
# cmd_version — line 157-161
# ---------------------------------------------------------------------------


class TestCmdVersion:
    def test_prints_version(self, capsys):
        args = argparse.Namespace()
        cmd_version(args)
        captured = capsys.readouterr()
        assert "LandmarkDiff v" in captured.out


# ---------------------------------------------------------------------------
# cmd_config — lines 107-127
# ---------------------------------------------------------------------------


class TestCmdConfig:
    def test_show_default_config(self, capsys):
        """cmd_config with no file and no --validate prints YAML."""
        args = argparse.Namespace(file=None, validate=False)
        cmd_config(args)
        captured = capsys.readouterr()
        # ExperimentConfig has a project_name field
        assert "project_name" in captured.out or len(captured.out) > 10

    def test_validate_no_warnings(self, capsys):
        """cmd_config --validate with valid config shows 'no warnings'."""
        args = argparse.Namespace(file=None, validate=True)
        with patch(
            "landmarkdiff.config.validate_config",
            return_value=[],
        ):
            cmd_config(args)
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower() or "no warnings" in captured.out.lower()

    def test_validate_with_warnings(self, capsys):
        """cmd_config --validate with warnings prints them."""
        args = argparse.Namespace(file=None, validate=True)
        with patch(
            "landmarkdiff.config.validate_config",
            return_value=["learning_rate too high", "batch_size not power of 2"],
        ):
            cmd_config(args)
        captured = capsys.readouterr()
        assert "learning_rate too high" in captured.out
        assert "batch_size not power of 2" in captured.out


# ---------------------------------------------------------------------------
# main() — lines 164-280
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_exits(self):
        """main([]) with no subcommand prints help and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_version_subcommand(self, capsys):
        """main(["version"]) prints version string."""
        main(["version"])
        captured = capsys.readouterr()
        assert "LandmarkDiff v" in captured.out

    def test_config_show_subcommand(self, capsys):
        """main(["config"]) shows default YAML config."""
        main(["config"])
        captured = capsys.readouterr()
        assert len(captured.out) > 10

    def test_config_validate_subcommand(self, capsys):
        """main(["config", "--validate"]) runs validation."""
        with patch(
            "landmarkdiff.config.validate_config",
            return_value=[],
        ):
            main(["config", "--validate"])
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower() or "no warnings" in captured.out.lower()
