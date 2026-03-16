"""Tests for __main__.py CLI entry point (30% coverage currently)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestValidateImagePath:
    """Tests for _validate_image_path."""

    def test_valid_path(self, tmp_path):
        from landmarkdiff.__main__ import _validate_image_path

        f = tmp_path / "test.png"
        f.write_bytes(b"fake")
        result = _validate_image_path(str(f))
        assert result == f

    def test_missing_path(self):
        from landmarkdiff.__main__ import _validate_image_path

        with pytest.raises(SystemExit):
            _validate_image_path("/nonexistent/image.png")

    def test_directory_path(self, tmp_path):
        from landmarkdiff.__main__ import _validate_image_path

        with pytest.raises(SystemExit):
            _validate_image_path(str(tmp_path))


class TestMainNoCommand:
    """Tests for main() with no command."""

    def test_no_command_prints_help(self, capsys):
        from landmarkdiff.__main__ import main

        with patch("sys.argv", ["landmarkdiff"]):
            main()
        # Should print help and return (not raise)

    def test_version_flag(self):
        from landmarkdiff.__main__ import main

        with (
            pytest.raises(SystemExit) as exc_info,
            patch("sys.argv", ["landmarkdiff", "--version"]),
        ):
            main()
        assert exc_info.value.code == 0


class TestMainLandmarks:
    """Tests for 'landmarks' subcommand."""

    def test_landmarks_missing_image(self):
        from landmarkdiff.__main__ import main

        with (
            pytest.raises(SystemExit),
            patch("sys.argv", ["landmarkdiff", "landmarks", "/nonexistent.png"]),
        ):
            main()


class TestMainInfer:
    """Tests for 'infer' subcommand."""

    def test_infer_missing_image(self):
        from landmarkdiff.__main__ import main

        with (
            pytest.raises(SystemExit),
            patch("sys.argv", ["landmarkdiff", "infer", "/nonexistent.png"]),
        ):
            main()

    def test_infer_bad_intensity(self, tmp_path):
        from landmarkdiff.__main__ import main

        img = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))

        with (
            pytest.raises(SystemExit),
            patch(
                "sys.argv",
                [
                    "landmarkdiff",
                    "infer",
                    str(img),
                    "--intensity",
                    "150",
                    "--mode",
                    "tps",
                ],
            ),
        ):
            main()


class TestMainDemo:
    """Tests for 'demo' subcommand."""

    def test_demo_import_error(self):
        from landmarkdiff.__main__ import main

        with (
            patch("sys.argv", ["landmarkdiff", "demo"]),
            patch.dict("sys.modules", {"scripts": None, "scripts.app": None}),
            pytest.raises(SystemExit),
        ):
            main()


class TestError:
    """Tests for _error helper."""

    def test_error_exits(self):
        from landmarkdiff.__main__ import _error

        with pytest.raises(SystemExit) as exc_info:
            _error("test error message")
        assert exc_info.value.code == 1
