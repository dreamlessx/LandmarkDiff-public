"""Tests for the logging module."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reset the module state before testing
import landmarkdiff.log as log_module


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    # Save original state
    orig_configured = log_module._CONFIGURED
    root = logging.getLogger("landmarkdiff")
    orig_handlers = root.handlers[:]
    orig_level = root.level

    yield

    # Restore state
    log_module._CONFIGURED = orig_configured
    root.handlers = orig_handlers
    root.setLevel(orig_level)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_configures_root_logger(self):
        log_module._CONFIGURED = False
        log_module.setup_logging(level="DEBUG")
        root = logging.getLogger("landmarkdiff")
        assert root.level == logging.DEBUG

    def test_string_level(self):
        log_module._CONFIGURED = False
        log_module.setup_logging(level="WARNING")
        root = logging.getLogger("landmarkdiff")
        assert root.level == logging.WARNING

    def test_int_level(self):
        log_module._CONFIGURED = False
        log_module.setup_logging(level=logging.ERROR)
        root = logging.getLogger("landmarkdiff")
        assert root.level == logging.ERROR

    def test_custom_stream(self):
        log_module._CONFIGURED = False
        buf = io.StringIO()
        log_module.setup_logging(level="INFO", stream=buf)

        logger = logging.getLogger("landmarkdiff.test_stream")
        logger.info("test message")
        output = buf.getvalue()
        assert "test message" in output

    def test_custom_format(self):
        log_module._CONFIGURED = False
        buf = io.StringIO()
        log_module.setup_logging(level="INFO", fmt="%(message)s", stream=buf)

        logger = logging.getLogger("landmarkdiff.test_fmt")
        logger.info("bare message")
        output = buf.getvalue()
        assert output.strip() == "bare message"

    def test_no_duplicate_handlers(self):
        log_module._CONFIGURED = False
        log_module.setup_logging()
        handler_count_1 = len(logging.getLogger("landmarkdiff").handlers)

        # Second call should NOT add another handler
        log_module.setup_logging(level="DEBUG")
        handler_count_2 = len(logging.getLogger("landmarkdiff").handlers)
        assert handler_count_2 == handler_count_1

    def test_propagate_disabled(self):
        log_module._CONFIGURED = False
        log_module.setup_logging()
        assert logging.getLogger("landmarkdiff").propagate is False


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        logger = log_module.get_logger("landmarkdiff.test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = log_module.get_logger("landmarkdiff.mymodule")
        assert logger.name == "landmarkdiff.mymodule"

    def test_auto_configures(self):
        log_module._CONFIGURED = False
        _ = log_module.get_logger("landmarkdiff.auto")
        assert log_module._CONFIGURED is True

    def test_child_inherits_level(self):
        log_module._CONFIGURED = False
        log_module.setup_logging(level="WARNING")
        logger = log_module.get_logger("landmarkdiff.child")
        # Child logger should effectively inherit WARNING level
        assert logger.getEffectiveLevel() == logging.WARNING
