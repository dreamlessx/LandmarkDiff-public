"""Centralized logging configuration for LandmarkDiff.

Usage:
    from landmarkdiff.log import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")

Configure globally:
    from landmarkdiff.log import setup_logging
    setup_logging(level="DEBUG")  # affects all LandmarkDiff loggers
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

# Default format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    fmt: str | None = None,
    stream: object = None,
    log_file: str | None = None,
) -> None:
    """Configure logging for the landmarkdiff package.

    Call once at application startup. Subsequent calls update the level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Custom format string. None uses the default.
        stream: Output stream. None defaults to stderr.
        log_file: Optional file path for logging to a file.
    """
    global _CONFIGURED

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("landmarkdiff")
    root_logger.setLevel(level)

    if not _CONFIGURED:
        formatter = logging.Formatter(
            fmt or LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
        )
        
        # Add stream handler
        stream_handler = logging.StreamHandler(stream or sys.stderr)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        
        # Add file handler if log_file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        root_logger.propagate = False
        _CONFIGURED = True
    else:
        # Just update the level
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a LandmarkDiff module.

    The returned logger is a child of the 'landmarkdiff' root logger,
    so setup_logging() controls its level and formatting.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logging.Logger instance.
    """
    # Ensure base configuration exists
    if not _CONFIGURED:
        setup_logging()

    return logging.getLogger(name)
