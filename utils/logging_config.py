"""
PRADAS Logging Configuration.

Structured logging with configurable levels, consistent formatting,
and optional file output.
"""

import logging
import sys
from typing import Optional


_INITIALIZED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    module_name: str = "pradas"
) -> logging.Logger:
    """
    Configure structured logging for the PRADAS system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        module_name: Root logger name.

    Returns:
        Configured logger instance.
    """
    global _INITIALIZED

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(module_name)
    logger.setLevel(numeric_level)

    if _INITIALIZED:
        return logger

    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create log file '{log_file}': {e}")

    # Prevent propagation to root logger
    logger.propagate = False

    _INITIALIZED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the pradas namespace.

    Args:
        name: Logger name (will be prefixed with 'pradas.').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"pradas.{name}")
