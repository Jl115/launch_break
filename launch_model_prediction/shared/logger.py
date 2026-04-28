"""Shared logger factory."""
import logging
import sys
from typing import Any

__all__ = ["get_logger"]


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with Rich-like formatting.

    Args:
        name: Logger name (usually __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
