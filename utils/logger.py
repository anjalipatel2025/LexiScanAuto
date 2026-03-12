"""
LexiScan Auto — Structured Logging Utility
============================================
Provides a production-grade logger with both console and rotating file
output.  Every module in the project imports this single factory function
so that log format, level, and destination stay consistent.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
_BACKUP_COUNT = 3


def configure_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Create or retrieve a named logger with console + optional file output.

    Parameters
    ----------
    name : str
        Hierarchical logger name, e.g. ``"LexiScanAuto.OCR"``.
    level : int
        Minimum severity level (default ``logging.INFO``).
    log_to_file : bool
        If *True*, also write to a rotating file under ``logs/lexiscan.log``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ──────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (rotating) ──────────────────────────────────────
    if log_to_file:
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(_LOG_DIR, "lexiscan.log"),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # Gracefully degrade — at least console logging works
            logger.warning("Could not create log directory; file logging disabled.")

    return logger
