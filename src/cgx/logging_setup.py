# src/cgx/logging_setup.py
from __future__ import annotations

"""
Centralized logging setup for cgx.

This module is **add-only** and safe to import from anywhere. It provides:
- setup_logging(): configure a root logger once (stdout + optional file).
- get_logger(name): convenience to obtain a configured child logger.
- temp_log_level(): context manager to temporarily change a logger's level.

It does NOT modify other modules' behavior unless you explicitly call setup_logging().
"""

from contextlib import contextmanager
import logging
import os
from typing import Optional


_DEFAULT_FMT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
    logfile: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure the root logger once. Safe to call multiple times.

    Args:
        level: Log level (name or int).
        fmt: Log message format.
        datefmt: Datetime format.
        logfile: Optional path for a file handler (created if missing).
        propagate: If True, allow logs to bubble up to parent handlers.

    Returns:
        The configured root logger.
    """
    lvl = logging.getLevelName(level) if isinstance(level, str) else int(level)
    root = logging.getLogger()
    root.setLevel(lvl)

    # If handlers already exist, we won't duplicate them.
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(sh)

    if logfile:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            root.addHandler(fh)
        except Exception as e:
            # Do not crash; keep stdout logging
            root.warning("logging_setup: failed to create file handler %r (%s)", logfile, e)

    root.propagate = bool(propagate)
    return root


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with sane defaults. If setup_logging() was not called yet,
    the first call initializes a basic configuration.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


@contextmanager
def temp_log_level(logger: logging.Logger, level: str | int):
    """
    Temporarily change the logger level within a context.
    """
    old = logger.level
    try:
        logger.setLevel(logging.getLevelName(level) if isinstance(level, str) else int(level))
        yield
    finally:
        logger.setLevel(old)
