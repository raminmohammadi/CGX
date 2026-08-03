

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
import re
from typing import Optional


_DEFAULT_FMT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


# --- Secret scrubbing -------------------------------------------------------
#
# The single source of truth for redacting credential material out of log
# lines. Providers put secrets in three shapes we ever emit: Gemini's
# ``?key=<secret>`` query parameter (which ``requests`` echoes back inside
# connection/SSL exception strings -- the leak this centralizes against),
# an ``api_key=<secret>`` form field, and an ``Authorization: Bearer
# <token>`` header. Each value is matched only up to the next delimiter so
# the surrounding log text is never mangled, and empty values become
# ``<missing>`` so a redacted secret is distinguishable from a request that
# carried none. ``scrub_secrets`` is imported by ``cgx.answer.ratelimit``
# and ``cgx.answer.providers`` so the redaction rule lives in exactly one
# place; ``SecretScrubbingFilter`` is the process-wide backstop attached to
# every handler by :func:`setup_logging`, catching anything a call site
# forgot to scrub itself.
_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    # ``key=`` / ``api_key=`` as a query param (``?``/``&`` lead) or a
    # standalone token (start-of-string or preceded by whitespace). The
    # value runs to the next ``&``, whitespace, or quote.
    (re.compile(r"((?:[?&]|^|(?<=\s))(?:api_key|key)=)([^&\s\"']*)",
                re.IGNORECASE), "query"),
    (re.compile(r"(Authorization\s*[:=]\s*Bearer\s+)([^\s\"']*)",
                re.IGNORECASE), "header"),
)


def scrub_secrets(text: str) -> str:
    """Redact known credential shapes from ``text``.

    Idempotent and safe on non-secret input: a string with no matching
    pattern is returned unchanged. Non-``str`` input is returned as-is so
    callers can pass it through unconditionally.
    """
    if not text or not isinstance(text, str):
        return text

    def _sub(m: "re.Match[str]") -> str:
        return m.group(1) + ("<redacted>" if m.group(2) else "<missing>")

    for pattern, _kind in _SECRET_PATTERNS:
        text = pattern.sub(_sub, text)
    return text


class SecretScrubbingFilter(logging.Filter):
    """Logging filter that scrubs credential material from every record.

    Attached to each handler configured by :func:`setup_logging`, so it
    runs after the emitting module has interpolated its ``msg % args`` but
    before the record reaches any handler's stream. We rewrite ``msg`` (and
    clear ``args``) rather than only the final formatted string so the
    redaction survives regardless of the handler's formatter -- including
    the JSONL handlers used by the agent-log / trace sinks.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive: bad % args
            return True
        scrubbed = scrub_secrets(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = None
        return True


def _install_scrubbing_filter(handler: logging.Handler) -> None:
    """Attach a :class:`SecretScrubbingFilter` to ``handler`` once."""
    if not any(isinstance(f, SecretScrubbingFilter) for f in handler.filters):
        handler.addFilter(SecretScrubbingFilter())


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

    # Backstop: guarantee no handler ever emits credential material, even
    # for a log site that forgot to scrub. Re-applied on every call (and
    # to pre-existing handlers) since setup_logging is idempotent and may
    # run after other code attached its own handlers.
    for handler in root.handlers:
        _install_scrubbing_filter(handler)

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
