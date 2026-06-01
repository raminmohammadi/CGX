"""Tests for :mod:`cgx.logging_setup` (centralized logger configuration)."""

from __future__ import annotations

import logging
import os

import pytest

from cgx.logging_setup import get_logger, setup_logging, temp_log_level


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Snapshot and restore root logger handlers/level around each test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_propagate = root.propagate
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    root.propagate = saved_propagate


def test_setup_logging_with_string_level_sets_root_level():
    root = setup_logging(level="DEBUG")
    assert root.level == logging.DEBUG


def test_setup_logging_with_int_level_sets_root_level():
    root = setup_logging(level=logging.WARNING)
    assert root.level == logging.WARNING


def test_setup_logging_is_idempotent_for_stream_handlers():
    setup_logging(level="INFO")
    n_first = len(logging.getLogger().handlers)
    setup_logging(level="INFO")
    n_second = len(logging.getLogger().handlers)
    assert n_first == n_second, "setup_logging must not duplicate handlers"


def test_setup_logging_adds_file_handler_when_logfile_given(tmp_path):
    log_path = tmp_path / "nested" / "averix.log"
    setup_logging(level="INFO", logfile=str(log_path))
    assert log_path.parent.exists()
    # File handler should exist on the root logger.
    file_handlers = [h for h in logging.getLogger().handlers
                     if isinstance(h, logging.FileHandler)]
    assert any(os.path.abspath(h.baseFilename) == os.path.abspath(str(log_path))
               for h in file_handlers)


def test_setup_logging_keeps_running_when_file_handler_creation_fails(tmp_path):
    # Use an invalid path on a non-existent root-owned directory to force
    # FileHandler creation to swallow the error gracefully.
    bad_path = "/proc/1/cannot-create-here.log"
    # Should not raise.
    setup_logging(level="INFO", logfile=bad_path)
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_propagate_flag():
    root = setup_logging(level="INFO", propagate=True)
    assert root.propagate is True
    setup_logging(level="INFO", propagate=False)
    assert logging.getLogger().propagate is False


def test_get_logger_returns_child_logger_with_handlers_configured():
    # Clear handlers to force auto-config branch in get_logger.
    logging.getLogger().handlers[:] = []
    log = get_logger("cgx.tests.x")
    assert log.name == "cgx.tests.x"
    assert logging.getLogger().handlers, "get_logger must trigger setup_logging"


def test_temp_log_level_restores_original_level_on_exit():
    log = logging.getLogger("cgx.tests.temp")
    log.setLevel(logging.WARNING)
    with temp_log_level(log, "DEBUG"):
        assert log.level == logging.DEBUG
    assert log.level == logging.WARNING


def test_temp_log_level_restores_level_when_body_raises():
    log = logging.getLogger("cgx.tests.temp2")
    log.setLevel(logging.INFO)
    with pytest.raises(RuntimeError):
        with temp_log_level(log, logging.DEBUG):
            raise RuntimeError("boom")
    assert log.level == logging.INFO
