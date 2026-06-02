"""Tests for the opt-in anonymous telemetry module.

These verify the (small but important) privacy guarantees:

* ``ping()`` is a no-op when the env var is missing/falsy.
* The install id is a UUID4 persisted under ``CGX_CONFIG_DIR``.
* The payload is the minimal {install_id, cgx_version, event} dict
  with no extra fields that could leak PII.
* Calls are at-most-once-per-process via the in-module latch.
* Network errors never propagate to the caller.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from cgx import telemetry


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Force CGX_CONFIG_DIR to a tmp path and reset telemetry state."""
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path))
    # Re-resolve module-level constants that captured the env at import.
    monkeypatch.setattr(telemetry, "CONFIG_DIR", Path(tmp_path))
    monkeypatch.setattr(telemetry, "INSTALL_ID_PATH", Path(tmp_path) / "install_id")
    telemetry.reset_for_tests()
    yield


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CGX_TELEMETRY", raising=False)
    assert telemetry.is_enabled() is False
    assert telemetry.ping() is False


def test_falsy_values_do_not_enable(monkeypatch):
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("CGX_TELEMETRY", val)
        assert telemetry.is_enabled() is False


def test_install_id_is_uuid_and_persisted(tmp_path):
    first = telemetry.get_install_id()
    second = telemetry.get_install_id()
    assert first == second, "install id must be stable across calls"
    # Must parse as a UUID and be a v4 (random) one.
    parsed = uuid.UUID(first)
    assert parsed.version == 4


def test_ping_dispatches_minimal_payload(monkeypatch):
    monkeypatch.setenv("CGX_TELEMETRY", "1")
    monkeypatch.setenv("CGX_TELEMETRY_URL", "https://example.invalid/ping")
    seen = {}

    def fake_post(url, payload, timeout=2.0):
        seen["url"] = url
        seen["payload"] = payload
        seen["timeout"] = timeout

    monkeypatch.setattr(telemetry, "_post", fake_post)
    # Disable the background thread to make the test deterministic; the
    # thread is purely for I/O latency, so calling _post inline is fine.
    monkeypatch.setattr(telemetry.threading, "Thread",
                        lambda target, args, daemon: _ImmediateThread(target, args))

    assert telemetry.ping() is True
    payload = seen["payload"]
    # Only the three documented fields are allowed.
    assert set(payload.keys()) == {"install_id", "cgx_version", "event"}
    assert payload["event"] == "startup"
    uuid.UUID(payload["install_id"])  # must parse


def test_ping_is_at_most_once_per_process(monkeypatch):
    monkeypatch.setenv("CGX_TELEMETRY", "1")
    monkeypatch.setenv("CGX_TELEMETRY_URL", "https://example.invalid/ping")
    monkeypatch.setattr(telemetry, "_post", lambda *a, **kw: None)
    monkeypatch.setattr(telemetry.threading, "Thread",
                        lambda target, args, daemon: _ImmediateThread(target, args))
    assert telemetry.ping() is True
    assert telemetry.ping() is False  # latched


def test_post_errors_are_swallowed(monkeypatch):
    monkeypatch.setenv("CGX_TELEMETRY", "1")
    monkeypatch.setenv("CGX_TELEMETRY_URL", "https://example.invalid/ping")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    # Patch requests so the real ``_post`` exception path is exercised.
    class _FakeRequests:
        @staticmethod
        def post(*a, **kw):
            raise RuntimeError("network down")

    import sys
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests)
    monkeypatch.setattr(telemetry.threading, "Thread",
                        lambda target, args, daemon: _ImmediateThread(target, args))
    # Should NOT raise.
    assert telemetry.ping() is True


def test_payload_has_no_pii_fields():
    """Belt-and-suspenders: enumerate forbidden keys."""
    forbidden = {"prompt", "query", "code", "file", "path", "user", "email",
                 "hostname", "ip", "token", "api_key"}
    payload = telemetry._payload()
    for k in payload.keys():
        assert k not in forbidden, f"telemetry payload leaks {k}"


class _ImmediateThread:
    """Tiny stand-in that runs the target synchronously."""

    def __init__(self, target, args):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)
