# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Ultra-minimal, opt-in anonymous telemetry for CGX.

Design constraints (non-negotiable):

* **Off by default.** No ping is ever sent unless ``CGX_TELEMETRY=1`` is
  set in the environment.
* **Anonymous.** The only payload field that identifies anything is a
  random UUID4 generated on first run and persisted under
  ``~/.cgx/install_id``. No prompts, file paths, code, model names, or
  PII are collected.
* **At-most-once-per-process.** :func:`ping` is idempotent; subsequent
  calls within the same process short-circuit.
* **Non-blocking and best-effort.** The HTTP POST runs in a daemon
  thread with a 2 s timeout and swallows every exception so a failed
  ping cannot affect application behaviour.

Disable globally with ``CGX_TELEMETRY=0`` or simply by leaving the env
var unset (the default). To inspect or rotate the install id, delete
``~/.cgx/install_id`` and restart.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Optional

# Imported lazily inside ``_post`` so the telemetry module can be imported
# in environments without ``requests`` installed.

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
INSTALL_ID_PATH = CONFIG_DIR / "install_id"

# Default endpoint is intentionally empty — telemetry is a self-hosted
# extension point. Set ``CGX_TELEMETRY_URL`` to your collector to enable.
DEFAULT_ENDPOINT = ""

# Module-level guard so multiple imports/launches share state.
_HAS_PINGED = False
_LOCK = threading.Lock()


def is_enabled() -> bool:
    """Return True iff the opt-in env var is set to a truthy value."""
    val = os.environ.get("CGX_TELEMETRY", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def get_install_id() -> str:
    """Return (and lazily create) a persistent random install id.

    The id is a UUID4 stored as plain text under ``~/.cgx/install_id``
    with mode ``0600`` when possible. If the file already exists but
    contains an obviously bogus value (empty, very long, non-printable)
    we regenerate it rather than reusing it.
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to an ephemeral id; we still want telemetry to be
        # totally non-fatal on read-only filesystems.
        return str(uuid.uuid4())

    if INSTALL_ID_PATH.exists():
        try:
            raw = INSTALL_ID_PATH.read_text(encoding="utf-8").strip()
            if 16 <= len(raw) <= 64 and raw.isprintable():
                return raw
        except Exception:
            pass

    new_id = str(uuid.uuid4())
    try:
        INSTALL_ID_PATH.write_text(new_id, encoding="utf-8")
        os.chmod(INSTALL_ID_PATH, 0o600)
    except Exception:
        pass
    return new_id


def _cgx_version() -> str:
    try:
        from importlib.metadata import version  # py3.8+
        return version("cgx")
    except Exception:
        return "unknown"


def _post(endpoint: str, payload: dict, timeout: float = 2.0) -> None:
    """Best-effort POST. Swallows every exception."""
    try:
        import requests  # imported here so telemetry never breaks imports.
        requests.post(endpoint, json=payload, timeout=timeout)
    except Exception:
        pass


def _payload() -> dict:
    """Return the (intentionally minimal) telemetry payload."""
    return {
        "install_id": get_install_id(),
        "cgx_version": _cgx_version(),
        "event": "startup",
    }


def ping(endpoint: Optional[str] = None) -> bool:
    """Fire an anonymous startup ping if telemetry is enabled.

    Returns ``True`` if a ping was dispatched (in a background thread),
    ``False`` otherwise. Idempotent within a process.
    """
    global _HAS_PINGED
    if not is_enabled():
        return False
    with _LOCK:
        if _HAS_PINGED:
            return False
        _HAS_PINGED = True
    url = endpoint or os.environ.get("CGX_TELEMETRY_URL", DEFAULT_ENDPOINT)
    if not url:
        return False
    t = threading.Thread(target=_post, args=(url, _payload()), daemon=True)
    t.start()
    return True


def reset_for_tests() -> None:
    """Test hook: clear the in-process ping latch."""
    global _HAS_PINGED
    with _LOCK:
        _HAS_PINGED = False
