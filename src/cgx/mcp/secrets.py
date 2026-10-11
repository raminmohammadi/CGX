"""Secret storage for MCP servers (tokens, API keys) -- never in ``mcp.json``.

A catalog server often needs a credential (``GITHUB_PERSONAL_ACCESS_TOKEN``,
``BRAVE_API_KEY`` ...). The roster only ever records the *variable name*; the
value lives in the OS keyring when the optional ``keyring`` package is
available (Keychain / Secret Service / Credential Manager), else in a
permission-restricted JSON file beside the roster -- the same two-tier scheme
``cgx.answer.profiles`` uses for LLM API keys.

Resolution order at launch (:func:`resolve_env`): a variable already present
in the CGX process environment wins (so ``export GITHUB_PERSONAL_ACCESS_TOKEN``
keeps working), then the stored secret.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "cgx-mcp"


def _secrets_path() -> Path:
    from cgx.mcp.config import default_config_path
    return default_config_path().with_name("mcp-secrets.json")


def _keyring():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception:
        return None


def _key(server: str, var: str) -> str:
    return f"{server}/{var}"


def _read_file() -> Dict[str, str]:
    p = _secrets_path()
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} \
            if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_file(data: Dict[str, str]) -> None:
    p = _secrets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover
        pass


def set_secret(server: str, var: str, value: str) -> str:
    """Store ``value`` for ``server``/``var``; returns ``"keyring"`` or ``"file"``."""
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, _key(server, var), value)  # codeql[py/clear-text-storage-sensitive-data]
            return "keyring"
        except Exception as e:  # pragma: no cover - backend dependent
            logger.warning("mcp secrets: keyring unavailable (%s); using the secrets file", type(e).__name__)
    data = _read_file()
    data[_key(server, var)] = value
    _write_file(data)
    return "file"


def get_secret(server: str, var: str) -> Optional[str]:
    kr = _keyring()
    if kr is not None:
        try:
            val = kr.get_password(KEYRING_SERVICE, _key(server, var))
            if val:
                return val
        except Exception:  # pragma: no cover
            pass
    val = _read_file().get(_key(server, var))
    return val or None


def has_secret(server: str, var: str) -> bool:
    return get_secret(server, var) is not None


def delete_secrets(server: str, var: Optional[str] = None) -> int:
    """Delete one variable's secret (or every secret of ``server``). Returns count."""
    removed = 0
    data = _read_file()
    keys = [k for k in list(data) if k.startswith(server + "/") and (var is None or k == _key(server, var))]
    kr = _keyring()
    targets = set(keys)
    if var is not None:
        targets.add(_key(server, var))
    for k in targets:
        if kr is not None:
            try:
                kr.delete_password(KEYRING_SERVICE, k)
                removed += 1
            except Exception:
                pass
        if k in data:
            data.pop(k, None)
            removed += 1
    if keys:
        _write_file(data)
    return removed


def resolve_env(server: str, var: str) -> Optional[str]:
    """Process environment first, then the stored secret."""
    val = os.environ.get(var)
    if val:
        return val
    return get_secret(server, var)
