"""Secret redaction helpers for logs, traces, and stored payloads.

Masks common credential shapes (API keys, bearer tokens, ``key=value``
assignments, provider-specific key prefixes) before text is written to
the project ``agent.log``, the fallback trace log, or the session store.
The goal is defence-in-depth: prompts and responses can legitimately
echo configuration, and a raw secret must never persist.

The helpers are deliberately conservative -- they target recognisable
credential shapes rather than blanket-redacting long tokens, so ordinary
code and prose survive intact. This module depends only on the standard
library so it is safe to import from anywhere (including ``cgx.trace``).
"""

from __future__ import annotations

import json
import re
from typing import Any, List

_PLACEHOLDER = "<redacted>"

# Keys whose *value* is always sensitive. Longer keys are listed before
# their prefixes so ordered alternation prefers the specific match.
_SENSITIVE_KEYS = (
    "authorization", "access_token", "refresh_token", "client_secret",
    "private_key", "api_key", "apikey", "api-key", "password", "passwd",
    "secret", "token", "auth",
)
_SENSITIVE_KEY_SET = {k.lower() for k in _SENSITIVE_KEYS}
_KEY_ALT = "|".join(re.escape(k) for k in _SENSITIVE_KEYS)

# (pattern, replacement). Order matters: bearer runs before the bare
# ``key=value`` rule so ``Authorization: Bearer <tok>`` fully collapses.
_RULES = [
    # URL query params: ?key=... / &access_token=... (Gemini/OAuth style)
    (re.compile(r"([?&](?:key|api_key|access_token|token)=)[^&\s\"']+", re.I),
     r"\1" + _PLACEHOLDER),
    # JSON / dict literal: "api_key": "secret"  'token' : 'secret'
    (re.compile(r"(['\"]?(?:%s)['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])" % _KEY_ALT,
                re.I), r"\1" + _PLACEHOLDER + r"\2"),
    # Authorization: Bearer <token>
    (re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1" + _PLACEHOLDER),
    # Bare assignment: api_key=secret   token = secret (no quotes)
    (re.compile(r"\b((?:%s)\s*[:=]\s*)[^\s,;&'\"]+" % _KEY_ALT, re.I),
     r"\1" + _PLACEHOLDER),
    # Provider-specific key prefixes.
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), _PLACEHOLDER),
    (re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}\b"), _PLACEHOLDER),
    (re.compile(r"\bgh[posru]_[A-Za-z0-9]{20,}\b"), _PLACEHOLDER),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), _PLACEHOLDER),
]

_DEFAULT_PREVIEW_CAP = 240
_MAX_DEPTH = 6


def redact_text(text: Any) -> str:
    """Return ``text`` with recognised secret shapes masked.

    Non-strings are coerced via ``str``; ``None`` becomes an empty string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    out = text
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    return out


def redact_mapping(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively redact a JSON-like structure.

    Values under a sensitive key are dropped entirely; every other string
    is passed through :func:`redact_text`. Depth is bounded to avoid
    pathological nesting.
    """
    if _depth > _MAX_DEPTH:
        return "<max-depth>"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEY_SET:
                out[k] = _PLACEHOLDER
            else:
                out[k] = redact_mapping(v, _depth=_depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_mapping(v, _depth=_depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"…[+{len(text) - cap} chars]"


def preview_text(text: Any, cap: int = _DEFAULT_PREVIEW_CAP) -> str:
    """Redacted, length-capped snippet suitable for a trace field."""
    return _truncate(redact_text(text), cap)


def preview_mapping(obj: Any, cap: int = _DEFAULT_PREVIEW_CAP) -> str:
    """Redacted, length-capped JSON snippet for a dict/list trace field."""
    try:
        rendered = json.dumps(redact_mapping(obj), default=str,
                              ensure_ascii=False, sort_keys=True)
    except Exception:  # pragma: no cover - defensive
        rendered = redact_text(obj)
    return _truncate(rendered, cap)


def flatten_messages(messages: Any) -> str:
    """Flatten a chat ``messages`` list into ``[role] content`` lines."""
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)
