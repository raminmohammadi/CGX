"""PII / DLP scanning beyond credential redaction (Subsystem M).

:mod:`cgx.redact` masks *credential* shapes (API keys, bearer tokens); this
module is the second, orthogonal pass that finds and scrubs *personal* data
-- emails, phone numbers, IPv4 addresses, and card-like number runs -- so an
operator can (a) audit how much PII a stored payload carries and (b) scrub it
before it is persisted when the ``scrub_pii`` policy toggle is on.

The detectors are deliberately conservative to keep false positives low: they
target well-shaped tokens rather than blanket-matching digits, and each match
collapses to a typed placeholder (``<email>``, ``<phone>`` ...) so the scrubbed
text stays readable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_PATTERNS: List[tuple[str, "re.Pattern[str]"]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # 13-16 digit runs allowing space/dash grouping (card-like); anchored on
    # word boundaries so it won't swallow a longer identifier.
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    # Phone: optional +, 7+ digits with common separators. Runs after card so
    # long card sequences are consumed first.
    ("phone", re.compile(r"(?<![\w.])\+?\d(?:[\d\-\s().]{6,}\d)(?![\w.])")),
]

_PLACEHOLDER = {name: f"<{name}>" for name, _ in _PATTERNS}


def scan_pii(text: Any) -> List[Dict[str, Any]]:
    """Return ``[{"type", "count"}]`` for each PII class present in ``text``.

    Non-strings coerce via ``str``; empty/None yields ``[]``. Detection is
    non-destructive -- this is the audit path used by the scan API and by the
    admin overview to quantify exposure without mutating stored data.
    """
    if not text:
        return []
    if not isinstance(text, str):
        text = str(text)
    out: List[Dict[str, Any]] = []
    # Consume matches in the same order as :func:`scrub_pii` so the counts are
    # non-overlapping -- an IP or card run is not also counted as a phone.
    work = text
    for name, pat in _PATTERNS:
        n = len(pat.findall(work))
        if n:
            out.append({"type": name, "count": n})
            work = pat.sub(_PLACEHOLDER[name], work)
    return out


def scrub_pii(text: Any) -> str:
    """Return ``text`` with every recognised PII token replaced by ``<type>``.

    Applied in the same order as :data:`_PATTERNS` so the greedier card rule
    runs before the looser phone rule and the two do not fight over a match.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    out = text
    for name, pat in _PATTERNS:
        out = pat.sub(_PLACEHOLDER[name], out)
    return out


def scrub_mapping(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively scrub PII from a JSON-like structure (strings only)."""
    if _depth > 6:
        return obj
    if isinstance(obj, dict):
        return {k: scrub_mapping(v, _depth=_depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_mapping(v, _depth=_depth + 1) for v in obj]
    if isinstance(obj, str):
        return scrub_pii(obj)
    return obj


def has_pii(text: Any) -> bool:
    return bool(scan_pii(text))
