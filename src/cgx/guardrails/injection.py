

"""Prompt-injection heuristics for user + retrieved content (Subsystem K).

Indirect prompt injection -- an instruction smuggled into a *retrieved* code
chunk or the user's question that tries to override the system prompt or
exfiltrate secrets -- is the highest-signal risk once RAG feeds untrusted repo
text into the model. These scanners are deliberately conservative (recognisable
attack phrasings only, not blanket long-token matching) so ordinary code and
prose survive; every hit is a :class:`~cgx.guardrails.policy.Finding` the
caller records + surfaces, never a silent mutation of the prompt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from cgx.guardrails.policy import Finding

# (code, severity, compiled pattern). Anchored on override / exfiltration
# phrasings rather than generic keywords to keep the false-positive rate low.
_PATTERNS = [
    ("override_instructions", "warning", re.compile(
        r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|"
        r"earlier|preceding)\s+instructions", re.I)),
    ("override_instructions", "warning", re.compile(
        r"disregard\s+(?:the\s+)?(?:previous|prior|above|system)", re.I)),
    ("role_reassignment", "warning", re.compile(
        r"you\s+are\s+now\s+(?:a|an|the|no\s+longer)", re.I)),
    ("system_prompt_probe", "warning", re.compile(
        r"(?:reveal|print|show|repeat|output|display)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:system\s+prompt|instructions|prompt|rules)", re.I)),
    ("secret_exfiltration", "critical", re.compile(
        r"(?:reveal|print|show|send|leak|exfiltrate|output)\s+(?:me\s+)?"
        r"(?:your|the|any)\s+(?:api[\s_-]?key|secret|token|password|"
        r"credential)", re.I)),
    ("delimiter_injection", "warning", re.compile(
        r"(?:<\|im_start\|>|<\|system\|>|\[/?INST\]|```+\s*system)", re.I)),
    ("new_instructions", "warning", re.compile(
        r"(?:new|updated|revised)\s+(?:instructions|system\s+prompt)\s*:", re.I)),
]


def scan_text(text: Any, *, source: str = "input") -> List[Finding]:
    """Return injection :class:`Finding`\\ s for one blob of text."""
    if not isinstance(text, str) or not text:
        return []
    seen: set = set()
    out: List[Finding] = []
    for code, severity, pattern in _PATTERNS:
        m = pattern.search(text)
        if not m or code in seen:
            continue
        seen.add(code)
        out.append(Finding(
            code=code, severity=severity,
            message=f"possible prompt injection in {source}: {code}",
            detail=_excerpt(text, m.start(), m.end())))
    return out


def scan_context(hits: List[Dict[str, Any]], *,
                 max_hits: int = 50) -> List[Finding]:
    """Scan retrieved chunks for *indirect* injection (repo text as attacker).

    Each hit's text is scanned; findings are de-duplicated by ``code`` across
    the batch so one poisoned corpus doesn't flood the alert store. ``source``
    is tagged ``context`` so the caller can distinguish it from user input.
    """
    if not isinstance(hits, list):
        return []
    seen: set = set()
    out: List[Finding] = []
    for hit in hits[:max_hits]:
        if not isinstance(hit, dict):
            continue
        text = hit.get("text") or hit.get("content") or hit.get("code") or ""
        for f in scan_text(text, source="context"):
            if f.code in seen:
                continue
            seen.add(f.code)
            out.append(f)
    return out


def _excerpt(text: str, start: int, end: int, *, pad: int = 40) -> str:
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    snippet = text[lo:hi].replace("\n", " ").strip()
    return (("…" if lo > 0 else "") + snippet + ("…" if hi < len(text) else ""))


__all__ = ["scan_text", "scan_context"]
