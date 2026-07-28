"""Cross-session lesson store (Phase 7.1).

Persists ``{trigger_signature, classification, applied_fix, scope}``
records to ``~/.cgx/lessons.jsonl`` (override via ``$CGX_LESSONS_PATH``)
every time a REPAIR cycle is observed to repair the failure --
i.e. a downstream VERIFY in the same chain finishes ``outcome=passed``.

Reads back via :func:`relevant_lessons`, which scores entries against a
SCAFFOLD's objective + stack and returns the highest-ranking ones for
prompt injection. The store is intentionally append-only and crash-safe
(one JSON object per line, ``json.dumps(sort_keys=True)`` so diffs are
review-friendly); corrupt lines are skipped on read.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".cgx" / "lessons.jsonl"
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
# Words that pollute the keyword index without adding signal.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "that", "this", "use",
    "using", "app", "api", "build", "make", "create", "write", "new",
    "project", "code", "file", "files",
})


def lessons_path() -> Path:
    """Return the on-disk lessons file (env-override aware)."""
    override = os.environ.get("CGX_LESSONS_PATH")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_PATH


def record_lesson(
    *,
    trigger_signature: str,
    classification: str,
    applied_fix: Dict[str, Any],
    scope: Dict[str, Any],
    session_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Append a lesson; return the persisted record or ``None`` on failure.

    The record carries a generated ``lesson_id`` + ISO-8601 ``created_at``
    so the file is amenable to time-based filtering later. Disk failures
    are swallowed with a warning -- learning is best-effort and must
    not break the agent loop.
    """
    target = path or lessons_path()
    sig = str(trigger_signature or "").strip()
    cls = str(classification or "").strip()
    if not sig or not cls:
        return None
    entry: Dict[str, Any] = {
        "lesson_id": "lesson_" + uuid.uuid4().hex[:12],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger_signature": sig,
        "classification": cls,
        "applied_fix": dict(applied_fix or {}),
        "scope": dict(scope or {}),
        "session_id": session_id,
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("lessons: failed to record lesson at %s: %s",
                       target, exc)
        return None
    return entry


def load_lessons(*, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read every lesson; skip blank or malformed lines silently."""
    target = path or lessons_path()
    if not target.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("lessons: failed to read %s: %s", target, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def relevant_lessons(
    *,
    objective: str = "",
    stack: Sequence[str] = (),
    limit: int = 5,
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return up to ``limit`` lessons whose scope overlaps the call site.

    Scoring is intentionally simple:

    * +2 per stack overlap (case-insensitive, normalised package names),
    * +1 per objective-keyword overlap (after stopword removal),
    * ties broken by ``created_at`` (more recent first).

    Returns ``[]`` if the store is empty or no lesson scored above zero.
    """
    lessons = load_lessons(path=path)
    if not lessons:
        return []
    stack_norm = {_normalise(s) for s in stack if str(s).strip()}
    obj_words = {w.lower() for w in _WORD_RE.findall(objective or "")
                 if w.lower() not in _STOPWORDS and len(w) >= 3}
    scored: List = []
    for lesson in lessons:
        scope = lesson.get("scope") or {}
        scope_stack = {_normalise(s) for s in (scope.get("stack") or [])}
        scope_kw = {str(w).lower() for w in (scope.get("objective_keywords") or [])}
        score = 2 * len(stack_norm & scope_stack) + len(obj_words & scope_kw)
        if score > 0:
            scored.append((score, lesson.get("created_at") or "", lesson))
    scored.sort(key=lambda row: (-row[0], _neg(row[1])))
    return [row[2] for row in scored[:max(limit, 0)]]


def extract_objective_keywords(objective: str) -> List[str]:
    """Return the keyword tokens used to score a SCAFFOLD's objective."""
    return sorted({w.lower() for w in _WORD_RE.findall(objective or "")
                   if w.lower() not in _STOPWORDS and len(w) >= 3})


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _neg(s: str) -> str:
    # Sort descending by created_at without flipping the lesson type.
    return "".join(chr(255 - ord(c)) if ord(c) < 256 else c for c in s)
