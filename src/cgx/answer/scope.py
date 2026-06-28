

"""
Scope classifier for retrieval -- orthogonal to :mod:`cgx.answer.intent`.

Where :func:`cgx.answer.intent.detect_intent` answers *what shape of answer
the user wants* (overview, symbol_explain, change_plan, ...), this module
answers *what code scope the user is asking about*: production source,
tests, or "both". The result is consumed by :func:`run_query_auto` to
soft-penalize off-scope chunks so the LLM doesn't ground production
answers in test stubs (e.g. ``CountingHashEmbedder`` from
``tests/test_incremental_index.py``).

Defaults
--------
- ``"src"`` is the default: assume the user is asking about real code.
- ``"tests"`` only when the question is *about* tests.
- ``"any"`` only when the user explicitly opts in (or the intent is
  ``change_plan`` -- new code typically needs to extend existing tests,
  which lives in caller logic, not here).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal

Scope = Literal["src", "tests", "any"]

_TESTS_PATH_RE = re.compile(r"(?:^|/)tests?/")
_TEST_FILE_RE = re.compile(r"(?:^|/)test_[A-Za-z0-9_]+\.py|(?:^|/)[A-Za-z0-9_]+_test\.py")

_TESTS_WORD_RE = re.compile(
    r"\b("
    r"tests?|tested|testing|"
    r"pytest|conftest|"
    r"fixtures?|mocks?|mocking|stubs?|"
    r"test[ _-]suite|test[ _-]cases?|test[ _-]coverage"
    r")\b"
)

_ANY_PHRASES = (
    "including tests", "and tests", "with tests",
    "whole codebase", "entire codebase", "across the codebase",
    "everything", "all files", "every file",
)


def detect_scope(question: str) -> Scope:
    """Classify the code scope the question targets.

    Rule-based, matching the style of :func:`detect_intent`. ``"any"`` wins
    over ``"tests"`` when both are present so an explicit opt-in like
    "explain the parser including its tests" stays inclusive.
    """
    q = (question or "").strip().lower()
    if not q:
        return "src"
    if any(p in q for p in _ANY_PHRASES):
        return "any"
    if _TESTS_WORD_RE.search(q):
        return "tests"
    return "src"


def _chunk_path(hit: Dict[str, Any]) -> str:
    """Best-effort extract of the file path from a hit dict.

    Hybrid retriever hits carry ``chunk_id`` in the form
    ``<abs_path>::<kind>::<symbol>``; downstream code may also attach a
    ``path`` field. Try ``path`` first, then split ``chunk_id``.
    """
    p = hit.get("path")
    if isinstance(p, str) and p:
        return p
    cid = hit.get("chunk_id") or ""
    if isinstance(cid, str) and "::" in cid:
        return cid.split("::", 1)[0]
    return str(cid)


def _is_test_path(path: str) -> bool:
    if not path:
        return False
    if _TESTS_PATH_RE.search(path):
        return True
    if _TEST_FILE_RE.search(path):
        return True
    return False


def apply_scope_penalty(
    hits: List[Dict[str, Any]],
    scope: Scope,
    *,
    penalty: float = 0.3,
) -> List[Dict[str, Any]]:
    """Return ``hits`` with off-scope chunks soft-down-weighted.

    Multiplies the ``score`` field of off-scope hits by ``penalty`` (default
    ``0.3``) and re-sorts descending. ``scope="any"`` is a no-op. Hits
    without a ``score`` field are left alone but still re-positioned by the
    sort, so callers that care about original ordering should pass a copy.

    The returned list is a *new* list of the same dicts (mutated in place
    for the ``score`` field). Original ranks are preserved via a sidecar
    ``scope_demoted`` flag so the UI can surface why a result was demoted.
    """
    if scope == "any" or not hits:
        return list(hits)
    out: List[Dict[str, Any]] = []
    for h in hits:
        path = _chunk_path(h)
        is_test = _is_test_path(path)
        off_scope = (scope == "src" and is_test) or (scope == "tests" and not is_test)
        if off_scope and isinstance(h.get("score"), (int, float)):
            h["score"] = float(h["score"]) * float(penalty)
            h["scope_demoted"] = True
        out.append(h)
    out.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    return out


def resolve_scope_for_intent(question: str, intent: str) -> Scope:
    """Combine question-level scope detection with intent-level override.

    ``change_plan`` always returns ``"any"`` because plans typically need
    to read and extend existing tests. All other intents defer to
    :func:`detect_scope`.
    """
    if intent == "change_plan":
        return "any"
    return detect_scope(question)
