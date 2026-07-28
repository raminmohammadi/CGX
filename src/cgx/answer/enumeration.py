

"""Deterministic endpoint enumeration for counting/listing queries.

When a user asks "how many API endpoints does X have?" or "list all
routes", semantic retrieval can't reliably produce an aggregate answer --
the truth is scattered across many route-decorator chunks rather than
sitting in one text block, so ranking surfaces a handful and the count is
wrong. This module answers such questions deterministically by filtering
index records that carry ``route`` metadata (stamped at parse time; see
``parse_codebase._detect_route``) and rendering an exact count + list.

Pure and side-effect free: callers load records and pass them in.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Tokens that describe the *question shape* rather than the subject being
# asked about. Dropped when extracting the subject term so "how many api
# endpoints does scanai have?" reduces to ["scanai"].
_ENUM_STOPWORDS = {
    "how", "many", "much", "list", "all", "every", "the", "does", "do",
    "did", "is", "are", "there", "have", "has", "what", "show", "me", "of",
    "a", "an", "and", "in", "for", "to", "count", "number", "total",
    "which", "give", "enumerate", "expose", "exposes", "exposed",
    "api", "apis", "endpoint", "endpoints", "route", "routes",
    "http", "url", "urls", "rest", "restful",
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_subject_terms(question: str) -> List[str]:
    """Return candidate subject tokens (lowercased) from an enumerate query.

    Enumeration stopwords and 1-char tokens are dropped. E.g. "how many
    api endpoints does scanai have?" -> ``["scanai"]``. When the question
    is pure boilerplate ("list all endpoints") the result is empty, which
    the caller treats as "enumerate across the whole index".
    """
    out: List[str] = []
    for m in _WORD_RE.findall(question or ""):
        t = m.lower()
        if t in _ENUM_STOPWORDS or len(t) < 2:
            continue
        out.append(t)
    return out


def _record_route(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the record's route dict if it carries usable route metadata."""
    r = rec.get("route")
    if isinstance(r, dict) and (r.get("methods") or r.get("path")):
        return r
    return None


def _matches_subject(rec: Dict[str, Any], subjects: List[str]) -> bool:
    """True when any subject token appears in the record's file/module/name."""
    if not subjects:
        return True
    hay = " ".join(
        str(rec.get(k) or "").lower()
        for k in ("file", "module_path", "name", "class_name")
    )
    return any(s in hay for s in subjects)


def collect_endpoints(
    records: List[Dict[str, Any]],
    subjects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter ``records`` to route-bearing ones (optionally matching
    ``subjects``), deduped by (methods, path, chunk id) and sorted by path.

    Each entry: ``{chunk_id, methods, path, file, name, start_line}``.
    """
    subjects = subjects or []
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for rec in records:
        route = _record_route(rec)
        if route is None:
            continue
        if not _matches_subject(rec, subjects):
            continue
        methods = [str(m).upper() for m in (route.get("methods") or [])] or ["GET"]
        path = route.get("path")
        cid = str(rec.get("id") or "")
        key = (tuple(sorted(methods)), path, cid)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "chunk_id": cid,
            "methods": methods,
            "path": path,
            "file": rec.get("file"),
            "name": rec.get("name"),
            "start_line": int(rec.get("start_line") or 0),
        })
    out.sort(key=lambda e: (
        str(e.get("path") or "~"), str(e.get("file") or ""), e.get("start_line") or 0
    ))
    return out


def render_enumeration(
    endpoints: List[Dict[str, Any]],
    subjects: Optional[List[str]] = None,
    *,
    scoped: bool = False,
) -> Dict[str, Any]:
    """Build the ``('done', payload)`` result dict for an enumeration."""
    subjects = subjects or []
    subj_txt = f" matching `{' '.join(subjects)}`" if (scoped and subjects) else ""
    n = len(endpoints)
    if n == 0:
        return {
            "answer_md": (
                f"I found **no API endpoints**{subj_txt} in the current index. "
                "If the routes use a framework/decorator style CGX doesn't yet "
                "recognize, they won't be counted -- re-index after upgrading."
            ),
            "citations": [],
            "suggested_changes": [],
            "confidence": 0.5,
            "debug": {"mode": "enumerate", "endpoint_count": 0, "subjects": subjects},
        }
    lines = [f"**{n} API endpoint{'s' if n != 1 else ''}**{subj_txt}:", ""]
    for e in endpoints:
        methods = ", ".join(e["methods"])
        path = e.get("path") or "(path not detected)"
        loc = e.get("file") or ""
        lines.append(f"- `{methods} {path}` \u2014 {e.get('name')} ({loc})")
    return {
        "answer_md": "\n".join(lines),
        "citations": [{"chunk_id": e["chunk_id"]} for e in endpoints if e["chunk_id"]],
        "suggested_changes": [],
        "confidence": 0.95,
        "debug": {"mode": "enumerate", "endpoint_count": n, "subjects": subjects},
    }


def answer_endpoint_enumeration(
    records: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    """Top-level: extract the subject, prefer subject-scoped endpoints, and
    fall back to enumerating every endpoint when the subject matches none."""
    subjects = extract_subject_terms(question)
    scoped = collect_endpoints(records, subjects)
    if scoped:
        return render_enumeration(scoped, subjects, scoped=True)
    return render_enumeration(collect_endpoints(records, []), subjects, scoped=False)
