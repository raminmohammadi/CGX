"""Append-only, auditable, reversible store for learned lessons.

Two scopes, per the design:

* **project** -- ``<root>/.cgx/lessons.jsonl`` (+ a rendered ``.cgx/lessons.md``)
  for quirks of one repo (its layout, its fixtures, its config).
* **skill** -- ``skills/<name>/lessons.jsonl`` (+ ``skills/<name>/lessons.md``)
  for stack-level gotchas that apply to EVERY future run using that skill
  (e.g. "a Flask blueprint route must not repeat its own url_prefix").

The JSONL ledger is the source of truth; the ``.md`` is rendered from it for
human reading. Recording appends one entry; the ``.md`` is re-rendered. Pruning
removes the entry by id and re-renders -- so a wrong lesson is fully reversible.
Every function is defensive: a learning-store failure must never sink a run.

Design invariants:

* **Auditable** -- each entry carries ``created_at``, ``session_id``,
  ``task_id``, the ``failure_signature`` it came from, and the ``root_cause``.
* **Bounded** -- a scope keeps at most :data:`_MAX_LESSONS` entries (newest
  win); duplicates (same normalised lesson text) are collapsed, not appended.
* **Determinism-friendly** -- lessons are recorded post-hoc and are inspectable;
  a run's behaviour change is always traceable to a specific, removable entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Where skills live (repo-level ``skills/`` package) -- resolved from this file
# so it holds regardless of CWD. skills/<name>/lessons.{jsonl,md}.
_SKILLS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "skills"))

_MAX_LESSONS = 40          # per scope; oldest pruned when exceeded
_MAX_INJECT_CHARS = 1500   # clip when injected into a prompt (num_ctx guard)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _skill_dir(skill: str) -> str:
    return os.path.join(_SKILLS_DIR, re.sub(r"[^\w\-]", "", skill or ""))


def _ledger_path(scope: str, skill: str, root: Optional[str]) -> Optional[str]:
    if scope == "skill" and skill:
        return os.path.join(_skill_dir(skill), "lessons.jsonl")
    if scope == "project" and root:
        return os.path.join(root, ".cgx", "lessons.jsonl")
    return None


def _md_path(scope: str, skill: str, root: Optional[str]) -> Optional[str]:
    p = _ledger_path(scope, skill, root)
    return p[: -len(".jsonl")] + ".md" if p else None


# --------------------------------------------------------------------------- #
# Read / render
# --------------------------------------------------------------------------- #
def _read_ledger(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:  # pragma: no cover - store read is best-effort
        return []
    return out


def read_lessons(scope: str, *, skill: str = "",
                 root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the ledger entries for one scope (newest last)."""
    return _read_ledger(_ledger_path(scope, skill, root))


def _render_md(scope: str, skill: str, entries: List[Dict[str, Any]]) -> str:
    who = f"skill `{skill}`" if scope == "skill" else "this project"
    head = (f"# Lessons learned for {who}\n\n"
            "Auto-recorded by CGX after failed/repaired runs. Each is a durable "
            "rule fed back into future generation. Safe to prune: delete a line "
            "here and its matching entry in the `.jsonl` beside it.\n")
    lines = [head]
    for e in entries:
        ts = e.get("created_at")
        when = ""
        if isinstance(ts, (int, float)):
            when = time.strftime("%Y-%m-%d", time.gmtime(ts))
        cause = str(e.get("root_cause") or "").strip()
        lesson = str(e.get("lesson") or "").strip()
        tag = f"  <!-- id={e.get('id','')} {when} -->"
        lines.append(f"- {lesson}" + (f" (root cause: {cause})" if cause else "")
                     + tag)
    return "\n".join(lines) + "\n"


def _rewrite(scope: str, skill: str, root: Optional[str],
             entries: List[Dict[str, Any]]) -> None:
    ledger = _ledger_path(scope, skill, root)
    md = _md_path(scope, skill, root)
    if not ledger:
        return
    try:
        os.makedirs(os.path.dirname(ledger), exist_ok=True)
        with open(ledger, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        if md:
            with open(md, "w", encoding="utf-8") as fh:
                fh.write(_render_md(scope, skill, entries))
    except Exception:  # pragma: no cover - store write is best-effort
        pass


# --------------------------------------------------------------------------- #
# Record / prune
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def record_lesson(scope: str, lesson: str, *, skill: str = "",
                  root: Optional[str] = None, root_cause: str = "",
                  failure_signature: Optional[List[str]] = None,
                  source_files: Optional[List[str]] = None,
                  session_id: str = "", task_id: str = "",
                  now: Optional[float] = None) -> Optional[str]:
    """Append a lesson to its scope ledger (idempotent on lesson text).

    Returns the lesson id, or ``None`` when the scope is unaddressable (e.g.
    ``project`` with no root) or the text is empty. A lesson whose normalised
    text already exists is refreshed in place (timestamp bumped, moved to the
    end) rather than duplicated, so repeated failures do not bloat the store.
    Never raises.
    """
    lesson = (lesson or "").strip()
    if not lesson or scope not in ("project", "skill"):
        return None
    if scope == "skill" and not skill:
        return None
    if scope == "project" and not root:
        return None
    entries = read_lessons(scope, skill=skill, root=root)
    key = _norm(lesson)
    entries = [e for e in entries if _norm(str(e.get("lesson"))) != key]
    ts = now if now is not None else time.time()
    lid = "lsn_" + hashlib.sha1(
        f"{scope}:{skill}:{key}".encode("utf-8")).hexdigest()[:12]
    entries.append({
        "id": lid,
        "created_at": ts,
        "scope": scope,
        "skill": skill,
        "lesson": lesson,
        "root_cause": (root_cause or "").strip(),
        "failure_signature": sorted(failure_signature or []),
        "source_files": list(source_files or []),
        "session_id": session_id,
        "task_id": task_id,
    })
    if len(entries) > _MAX_LESSONS:
        entries = entries[-_MAX_LESSONS:]
    _rewrite(scope, skill, root, entries)
    return lid


def prune_lesson(lesson_id: str, *, scope: str, skill: str = "",
                 root: Optional[str] = None) -> bool:
    """Remove a lesson by id and re-render (full reversibility). Never raises."""
    entries = read_lessons(scope, skill=skill, root=root)
    kept = [e for e in entries if e.get("id") != lesson_id]
    if len(kept) == len(entries):
        return False
    _rewrite(scope, skill, root, kept)
    return True


# --------------------------------------------------------------------------- #
# Inject (prompt-facing text)
# --------------------------------------------------------------------------- #
def _inject_text(entries: List[Dict[str, Any]], header: str) -> str:
    if not entries:
        return ""
    bullets = []
    for e in entries:
        lesson = str(e.get("lesson") or "").strip()
        if lesson:
            bullets.append(f"- {lesson}")
    if not bullets:
        return ""
    body = header + "\n" + "\n".join(bullets)
    if len(body) > _MAX_INJECT_CHARS:
        body = body[:_MAX_INJECT_CHARS].rstrip() + "\n- [... older lessons omitted ...]"
    return body


def skill_lessons_text(skill: str) -> str:
    """Prompt-ready lessons for a skill (empty when none). Clipped for num_ctx."""
    return _inject_text(
        read_lessons("skill", skill=skill),
        f"LESSONS FROM PAST RUNS (skill: {skill} -- avoid these mistakes):")


def project_lessons_text(root: Optional[str]) -> str:
    """Prompt-ready lessons for a project (empty when none). Clipped."""
    if not root:
        return ""
    return _inject_text(
        read_lessons("project", root=root),
        "PROJECT MEMORY (lessons from past runs on this project):")
