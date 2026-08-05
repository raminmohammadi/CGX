

"""The data flywheel: turn feedback into eval candidates + a unified view.

Two jobs close the loop opened by the feedback store:

* :func:`export_eval_candidates` drains the down-votes into a JSONL file of
  *candidate* golden rows (``query`` for ask, ``task`` for plan) that a human
  triages into ``evals/retrieval_golden.jsonl`` / ``evals/codegen_golden.jsonl``
  -- negatives are the highest-signal cases to add to the regression set.
* :func:`unify_with_lessons` merges the feedback signal with the cross-session
  ``lessons.jsonl`` store (Subsystem 7.1) so the admin/activity view has one
  place to see "what went wrong and what we learned", joined by ``run_id`` /
  ``session_id`` where available.

Stdlib-only and best-effort: a missing store or file degrades to empty output
rather than raising.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cgx.feedback.store import FeedbackStore, get_default_store

logger = logging.getLogger(__name__)


def default_candidates_path() -> Path:
    """Where exported eval candidates land (``$CGX_EVAL_CANDIDATES_PATH`` or
    ``<CGX_CONFIG_DIR or ~/.cgx>/eval_candidates.jsonl``)."""
    override = os.environ.get("CGX_EVAL_CANDIDATES_PATH")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("CGX_CONFIG_DIR") or str(Path.home() / ".cgx")
    return Path(base) / "eval_candidates.jsonl"


def _candidate_row(fb: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a down-voted feedback row into an eval-golden candidate.

    Ask rows carry ``query`` (retrieval golden shape); plan rows carry
    ``task`` (codegen golden shape). ``relevant``/``expect`` are left for the
    human triager to fill -- this is a candidate, not a finished golden case.
    """
    kind = fb.get("kind") or "ask"
    key = "task" if kind == "plan" else "query"
    return {
        "source": "user_feedback",
        "kind": kind,
        "run_id": fb.get("run_id"),
        "session_id": fb.get("session_id"),
        key: fb.get("question") or "",
        "comment": fb.get("comment") or "",
        "model": fb.get("model"),
        "prompt_version": fb.get("prompt_version"),
        "created_at": fb.get("created_at"),
    }


def export_eval_candidates(
    *,
    store: Optional[FeedbackStore] = None,
    out_path: Optional[str | Path] = None,
    kind: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Append every down-vote to the eval-candidates JSONL. Returns a summary.

    Idempotency is by ``feedback_id``: rows already present in the target file
    are skipped, so this is safe to run repeatedly (e.g. on a schedule).
    """
    store = store or get_default_store()
    target = Path(out_path) if out_path else default_candidates_path()
    negatives = store.recent(rating="down", kind=kind, since=since, limit=limit)

    seen: set[str] = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(str(json.loads(line).get("feedback_id")))
            except ValueError:
                continue

    written = 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for fb in negatives:
                fid = str(fb.get("feedback_id"))
                if fid in seen:
                    continue
                row = _candidate_row(fb)
                row["feedback_id"] = fid
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    except OSError as exc:  # pragma: no cover - disk failure is best-effort
        logger.warning("flywheel: failed to write candidates at %s: %s",
                       target, exc)
    return {"path": str(target), "candidates": len(negatives),
            "written": written, "skipped": len(negatives) - written}


def unify_with_lessons(
    *,
    store: Optional[FeedbackStore] = None,
    lessons_path: Optional[Path] = None,
    since: Optional[float] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Merge feedback + lessons into one negative-signal view for the flywheel.

    Returns ``{stats, lessons_count, signals}`` where ``signals`` is a
    time-ordered list of ``{type, ...}`` rows (``type="feedback"`` for
    down-votes, ``type="lesson"`` for repaired-failure lessons).
    """
    from cgx.session import lessons as _lessons

    store = store or get_default_store()
    stats = store.stats(since=since)
    negatives = store.recent(rating="down", since=since, limit=limit)
    lessons = _lessons.load_lessons(path=lessons_path)

    signals: List[Dict[str, Any]] = []
    for fb in negatives:
        signals.append({"type": "feedback", "at": fb.get("created_at"),
                        "run_id": fb.get("run_id"), "kind": fb.get("kind"),
                        "detail": fb.get("comment") or fb.get("question")})
    for les in lessons:
        signals.append({"type": "lesson", "at": les.get("created_at"),
                        "session_id": les.get("session_id"),
                        "classification": les.get("classification"),
                        "detail": les.get("trigger_signature")})
    signals.sort(key=lambda s: str(s.get("at") or ""), reverse=True)
    return {"stats": stats, "lessons_count": len(lessons),
            "signals": signals[:limit], "generated_at": time.time()}
