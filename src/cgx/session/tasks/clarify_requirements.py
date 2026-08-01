

"""CLARIFY_REQUIREMENTS executor: surface a typed list of clarifying questions.

The first step of the greenfield loop. The user has stated *what* they
want ("a FastAPI todo app with SQLite and pytest tests"), but the
agent still needs to know the *how* (Python version? auth? Docker?
target deploy?). We ask the LLM for 3-6 short, concrete questions
covering tech stack, must-haves, and target environment, then surface
them via an ``ASK_USER(CLARIFY_ANSWERS)`` follow-up.

A deterministic fallback bank covers the case where the LLM returns
malformed JSON or the call fails, so the greenfield path never stalls.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from cgx.answer.schemas import (
    CLARIFY_QUESTIONS_SCHEMA,
    validate_json_schema,
)
from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You help a coding agent scope a brand-new project. The user has "
    "stated a high-level goal; your job is to surface 3 to 6 short, "
    "concrete clarifying questions that would meaningfully change the "
    "scaffold (tech stack, key features, target environment, testing "
    "expectations). Output STRICT JSON only, no prose. Schema:\n"
    '{"questions": [{"id": "q1", "prompt": "...", '
    '"hint": "<optional one-line hint>", '
    '"suggested": ["option a", "option b"]}]}\n'
    "Rules:\n"
    "- 3 to 6 questions. Each must be answerable in one short sentence.\n"
    "- Cover at least one of: language/framework, persistence, auth, "
    "testing, deployment.\n"
    "- ``suggested`` is optional (1-4 short example answers).\n"
    "- Do NOT ask the user to restate the goal; assume it is given.\n"
)


_FALLBACK_QUESTIONS: List[Dict[str, Any]] = [
    {"id": "q1",
     "prompt": "Which language and framework should this use?",
     "hint": "e.g. Python + FastAPI, Node + Express, Go + chi",
     "suggested": ["Python + FastAPI", "Node + Express", "Go + chi"]},
    {"id": "q2",
     "prompt": "Where should the data live?",
     "hint": "Pick a storage layer",
     "suggested": ["SQLite", "Postgres", "In-memory", "Filesystem only"]},
    {"id": "q3",
     "prompt": "Does the project need authentication?",
     "hint": "Pick a scope",
     "suggested": ["None", "API key", "Session login", "OAuth"]},
    {"id": "q4",
     "prompt": "What is the testing expectation?",
     "hint": "Pick a target",
     "suggested": ["Smoke tests only", "Unit tests", "Unit + integration"]},
    {"id": "q5",
     "prompt": "How will this be run / deployed?",
     "hint": "e.g. local dev only, Docker, cloud",
     "suggested": ["Local dev only", "Docker image", "Cloud (AWS/GCP)"]},
]


@register_executor(TaskKind.CLARIFY_REQUIREMENTS)
def run_clarify_requirements(task: TaskNode,
                             deps: ExecutorDeps) -> ExecutorResult:
    """Generate clarifying questions for a new-project objective."""
    goal = str(task.inputs.get("goal") or "").strip()
    if not goal:
        return ExecutorResult(failure="CLARIFY_REQUIREMENTS: empty goal")

    questions = _ask_llm_for_questions(goal, deps) or _FALLBACK_QUESTIONS

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REQUIREMENTS_SHEET,
        content={
            "goal": goal,
            "questions": questions,
            "source": "llm" if questions is not _FALLBACK_QUESTIONS else "fallback",
        },
    )
    return ExecutorResult(
        outputs={
            "requirements_artifact_id": artifact.artifact_id,
            "question_count": len(questions),
        },
        artifact=artifact,
    )


def _ask_llm_for_questions(goal: str,
                           deps: ExecutorDeps) -> List[Dict[str, Any]]:
    """Round-trip the prompt. Returns ``[]`` when the model can't help.

    The call requests schema-constrained decoding; when the backend
    ignores the schema and the reply still parses to no usable question,
    one bounded re-ask folds the concrete violations back to the model.
    A second miss returns ``[]`` and the fallback bank takes over.
    """
    if deps.provider is None:
        return []
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}"},
    ]
    try:
        resp = deps.provider.chat(
            messages=messages,
            temperature=0.2,
            force_json=True,
            json_schema=CLARIFY_QUESTIONS_SCHEMA,
        )
    except Exception as exc:
        logger.warning("CLARIFY_REQUIREMENTS: provider.chat failed: %s: %s",
                       type(exc).__name__, exc)
        return []
    raw = (resp or {}).get("content") or ""
    questions = _parse_questions(raw)
    if questions or not raw.strip():
        return questions
    try:
        parsed = json.loads(raw.strip())
    except Exception:
        parsed = {}
    violations = (validate_json_schema(parsed, CLARIFY_QUESTIONS_SCHEMA)
                  or ["$.questions: fewer than 3 well-formed questions "
                      "(each needs a non-empty prompt)"])
    logger.warning("CLARIFY_REQUIREMENTS: schema violation, re-asking -- %s",
                   "; ".join(violations[:4]))
    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": (
        "Your reply did not match the required schema. Violations:\n- "
        + "\n- ".join(violations[:8])
        + "\n\nReply again with STRICT JSON only, exactly: "
          '{"questions": [{"id": "q1", "prompt": "...", '
          '"hint": "...", "suggested": ["..."]}]} with 3 to 6 questions. '
          "No prose outside JSON.")})
    try:
        resp = deps.provider.chat(
            messages=messages,
            temperature=0.0,
            force_json=True,
            json_schema=CLARIFY_QUESTIONS_SCHEMA,
        )
    except Exception as exc:
        logger.warning("CLARIFY_REQUIREMENTS: re-ask failed: %s: %s",
                       type(exc).__name__, exc)
        return []
    return _parse_questions((resp or {}).get("content") or "")


_HINT_MARKERS = (
    "for example:", "for example", "examples:", "example:",
    "e.g.:", "e.g.", "e.g", "eg.", "ex:",
)


def _chips_from_hint(hint: str) -> List[str]:
    """Derive suggestion chips from a hint when the model omits ``suggested``.

    Weaker models often return the ``hint`` (``"e.g. SQLite, Postgres"``)
    but drop the ``suggested`` array, which would force the user to type
    every answer. We recover chips deterministically: only when the hint
    carries an explicit example marker or an ``or``-list, so plain
    instructions like ``"Pick a storage layer"`` yield nothing.
    """
    text = (hint or "").strip()
    if not text:
        return []
    low = text.lower()
    has_marker = False
    for marker in _HINT_MARKERS:
        idx = low.find(marker)
        if idx != -1:
            text = text[idx + len(marker):].strip(" :")
            has_marker = True
            break
    parts: List[str] = []
    for piece in text.split(","):
        for sub in piece.replace(" OR ", " or ").split(" or "):
            t = sub.strip(" .")
            if t and len(t) <= 40:
                parts.append(t)
    # Dedupe (case-insensitive) while preserving order.
    seen: set = set()
    chips: List[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            chips.append(p)
    if has_marker:
        return chips[:4]
    return chips[:4] if len(chips) >= 2 else []


def _parse_questions(raw: str) -> List[Dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    items = obj.get("questions") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, q in enumerate(items[:6]):
        if not isinstance(q, dict):
            continue
        prompt = str(q.get("prompt") or "").strip()
        if not prompt:
            continue
        hint = str(q.get("hint") or "").strip()
        suggested = q.get("suggested")
        suggested_clean: List[str] = []
        if isinstance(suggested, list):
            for s in suggested[:4]:
                t = str(s or "").strip()
                if t:
                    suggested_clean.append(t)
        if not suggested_clean:
            suggested_clean = _chips_from_hint(hint)
        out.append({
            "id": str(q.get("id") or f"q{idx + 1}"),
            "prompt": prompt,
            "hint": hint,
            "suggested": suggested_clean,
        })
    return out if len(out) >= 3 else []
