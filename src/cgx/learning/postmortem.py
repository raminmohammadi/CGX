"""Post-mortem lesson extraction -- the model-in-the-loop "learn" step.

Run ONCE at the end of a Swarm verify (never per-file), and only when the run
actually had failures worth learning from. It hands the model the concrete
incidents (what broke, how it was fixed or that it stayed broken) and asks for
*generalised* lessons via the JEV typed-decision layer, then records each to its
scope. Prevention beats cure: a recorded skill lesson is injected into the
scaffold prompt of every future run of that technology, so the mistake is not
made again -- the deterministic Layer-1 fixer then rarely even has to fire.

Gated by ``CGX_SWARM_LEARNING`` (default on; set ``off``/``0``/``false`` to
disable). Fully defensive: a learning failure never affects the run's outcome.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from cgx.answer.schemas import LESSON_SCHEMA
from cgx.learning import lessons as _lessons

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior engineer writing durable post-mortem lessons for an "
    "autonomous coding agent. Given the concrete failures from one build, "
    "distil each into a SHORT, GENERALISED rule that would PREVENT the class of "
    "mistake next time -- not a restatement of this one incident. Choose scope: "
    "'skill' when the rule applies to EVERY future project using that technology "
    "(a framework/stack gotcha) and set 'skill' to the technology name; "
    "'project' when it is specific to THIS repository's layout or config. Prefer "
    "'skill' for framework rules. Write each lesson as an imperative one-liner "
    "(e.g. 'A Flask blueprint route path must not repeat the url_prefix it is "
    "registered under'). Return an empty list if nothing is durably "
    "generalisable. Never invent a lesson unrelated to the given failures.")


def _enabled() -> bool:
    return os.environ.get("CGX_SWARM_LEARNING", "on").strip().lower() not in (
        "off", "0", "false", "no")


def _format_incidents(incidents: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i in incidents:
        status = "FIXED" if i.get("fixed") else "UNRESOLVED"
        files = ", ".join(i.get("files") or []) or "-"
        sig = ", ".join(i.get("signature") or [])
        detail = str(i.get("detail") or "").strip()
        lines.append(f"- [{status}] {detail} (files: {files}"
                     + (f"; signature: {sig}" if sig else "") + ")")
    return "\n".join(lines)


def run_postmortem(provider: Any, *, goal: str, skills: List[str],
                   root: Optional[str], incidents: List[Dict[str, Any]],
                   session_id: str = "", task_id: str = "") -> List[Dict[str, str]]:
    """Extract + record generalised lessons from a run's incidents.

    Returns ``[{id, scope, skill, lesson}]`` for what was recorded (empty when
    disabled, no incidents, or the model found nothing durable). Never raises.
    """
    if not _enabled() or not incidents or provider is None:
        return []
    active = [str(s) for s in (skills or [])]
    try:
        from cgx.answer.jev import decide
        prompt = (
            f"BUILD GOAL:\n{goal}\n\n"
            f"ACTIVE SKILLS: {', '.join(active) or '(none)'}\n\n"
            "FAILURES FROM THIS RUN:\n" + _format_incidents(incidents) + "\n\n"
            "Write the generalised lessons (JSON per the schema).")
        decision = decide(provider, prompt, LESSON_SCHEMA,
                          decision_key="lessons", system=_SYSTEM,
                          temperature=0.0, max_tokens=700)
    except Exception as e:  # pragma: no cover - extraction is best-effort
        logger.debug("postmortem extraction failed: %r", e)
        return []
    if not decision.ok or not isinstance(decision.value, list):
        return []

    # Known skill names, so a 'skill' lesson can only bind to a real skill the
    # run actually used (guards against the model inventing an off-topic scope).
    try:
        import skills as _sk
        known = set(_sk.known_skill_names())
    except Exception:  # pragma: no cover - defensive
        known = set(active)

    recorded: List[Dict[str, str]] = []
    for item in decision.value:
        if not isinstance(item, dict):
            continue
        if item.get("generalizable") is False:
            continue
        lesson = str(item.get("lesson") or "").strip()
        scope = str(item.get("scope") or "").strip().lower()
        if not lesson:
            continue
        skill = str(item.get("skill") or "").strip()
        if scope == "skill":
            # A stack lesson may ONLY bind to a skill the run actually used --
            # otherwise it would poison an unrelated stack. When the run used
            # exactly one skill, an unnamed/mismatched target is unambiguous, so
            # snap it to that skill. In every other case (no active skills, or a
            # name not in the active set) we cannot safely attribute it -> demote
            # to a project lesson. (``known`` is only a sanity check that the name
            # is a real skill; membership in ``active`` is what actually gates.)
            if len(active) == 1 and (skill not in active):
                skill = active[0]
            if not skill or skill not in active or skill not in known:
                scope, skill = "project", ""
        if scope not in ("project", "skill"):
            scope = "project"
        lid = _lessons.record_lesson(
            scope, lesson, skill=skill, root=root,
            root_cause=str(item.get("root_cause") or ""),
            session_id=session_id, task_id=task_id)
        if lid:
            recorded.append({"id": lid, "scope": scope, "skill": skill,
                             "lesson": lesson})
    return recorded


__all__ = ["run_postmortem"]
