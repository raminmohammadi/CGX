"""Averix Skills — pluggable technology-specific knowledge bundles.

A *skill* encapsulates everything Averix knows about one technology
(framework, language runtime, library, build tool). Skills are
consulted by:

* :mod:`cgx.agents.planner` — to decide whether a goal describes a
  scaffold/change involving a known technology (the new
  scaffold-detection signal).
* :mod:`cgx.answer.engine` — to compose technology-specific
  instructions into the LLM system prompt for scaffold + plan tasks.
* :mod:`cgx.agents.judge` — to run technology-specific structural
  checks on produced diffs.

Skills are listed explicitly in :data:`SKILLS` so the surface is
auditable and import order is deterministic. To add a new skill: write
``skills/<name>.py`` exposing a single ``Skill`` subclass, import it
here, and append an instance to :data:`SKILLS`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from skills.base import (
    SKILL_DETECT_THRESHOLD,
    Skill,
    SkillVerdict,
    file_paths,
    file_with_content,
    has_any_ext,
)
from skills.django import DjangoSkill
from skills.express import ExpressSkill
from skills.fastapi import FastAPISkill
from skills.flask import FlaskSkill
from skills.nextjs import NextJsSkill
from skills.python_cli import PythonCliSkill
from skills.react import ReactSkill
from skills.sqlite import SQLiteSkill
from skills.tailwind import TailwindSkill
from skills.vue import VueSkill


#: Master registry. Order is significant only in that earlier entries
#: are checked first for conflict-resolution logging; the registry
#: itself doesn't enforce role exclusivity (a goal can legitimately
#: trigger React + FastAPI + SQLite + Tailwind at once).
SKILLS: List[Skill] = [
    # Frontend frameworks
    ReactSkill(),
    NextJsSkill(),
    VueSkill(),
    # Backend frameworks (Python)
    FastAPISkill(),
    FlaskSkill(),
    DjangoSkill(),
    # Backend frameworks (Node)
    ExpressSkill(),
    # CLI / scripts
    PythonCliSkill(),
    # Data layer
    SQLiteSkill(),
    # Styling addons
    TailwindSkill(),
]


def detect_skills(goal: str,
                  threshold: float = SKILL_DETECT_THRESHOLD) -> List[Skill]:
    """Return skills whose ``detect(goal)`` score meets ``threshold``.

    Results are sorted by descending detection confidence so callers
    that need a "primary" skill can take the head of the list.
    """
    if not goal or not goal.strip():
        return []
    scored: List[Tuple[Skill, float]] = []
    for s in SKILLS:
        try:
            score = float(s.detect(goal))
        except Exception:
            score = 0.0
        if score >= threshold:
            scored.append((s, score))
    scored.sort(key=lambda x: -x[1])
    return [s for s, _ in scored]


def skill_names(skills: List[Skill]) -> List[str]:
    """Return the ``name`` of each skill in ``skills`` (preserves order)."""
    return [s.name for s in skills]


def skills_by_names(names: List[str]) -> List[Skill]:
    """Resolve a list of skill ``name`` strings to ``Skill`` instances.

    Unknown names are silently skipped so a stale ``task.inputs['skills']``
    payload (carried across versions) doesn't crash a run. Order is
    preserved from ``names``.
    """
    if not names:
        return []
    lookup: Dict[str, Skill] = {s.name: s for s in SKILLS}
    out: List[Skill] = []
    for n in names:
        s = lookup.get(str(n).strip())
        if s is not None:
            out.append(s)
    return out


def compose_scaffold_prompt(skills: List[Skill]) -> str:
    """Join non-empty ``scaffold_system_prompt`` fragments with blank lines."""
    parts = [s.scaffold_system_prompt().strip() for s in skills]
    return "\n\n".join(p for p in parts if p)


def compose_plan_prompt(skills: List[Skill]) -> str:
    """Join non-empty ``plan_system_prompt`` fragments with blank lines."""
    parts = [s.plan_system_prompt().strip() for s in skills]
    return "\n\n".join(p for p in parts if p)


def validate_scaffold(skills: List[Skill],
                      diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
    """Run each skill's scaffold validator; return the first fatal failure.

    Verdicts with ``severity="warning"`` are skipped here; callers that
    want them should use :func:`collect_scaffold_warnings`. Returns
    ``None`` when every skill abstained or passed.
    """
    for s in skills:
        v = s.validate_scaffold(diffs, goal=goal)
        if v is None:
            continue
        if not v.passed and v.severity != "warning":
            if not v.skill:
                v.skill = s.name
            return v
    return None


def collect_scaffold_warnings(skills: List[Skill],
                              diffs: List[Dict[str, Any]],
                              goal: str = "") -> List[SkillVerdict]:
    """Return all advisory (``severity="warning"``) verdicts.

    Aggregates from both :meth:`Skill.scaffold_warnings` and any
    warning-severity verdicts returned by :meth:`Skill.validate_scaffold`,
    so callers see every soft issue (missing tests, missing READMEs,
    etc.) in one list.
    """
    out: List[SkillVerdict] = []
    for s in skills:
        for w in s.scaffold_warnings(diffs, goal=goal) or []:
            if not w.skill:
                w.skill = s.name
            out.append(w)
        v = s.validate_scaffold(diffs, goal=goal)
        if v is not None and v.severity == "warning" and not v.passed:
            if not v.skill:
                v.skill = s.name
            out.append(v)
    return out


def validate_plan(skills: List[Skill],
                  diffs: List[Dict[str, Any]],
                  goal: str = "") -> Optional[SkillVerdict]:
    """Run each skill's plan validator; return the first failure."""
    for s in skills:
        v = s.validate_plan(diffs, goal=goal)
        if v is None:
            continue
        if not v.passed:
            if not v.skill:
                v.skill = s.name
            return v
    return None


__all__ = [
    "SKILLS",
    "SKILL_DETECT_THRESHOLD",
    "Skill",
    "SkillVerdict",
    "collect_scaffold_warnings",
    "compose_plan_prompt",
    "compose_scaffold_prompt",
    "detect_skills",
    "file_paths",
    "file_with_content",
    "has_any_ext",
    "skill_names",
    "skills_by_names",
    "validate_plan",
    "validate_scaffold",
]
