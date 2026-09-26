"""CGX Skills -- pluggable technology-specific knowledge bundles.

A *skill* encapsulates everything CGX knows about one technology
(framework, language runtime, library, build tool). Skills are
consulted by:

* :mod:`cgx.answer.engine` -- to detect which technologies a goal
  involves (``detect_skills``) and to compose technology-specific
  instructions into the LLM system prompt for scaffold + plan tasks
  (``compose_scaffold_prompt`` / ``compose_plan_prompt``).
* :mod:`cgx.session.tasks.scaffold` -- to run technology-specific
  structural checks on the produced diffs (``validate_scaffold`` /
  ``collect_scaffold_warnings``); a fatal verdict drives the router's
  whole-tree scaffold regenerate.
* :mod:`cgx.session.tasks.plan_change` -- to run ``validate_plan``
  over a proposed code-change plan and surface the verdict at the
  approval gate.

Skills are listed explicitly in :data:`SKILLS` so the surface is
auditable and import order is deterministic. To add a new built-in
skill: write ``skills/<name>.py`` exposing a single ``Skill`` subclass,
import it here, and append an instance to :data:`SKILLS`.

Users can also add **custom skills** at runtime without touching this
package: a single ``Skill`` subclass per ``.py`` file under
``~/.cgx/skills/`` (see :mod:`skills.loader`). Custom skills are loaded
lazily and merged with :data:`SKILLS` by every public function below
(:func:`detect_skills`, :func:`skills_by_names`, :func:`describe_skills`)
via the internal ``_all_skills()`` helper, so they participate in
detection/prompt-composition/validation identically to built-ins.
"""

from __future__ import annotations

import inspect
from pathlib import Path
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
from skills.loader import (
    load_custom_skills,
    load_markdown_skills,
    load_repo_markdown_skills,
)
from skills.markdown_skill import MarkdownSkill
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


def _all_skills(project_root: Optional[str] = None) -> List[Skill]:
    """Built-in skills plus every user-authored skill in scope.

    Three user tiers merge on top of the built-ins:

    * Python custom skills   (``~/.cgx/skills/*.py``)
    * global markdown skills (``~/.cgx/skills/<name>/SKILL.md``)
    * per-repo markdown skills (``<project_root>/.cgx/skills/``) -- only when
      ``project_root`` is supplied.

    Loading fails soft to just the built-ins on any error so a broken
    loader/directory never breaks detection or resolution.
    """
    try:
        skills = list(SKILLS) + load_custom_skills() + load_markdown_skills()
    except Exception:
        return list(SKILLS)
    if project_root:
        try:
            skills += load_repo_markdown_skills(project_root)
        except Exception:
            pass
    return skills


def _skill_format(s: Skill, builtin_names: set) -> str:
    """Classify a skill by its authoring format for the UI."""
    if s.name in builtin_names:
        return "builtin"
    if isinstance(s, MarkdownSkill):
        return "markdown"
    return "python"


def describe_skills(project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """UI-facing listing of every skill in scope.

    Each entry carries ``format`` (builtin/python/markdown), and, for
    markdown skills, the ``surfaces`` it targets, whether it is ``always_on``
    and its ``scope`` (global/repo) so the Skills tab can render all three
    tiers side by side.
    """
    builtin_names = {s.name for s in SKILLS}
    out: List[Dict[str, Any]] = []
    for s in _all_skills(project_root):
        fmt = _skill_format(s, builtin_names)
        entry: Dict[str, Any] = {
            "name": s.name,
            "role": s.role,
            "aliases": list(s.aliases),
            "description": getattr(s, "description", ""),
            "is_custom": s.name not in builtin_names,
            "format": fmt,
        }
        if fmt == "markdown":
            entry["surfaces"] = list(getattr(s, "surfaces", ()))
            entry["always_on"] = bool(getattr(s, "always_on", False))
            entry["scope"] = getattr(s, "scope", "global")
        out.append(entry)
    return out


def known_skill_names(exclude: Optional[str] = None,
                      project_root: Optional[str] = None) -> set:
    """Lower-cased set of every skill's name + aliases, for collision checks."""
    out: set = set()
    for s in _all_skills(project_root):
        if s.name == exclude:
            continue
        out.add(s.name.lower())
        out.update(a.lower() for a in s.aliases)
    return out


def read_skill_source(name: str) -> Optional[str]:
    """Return the Python source for any skill, built-in or custom.

    Custom skills are read straight from their file under
    ``~/.cgx/skills/``; built-ins are located via :func:`inspect` (their
    source lives in this package's own tree) so the Skills tab can show
    "how does this skill work?" for every entry, not just user-authored
    ones. Read-only either way -- editing/deleting is still restricted
    to custom skills at the route layer.
    """
    from skills.loader import read_custom_skill_source, read_markdown_skill_source
    custom = read_custom_skill_source(name)
    if custom is not None:
        return custom
    md = read_markdown_skill_source(name)
    if md is not None:
        return md
    for s in SKILLS:
        if s.name == name:
            try:
                path = inspect.getsourcefile(type(s))
                return Path(path).read_text(encoding="utf-8") if path else None
            except Exception:
                return None
    return None


def detect_skills(goal: str,
                  threshold: float = SKILL_DETECT_THRESHOLD,
                  project_root: Optional[str] = None) -> List[Skill]:
    """Return skills whose ``detect(goal)`` score meets ``threshold``.

    Results are sorted by descending detection confidence, then by a
    skill's ``priority`` (markdown skills only; default 0), so callers that
    need a "primary" skill can take the head of the list. ``always_on``
    markdown skills score 1.0 and therefore always lead. Pass
    ``project_root`` to also consider per-repo markdown skills.
    """
    pool = _all_skills(project_root)
    if not goal or not goal.strip():
        # always_on skills activate even for an empty goal (they are the
        # scoped equivalent of an always-loaded CGX.md).
        return [s for s in pool if getattr(s, "always_on", False)]
    scored: List[Tuple[Skill, float]] = []
    for s in pool:
        try:
            score = float(s.detect(goal))
        except Exception:
            score = 0.0
        if score >= threshold:
            scored.append((s, score))
    scored.sort(key=lambda x: (-x[1], -int(getattr(x[0], "priority", 0))))
    return [s for s, _ in scored]


def skill_names(skills: List[Skill]) -> List[str]:
    """Return the ``name`` of each skill in ``skills`` (preserves order)."""
    return [s.name for s in skills]


def skills_by_names(names: List[str],
                    project_root: Optional[str] = None) -> List[Skill]:
    """Resolve a list of skill ``name`` strings to ``Skill`` instances.

    Unknown names are silently skipped so a stale ``task.inputs['skills']``
    payload (carried across versions) doesn't crash a run. Order is
    preserved from ``names``.
    """
    if not names:
        return []
    lookup: Dict[str, Skill] = {s.name: s for s in _all_skills(project_root)}
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


def compose_ask_prompt(skills: List[Skill]) -> str:
    """Join non-empty ``ask_system_prompt`` fragments with blank lines.

    Used to inject knowledge/house-style skills into conversational
    surfaces (the Ask chatbot and read-only agent tasks). Each fragment is
    headed by the skill name so a weak model can tell them apart.
    """
    parts: List[str] = []
    for s in skills:
        frag = s.ask_system_prompt().strip()
        if frag:
            name = getattr(s, "name", "") or "skill"
            parts.append(f"### {name}\n{frag}")
    return "\n\n".join(parts)


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
    "MarkdownSkill",
    "Skill",
    "SkillVerdict",
    "collect_scaffold_warnings",
    "compose_ask_prompt",
    "compose_plan_prompt",
    "compose_scaffold_prompt",
    "describe_skills",
    "detect_skills",
    "file_paths",
    "file_with_content",
    "has_any_ext",
    "known_skill_names",
    "read_skill_source",
    "skill_names",
    "skills_by_names",
    "validate_plan",
    "validate_scaffold",
]
