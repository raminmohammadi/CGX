"""Core types for the CGX Skills system.

A *skill* is a self-contained module bundling everything CGX needs to
know about one technology (framework, runtime, library, toolchain).
Every skill answers three orthogonal questions:

1. **Does this goal involve me?** — :meth:`Skill.detect` returns a
   confidence in ``[0.0, 1.0]``. Scores above
   :data:`SKILL_DETECT_THRESHOLD` mark the skill as *active* for the
   current goal.
2. **What should the LLM know to do my job well?** —
   :meth:`Skill.scaffold_system_prompt` and
   :meth:`Skill.plan_system_prompt` return small, composable prompt
   fragments that get concatenated into the relevant system message.
3. **Did the produced output actually use me correctly?** —
   :meth:`Skill.validate_scaffold` and :meth:`Skill.validate_plan`
   inspect the produced ``diffs`` and return a :class:`SkillVerdict`
   (or ``None`` for "no opinion").

Skills live under the top-level :mod:`skills` package so they can be
discovered, audited, and extended without touching the agent layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Detection confidence at or above this is treated as "skill is active".
SKILL_DETECT_THRESHOLD: float = 0.5


# Valid roles. Multi-role skills are allowed (e.g. Next.js is both
# frontend and backend); store them as a tuple on the skill class.
ROLES: Tuple[str, ...] = (
    "frontend",   # UI / browser-rendered code
    "backend",    # server-side application code
    "fullstack",  # frameworks that span both (Next.js, Remix, ...)
    "data",       # databases, ORMs, persistence layers
    "cli",        # command-line tools / scripts
    "style",      # styling-only addons (Tailwind, Bootstrap, ...)
    "infra",      # build tooling, deploy config, containers
)


@dataclass
class SkillVerdict:
    """Validation outcome from a skill's check on produced diffs.

    Mirrors :class:`cgx.agents.judge.Verdict` so the Judge can pass the
    skill's verdict through verbatim. 
    
    ``passed=False`` with the default ``severity="error"`` short-circuits
    the Judge to FAIL with this skill's rationale; 
    
    ``severity="warning"`` is advisory only the Judge surfaces the rationale but does
    not fail the task on it.
    """

    passed: bool
    confidence: float
    rationale: str
    skill: str = ""  # filled by registry helpers when missing
    severity: str = "error"  # "error" (fatal) or "warning" (advisory)


class Skill(ABC):
    """Base class for a CGX skill.

    Subclasses set the class attributes (``name``, ``role``, ``aliases``)
    and override :meth:`detect`. Prompt fragments and validators are
    optional — override only the ones that make sense for the technology.
    """

    #: Stable, lower-snake identifier used in logs / task inputs.
    name: str = ""

    #: One of :data:`ROLES`, or a tuple of multiple roles for fullstack
    #: frameworks. Used by the registry for grouping and conflict
    #: resolution (e.g. only one "frontend" framework should fire).
    role: str = "infra"

    #: Optional surface-form aliases (display name + common
    #: misspellings). Only used for diagnostics / docs.
    aliases: Tuple[str, ...] = ()

    # ---- Required ----------------------------------------------------
    @abstractmethod
    def detect(self, goal: str) -> float:
        """Return ``[0.0, 1.0]`` confidence the goal involves this skill."""

    # ---- Optional: prompt composition --------------------------------
    def scaffold_system_prompt(self) -> str:
        """Prompt fragment to add when scaffolding a project."""
        return ""

    def plan_system_prompt(self) -> str:
        """Prompt fragment to add when planning a code change."""
        return ""

    # ---- Optional: post-generation validation ------------------------
    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        """Inspect the scaffold's diffs; ``None`` means no opinion."""
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        """Inspect a code-change plan's diffs; ``None`` means no opinion."""
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        """Return non-fatal advisory verdicts about the scaffold's diffs.

        Default implementation returns ``[]``. Override to flag soft
        quality issues (e.g. "no test files were generated") without
        failing the task. Each returned verdict should have
        ``severity="warning"`` and ``passed=False``.
        """
        return []

    # ---- Convenience helpers -----------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<Skill {self.name!r} role={self.role}>"


def file_paths(diffs: List[Dict[str, Any]]) -> List[str]:
    """Return the list of file paths from a ``diffs`` payload.

    Diffs use either ``{"file": ..., "patch": ...}`` (scaffold/plan) or
    ``{"path": ..., "diff": ...}`` (legacy). This helper handles both.
    """
    out: List[str] = []
    for d in diffs or []:
        if not isinstance(d, dict):
            continue
        p = str(d.get("file") or d.get("path") or "").strip()
        if p:
            out.append(p)
    return out


def has_any_ext(paths: List[str], exts: Tuple[str, ...]) -> bool:
    """Return ``True`` when any path in ``paths`` ends with one of ``exts``."""
    el = tuple(e.lower() for e in exts)
    return any(p.lower().endswith(el) for p in paths)


def file_with_content(diffs: List[Dict[str, Any]],
                      needle: str) -> Optional[str]:
    """Return the path of the first file whose patch contains ``needle``."""
    n = needle.lower()
    for d in diffs or []:
        if not isinstance(d, dict):
            continue
        body = str(d.get("patch") or d.get("diff") or "").lower()
        if n in body:
            return str(d.get("file") or d.get("path") or "")
    return None


def has_python_test_file(paths: List[str]) -> bool:
    """Return True when any path looks like a pytest-discoverable test."""
    for p in paths or []:
        pl = p.lower().replace("\\", "/")
        name = pl.rsplit("/", 1)[-1]
        if not name.endswith(".py"):
            continue
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if pl.startswith("tests/") and name != "__init__.py":
            return True
    return False


def has_js_test_file(paths: List[str]) -> bool:
    """Return True when any path looks like a Jest / Vitest test file."""
    for p in paths or []:
        pl = p.lower().replace("\\", "/")
        if pl.endswith((".test.js", ".test.jsx", ".test.ts", ".test.tsx",
                        ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx")):
            return True
        if pl.startswith("tests/") and pl.endswith(
            (".js", ".jsx", ".ts", ".tsx")
        ):
            return True
    return False
