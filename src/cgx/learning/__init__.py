"""CGX learning layer -- the self-improving loop beside the deterministic core.

Layer 1 (in ``skills`` + ``swarm_verify``) fixes the mechanical, 100%-decidable
bugs with certainty. This package is Layer 2: when a run hits a failure Layer 1
does not recognise, a bounded post-mortem distils a *generalised lesson* from the
diagnosis and records it -- to the project (``.cgx/lessons.md``) for project
quirks and to the skill (``skills/<name>/lessons.md``) for stack-level gotchas
that condition every future run of that technology. Lessons are append-only,
timestamped, traced, and reversible (delete the JSONL entry, re-render).
"""

from __future__ import annotations

from cgx.learning.lessons import (
    project_lessons_text,
    prune_lesson,
    read_lessons,
    record_lesson,
    skill_lessons_text,
)

__all__ = [
    "project_lessons_text",
    "prune_lesson",
    "read_lessons",
    "record_lesson",
    "skill_lessons_text",
]
