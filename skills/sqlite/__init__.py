

"""SQLite persistence skill.

An addon skill that composes with whatever backend skill is also
active. It contributes guidance for using the stdlib ``sqlite3``
module (or SQLAlchemy when explicitly requested) and validates that
the scaffold actually wires up a database file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import Skill, SkillVerdict, file_with_content


_SQLITE_RE = re.compile(r"\bsqlite(?:3)?\b", re.IGNORECASE)


class SQLiteSkill(Skill):
    name = "sqlite"
    role = "data"
    aliases = ("SQLite", "sqlite3")

    def detect(self, goal: str) -> float:
        if _SQLITE_RE.search(goal or ""):
            return 0.9
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "DATA -- SQLite persistence\n"
            "- Use the stdlib `sqlite3` module unless the goal explicitly "
            "asks for an ORM (SQLAlchemy / Django ORM / Tortoise).\n"
            "- Put schema setup (CREATE TABLE IF NOT EXISTS ...) in a "
            "single `init_db()` function the application calls at startup; "
            "don't sprinkle CREATE statements across modules.\n"
            "- Parameterise every query with `?` placeholders -- never "
            "string-format user input into SQL.\n"
            "- Use `with sqlite3.connect(path) as conn:` for transaction "
            "scoping, or an explicit `conn.commit()` after writes.\n"
            "- Default database file at `./data/app.db` (mkdir the data "
            "directory at startup if absent)."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying SQLite-backed code:\n"
            "- Schema changes go in `init_db()` (idempotent CREATE … IF "
            "NOT EXISTS / ALTER TABLE) -- keep them backward-compatible.\n"
            "- All new queries must use `?` placeholders."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        if not diffs:
            return None
        if file_with_content(diffs, "sqlite3") is None \
                and file_with_content(diffs, "sqlalchemy") is None:
            return SkillVerdict(
                passed=False, confidence=0.75,
                rationale=("SQLite skill: no file imports `sqlite3` or "
                           "`sqlalchemy`. Wire up a connection in the "
                           "backend module."),
            )
        return None


__all__ = ["SQLiteSkill"]
