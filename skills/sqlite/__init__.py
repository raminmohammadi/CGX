

"""SQLite persistence skill.

An addon skill that composes with whatever backend skill is also
active. It contributes guidance for using the stdlib ``sqlite3``
module (or SQLAlchemy when explicitly requested) and validates that
the scaffold actually wires up a database file -- and that the db
handle lives in its own module rather than being created in, or
imported back from, the application entrypoint (the data-owner half
of the classic Flask circular-import failure).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import any_body_matches, app_backimports
from skills.base import Skill, SkillVerdict, file_with_content

_SQLITE_RE = re.compile(r"\bsqlite(?:3)?\b", re.IGNORECASE)

# Generic persistence intent: a goal that clearly wants to keep data around but
# names no engine. We take these (embedded SQLite is the sensible default) so
# the DB structural guidance loads.
_PERSIST_RE = re.compile(
    r"\b(?:members?|orders?|stores?|stored|persist(?:s|ed|ing|ence)?|"
    r"databases?|records?|users?|accounts?|inventory|inventories)\b",
    re.IGNORECASE,
)

# A competing engine named in the goal means it belongs to that engine's skill;
# never steal it (keep detection high-precision).
_COMPETING_ENGINE_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongo(?:db)?|redis)\b",
    re.IGNORECASE,
)

# The db handle a leaf module must never import *back* from the app entrypoint
# (app.py / main.py / server.py / wsgi.py / asgi.py). Restricted to db-flavoured
# names so the shared ``app_backimports`` check stays high precision.
_DB_APP_OBJECTS = (
    "db", "engine", "conn", "connection", "SessionLocal",
    "get_db", "get_connection", "get_db_connection",
)

# SQL built by string formatting and handed to execute() -- an injection risk
# that the `?`-placeholder rule exists to prevent.
_SQL_INJECT_RE = (
    r"\bexecute(?:script|many)?\s*\(\s*f['\"]"                          # f-string
    r"|\bexecute(?:script|many)?\s*\(\s*['\"][^'\"]*['\"]\s*%"          # %-format
    r"|\bexecute(?:script|many)?\s*\(\s*['\"][^'\"]*['\"]\s*\.format\("  # .format()
)

# An absolute database path (breaks portability -- the file should live relative
# to the project). Covers both sqlite3.connect("/...") and a SQLAlchemy
# ``sqlite:////abs`` URL (four slashes == absolute).
_ABS_DB_PATH_RE = (
    r"connect\(\s*['\"](?:/|[A-Za-z]:[\\/])"
    r"|sqlite:////"
)


class SQLiteSkill(Skill):
    name = "sqlite"
    role = "data"
    aliases = ("SQLite", "sqlite3")
    description = "SQLite persistence layer / embedded database access."

    def detect(self, goal: str) -> float:
        text = goal or ""
        # Explicit SQLite mention -- highest confidence (unchanged behaviour).
        if _SQLITE_RE.search(text):
            return 0.9
        # A goal that names another engine belongs to that engine's skill.
        if _COMPETING_ENGINE_RE.search(text):
            return 0.0
        # Generic persistence intent, no engine named -> load DB guidance with a
        # sensible embedded default.
        if _PERSIST_RE.search(text):
            return 0.55
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "DATA -- SQLite persistence\n"
            "- Use the stdlib `sqlite3` module unless the goal explicitly "
            "asks for an ORM (SQLAlchemy / Django ORM / Tortoise).\n"
            "- STRUCTURE (data-owner half of the classic Flask circular "
            "import): the connection/engine lives in its OWN module -- a "
            "`sqlite3` `get_db()` helper in `db.py`, or `db = SQLAlchemy()` / "
            "the SQLAlchemy `engine` in `db.py` or `extensions.py`. NEVER "
            "create the connection/engine in the app entrypoint "
            "(app.py/main.py/server.py/wsgi.py), and a route/model module must "
            "NEVER do `from app import db` (or import the connection/engine "
            "back from the app module) -- that is a circular import. Import the "
            "handle from `db`/`extensions` instead.\n"
            "- Put schema setup (CREATE TABLE IF NOT EXISTS ...) in a "
            "single `init_db()` function the application calls at startup; "
            "don't sprinkle CREATE statements across modules.\n"
            "- Parameterise every query with `?` placeholders -- never "
            "string-format (f-string / % / .format) user input into SQL.\n"
            "- Use `with sqlite3.connect(path) as conn:` for transaction "
            "scoping, or an explicit `conn.commit()` after writes.\n"
            "- Store the database file at a path RELATIVE to the project "
            "(default `./data/app.db`, or `sqlite:///./data/app.db` for a "
            "SQLAlchemy URL) -- never an absolute path like `/var/...` or "
            "`sqlite:////...`. You MUST import `os` and create the parent "
            "directory first (e.g. `import os; "
            "os.makedirs('./data', exist_ok=True)`) inside `init_db()`.\n"
            "- IMPORTANT: Do NOT call `init_db()` at the top-level of the module. Only call it "
            "inside `if __name__ == '__main__':` or let the main application entrypoint call it."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying SQLite-backed code:\n"
            "- Keep the connection/engine in its own module (db.py / "
            "extensions.py); routes/models import it from there and never "
            "import it back from the app entrypoint (a circular import).\n"
            "- Schema changes go in `init_db()` (idempotent CREATE … IF "
            "NOT EXISTS / ALTER TABLE) -- keep them backward-compatible.\n"
            "- All new queries must use `?` placeholders."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        if not diffs:
            return None
        # THE data-owner failure: a leaf module importing the db handle/engine/
        # connection back from the app entrypoint = circular import (breaks even
        # pytest collection). High precision via the shared structural check.
        cyc = app_backimports(diffs, app_objects=_DB_APP_OBJECTS)
        if cyc:
            worst = cyc[0]
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=(f"SQLite skill: circular import -- {worst['file']} "
                           f"does `from ...{worst['from_module']} import "
                           f"{worst['imported']}`. The db "
                           "connection/engine must live in its own module "
                           "(db.py / extensions.py) and be imported from "
                           "there -- never created in, nor imported back from, "
                           "the app entrypoint."),
            )
        # No DB wiring at all. Only FATAL when the goal explicitly asked for
        # SQLite; a generic persistence goal may legitimately use a different
        # store, so we do not hard-reject it here (stay high precision).
        if file_with_content(diffs, "sqlite3") is None \
                and file_with_content(diffs, "sqlalchemy") is None \
                and _SQLITE_RE.search(goal or ""):
            return SkillVerdict(
                passed=False, confidence=0.75,
                rationale=("SQLite skill: no file imports `sqlite3` or "
                           "`sqlalchemy`. Wire up a connection in a dedicated "
                           "db.py module."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        warns: List[SkillVerdict] = []
        if any_body_matches(diffs, _SQL_INJECT_RE, only_ext=(".py",)):
            warns.append(SkillVerdict(
                passed=False, confidence=0.7, severity="warning",
                rationale=("SQLite skill: SQL built with an f-string / % / "
                           ".format and passed to execute() -- an injection "
                           "risk. Use `?` placeholders and pass the values as "
                           "execute()'s second argument."),
            ))
        if any_body_matches(diffs, _ABS_DB_PATH_RE, only_ext=(".py",)):
            warns.append(SkillVerdict(
                passed=False, confidence=0.65, severity="warning",
                rationale=("SQLite skill: the database file uses an absolute "
                           "path. Store it relative to the project (e.g. "
                           "`./data/app.db`) so the app stays portable."),
            ))
        return warns


__all__ = ["SQLiteSkill"]
