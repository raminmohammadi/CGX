"""Tests for the SQLiteSkill: broadened persistence detection, the data-owner
circular-import validator, and parameterized-query / relative-path warnings."""

from __future__ import annotations

import pytest

import skills as registry
from skills.base import SKILL_DETECT_THRESHOLD
from skills.sqlite import SQLiteSkill


@pytest.fixture()
def skill() -> SQLiteSkill:
    return SQLiteSkill()


def _diff(path, patch=""):
    return {"file": path, "patch": patch}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "store data in sqlite",
    "use sqlite3 for persistence",
    "an embedded SQLite database",
])
def test_detects_literal_sqlite(skill, goal):
    assert skill.detect(goal) == 0.9


@pytest.mark.parametrize("goal", [
    "an app to store user accounts",
    "track members and their orders",
    "persist records to disk",
    "a database of inventory items",
    "manage users and their data",
])
def test_detects_persistence_intent(skill, goal):
    score = skill.detect(goal)
    assert score == pytest.approx(0.55)
    assert score >= SKILL_DETECT_THRESHOLD  # activates


@pytest.mark.parametrize("goal", [
    "store user accounts in postgres",
    "persist members to a mysql database",
    "keep orders in mongodb",
    "cache records in redis",
    "an inventory service backed by PostgreSQL",
])
def test_does_not_steal_when_competing_engine_named(skill, goal):
    # A named competing engine wins -- sqlite must abstain (high precision)...
    assert skill.detect(goal) == 0.0


def test_explicit_sqlite_wins_even_with_other_engine(skill):
    # ...unless the user explicitly asked for sqlite too.
    assert skill.detect("migrate from postgres to sqlite") == 0.9


@pytest.mark.parametrize("goal", [
    "what does the auth module do?",
    "build a calculator UI",
    "restore a backup archive",  # 'restore' must NOT match the 'store' token
    "",
])
def test_no_false_positive_detection(skill, goal):
    assert skill.detect(goal) == 0.0


def test_registry_activates_on_persistence_but_not_competing_engine():
    persist = [s.name for s in registry.detect_skills("store user accounts")]
    assert "sqlite" in persist
    mongo = [s.name for s in registry.detect_skills(
        "store user accounts in mongodb")]
    assert "sqlite" not in mongo
    # An unrelated goal still yields nothing (locks in existing contract).
    assert registry.detect_skills("what does the auth module do?") == []


# --- prompt content --------------------------------------------------------

def test_scaffold_prompt_teaches_own_module_and_no_backimport(skill):
    p = skill.scaffold_system_prompt().lower()
    assert "db.py" in p and "extensions.py" in p
    assert "circular import" in p
    assert "from app import db" in p
    assert "own module" in p


def test_scaffold_prompt_teaches_params_and_relative_path(skill):
    p = skill.scaffold_system_prompt().lower()
    assert "?" in p and "placeholder" in p
    assert "relative" in p
    assert "never an absolute path" in p


# --- scaffold validation: circular import (data-owner half) ----------------

def test_validate_scaffold_rejects_backimport_of_db(skill):
    diffs = [
        _diff("app.py", "import sqlite3\napp = object()\n"),
        _diff("routes/members.py",
              "from app import db\n\ndef list_members():\n    return db\n"),
    ]
    v = skill.validate_scaffold(diffs, goal="store members in sqlite")
    assert v is not None and not v.passed and v.severity == "error"
    assert "circular import" in v.rationale
    assert "routes/members.py" in v.rationale


def test_validate_scaffold_rejects_backimport_of_get_db(skill):
    diffs = [
        _diff("db.py", "import sqlite3\n"),
        _diff("main.py", "import sqlite3\n"),
        _diff("routes.py", "from main import get_db\n"),
    ]
    v = skill.validate_scaffold(diffs, goal="users store")
    assert v is not None and not v.passed
    assert "circular import" in v.rationale


def test_validate_scaffold_allows_import_from_own_db_module(skill):
    # Importing the handle from a dedicated db module is the SANCTIONED layout.
    diffs = [
        _diff("db.py", "import sqlite3\n\ndef get_db():\n    "
                       "return sqlite3.connect('./data/app.db')\n"),
        _diff("app.py", "from db import get_db\n"),
        _diff("routes/members.py",
              "from db import get_db\n\ndef list_members():\n    "
              "return get_db().execute('SELECT * FROM members').fetchall()\n"),
    ]
    assert skill.validate_scaffold(diffs, goal="store members in sqlite") is None


# --- scaffold validation: DB-library presence ------------------------------

def test_validate_scaffold_fatal_when_sqlite_named_but_no_db_lib(skill):
    diffs = [_diff("main.py", "print('hello')\n")]
    v = skill.validate_scaffold(diffs, goal="build a sqlite todo app")
    assert v is not None and not v.passed
    assert "sqlite3" in v.rationale or "sqlalchemy" in v.rationale


def test_validate_scaffold_lenient_for_generic_persistence_goal(skill):
    # Persistence goal that did NOT name sqlite -> the model may have chosen a
    # different store; do not fatally reject (high precision).
    diffs = [_diff("main.py", "DATA = {}\nprint('hello')\n")]
    assert skill.validate_scaffold(diffs, goal="store user accounts") is None


def test_validate_scaffold_none_on_empty(skill):
    assert skill.validate_scaffold([], goal="store users in sqlite") is None


def test_validate_scaffold_passes_clean_sqlite_scaffold(skill):
    diffs = [
        _diff("db.py", "import sqlite3\n\ndef get_db():\n    "
                       "return sqlite3.connect('./data/app.db')\n"),
        _diff("app.py", "from db import get_db\n"),
    ]
    assert skill.validate_scaffold(diffs, goal="build a sqlite app") is None


# --- warnings: parameterized queries + relative path -----------------------

def test_warns_on_fstring_sql(skill):
    diffs = [_diff("db.py",
                   "import sqlite3\n"
                   "def find(uid):\n"
                   "    c = sqlite3.connect('./app.db')\n"
                   "    return c.execute(f'SELECT * FROM t WHERE id={uid}')\n")]
    warns = skill.scaffold_warnings(diffs)
    assert any("injection" in w.rationale.lower() for w in warns)
    assert all(w.severity == "warning" for w in warns)


def test_warns_on_percent_formatted_sql(skill):
    diffs = [_diff("db.py",
                   "c.execute('SELECT * FROM t WHERE id=%s' % uid)\n")]
    warns = skill.scaffold_warnings(diffs)
    assert any("injection" in w.rationale.lower() for w in warns)


def test_no_warning_for_parameterized_query(skill):
    diffs = [_diff("db.py",
                   "import sqlite3\n"
                   "c = sqlite3.connect('./data/app.db')\n"
                   "c.execute('SELECT * FROM t WHERE id = ?', (uid,))\n")]
    assert skill.scaffold_warnings(diffs) == []


def test_warns_on_absolute_db_path(skill):
    diffs = [_diff("db.py", "sqlite3.connect('/var/lib/app/app.db')\n")]
    warns = skill.scaffold_warnings(diffs)
    assert any("absolute" in w.rationale.lower() for w in warns)


def test_warns_on_absolute_sqlalchemy_url(skill):
    diffs = [_diff("db.py",
                   "engine = create_engine('sqlite:////var/lib/app.db')\n")]
    warns = skill.scaffold_warnings(diffs)
    assert any("absolute" in w.rationale.lower() for w in warns)


def test_no_warning_for_relative_paths(skill):
    diffs = [
        _diff("db.py",
              "import sqlite3\n"
              "conn = sqlite3.connect('./data/app.db')\n"
              "engine = create_engine('sqlite:///./data/app.db')\n"
              "conn.execute('SELECT 1')\n"),
    ]
    assert skill.scaffold_warnings(diffs) == []
