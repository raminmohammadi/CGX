"""Tests for the FastAPISkill: detection, prompt content, and the scaffold /
plan validators (real FastAPI() instance, the router<->main circular import,
undeclared sqlalchemy, and the routers-need-an-entrypoint plan rule)."""

from __future__ import annotations

import pytest

import skills as registry
from skills.fastapi import FastAPISkill


@pytest.fixture()
def skill() -> FastAPISkill:
    return FastAPISkill()


def _diff(path, content=""):
    return {"path": path, "content": content}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a FastAPI service for todos",
    "a fast api backend with sqlalchemy",
    "create a calculator app using FastAPI and a react frontend",
])
def test_detects_fastapi_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_abstains_on_ambiguous_backend(skill):
    assert skill.detect("build a python backend") == 0.0
    assert skill.detect("") == 0.0


def test_registered_and_detected_via_registry():
    detected = [s.name for s in registry.detect_skills("build a FastAPI api")]
    assert "fastapi" in detected


# --- prompt content --------------------------------------------------------

def test_scaffold_prompt_teaches_router_split_and_deps(skill):
    p = skill.scaffold_system_prompt()
    assert "include_router" in p
    assert "APIRouter" in p
    assert "circular import" in p
    # steer shared deps to db.py, not back-imported from main
    assert "db.py" in p or "database.py" in p
    assert "pydantic" in p and "uvicorn" in p


def test_plan_prompt_teaches_entrypoint_and_no_backimport(skill):
    p = skill.plan_system_prompt()
    assert "main.py" in p
    assert "include_router" in p
    assert "circular import" in p


# --- scaffold validation: happy path --------------------------------------

def _sanctioned_diffs():
    return [
        _diff("backend/db.py",
              "from sqlalchemy import create_engine\n"
              "from sqlalchemy.orm import sessionmaker, declarative_base\n"
              "engine = create_engine('sqlite:///./app.db')\n"
              "SessionLocal = sessionmaker(bind=engine)\n"
              "Base = declarative_base()\n"
              "def get_db():\n    db = SessionLocal()\n    yield db\n"),
        _diff("backend/models.py",
              "from pydantic import BaseModel\n"
              "class Item(BaseModel):\n    name: str\n"),
        _diff("backend/routers/items.py",
              "from fastapi import APIRouter, Depends\n"
              "from backend.db import get_db\n"
              "router = APIRouter(prefix='/items')\n"
              "@router.get('/')\ndef list_items():\n    return []\n"),
        _diff("backend/main.py",
              "from fastapi import FastAPI\n"
              "from backend.routers.items import router as items_router\n"
              "app = FastAPI()\napp.include_router(items_router)\n"),
        _diff("requirements.txt", "fastapi\nuvicorn[standard]\npydantic\nsqlalchemy\n"),
        _diff("tests/test_main.py",
              "from fastapi.testclient import TestClient\n"
              "from backend.main import app\nclient = TestClient(app)\n"),
    ]


def test_validate_scaffold_passes_sanctioned_layout(skill):
    assert skill.validate_scaffold(_sanctioned_diffs(), goal="items") is None


# --- scaffold validation: fatal checks ------------------------------------

def test_no_python_files_rejected(skill):
    v = skill.validate_scaffold([_diff("README.md", "# hi")])
    assert v is not None and not v.passed and "Python" in v.rationale


def test_import_only_is_not_a_real_instance(skill):
    # The word 'fastapi'/'FastAPI' is present but nothing constructs the app.
    diffs = [
        _diff("backend/main.py", "from fastapi import FastAPI\n# TODO build app\n"),
        _diff("requirements.txt", "fastapi\nuvicorn\npydantic\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "FastAPI application instance" in v.rationale


def test_circular_import_router_imports_app_from_main(skill):
    diffs = [
        _diff("backend/main.py",
              "from fastapi import FastAPI\n"
              "from backend.routers.items import router\n"
              "app = FastAPI()\napp.include_router(router)\n"),
        _diff("backend/routers/items.py",
              "from fastapi import APIRouter\n"
              "from backend.main import app\n"
              "router = APIRouter()\n"),
        _diff("requirements.txt", "fastapi\nuvicorn\npydantic\n"),
    ]
    v = skill.validate_scaffold(diffs, goal="items")
    assert v is not None and not v.passed and "circular import" in v.rationale


def test_undeclared_sqlalchemy_rejected(skill):
    diffs = [
        _diff("backend/main.py",
              "from fastapi import FastAPI\n"
              "from sqlalchemy import create_engine\n"
              "app = FastAPI()\nengine = create_engine('sqlite://')\n"),
        _diff("requirements.txt", "fastapi\nuvicorn\npydantic\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "sqlalchemy" in v.rationale


def test_pydantic_undeclared_is_not_fatal(skill):
    # pydantic ships with fastapi, so importing it while only fastapi is
    # pinned still works -- must NOT be a fatal reject (high precision).
    diffs = [
        _diff("backend/main.py",
              "from fastapi import FastAPI\n"
              "from pydantic import BaseModel\n"
              "app = FastAPI()\n"),
        _diff("requirements.txt", "fastapi\nuvicorn\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_missing_requirements_rejected(skill):
    diffs = [
        _diff("backend/main.py", "from fastapi import FastAPI\napp = FastAPI()\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "requirements.txt" in v.rationale


# --- plan validation -------------------------------------------------------

def test_plan_flags_routers_without_entrypoint(skill):
    v = skill.validate_plan([
        _diff("backend/routers/items.py"),
        _diff("backend/routers/users.py"),
        _diff("backend/models.py"),
    ])
    assert v is not None and not v.passed and "main.py" in v.rationale


def test_plan_ok_when_main_present(skill):
    assert skill.validate_plan([
        _diff("backend/main.py"),
        _diff("backend/routers/items.py"),
    ]) is None


def test_plan_ok_when_no_routers(skill):
    # A single-module app (routes inline on the app) needs no routers dir.
    assert skill.validate_plan([_diff("app/service.py")]) is None


def test_plan_none_on_empty(skill):
    assert skill.validate_plan([]) is None


# --- warnings --------------------------------------------------------------

def test_warns_when_no_test_file(skill):
    warns = skill.scaffold_warnings([
        _diff("backend/main.py", "from fastapi import FastAPI\napp = FastAPI()\n"),
    ])
    assert warns and warns[0].severity == "warning"


def test_no_warning_when_test_present(skill):
    assert skill.scaffold_warnings([
        _diff("backend/main.py", "from fastapi import FastAPI\napp = FastAPI()\n"),
        _diff("tests/test_main.py", "def test_x():\n    assert True\n"),
    ]) == []
