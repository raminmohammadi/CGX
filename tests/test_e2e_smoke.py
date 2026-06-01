"""End-to-end smoke tests for the Planner → Tracker → Judge loop.

These exercise :func:`cgx.agents.run_agent` against a scripted LLM stub
so the full chain (skill detection → planner kind-policy → engine
prompt composition → tracker dispatch → judge structural validation)
runs without touching any real model, index, or network.

The scenarios mirror the documented happy paths in ``docs/usage.md``:

* multi-skill scaffold (React UI + FastAPI backend) with disk APPLY +
  VERIFY skipping because no tests are generated;
* single-skill scaffold (Django) that produces a manage.py;
* unsupported-tech scaffold goal (Tkinter) that still routes to SCAFFOLD
  via the ``_TECH_RE`` fallback even though no skill claims it;
* read-only Q&A goal that downgrades to ASK and never touches APPLY.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest

from cgx.agents import run_agent
from cgx.agents.types import AgentEvent, TaskKind, TaskStatus


class _ScriptedProvider:
    """Reply-script stub that records every call for assertions."""

    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self._replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, *args, **kw):  # noqa: ANN001 — match LLMProvider duck type
        # OllamaProvider.chat is called positionally or via messages kwarg.
        messages = kw.get("messages") if "messages" in kw else (args[0] if args else None)
        self.calls.append({"messages": messages,
                           "force_json": kw.get("force_json"),
                           "temperature": kw.get("temperature")})
        if not self._replies:
            return {"content": "", "error": "stub: replies exhausted"}
        return self._replies.pop(0)


def _events(stream) -> List[AgentEvent]:
    return list(stream)


# ---------------------------------------------------------------------------
# Scenario 1 — Multi-skill scaffold: React UI + FastAPI backend
# ---------------------------------------------------------------------------
def test_e2e_react_fastapi_calculator_scaffold(tmp_path):
    """*"create a React calculator with FastAPI backend"* runs end-to-end.

    New manifest-first flow: planner emits SCAFFOLD_MANIFEST + APPLY + VERIFY;
    the manifest call returns layers with paths/descriptions; the Tracker
    injects one SCAFFOLD_FILE task per file; each per-file call returns
    {"content": ...} and APPLY writes them to disk.
    """

    planner_reply = {"content": json.dumps({"tasks": [
        {"name": "UI", "description": "Generate React calculator UI",
         "kind": "scaffold", "criteria": ["renders calculator buttons"]},
        {"name": "Backend", "description": "Generate FastAPI compute endpoint",
         "kind": "scaffold", "criteria": ["POST /compute returns result"]},
    ]})}
    manifest_reply = {"content": json.dumps({
        "plan_md": "React UI + FastAPI backend calculator.",
        "layers": [
            {"name": "ui", "files": [
                {"path": "package.json", "description": "npm manifest with react dep"},
                {"path": "src/App.jsx", "description": "React App component"},
            ]},
            {"name": "backend", "files": [
                {"path": "backend/main.py", "description": "FastAPI app with POST /compute"},
                {"path": "backend/requirements.txt", "description": "fastapi+uvicorn"},
            ]},
        ],
    })}
    file_replies = [
        {"content": json.dumps({"content": '{"name":"calc","dependencies":{"react":"^18.0.0"}}'})},
        {"content": json.dumps({"content": "import React from 'react';\nexport default function App(){return <div>Calc</div>;}"})},
        {"content": json.dumps({"content": "from fastapi import FastAPI\napp = FastAPI()\n@app.post('/compute')\ndef compute():\n    return {'ok': True}\n"})},
        {"content": json.dumps({"content": "fastapi\nuvicorn\n"})},
    ]

    provider = _ScriptedProvider([planner_reply, manifest_reply, *file_replies])
    plan = run_agent(
        goal=("create a calculator project with React UI and FastAPI backend "
              "in which user can pass input and get the output"),
        provider=provider,
        project_root=str(tmp_path),
        stream=False, max_retries=0,
    )

    kinds = [t.kind for t in plan.tasks]
    assert kinds[0] == TaskKind.SCAFFOLD_MANIFEST
    assert kinds[-2:] == [TaskKind.APPLY, TaskKind.VERIFY]

    # Manifest must carry the goal-level skill set so the per-file generator
    # gets the React + FastAPI prompts.
    manifest_task = next(t for t in plan.tasks if t.kind == TaskKind.SCAFFOLD_MANIFEST)
    attached = manifest_task.inputs.get("skills") or []
    assert "react" in attached and "fastapi" in attached, attached

    # The Tracker injected one SCAFFOLD_FILE task per manifest file —
    # 4 from the LLM manifest plus a deterministic ``backend/__init__.py``
    # package-marker injected by ``_inject_python_package_inits`` so
    # pytest can resolve ``from backend.main import …`` on disk.
    file_tasks = [t for t in plan.tasks if t.kind == TaskKind.SCAFFOLD_FILE]
    task_names = [t.name for t in file_tasks]
    assert len(file_tasks) == 5, task_names
    assert "Generate backend/__init__.py" in task_names, task_names
    for t in file_tasks:
        assert t.status == TaskStatus.DONE, f"{t.name} failed: {t.error}"
        assert (t.output or {}).get("syntax_ok", True)

    apply_task = next(t for t in plan.tasks if t.kind == TaskKind.APPLY)
    assert apply_task.status == TaskStatus.DONE
    written = (apply_task.output or {}).get("applied_files") or []
    written_str = " ".join(written)
    assert "App.jsx" in written_str and "main.py" in written_str
    for rel in ("package.json", "src/App.jsx", "backend/main.py",
                "backend/requirements.txt", "backend/__init__.py"):
        assert (tmp_path / rel).exists(), f"missing written file: {rel}"
    assert "import React" in (tmp_path / "src/App.jsx").read_text()
    assert "FastAPI" in (tmp_path / "backend/main.py").read_text()


# ---------------------------------------------------------------------------
# Scenario 2 — Read-only goal must NOT scaffold or apply
# ---------------------------------------------------------------------------
def test_e2e_readonly_goal_downgrades_to_ask_only():
    """A plain question never triggers SCAFFOLD/APPLY even with no index."""

    plan = run_agent(
        goal="What does the Planner module do?",
        provider=None,  # forces deterministic fallback
        project_root=None,
        stream=False, max_retries=0,
    )
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.ASK]
    # No project_root + no provider means ASK fails fast — but the kind
    # itself must be ASK; the failure is allowed.
    assert TaskKind.APPLY not in kinds and TaskKind.SCAFFOLD not in kinds


# ---------------------------------------------------------------------------
# Scenario 3 — Tech-paired scaffold goal still routes via _TECH_RE fallback
# ---------------------------------------------------------------------------
def test_e2e_unsupported_tech_routes_to_scaffold_via_regex():
    """Tkinter has no skill but the verb+tech regex still catches it."""

    # Provider=None forces deterministic plan: single SCAFFOLD task + APPLY+VERIFY.
    plan = run_agent(
        goal="Create a Tkinter desktop app for converting CSV files to JSON",
        provider=None, project_root=None,
        stream=False, max_retries=0,
    )
    kinds = [t.kind for t in plan.tasks]
    assert kinds[0] == TaskKind.SCAFFOLD_MANIFEST
    assert kinds[-2:] == [TaskKind.APPLY, TaskKind.VERIFY]
    # The manifest task carries an empty skills list (no skill claims tkinter)
    # but the goal text is preserved for the engine.
    sc_task = plan.tasks[0]
    assert "tkinter" in sc_task.inputs.get("goal", "").lower()


# ---------------------------------------------------------------------------
# Scenario 4 — Goal-level skill detection on the manifest task
# ---------------------------------------------------------------------------
def test_e2e_planner_attaches_goal_skills_to_manifest():
    """A React+FastAPI scaffold goal is collapsed to a single
    SCAFFOLD_MANIFEST task carrying both skills (the per-file generator
    consumes that list)."""

    planner_reply = {"content": json.dumps({"tasks": [
        {"name": "UI", "description": "Generate React calculator UI",
         "kind": "scaffold"},
        {"name": "Backend", "description": "Generate FastAPI compute endpoint",
         "kind": "scaffold"},
    ]})}
    from cgx.agents import Planner
    provider = _ScriptedProvider([planner_reply])
    plan = Planner(provider=provider).plan(
        "create a calculator project with React UI and FastAPI backend")

    manifests = [t for t in plan.tasks if t.kind == TaskKind.SCAFFOLD_MANIFEST]
    assert len(manifests) == 1
    skills = manifests[0].inputs.get("skills") or []
    assert "react" in skills and "fastapi" in skills, skills


def test_e2e_planner_falls_back_to_goal_skills_when_task_desc_is_generic():
    """A scaffold goal with a non-tech task description must still surface
    the goal-level skill set on the SCAFFOLD_MANIFEST task."""

    planner_reply = {"content": json.dumps({"tasks": [
        {"name": "Generic", "description": "Build the project",
         "kind": "scaffold"},
    ]})}
    from cgx.agents import Planner
    provider = _ScriptedProvider([planner_reply])
    plan = Planner(provider=provider).plan(
        "create a Vue dashboard with SQLite storage")

    sc = [t for t in plan.tasks if t.kind == TaskKind.SCAFFOLD_MANIFEST][0]
    skills_attached = sc.inputs.get("skills") or []
    assert "vue" in skills_attached
    assert "sqlite" in skills_attached


# ---------------------------------------------------------------------------
# Scenario 5 — Change-goal: PLAN/APPLY/VERIFY chain with stub capabilities
# ---------------------------------------------------------------------------
def test_e2e_change_goal_runs_plan_apply_verify(tmp_path):
    """Change goal "add a function" runs the full plan→apply→verify chain
    using stub capabilities so no LLM or index is needed."""

    (tmp_path / "mymod.py").write_text("def add(a, b):\n    return a + b\n")

    def plan_cap(_text, **_kw):
        return {"plan_md": "Add mul().",
                "diffs": [{"file": "mymod.py",
                           "patch": ("--- a/mymod.py\n+++ b/mymod.py\n"
                                     "@@ -1,2 +1,4 @@\n def add(a, b):\n"
                                     "     return a + b\n+def mul(a, b):\n"
                                     "+    return a * b\n")}]}

    def apply_cap(prior, **_kw):
        from cgx.codegen.disk_apply import apply_diffs_to_disk
        diffs: List[Dict[str, Any]] = []
        for o in prior or []:
            diffs.extend(o.get("diffs") or [])
        return apply_diffs_to_disk(str(tmp_path), diffs)

    def verify_cap(_prior, **_kw):
        return {"ran": True, "tests_passed": True, "returncode": 0,
                "tests_selected": [], "stdout": "ok",
                "stderr": "", "skipped_reason": None, "mode": "impacted"}

    caps = {"plan": plan_cap, "apply": apply_cap, "verify": verify_cap}
    plan = run_agent(
        goal="Add a mul() function to mymod.py",
        provider=None, project_root=str(tmp_path),
        capabilities=caps, stream=False, max_retries=0,
    )
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY]
    statuses = [t.status for t in plan.tasks]
    assert statuses == [TaskStatus.DONE] * 3
    assert "def mul" in (tmp_path / "mymod.py").read_text()


# ---------------------------------------------------------------------------
# Scenario 6 — Streaming run yields ordered AgentEvent timeline
# ---------------------------------------------------------------------------
def test_e2e_streaming_emits_ordered_event_types(tmp_path):
    """``stream=True`` yields plan → task_start* → task_done* → summary."""

    def ask_cap(_q, **_kw):
        return {"answer_md": "stubbed"}

    events = _events(run_agent(
        goal="What is run_agent?",
        provider=None, project_root=None,
        capabilities={"ask": ask_cap},
        stream=True, max_retries=0,
    ))
    types = [e.type for e in events]
    assert types[0] == "plan"
    assert "task_start" in types
    assert "task_done" in types
    assert types[-1] == "summary"


# ---------------------------------------------------------------------------
# Scenario 7 — Per-file scaffold tasks coordinate file layout
# ---------------------------------------------------------------------------
def test_e2e_per_file_scaffolds_share_layout_and_strip_project_prefix(tmp_path):
    """Stray project-name prefixes (``calculator/``) emitted by the LLM
    inside the manifest are normalised to the canonical root, and each
    SCAFFOLD_FILE prompt sees the content of all prior files generated in
    the same run, so APPLY writes one coherent tree."""

    planner_reply = {"content": json.dumps({"tasks": [
        {"name": "UI", "description": "Generate React calculator UI",
         "kind": "scaffold"},
        {"name": "Backend", "description": "Generate FastAPI compute endpoint",
         "kind": "scaffold"},
    ]})}
    # Manifest emits a project-name prefix that must be stripped before the
    # per-file generator runs.
    manifest_reply = {"content": json.dumps({
        "plan_md": "React + FastAPI calculator.",
        "layers": [
            {"name": "ui", "files": [
                {"path": "calculator/package.json", "description": "npm manifest"},
                {"path": "calculator/src/App.jsx", "description": "React App"},
            ]},
            {"name": "backend", "files": [
                {"path": "backend/main.py", "description": "FastAPI app"},
                {"path": "pyproject.toml", "description": "Python project metadata"},
            ]},
        ],
    })}
    file_replies = [
        {"content": json.dumps({"content": '{"name":"calc"}'})},
        {"content": json.dumps({"content": "import React from 'react';\nexport default function App(){return <div/>;}"})},
        {"content": json.dumps({"content": "from fastapi import FastAPI\napp = FastAPI()\n"})},
        {"content": json.dumps({"content": "[project]\nname = 'calc'\n"})},
    ]
    provider = _ScriptedProvider([planner_reply, manifest_reply, *file_replies])
    plan = run_agent(
        goal="create a calculator with React UI and FastAPI backend",
        provider=provider, project_root=str(tmp_path),
        stream=False, max_retries=0,
    )

    file_tasks = [t for t in plan.tasks if t.kind == TaskKind.SCAFFOLD_FILE]
    paths = [(t.output or {}).get("file") for t in file_tasks]
    # Project-name prefix stripped on the per-file generator path.
    assert "package.json" in paths and "src/App.jsx" in paths, paths
    assert all(not (p or "").startswith("calculator/") for p in paths), paths

    # The 3rd per-file call (backend/main.py) sees the two UI files'
    # content as already-generated context.
    backend_user_msg = provider.calls[4]["messages"][1]["content"]
    assert "ALREADY GENERATED FILES" in backend_user_msg
    assert "src/App.jsx" in backend_user_msg
    assert "package.json" in backend_user_msg

    # APPLY wrote one coherent tree (no nested ``calculator/`` folder).
    for rel in ("package.json", "src/App.jsx", "backend/main.py"):
        assert (tmp_path / rel).exists(), f"missing: {rel}"
    assert not (tmp_path / "calculator").exists()
