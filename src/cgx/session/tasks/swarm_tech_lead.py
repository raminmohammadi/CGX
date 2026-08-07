"""The Tech Lead executor: author and validate a swarm WORK_PLAN.

The first cut let the Tech Lead free-form a prose "delegation" that the
Developer then interpreted however it liked -- the largest source of drift in
the loop. This rewrite makes the Tech Lead a *planner* under the
propose-then-validate discipline: it prompts the model for a draft JSON plan,
then deterministic invariants take over -- :func:`normalize_plan` dedupes files
and prunes dangling ``depends_on`` edges, the shared manifest toposort orders
the files dependency-first, and :func:`verify_plan` rejects a plan that could
not build coherently (unsafe paths, no runnable source, a mixed import rooting,
a dependency cycle, or an orphan test) with the exact problems fed back for one
corrective re-ask. The validated plan is persisted as a ``WORK_PLAN`` artifact
and the router drives the Developer over its ordered paths one file per turn.
"""

from __future__ import annotations

from typing import Any, Dict, List

from cgx.session.models import Artifact, ArtifactKind, TaskKind
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.swarm_log import swarm_beat
from cgx.session.tasks.swarm_plan import (
    ensure_scaffolding, ensure_test_coverage, normalize_plan, ordered_paths,
    parse_plan_reply, verify_plan)

# Bounded re-asks: the model gets one corrective retry per failure mode
# (unparseable / not buildable) before the plan is declared a dead end.
_MAX_PLAN_ATTEMPTS = 3

_SYSTEM_PROMPT = (
    "You are the Tech Lead in a two-agent swarm. You do NOT write code.\n"
    "You author a build PLAN as a single JSON object and nothing else.\n\n"
    "Schema:\n"
    "{\n"
    '  "goal": "one sentence restating the objective",\n'
    '  "layers": [\n'
    '    {"name": "models|core|api|tests|...",\n'
    '     "files": [\n'
    '       {"path": "src/foo.py",\n'
    '        "description": "what this file must contain",\n'
    '        "depends_on": ["src/bar.py"]}\n'
    "     ]}\n"
    "  ],\n"
    '  "contracts": {\n'
    '    "functions": [\n'
    '      {"name": "total_area", "module": "src/foo.py",\n'
    '       "parameters": [{"name": "circles", "type": "list"}],\n'
    '       "return_type": "float",\n'
    '       "description": "sum of each circle area"},\n'
    '      {"name": "Circle.area", "module": "src/foo.py",\n'
    '       "parameters": [], "return_type": "float",\n'
    '       "description": "area of this circle"}\n'
    "    ],\n"
    '    "schemas": [\n'
    '      {"name": "Circle", "module": "src/foo.py",\n'
    '       "fields": {"radius": "float"}}\n'
    "    ]\n"
    "  }\n"
    "}\n\n"
    "Rules: every file has a unique relative path; depends_on lists ONLY\n"
    "other planned paths; order files so dependencies come first; include at\n"
    "least one runnable non-test source file. Commit to ONE layout -- put\n"
    "every source module under 'src/' OR every module at the top level, never\n"
    "a mix -- and give each test file a depends_on edge to the module it\n"
    "exercises so imports stay consistent.\n"
    "SCAFFOLDING IS MANDATORY: every plan MUST also include, as first-class\n"
    "files with their own descriptions, a top-level 'README.md' (project\n"
    "overview, install, and usage) and a top-level 'requirements.txt' (the\n"
    "runtime and test dependencies, e.g. 'pytest'). When you use a 'src/'\n"
    "layout, ALSO include a root 'conftest.py' so pytest can import the\n"
    "package. These are real deliverables, not source code -- give them a\n"
    "'description' but do NOT put them in 'contracts'.\n"
    "CONTRACTS ARE MANDATORY AND BINDING: every function, method, and class the\n"
    "objective requires MUST appear in contracts with a \"module\" naming the\n"
    "EXACT planned path that defines it. Name a method as \"ClassName.method\".\n"
    "Give each function real \"parameters\" and a \"return_type\". Do not invent\n"
    "symbols, files, or dependencies the objective did not ask for.\n"
    "Output ONLY the JSON."
)


def _ask_for_plan(provider: Any, goal: str,
                  correction: str) -> Dict[str, Any]:
    """Prompt the model for a draft plan JSON (empty dict on failure)."""
    user = f"Objective:\n{goal}\n"
    if correction:
        user += f"\nYour previous plan was rejected: {correction}\nFix it."
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        res = provider.chat(messages=messages, force_json=True)
    except Exception:  # pragma: no cover - defensive: provider crash
        return {}
    parsed = parse_plan_reply(str(res.get("content", "")))
    return parsed or {}


@register_executor(TaskKind.SWARM_TECH_LEAD)
def swarm_tech_lead(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Author, validate, and persist the swarm WORK_PLAN."""
    if deps.provider is None:
        return ExecutorResult(
            failure="No provider configured for Swarm mode.", retryable=False)

    goal = str(task.inputs.get("goal") or "")
    project_root = task.inputs.get("project_root") or deps.project_root or "."
    swarm_beat(project_root, "tech_lead", "plan", goal=goal[:200])

    correction = ""
    plan: Dict[str, Any] = {}
    paths: List[str] = []
    problems: List[str] = []
    for attempt in range(1, _MAX_PLAN_ATTEMPTS + 1):
        draft = _ask_for_plan(deps.provider, goal, correction)
        # Test coverage and scaffolding are injected deterministically rather
        # than re-asked: a pytest module is added for every uncovered source
        # module, and the README / dependency manifest / conftest the Developer
        # synthesises from source-derived templates or a grounded free-form
        # call are appended, so a weak model dropping either can no longer ship
        # an untested tree or abort planning over missing boilerplate. Coverage
        # runs first so the manifest the scaffolding scans includes the tests.
        plan = ensure_scaffolding(ensure_test_coverage(normalize_plan(draft)))
        paths = ordered_paths(plan)
        swarm_beat(project_root, "tech_lead", "normalize",
                   attempt=attempt, file_count=len(paths))
        if not paths:
            problems = ["the plan listed no valid files"]
            correction = problems[0]
            continue
        # Propose-then-validate: a plan that could not build coherently
        # (unsafe paths, mixed rooting, a dependency cycle, an orphan test)
        # is re-asked with the exact problems before any Developer is spawned.
        problems = verify_plan(plan)
        if problems:
            correction = "; ".join(problems)
            swarm_beat(project_root, "tech_lead", "plan_rejected",
                       attempt=attempt, problems=problems)
            continue
        break

    if not paths or problems:
        # Planning is a dead end: surface a zero-file plan so the router
        # ends the session FAILED rather than spawning an empty Developer
        # chain. (A hard failure would also work, but this keeps the
        # partial plan visible for debugging.)
        swarm_beat(project_root, "tech_lead", "report", ok=False,
                   reason=correction or "no buildable plan")
        return ExecutorResult(
            outputs={"file_count": 0, "reason": correction
                     or "Tech Lead could not produce a buildable plan."})

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.WORK_PLAN,
        content={
            "goal": goal,
            "layers": plan["layers"],
            "contracts": plan.get("contracts") or {},
            "paths": paths,
            "project_root": project_root,
        },
    )
    swarm_beat(project_root, "tech_lead", "report", ok=True,
               file_count=len(paths))
    return ExecutorResult(
        artifact=artifact,
        outputs={"work_plan_artifact_id": artifact.artifact_id,
                 "swarm_paths": paths,
                 "file_count": len(paths),
                 "goal": goal,
                 "project_root": project_root})
