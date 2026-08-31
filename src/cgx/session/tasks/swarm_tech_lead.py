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

from typing import Any, Dict, List, Optional

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
    '    "endpoints": [\n'
    '      {"path": "/ping", "method": "GET",\n'
    '       "description": "Returns pong"}\n'
    '    ],\n'
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
    "    ],\n"
    '    "third_party_dependencies": ["fastapi", "pydantic", "pytest"]\n'
    "  }\n"
    "}\n\n"
    "Rules: every file has a unique relative path; depends_on lists ONLY\n"
    "other planned paths; order files so dependencies come first; include at\n"
    "least one runnable non-test source file; give each test file a depends_on\n"
    "edge to the module it exercises. Use each file's REAL extension for its\n"
    "language (.py, .jsx/.tsx, .go, ...); contract \"module\" values name the\n"
    "exact planned path that defines the symbol.\n"
    "MULTIPLE COMPONENTS: a project may span more than one component and\n"
    "language -- e.g. a Python backend AND a JavaScript/React frontend. Plan\n"
    "files for EVERY component the objective asks for; do not silently drop\n"
    "one (a 'Flask + React' app needs both the Flask API and the React UI).\n"
    "Give each component its own top-level directory (e.g. 'backend/' and\n"
    "'frontend/') and keep each component's imports rooted consistently.\n"
    "SCAFFOLDING IS MANDATORY, PER COMPONENT: every runnable component MUST\n"
    "ship its dependency manifest and the project MUST ship a top-level\n"
    "'README.md'. For a Python component include a 'requirements.txt' (and,\n"
    "for a 'src/' layout, a root 'conftest.py'). For a JavaScript/TypeScript\n"
    "component include a 'package.json' (dependencies, devDependencies, and\n"
    "dev/build/test scripts). These are real deliverables with a\n"
    "'description' but NOT part of 'contracts'.\n"
    "CONTRACTS ARE MANDATORY AND BINDING: every function, method, and class the\n"
    "objective requires MUST appear in contracts with a \"module\" naming the\n"
    "EXACT planned path that defines it. Name a method as \"ClassName.method\".\n"
    "Give each function real \"parameters\" and a \"return_type\". Do not invent\n"
    "symbols, files, or dependencies the objective did not ask for.\n"
    "ENTRYPOINT IS MANDATORY: each runnable component needs an entrypoint that\n"
    "initializes it (e.g. a Python 'app = Flask(__name__)'/'FastAPI()' module,\n"
    "or a JS 'src/main.jsx' that mounts the app). Do NOT expect tests to run\n"
    "without an application instance to import.\n"
    "THIRD-PARTY LIBRARIES: if you use third-party libraries, you MAY search\n"
    "the web to retrieve their latest API signatures and include them in the\n"
    "contracts. Use tools by outputting: "
    "<call_tool name=\"search_web\">{\"query\": \"...\"}</call_tool>.\n"
    "If you call a tool, wait for the response before outputting the final JSON.\n"
    "Follow any ACTIVE SKILL guidance below for the specific frameworks.\n"
    "Output ONLY the JSON when you are ready to finalize the plan."
)


# The Tech Lead may search the web (to fetch third-party API signatures) while
# planning, plus any configured MCP tools; everything else is deterministic
# plan validation.
def _planner_tools() -> tuple:
    from cgx.session.tasks.swarm_tools import mcp_tools_if_configured
    return ("search_web",) + mcp_tools_if_configured()


def _ask_for_plan(provider: Any, goal: str,
                  correction: str,
                  project_root: Optional[str] = None,
                  skill_prompt: str = "") -> Dict[str, Any]:
    """Prompt the model for a draft plan JSON (empty dict on failure).

    Tool calls (``search_web``) are resolved through the shared tool registry,
    so the planner shares one dispatch path with the Developer instead of a
    bespoke regex/if-chain. A non-tool reply is parsed as the plan JSON, with
    one corrective re-ask on unparseable output.
    """
    from cgx.session.tasks.swarm_log import swarm_beat
    from cgx.session.tasks.tool_registry import (
        REGISTRY, ToolContext, parse_tool_calls)
    import cgx.session.tasks.swarm_tools  # noqa: F401  (populate the registry)

    user = f"Objective:\n{goal}\n"
    if correction:
        user += f"\nYour previous plan was rejected: {correction}\nFix it."
    system = _SYSTEM_PROMPT
    if skill_prompt.strip():
        # Framework-specific planning guidance (e.g. React's Vite layout,
        # Flask's app/blueprint conventions) for the stacks detected in the
        # goal -- this is what teaches the planner to include the frontend.
        system += "\n\nACTIVE SKILLS (follow this guidance):\n" + skill_prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    ctx = ToolContext(root=project_root or ".", log_root=project_root)
    planner_tools = _planner_tools()
    for _ in range(5):
        try:
            # force_json=False so a tool tag isn't rejected by strict-JSON
            # providers; the plan is parsed leniently from the final reply.
            res = provider.chat(messages=messages, force_json=False)
        except Exception:  # pragma: no cover
            return {}

        text = str(res.get("content", ""))
        calls = [c for c in parse_tool_calls(text) if c.name in planner_tools]
        if calls:
            messages.append({"role": "assistant", "content": text})
            for call in calls:
                swarm_beat(project_root, "tech_lead", "tool_call",
                           tool=call.name, args=call.raw_args)
                out = REGISTRY.dispatch(call, ctx)
                messages.append({
                    "role": "user",
                    "content": f"<tool_response name=\"{call.name}\">{out}"
                               "</tool_response>"})
            continue
        parsed = parse_plan_reply(text)
        if parsed:
            return parsed
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "<tool_response>Tool error: invalid JSON generated. Ensure your output is purely JSON without markdown formatting.</tool_response>"})
    return {}


def _auto_repair_plan_dependencies(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-inject depends_on for test files if the model forgot."""
    source_map = {}
    for layer in plan.get("layers", []):
        for f in layer.get("files", []):
            path = f.get("path", "")
            base = path.split("/")[-1]
            if path.endswith(".py") and not (base.startswith("test_") or base.endswith("_test.py")):
                source_map[base[:-3]] = path

    for layer in plan.get("layers", []):
        for f in layer.get("files", []):
            path = f.get("path", "")
            base = path.split("/")[-1]
            is_test = base.startswith("test_") or base.endswith("_test.py")
            if path.endswith(".py") and is_test and not f.get("depends_on"):
                target_base = None
                if base.startswith("test_"):
                    target_base = base[5:-3]
                elif base.endswith("_test.py"):
                    target_base = base[:-8]
                
                if target_base and target_base in source_map:
                    f["depends_on"] = [source_map[target_base]]
    return plan


@register_executor(TaskKind.SWARM_TECH_LEAD)
def swarm_tech_lead(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Author, validate, and persist the swarm WORK_PLAN."""
    if deps.provider is None:
        return ExecutorResult(
            failure="No provider configured for Swarm mode.", retryable=False)

    import os
    goal = str(task.inputs.get("goal") or "")
    project_root = task.inputs.get("project_root") or deps.project_root or "."
    project_root = os.path.abspath(project_root)
    if not os.path.exists(project_root):
        os.makedirs(project_root, exist_ok=True)
        
    swarm_beat(project_root, "tech_lead", "plan", goal=goal[:200])

    # Resolve the frameworks in play once: an explicit session pin (from an
    # Agent Profile) wins, else auto-detect from the goal text. Their plan-time
    # guidance is injected into the prompt and their validators gate the draft,
    # so a polyglot objective (e.g. Flask + React) plans every component instead
    # of collapsing to a single Python package.
    from cgx.session.tasks.base import session_skills as _session_skills
    from skills import (compose_plan_prompt, compose_scaffold_prompt,
                        detect_skills, skill_names, skills_by_names,
                        validate_plan)
    pinned = _session_skills(task, deps)
    active_skills = skills_by_names(pinned) if pinned else detect_skills(goal)
    # The planner needs the framework's *structural* requirements (e.g. React's
    # Vite layout mandates an index.html entry + main.jsx + package.json), which
    # live in the scaffold fragment; the plan fragment adds modify-time rules.
    # Composing both is what makes the plan include a buildable file set (the
    # missing-index.html build failure came from planning with the terse plan
    # fragment alone).
    _struct = compose_scaffold_prompt(active_skills)
    _rules = compose_plan_prompt(active_skills)
    skill_prompt = "\n\n".join(p for p in (_struct, _rules) if p.strip())
    active_skill_names = skill_names(active_skills)
    if active_skill_names:
        swarm_beat(project_root, "tech_lead", "skills",
                   skills=active_skill_names)

    correction = ""
    plan: Dict[str, Any] = {}
    paths: List[str] = []
    problems: List[str] = []

    is_debate = deps.extra.get("multi_agent_debate", False)
    
    for attempt in range(1, _MAX_PLAN_ATTEMPTS + 1):
        if is_debate:
            swarm_beat(project_root, "tech_lead", "debate_generation", attempt=attempt)
            draft1 = _ask_for_plan(deps.provider, goal, correction, project_root, skill_prompt)
            draft2 = _ask_for_plan(deps.provider, goal, correction, project_root, skill_prompt)
            
            # Judge decides
            judge_prompt = (
                "You are the Lead Architect. Two developers have proposed plans for the following objective:\n"
                f"OBJECTIVE: {goal}\n\n"
                f"PLAN A: {draft1}\n\n"
                f"PLAN B: {draft2}\n\n"
                "Evaluate both plans based on completeness, modularity, and adherence to the objective. "
                "On the first line output ONLY the winner letter ('A' or 'B'). "
                "On the next line give one sentence explaining why."
            )
            from cgx.session.tasks.swarm_tools import judge_decision
            decision, reason = judge_decision(deps.provider, judge_prompt)
            draft = draft1 if decision == "A" else draft2
            swarm_beat(project_root, "tech_lead", "debate_decision",
                       decision=decision, reason=reason)
        else:
            draft = _ask_for_plan(deps.provider, goal, correction, project_root, skill_prompt)

        # Test coverage and scaffolding are injected deterministically rather
        # than re-asked: a pytest module is added for every uncovered source
        # module, and the README / dependency manifest / conftest the Developer
        # synthesises from source-derived templates or a grounded free-form
        # call are appended, so a weak model dropping either can no longer ship
        # an untested tree or abort planning over missing boilerplate. Coverage
        # runs first so the manifest the scaffolding scans includes the tests.
        plan = ensure_scaffolding(ensure_test_coverage(normalize_plan(draft)))
        plan = _auto_repair_plan_dependencies(plan)
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
        # Framework-level validation: a skill can veto a plan that omits what
        # the stack requires (e.g. the React skill flags a plan with no
        # frontend files). A fatal verdict is fed back as a corrective re-ask.
        verdict = validate_plan(active_skills, [{"path": p} for p in paths],
                                goal)
        if verdict is not None and not verdict.passed:
            problems = [f"{verdict.skill or 'skill'}: {verdict.rationale}"]
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
            # Persist the resolved stack so the Developer + Verifier reuse it
            # (skill-guided generation, polyglot verification) without
            # re-detecting per file.
            "skills": active_skill_names,
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
                 "project_root": project_root,
                 "skills": active_skill_names})
