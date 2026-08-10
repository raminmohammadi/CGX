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
    "ENTRYPOINT IS MANDATORY: If the objective is an application or API (e.g. FastAPI), you MUST include a main entrypoint file (like 'src/main.py' or 'main.py') that initializes the application instance (e.g. 'app = FastAPI()'). Do NOT expect tests to run without a main application instance to import.\n"
    "THIRD-PARTY LIBRARIES: If you use third-party libraries (e.g. FastAPI, Pydantic), you MUST "
    "actively search the web to retrieve their latest API signatures and include them in the contracts.\n"
    "You can use tools by outputting: <call_tool name=\"search_web\">{\"query\": \"...\"}</call_tool>.\n"
    "If you call a tool, wait for the response before outputting the final JSON.\n"
    "Output ONLY the JSON when you are ready to finalize the plan."
)


def _ask_for_plan(provider: Any, goal: str,
                  correction: str) -> Dict[str, Any]:
    """Prompt the model for a draft plan JSON (empty dict on failure), supporting tool calls."""
    from cgx.session.tasks.swarm_tools import search_web
    import re
    import json
    
    user = f"Objective:\n{goal}\n"
    if correction:
        user += f"\nYour previous plan was rejected: {correction}\nFix it."
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    for _ in range(5):
        try:
            # We don't force_json=True here immediately because we allow tool tags.
            # Some providers fail if force_json is True and it outputs a tool tag instead of JSON.
            # Actually, we can use force_json=False.
            res = provider.chat(messages=messages, force_json=False)
        except Exception:  # pragma: no cover
            return {}
        
        text = str(res.get("content", ""))
        tool_match = re.search(r'<call_tool name="(.*?)">(.*?)</call_tool>', text, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_args_str = tool_match.group(2)
            messages.append({"role": "assistant", "content": text})
            from cgx.session.tasks.swarm_log import swarm_beat
            swarm_beat(None, "tech_lead", "tool_call", tool=tool_name, args=tool_args_str)
            if tool_name == "search_web":
                try:
                    args = json.loads(tool_args_str)
                    tool_res = search_web(args.get("query", ""))
                except Exception as e:
                    tool_res = f"Tool error: {e}"
            else:
                tool_res = f"Unknown tool: {tool_name}"
            messages.append({"role": "user", "content": f"<tool_response>{tool_res}</tool_response>"})
        else:
            parsed = parse_plan_reply(text)
            if parsed:
                with open("last_plan_parsed.json", "w") as f:
                    import json as _json
                    _json.dump(parsed, f, indent=2)
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

    correction = ""
    plan: Dict[str, Any] = {}
    paths: List[str] = []
    problems: List[str] = []
    
    is_debate = deps.extra.get("multi_agent_debate", False)
    
    for attempt in range(1, _MAX_PLAN_ATTEMPTS + 1):
        if is_debate:
            swarm_beat(project_root, "tech_lead", "debate_generation", attempt=attempt)
            draft1 = _ask_for_plan(deps.provider, goal, correction)
            draft2 = _ask_for_plan(deps.provider, goal, correction)
            
            # Judge decides
            judge_prompt = (
                "You are the Lead Architect. Two developers have proposed plans for the following objective:\n"
                f"OBJECTIVE: {goal}\n\n"
                f"PLAN A: {draft1}\n\n"
                f"PLAN B: {draft2}\n\n"
                "Evaluate both plans based on completeness, modularity, and adherence to the objective. "
                "Output ONLY 'A' or 'B'."
            )
            try:
                res = deps.provider.chat([{"role": "user", "content": judge_prompt}])
                decision = str(res.get("content", "")).strip().upper()
            except Exception:
                decision = "A"
            
            draft = draft1 if decision == "A" else draft2
            swarm_beat(project_root, "tech_lead", "debate_decision", decision=decision)
        else:
            draft = _ask_for_plan(deps.provider, goal, correction)
            
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
