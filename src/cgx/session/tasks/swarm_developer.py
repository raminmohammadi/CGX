"""The Developer executor: generate exactly one planned file per turn.

Under the incremental driver the router spawns one Developer task per file in
the plan's toposorted order, threading a ``file_index`` cursor and an
accumulating ``failed_paths`` list. Each turn this executor looks up its file
in the WORK_PLAN artifact, runs the :mod:`swarm_generate` ladder (full-file
grounded on on-disk dependencies, then AST fallback), writes the result with
:func:`edit_file`, and emits a per-file progress beat so the UI shows the same
shrinking countdown the greenfield SCAFFOLD does. A file that fails the whole
ladder is recorded in ``failed_paths`` but does not halt the chain -- the
router decides COMPLETED vs FAILED once every file has been attempted.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from cgx.session.models import TaskKind
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.scaffold import _emit_scaffold_progress
from cgx.session.tasks.swarm_generate import generate_file
from cgx.session.tasks.swarm_log import swarm_beat
from cgx.session.tasks.swarm_plan import plan_specs
from cgx.session.tasks.swarm_tools import edit_file, judge_decision


def _judge_decision(provider: Any, prompt: str) -> tuple:
    """Thin alias so the debate path reads naturally; see ``judge_decision``."""
    return judge_decision(provider, prompt)


def _load_plan(deps: ExecutorDeps, artifact_id: str) -> Dict[str, Any]:
    """The WORK_PLAN artifact content, or ``{}`` when unavailable."""
    if not artifact_id or deps.store is None:
        return {}
    art = deps.store.get_artifact(artifact_id)
    return dict(art.content) if art and art.content else {}


def _contract_item_key(item: Any) -> Any:
    """Stable identity for a contract list entry (function/schema/endpoint)."""
    if isinstance(item, dict):
        if item.get("name"):
            return ("name", item["name"])
        if item.get("path"):
            return ("path", item.get("path"), item.get("method"))
    return ("raw", repr(item))


def merge_contracts(old: Dict[str, Any],
                    new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge renegotiated contracts into the plan's contracts as a *superset*.

    A weak model asked to "output the complete updated contracts" routinely
    returns only the entries for the file it just wrote, which -- if written
    back verbatim -- silently drops the contracts for every file not yet
    generated, so those files can no longer be contract-checked. This merges
    instead: list sections (functions/schemas/endpoints) are unioned by symbol
    identity with the new entry overriding a same-named old one, so a
    renegotiated signature still takes effect but no previously-declared symbol
    is ever lost. ``third_party_dependencies`` is set-unioned. Only when ``new``
    is not a dict is the original returned unchanged.
    """
    if not isinstance(new, dict):
        return dict(old or {})
    merged: Dict[str, Any] = dict(old or {})
    for key, new_val in new.items():
        old_val = merged.get(key)
        if isinstance(new_val, list) and isinstance(old_val, list):
            if key == "third_party_dependencies":
                merged[key] = sorted(
                    {str(x) for x in old_val} | {str(x) for x in new_val})
            else:
                by_key: Dict[Any, Any] = {}
                for item in old_val:
                    by_key[_contract_item_key(item)] = item
                for item in new_val:  # new overrides same-identity old
                    by_key[_contract_item_key(item)] = item
                merged[key] = list(by_key.values())
        else:
            merged[key] = new_val
    return merged


@register_executor(TaskKind.SWARM_DEVELOPER)
def swarm_developer(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate, gate, and write one planned file."""
    if deps.provider is None:
        return ExecutorResult(
            failure="No provider configured for Swarm mode.", retryable=False)

    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "")
    file_index = int(task.inputs.get("file_index") or 0)
    failed_paths: List[str] = list(task.inputs.get("failed_paths") or [])
    content = _load_plan(deps, work_plan_id)
    paths: List[str] = list(content.get("paths") or [])
    file_count = int(task.inputs.get("file_count") or len(paths))
    goal = str(task.inputs.get("goal") or content.get("goal") or "")
    project_root = (task.inputs.get("project_root")
                    or content.get("project_root")
                    or deps.project_root or ".")

    base = {"work_plan_artifact_id": work_plan_id, "file_count": file_count,
            "goal": goal, "project_root": project_root}
    if file_index >= len(paths):
        return ExecutorResult(outputs={**base, "file_index": file_index,
                                       "failed_paths": failed_paths})

    specs = plan_specs({"layers": content.get("layers") or []})
    path = paths[file_index]
    spec = specs.get(path, {})
    contracts = content.get("contracts") or {}
    # The stack the Tech Lead resolved: threaded into generation so the file is
    # authored with the right framework conventions (React Vite layout, Flask
    # routes, package.json shape, ...) instead of Python defaults.
    skills = list(content.get("skills") or [])

    swarm_beat(project_root, "developer", "generate", file=path,
               index=file_index + 1, total=file_count)
    _emit_scaffold_progress(deps, task, file=path, layer="swarm",
                            index=file_index + 1, total=file_count,
                            status="start", failed_count=len(failed_paths))

    started = time.time()
    is_debate = deps.extra.get("multi_agent_debate", False)
    
    if is_debate:
        swarm_beat(project_root, "developer", "debate_generation", file=path)
        outcome1 = generate_file(
            path=path, description=str(spec.get("description") or ""),
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=project_root, provider=deps.provider,
            layer=path, manifest_paths=paths, log_root=project_root,
            skills=skills)

        outcome2 = generate_file(
            path=path, description=str(spec.get("description") or ""),
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=project_root, provider=deps.provider,
            layer=path, manifest_paths=paths, log_root=project_root,
            skills=skills)
            
        if outcome1.ok and outcome2.ok:
            judge_prompt = (
                "You are the Lead Code Reviewer. Two developers have written code for the following file:\n"
                f"FILE: {path}\n"
                f"OBJECTIVE: {spec.get('description')}\n\n"
                f"CODE A:\n```python\n{outcome1.content}\n```\n\n"
                f"CODE B:\n```python\n{outcome2.content}\n```\n\n"
                "Evaluate both implementations based on correctness, simplicity, and adherence to the objective. "
                "On the first line output ONLY the winner letter ('A' or 'B'). "
                "On the next line give one sentence explaining why."
            )
            decision, reason = _judge_decision(deps.provider, judge_prompt)
            outcome = outcome1 if decision == "A" else outcome2
            # Record the rationale (not just the letter) so a debate run is
            # auditable and its extra cost is justified in the trace.
            swarm_beat(project_root, "developer", "debate_decision", file=path,
                       decision=decision, reason=reason)
        else:
            outcome = outcome1 if outcome1.ok else outcome2
    else:
        outcome = generate_file(
            path=path, description=str(spec.get("description") or ""),
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=project_root, provider=deps.provider,
            layer=path, manifest_paths=paths, log_root=project_root,
            skills=skills)

    if outcome.ok:
        if outcome.renegotiated_contracts and deps.store:
            art = deps.store.get_artifact(work_plan_id)
            if art and isinstance(art.content, dict):
                # Merge (never replace): a partial renegotiated blob must not
                # drop contracts for files not yet generated. See
                # :func:`merge_contracts`.
                merged = merge_contracts(art.content.get("contracts") or {},
                                         outcome.renegotiated_contracts)
                art.content["contracts"] = merged
                deps.store.save_artifact(art)
                swarm_beat(project_root, "developer", "contracts_merged",
                           file=path)
                # also update our local copy for the rest of this function
                contracts = merged
        
        write_msg = edit_file(path, outcome.content, project_root)
        swarm_beat(project_root, "developer", "write", file=path,
                   ok=True, method=outcome.method, bytes=outcome.bytes,
                   detail=write_msg)
    else:
        failed_paths.append(path)
        swarm_beat(project_root, "developer", "write", file=path,
                   ok=False, error=outcome.error)

    elapsed_ms = int((time.time() - started) * 1000)
    _emit_scaffold_progress(
        deps, task, file=path, layer="swarm", index=file_index + 1,
        total=file_count, status="done" if outcome.ok else "failed",
        bytes=outcome.bytes if outcome.ok else None,
        elapsed_ms=elapsed_ms, failed_count=len(failed_paths))

    return ExecutorResult(outputs={
        **base, "file_index": file_index, "path": path,
        "file_ok": outcome.ok, "method": outcome.method,
        "failed_paths": failed_paths})
