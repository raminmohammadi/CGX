

"""SCAFFOLD executor: generate file contents for an approved WORK_PLAN.

Iterates through ``WORK_PLAN.layers`` in declaration order, calling
:func:`cgx.answer.engine.generate_single_scaffold_file` once per file.
Each generated file's content is fed back into the context for the
next file so cross-file imports resolve correctly (the legacy
scaffold pipeline does the same thing).

Emits an :class:`Artifact` of kind ``SCAFFOLD_PATCHES`` whose
``diffs`` list is shaped for :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`
-- the downstream APPLY task can therefore reuse the existing
disk-apply path without special casing greenfield.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.pypi_client import PyPIClient
from cgx.session.scaffold_validate import validate_scaffold_diffs
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.SCAFFOLD)
def run_scaffold(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate file contents for every entry in the work plan."""
    if deps.provider is None:
        return ExecutorResult(failure="SCAFFOLD requires an LLM provider")
    if deps.store is None:
        return ExecutorResult(
            failure="SCAFFOLD requires a session store in deps")

    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "").strip()
    if not work_plan_id:
        return ExecutorResult(
            failure="SCAFFOLD missing work_plan_artifact_id")
    work_plan = deps.store.get_artifact(work_plan_id)
    if work_plan is None or work_plan.kind is not ArtifactKind.WORK_PLAN:
        return ExecutorResult(
            failure=f"SCAFFOLD: work plan {work_plan_id!r} missing")

    content = work_plan.content or {}
    layers = content.get("layers") or []
    goal = str(content.get("composed_goal")
               or content.get("prior_goal") or "").strip()
    if not layers:
        return ExecutorResult(
            failure="SCAFFOLD: work plan carries no layers")

    # Phase 6.1: when REPAIR routed a regenerate verdict here, fold the
    # accumulated constraint payloads into the goal so the per-file
    # generator sees the prior-failure context. Each entry is a small
    # ``{kind, rationale, ...}`` dict shaped by the classifier; the
    # join keeps the prompt human-readable without leaking JSON syntax.
    regenerate_constraints = task.inputs.get("regenerate_constraints")
    if isinstance(regenerate_constraints, list) and regenerate_constraints:
        goal = _augment_goal_with_constraints(goal, regenerate_constraints)

    # Phase 7.1: pull matching cross-session lessons (recorded after
    # prior REPAIR -> VERIFY-pass cycles) and inject them as additional
    # constraints. Scored by stack overlap + objective-keyword overlap;
    # noop when the store is empty so the happy path stays unchanged.
    lesson_constraints = _lessons_as_constraints(goal, content)
    if lesson_constraints:
        goal = _augment_goal_with_constraints(goal, lesson_constraints,
                                              header="Lessons from prior "
                                              "sessions to apply:")

    # Lazy import: drags the scaffold prompt templates.
    from cgx.answer.engine import generate_single_scaffold_file

    diffs: List[Dict[str, str]] = []
    existing_with_content: List[Dict[str, str]] = []
    generated: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "project").strip()
        for entry in (layer.get("files") or []):
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            description = str(entry.get("description") or path).strip()
            if not path:
                continue
            try:
                result = generate_single_scaffold_file(
                    path, description, deps.provider,
                    layer=layer_name,
                    existing_files_with_content=list(existing_with_content),
                    goal=goal,
                )
            except Exception as exc:
                logger.exception(
                    "SCAFFOLD: generate_single_scaffold_file crashed for %s",
                    path)
                failed.append({
                    "file": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            file_path = str(result.get("file") or path).strip()
            patch = str(result.get("patch") or "")
            file_content = str(result.get("content") or "")
            if not file_path or not patch:
                failed.append({
                    "file": path,
                    "error": "generator returned empty patch",
                })
                continue

            diffs.append({"file": file_path, "patch": patch})
            existing_with_content.append({
                "path": file_path, "content": file_content,
            })
            generated.append({
                "file": file_path,
                "layer": layer_name,
                "syntax_ok": bool(result.get("syntax_ok")),
                "confidence": result.get("confidence"),
                "bytes": len(file_content),
            })

    if not diffs:
        return ExecutorResult(
            failure="SCAFFOLD: every file generation failed")

    # Phase 4.1: tighten upper bounds on known-fragile peers using the
    # consumer's PyPI ``requires_dist`` *before* APPLY writes the file.
    # Network / fetch failures degrade to no-op (returns the original
    # diffs and empty adjustments) so SCAFFOLD never blocks on PyPI.
    pin_adjustments: List[Dict[str, Any]] = []
    try:
        pypi_client = _resolve_pypi_client(deps)
        file_contents = {e["path"]: e["content"]
                         for e in existing_with_content
                         if e.get("path") and isinstance(e.get("content"), str)}
        diffs, _, pin_adjustments = validate_scaffold_diffs(
            diffs, file_contents, pypi_client=pypi_client)
    except Exception:  # pragma: no cover - defensive: validator is best-effort
        logger.exception(
            "SCAFFOLD: pin validator raised; emitting unmodified diffs")
        pin_adjustments = []

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.SCAFFOLD_PATCHES,
        content={
            "work_plan_artifact_id": work_plan_id,
            "prior_goal": content.get("prior_goal"),
            "composed_goal": goal,
            "diffs": diffs,
            "generated": generated,
            "failed": failed,
            "pin_adjustments": pin_adjustments,
        },
    )
    return ExecutorResult(
        outputs={
            "scaffold_artifact_id": artifact.artifact_id,
            "generated_count": len(generated),
            "failed_count": len(failed),
            "pin_adjustments_count": len(pin_adjustments),
        },
        artifact=artifact,
    )


def _resolve_pypi_client(deps: ExecutorDeps) -> PyPIClient:
    """Return the injected PyPI client, or build a default."""
    injected = (deps.extra or {}).get("pypi_client")
    if isinstance(injected, PyPIClient):
        return injected
    return PyPIClient()


def _augment_goal_with_constraints(
        goal: str, constraints: List[Dict[str, Any]],
        *, header: str = "Prior-attempt failures to avoid this time:") -> str:
    """Append ``constraints`` to ``goal`` under ``header`` as a bulleted tail.

    Each entry is rendered as ``- <kind>: <rationale>`` so the LLM
    sees the failure as a structured caveat rather than free-form
    context. The original goal is preserved verbatim; the tail is
    delimited by a blank line + header so prompt-builders that split on
    line ranges keep working. The header is configurable so Phase 7.1
    lessons can re-use the same shape with a clearer call-out.
    """
    lines = ["", header]
    for entry in constraints:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "constraint").strip()
        rationale = str(entry.get("rationale") or "").strip()
        if rationale:
            lines.append(f"- {kind}: {rationale}")
        else:
            lines.append(f"- {kind}")
    if len(lines) == 2:
        return goal
    return f"{goal}\n" + "\n".join(lines) if goal else "\n".join(lines[1:])


def _lessons_as_constraints(
        goal: str, work_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate matching :mod:`cgx.session.lessons` records into constraint dicts.

    Best-effort: any failure in the lesson store (missing file, OS
    error, malformed JSON) is swallowed by the underlying
    :func:`relevant_lessons` and surfaces here as an empty list.
    """
    from cgx.session.lessons import relevant_lessons

    stack = _extract_stack_packages(work_plan)
    try:
        lessons = relevant_lessons(objective=goal, stack=stack, limit=3)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("scaffold: relevant_lessons failed: %s", exc)
        return []
    out: List[Dict[str, Any]] = []
    for lesson in lessons:
        cls = str(lesson.get("classification") or "lesson").strip()
        sig = str(lesson.get("trigger_signature") or "").strip()
        fix = lesson.get("applied_fix") or {}
        files = fix.get("files") or []
        files_part = (f" (touched {', '.join(files[:3])}"
                      f"{'...' if len(files) > 3 else ''})") if files else ""
        rationale = (
            f"A previous session hit {sig!r} and fixed it via "
            f"{fix.get('strategy') or 'patch'}{files_part}; avoid the "
            "same trigger here.")
        out.append({"kind": f"lesson:{cls}", "rationale": rationale})
    return out


def _extract_stack_packages(work_plan: Dict[str, Any]) -> List[str]:
    """Return the package list a SCAFFOLD goal is targeting.

    Reads from ``requirements_pins`` first (the curated list a DECOMPOSE
    leaves on the WORK_PLAN) and falls back to the layer-derived
    ``pins`` field; either way the result is a list of bare package
    names with no version specifier.
    """
    out: List[str] = []
    for key in ("requirements_pins", "pins", "stack"):
        pins = work_plan.get(key)
        if isinstance(pins, list):
            for entry in pins:
                if isinstance(entry, str):
                    name = entry.split("==")[0].split(">=")[0]
                    name = name.split("<")[0].split(";")[0].strip()
                    if name:
                        out.append(name)
                elif isinstance(entry, dict):
                    name = str(entry.get("name") or "").strip()
                    if name:
                        out.append(name)
    return out
