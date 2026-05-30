"""High-level :func:`run_agent` entrypoint.

Wires :class:`~cgx.agents.planner.Planner`,
:class:`~cgx.agents.tracker.Tracker`, and
:class:`~cgx.agents.judge.Judge` to the existing Averix capabilities:

* ``ask``      → :func:`cgx.answer.engine.answer_with_llm`
* ``plan``     → :func:`cgx.answer.engine.generate_code_plan`
* ``scaffold`` → :func:`cgx.answer.engine.generate_project_scaffold`
* ``search``   → :func:`cgx.pipeline.auto.run_query_auto`
* ``summarize``→ inline LLM condensation of prior outputs
* ``apply``    → :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`
* ``verify``   → :func:`cgx.codegen.test_runner.run_tests_on_disk`

``scaffold`` is the only capability that does **not** require an index —
it generates a brand-new project from a plain-language idea and stores
its output as ``--- /dev/null`` new-file unified diffs so the ``apply``
capability can write them to ``project_root`` without special handling.

The capability callables are imported lazily inside :func:`run_agent`
so the agent module stays usable in environments that don't have the
embedding stack or a populated index (e.g. unit tests).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from cgx.agents.judge import Judge
from cgx.agents.planner import Planner
from cgx.agents.tracker import Tracker
from cgx.agents.types import AgentEvent, Plan

logger = logging.getLogger(__name__)


def _build_default_capabilities(
    *,
    provider: Any,
    index_dir: Optional[str],
    records_path: Optional[str],
    project_root: Optional[str],
) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Return capability callables backed by the real engine.

    Each capability tolerates ``index_dir`` / ``records_path`` being
    ``None`` by raising ``ValueError`` — the Tracker will record the
    failure and (by default) stop the plan.
    """
    def _need_index() -> None:
        if not index_dir or not records_path:
            raise ValueError("index_dir and records_path are required for this capability")

    def ask(question: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        from cgx.answer.engine import answer_with_llm
        return answer_with_llm(index_dir, records_path, question, provider, **kw)

    def plan(task_text: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        from cgx.answer.engine import generate_code_plan
        # Always honour project_root for self-test sandboxing if available.
        # Default to ``self_test=True`` so the Judge can inspect a
        # codegen_report; the downstream ``verify`` task runs the actual
        # pytest pass so we don't double-test here.
        kw.setdefault("project_root", project_root)
        if project_root:
            kw.setdefault("self_test", True)
            kw.setdefault("run_tests", False)
            kw.setdefault("max_retries", 1)
        return generate_code_plan(index_dir, records_path, task_text, provider, **kw)

    def search(query: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        from cgx.pipeline.auto import run_query_auto
        return run_query_auto(index_dir, records_path, query, **kw)

    def summarize(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        # Compose a single text blob then ask the LLM to summarise.
        if provider is None:
            return {"answer_md": ""}
        body = "\n\n---\n\n".join(
            str(o.get("answer_md") or o.get("plan_md") or o.get("hits") or o)
            for o in (prior or [])
        )[:6000]
        resp = provider.chat(messages=[
            {"role": "system", "content": "Summarise the following work products in <=8 bullets."},
            {"role": "user", "content": body},
        ], temperature=0.1, max_tokens=400, force_json=False)
        return {"answer_md": str((resp or {}).get("content") or "")}

    def scaffold(task_text: str, **kw: Any) -> Dict[str, Any]:
        # No index required — generates an entire project from scratch.
        from cgx.answer.engine import generate_project_scaffold
        kw.setdefault("project_root", project_root)
        return generate_project_scaffold(task_text, provider, **kw)

    def apply(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        if not project_root:
            raise ValueError("apply requires project_root to be set")
        from cgx.codegen.disk_apply import apply_diffs_to_disk
        diffs = _extract_prior_diffs(prior)
        if not diffs:
            return {
                "applied_files": [], "failed_files": [],
                "diffs": [], "error": "no diffs found in prior task outputs",
            }
        return apply_diffs_to_disk(project_root, diffs)

    def verify(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        if not project_root:
            raise ValueError("verify requires project_root to be set")
        from cgx.codegen.test_runner import (
            discover_all_tests, run_pytest_paths, run_tests_on_disk,
        )
        changed = _changed_files_from_prior(prior)
        # Standalone verify (no prior APPLY) — sweep all discovered tests
        # so a goal like "do the tests pass?" actually executes pytest
        # against the working tree instead of skipping.
        mode = "impacted"
        if not changed:
            discovered = discover_all_tests(project_root)
            if discovered:
                outcome = run_pytest_paths(
                    project_root, discovered,
                    timeout_seconds=float(kw.get("timeout", 180.0)),
                )
                mode = "all"
            else:
                outcome = run_tests_on_disk(
                    project_root, changed,
                    timeout_seconds=float(kw.get("timeout", 180.0)),
                )
        else:
            outcome = run_tests_on_disk(
                project_root, changed,
                timeout_seconds=float(kw.get("timeout", 180.0)),
            )
        return {
            "ran": outcome.ran,
            "tests_passed": outcome.ran and outcome.returncode == 0,
            "returncode": outcome.returncode,
            "tests_selected": outcome.tests_selected,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "skipped_reason": outcome.skipped_reason,
            "mode": mode,
        }

    return {"ask": ask, "plan": plan, "scaffold": scaffold, "search": search,
            "summarize": summarize, "apply": apply, "verify": verify}


def _extract_prior_diffs(prior: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Collect and merge diffs from ALL prior task outputs.

    Walks every output in order so that multi-scaffold plans (one task per
    layer) contribute all their files to the final APPLY step. Later entries
    win on file conflicts, allowing a follow-up scaffold task to refine a
    file produced by an earlier one.
    """
    merged: Dict[str, str] = {}
    for out in (prior or []):
        if not isinstance(out, dict):
            continue
        diffs = out.get("diffs")
        if not isinstance(diffs, list):
            continue
        for d in diffs:
            if not isinstance(d, dict):
                continue
            fp = str(d.get("file") or d.get("path") or "").strip()
            patch = str(d.get("patch") or d.get("diff") or "")
            if fp and patch:
                merged[fp] = patch
    return [{"file": fp, "patch": patch} for fp, patch in merged.items()]


def _changed_files_from_prior(prior: List[Dict[str, Any]]) -> List[str]:
    """Return the union of files written by any preceding APPLY task."""
    out: List[str] = []
    seen: set = set()
    for o in (prior or []):
        if not isinstance(o, dict):
            continue
        for f in (o.get("applied_files") or []):
            fp = str(f)
            if fp and fp not in seen:
                seen.add(fp)
                out.append(fp)
    return out


def _build_default_retriever(
    index_dir: Optional[str], records_path: Optional[str],
) -> Optional[Callable[[str], Dict[str, Any]]]:
    """Return a Planner-compatible retriever bound to the project's index.

    Returns ``None`` when an index isn't configured so the Planner falls
    back to LLM-only decomposition rather than failing on the first
    query call.
    """
    if not index_dir or not records_path:
        return None
    if not os.path.isdir(index_dir):
        return None

    def _retrieve(goal: str) -> Dict[str, Any]:
        from cgx.pipeline.auto import run_query_auto
        return run_query_auto(
            index_dir=index_dir, records_path=records_path, query=goal,
            top_k_per_view=10, neighbor_depth=1, use_lexical=True,
        )

    return _retrieve


def _extract_verify_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return outputs of VERIFY tasks that ran and failed (rc != 0)."""
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind != TaskKind.VERIFY or t.status != TaskStatus.FAILED:
            continue
        out = t.output or {}
        if not out.get("ran"):
            continue
        failures.append({
            "tests_selected": out.get("tests_selected") or [],
            "stdout_tail": str(out.get("stdout") or "")[-1200:],
            "stderr_tail": str(out.get("stderr") or "")[-600:],
            "returncode": out.get("returncode"),
            "error": t.error or "",
        })
    return failures


def _build_fix_goal(original_goal: str, failures: List[Dict[str, Any]],
                    plan: Any, project_root: str) -> str:
    """Compose a targeted fix-goal from failing test output."""
    from cgx.agents.types import TaskKind
    applied: List[str] = []
    for t in plan.tasks:
        if t.kind == TaskKind.APPLY:
            applied.extend((t.output or {}).get("applied_files") or [])

    parts = [
        f"Fix test failures in the project at: {project_root}",
        f"Original goal: {original_goal}",
    ]
    if applied:
        preview = "\n".join(f"  - {f}" for f in applied[:20])
        parts.append(f"Files written to disk:\n{preview}")
    for i, f in enumerate(failures[:3], 1):
        parts.append(f"Test failure {i}:")
        if f.get("tests_selected"):
            parts.append("  Tests: " + ", ".join(f["tests_selected"][:5]))
        if f.get("stdout_tail"):
            parts.append("  Output:\n" + f["stdout_tail"])
        if f.get("stderr_tail"):
            parts.append("  Stderr:\n" + f["stderr_tail"])
    parts.append(
        "Analyse the failures above, fix any bugs, missing imports, or logic "
        "errors in the project files, then re-apply and verify."
    )
    return "\n\n".join(parts)


def _extract_apply_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return APPLY tasks that failed due to syntax/patch errors in generated files.

    This catches the case where scaffold/plan tasks produce code that passes
    the in-pipeline self-test but is rejected by ``apply_diffs_to_disk``'s
    stricter smoke test (e.g. unterminated string literals, bad patches).
    """
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind != TaskKind.APPLY or t.status != TaskStatus.FAILED:
            continue
        out = t.output or {}
        failed_files: List[Dict[str, Any]] = out.get("failed_files") or []
        error = t.error or ((t.judge or {}).get("rationale") or "") if t.judge else t.error or ""
        if failed_files or error:
            failures.append({"failed_files": failed_files, "error": error})
    return failures


def _build_apply_fix_goal(original_goal: str, apply_failures: List[Dict[str, Any]]) -> str:
    """Compose a retry goal when apply fails due to syntax / patch errors."""
    file_errors: List[str] = []
    for f in apply_failures:
        for ff in (f.get("failed_files") or []):
            if not isinstance(ff, dict):
                continue
            fname = str(ff.get("file") or "").strip()
            err   = str(ff.get("error") or "").strip()
            if fname and err:
                file_errors.append(f"  - {fname}: {err}")
            elif fname:
                file_errors.append(f"  - {fname}: patch or syntax error")

    parts = [original_goal]
    parts.append(
        "CRITICAL: The previous attempt generated code that failed the syntax smoke "
        "test and was NOT written to disk. You MUST regenerate the files with "
        "correct syntax. Common causes:\n"
        "  • Unterminated string literals (missing closing quote or triple-quote)\n"
        "  • Unmatched brackets, parentheses, or braces\n"
        "  • Truncated function/class bodies\n"
        "  • Invalid escape sequences inside strings\n"
        "Generate every file in full — do NOT truncate or abbreviate any section."
    )
    if file_errors:
        parts.append("Specific errors to fix:\n" + "\n".join(file_errors))
    return "\n\n".join(parts)


def _extract_core_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return failed SCAFFOLD or PLAN tasks (judge rejections or execution errors)."""
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind not in (TaskKind.SCAFFOLD, TaskKind.PLAN):
            continue
        if t.status != TaskStatus.FAILED:
            continue
        failures.append({
            "kind": t.kind.value,
            "name": t.name or t.description,
            "error": t.error or "",
            "judge_rationale": ((t.judge or {}).get("rationale") or "") if t.judge else "",
        })
    return failures


def _build_core_fix_goal(original_goal: str, failures: List[Dict[str, Any]]) -> str:
    """Compose a retry goal from scaffold/plan failure diagnostics."""
    parts = [original_goal]
    issues = []
    for f in failures:
        reason = f["judge_rationale"] or f["error"]
        if reason:
            issues.append(f"{f['name']} ({f['kind']}): {reason}")
    if issues:
        parts.append(
            "The previous attempt had the following issues that MUST be fixed:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    parts.append(
        "Regenerate the project from scratch, ensuring all files are syntactically "
        "complete, all brackets/braces closed, all imports resolve, and all logic is "
        "fully implemented. Do not truncate any file."
    )
    return "\n\n".join(parts)


def _stream_with_retry(
    plan_obj: Any,
    tracker: Tracker,
    planner: Planner,
    capabilities: Dict[str, Callable[..., Dict[str, Any]]],
    judge: Optional[Judge],
    goal: str,
    project_root: Optional[str],
    stop_on_fail: bool,
    progress_interval: float,
    max_retries: int,
    attempt: int = 1,
) -> Iterator[Any]:
    """Yield events from the initial plan, then auto-retry if tasks failed."""
    from cgx.agents.types import AgentEvent
    for ev in tracker.stream(plan_obj):
        yield ev

    if max_retries <= 0:
        return

    # Priority 1: test failures (VERIFY) — build a targeted fix goal.
    verify_failures = _extract_verify_failures(plan_obj)
    if verify_failures:
        if not project_root:
            return
        logger.info("run_agent: %d verify failure(s) on attempt %d — re-planning",
                    len(verify_failures), attempt)
        fix_goal = _build_fix_goal(goal, verify_failures, plan_obj, project_root)
        retry_reason = f"{len(verify_failures)} test failure(s) detected — re-planning to fix"
    else:
        # Priority 2: apply failures (syntax / patch errors in generated files).
        # This fires when scaffold/plan tasks produced syntactically invalid code
        # that passed the in-pipeline check but was rejected by apply's smoke test.
        apply_failures = _extract_apply_failures(plan_obj)
        if apply_failures:
            logger.info("run_agent: %d apply failure(s) on attempt %d — regenerating",
                        len(apply_failures), attempt)
            fix_goal = _build_apply_fix_goal(goal, apply_failures)
            retry_reason = (
                f"{sum(len(f['failed_files']) for f in apply_failures)} file(s) had "
                "syntax / patch errors — regenerating with fixes"
            )
        else:
            # Priority 3: scaffold/plan generation failures — retry the generation.
            core_failures = _extract_core_failures(plan_obj)
            if not core_failures:
                return
            logger.info("run_agent: %d core failure(s) on attempt %d — re-planning",
                        len(core_failures), attempt)
            fix_goal = _build_core_fix_goal(goal, core_failures)
            retry_reason = f"{len(core_failures)} generation failure(s) — retrying with fixes"

    fix_plan = planner.plan(fix_goal)
    fix_tracker = Tracker(
        capabilities=capabilities, judge=judge,
        stop_on_fail=stop_on_fail, progress_interval=progress_interval,
    )

    # Signal the UI that a retry is beginning, then re-emit the new plan
    # as "retry_plan" so the frontend can append tasks rather than replace them.
    yield AgentEvent(
        type="retry_start",
        payload={"attempt": attempt + 1, "reason": retry_reason},
    )

    for ev in fix_tracker.stream(fix_plan):
        if ev.type == "plan":
            yield AgentEvent(type="retry_plan", payload=ev.payload)
        else:
            yield ev

    # Recurse for additional retries if still failing.
    retry_verify = _extract_verify_failures(fix_plan)
    retry_core = _extract_core_failures(fix_plan)
    if (retry_verify or retry_core) and max_retries - 1 > 0:
        for ev in _stream_with_retry(
            fix_plan, fix_tracker, planner, capabilities, judge,
            fix_goal, project_root, stop_on_fail, progress_interval,
            max_retries - 1, attempt + 1,
        ):
            yield ev


def run_agent(
    goal: str,
    *,
    provider: Any = None,
    index_dir: Optional[str] = None,
    records_path: Optional[str] = None,
    project_root: Optional[str] = None,
    capabilities: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    planner: Optional[Planner] = None,
    judge: Optional[Judge] = None,
    stop_on_fail: bool = True,
    stream: bool = False,
    progress_interval: float = 2.0,
    max_retries: int = 1,
) -> Any:
    """Run a Planner → Tracker → Judge loop for ``goal``.

    Parameters
    ----------
    goal
        Natural-language user request.
    provider
        Optional :class:`~cgx.answer.providers.LLMProvider`. Required for
        the planner's LLM path and for any ``ask``/``plan`` capability;
        the deterministic fallback plan still runs without it.
    index_dir, records_path
        Paths to the indexed artifacts. Required by the default
        capabilities; ignored when ``capabilities`` is supplied.
    project_root
        Forwarded to ``generate_code_plan`` for the self-test sandbox.
    capabilities
        Override the default capability table (useful for tests).
    planner / judge
        Override the default ``Planner`` / ``Judge`` (e.g. to inject a
        deterministic stub).
    stop_on_fail
        Halt the plan after the first failed task (default True).
    stream
        If True, return a generator of :class:`AgentEvent`. If False,
        return the final :class:`Plan` after running to completion.
    """
    if planner is None:
        retriever = _build_default_retriever(index_dir, records_path)
        planner = Planner(provider=provider, retriever=retriever)
    plan_obj: Plan = planner.plan(goal)
    if capabilities is None:
        capabilities = _build_default_capabilities(
            provider=provider, index_dir=index_dir,
            records_path=records_path, project_root=project_root,
        )
    judge = judge if judge is not None else Judge(provider=provider)
    tracker = Tracker(capabilities=capabilities, judge=judge,
                      stop_on_fail=stop_on_fail,
                      progress_interval=progress_interval)
    if stream:
        return _stream_with_retry(
            plan_obj, tracker, planner, capabilities, judge,
            goal, project_root, stop_on_fail, progress_interval, max_retries,
        )
    tracker.run(plan_obj)
    if max_retries > 0:
        verify_failures = _extract_verify_failures(plan_obj)
        apply_failures  = _extract_apply_failures(plan_obj)
        core_failures   = _extract_core_failures(plan_obj)
        if verify_failures and project_root:
            fix_goal = _build_fix_goal(goal, verify_failures, plan_obj, project_root)
        elif apply_failures:
            fix_goal = _build_apply_fix_goal(goal, apply_failures)
        elif core_failures:
            fix_goal = _build_core_fix_goal(goal, core_failures)
        else:
            fix_goal = None
        if fix_goal:
            fix_plan = planner.plan(fix_goal)
            fix_tracker = Tracker(capabilities=capabilities, judge=judge,
                                  stop_on_fail=stop_on_fail,
                                  progress_interval=progress_interval)
            fix_tracker.run(fix_plan)
            return fix_plan
    return plan_obj
