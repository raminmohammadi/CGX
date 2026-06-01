"""Tracker: state machine that runs a Plan task-by-task.

The Tracker is the *only* part of the agent layer that actually executes
work. It delegates capabilities to caller-supplied callables so the
agents module stays decoupled from ``cgx.answer.engine`` import-time
side effects, and so tests can inject deterministic stubs.

Capabilities dispatched by task ``kind``:

* ``ask``      ⇒ ``capabilities["ask"](question: str, **task.inputs) -> dict``
* ``plan``     ⇒ ``capabilities["plan"](task_text: str, **task.inputs) -> dict``
* ``scaffold`` ⇒ ``capabilities["scaffold"](idea: str, **task.inputs) -> dict``
* ``search``   ⇒ ``capabilities["search"](query: str, **task.inputs) -> dict``
* ``summarize``⇒ ``capabilities["summarize"](prior_outputs: list, **task.inputs) -> dict``
* ``apply``    ⇒ ``capabilities["apply"](prior_outputs: list, **task.inputs) -> dict``
* ``verify``   ⇒ ``capabilities["verify"](prior_outputs: list, **task.inputs) -> dict``

``ask``, ``plan``, ``scaffold``, and ``search`` receive the task description
as their first positional argument.  ``summarize``, ``apply``, and ``verify``
receive the list of all prior task outputs so they can consume the diffs or
answers produced upstream.

If a capability is missing the task is marked SKIPPED rather than FAILED
so partial-feature deployments behave gracefully.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from cgx.agents.judge import Judge
from cgx.agents.types import AgentEvent, Plan, Task, TaskKind, TaskStatus

logger = logging.getLogger(__name__)


Capability = Callable[..., Dict[str, Any]]


# Default cadence at which task_progress heartbeats are emitted while a
# capability is blocked on an LLM call. Two seconds keeps the UI feeling
# alive without spamming the SSE channel.
_DEFAULT_PROGRESS_INTERVAL = 2.0

# Kinds whose failure is treated as local rather than plan-wide. When
# ``stop_on_fail`` is True, the Tracker still continues past these because
# a single bad file (e.g. an LLM-empty ``package.json`` or a syntax-fail
# on one component) shouldn't waste the siblings that already succeeded.
# The APPLY task then writes whatever survived, populating
# ``plan.owned_files`` so the retry loop can regenerate only the failed
# file(s) instead of re-running the entire manifest.
_SOFT_FAIL_KINDS = frozenset({TaskKind.SCAFFOLD_FILE})


def _summarize_task_output(task: Task) -> str:
    """Return a short, human-readable description of what ``task`` produced.

    Never raises: a malformed output falls back to listing the top-level
    keys so the UI can still tell the user *something* was returned.
    """
    out = task.output or {}
    if not isinstance(out, dict):
        return str(out)[:200]
    bits: List[str] = []
    try:
        if task.kind == TaskKind.ASK:
            ans = str(out.get("answer_md") or "").strip()
            if ans:
                first = ans.splitlines()[0] if ans.splitlines() else ans
                bits.append(first[:200] + ("…" if len(first) > 200 else ""))
            cites = out.get("citations") or []
            if cites:
                bits.append(f"{len(cites)} citation(s)")
            conf = out.get("confidence")
            if isinstance(conf, (int, float)):
                bits.append(f"confidence {float(conf):.2f}")
        elif task.kind == TaskKind.PLAN:
            diffs = out.get("diffs") or []
            files = sorted({str(d.get("file") or d.get("path") or "")
                            for d in diffs if isinstance(d, dict)} - {""})
            if diffs:
                bits.append(f"{len(diffs)} diff(s) across {len(files)} file(s)")
                if files:
                    preview = ", ".join(files[:3])
                    if len(files) > 3:
                        preview += f", +{len(files) - 3} more"
                    bits.append(preview)
            plan_md = str(out.get("plan_md") or "").strip()
            if plan_md:
                bits.append(plan_md.splitlines()[0][:160])
            cites = out.get("citations") or []
            if cites:
                bits.append(f"{len(cites)} citation(s)")
            # Surface self-test failure causes so the timeline row tells
            # the user *why* the plan was rejected.
            report = out.get("codegen_report") or {}
            summary = report.get("summary") or {} if isinstance(report, dict) else {}
            if isinstance(summary, dict) and summary.get("overall_ok") is False:
                causes: List[str] = []
                if summary.get("n_patches_failed"):
                    causes.append(f"{int(summary['n_patches_failed'])} patch fail(s)")
                if summary.get("n_syntax_failed"):
                    causes.append(f"{int(summary['n_syntax_failed'])} syntax error(s)")
                if summary.get("tests_ran") and not summary.get("tests_passed"):
                    causes.append("tests failed")
                if causes:
                    bits.append("self-test ✗: " + ", ".join(causes))
        elif task.kind == TaskKind.SCAFFOLD:
            diffs = out.get("diffs") or []
            files = sorted({str(d.get("file") or d.get("path") or "")
                            for d in diffs if isinstance(d, dict)} - {""})
            if files:
                bits.append(f"{len(files)} file(s) generated")
                preview = ", ".join(files[:3])
                if len(files) > 3:
                    preview += f", +{len(files) - 3} more"
                bits.append(preview)
            plan_md = str(out.get("plan_md") or "").strip()
            if plan_md:
                bits.append(plan_md.splitlines()[0][:160])
        elif task.kind == TaskKind.SCAFFOLD_MANIFEST:
            layers = out.get("layers") or []
            n_files = sum(len(lay.get("files") or []) for lay in layers
                          if isinstance(lay, dict))
            if n_files:
                bits.append(f"manifest: {n_files} file(s) across {len(layers)} layer(s)")
            plan_md = str(out.get("plan_md") or "").strip()
            if plan_md:
                bits.append(plan_md.splitlines()[0][:160])
        elif task.kind == TaskKind.SCAFFOLD_FILE:
            fp = str(out.get("file") or "").strip()
            if fp:
                bits.append(fp)
            ok = out.get("syntax_ok")
            if ok is not None:
                bits.append("syntax ok" if ok else "syntax ERROR")
        elif task.kind == TaskKind.SEARCH:
            hits = out.get("hits") or []
            if hits:
                bits.append(f"{len(hits)} hit(s)")
            top_files = out.get("top_files") or []
            names: List[str] = []
            for f in top_files[:3]:
                if isinstance(f, dict):
                    fp = str(f.get("file") or "")
                    if fp:
                        names.append(fp.rsplit("/", 1)[-1])
            if names:
                bits.append("top: " + ", ".join(names))
            impact = out.get("impact") or {}
            impacted = impact.get("impacted_files") if isinstance(impact, dict) else None
            if isinstance(impacted, list) and impacted:
                bits.append(f"{len(impacted)} impacted file(s)")
        elif task.kind == TaskKind.SUMMARIZE:
            ans = str(out.get("answer_md") or "").strip()
            if ans:
                first = ans.splitlines()[0] if ans.splitlines() else ans
                bits.append(first[:200] + ("…" if len(first) > 200 else ""))
        elif task.kind == TaskKind.APPLY:
            applied = out.get("applied_files") or []
            failed = out.get("failed_files") or []
            if applied:
                bits.append(f"applied {len(applied)} file(s)")
                preview = ", ".join(applied[:3])
                if len(applied) > 3:
                    preview += f", +{len(applied) - 3} more"
                bits.append(preview)
            if failed:
                bits.append(f"{len(failed)} failed")
            backup = out.get("backup_dir")
            if backup:
                bits.append(f"backups → {backup}")
        elif task.kind == TaskKind.FILL_LOGIC:
            fn = str(out.get("function_name") or "").strip()
            fp = str(out.get("file_path") or "").strip()
            if fn and fp:
                bits.append(f"{fn}() in {fp}")
            applied = out.get("applied")
            if applied is True:
                bits.append("stitched")
            elif applied is False:
                bits.append("stitch skipped")
            ok = out.get("syntax_ok")
            if ok is not None:
                bits.append("syntax ok" if ok else "syntax ERROR")
        elif task.kind == TaskKind.VERIFY:
            if out.get("tests_passed"):
                n = out.get("tests_selected") or []
                bits.append(f"tests passed ({len(n)} file(s))")
            elif out.get("ran") is False:
                bits.append(out.get("skipped_reason") or "tests skipped")
            else:
                rc = out.get("returncode")
                bits.append(f"tests FAILED (rc={rc})")
    except Exception:
        bits = []
    if not bits:
        keys = ", ".join(list(out.keys())[:6]) or "no output"
        bits.append(f"keys: {keys}")
    return " · ".join(bits)


def _log_codegen_diagnostics(task: Task) -> None:
    """Log the codegen self-test outcome of a PLAN task to the server log.

    Surfaces *why* the self-test failed (no diffs parsed, patch errors,
    syntax errors, or failing tests) so the operator can diagnose issues
    without having to expand the React panel.
    """
    if task.kind != TaskKind.PLAN:
        return
    out = task.output or {}
    report = out.get("codegen_report")
    if not isinstance(report, dict):
        return
    if report.get("error"):
        logger.warning("Tracker: codegen self-test errored task id=%s error=%s",
                       task.id, str(report["error"])[:200])
        return
    summary = report.get("summary") or {}
    if not isinstance(summary, dict):
        return
    if summary.get("overall_ok"):
        logger.info(
            "Tracker: codegen self-test OK task id=%s targets=%d patches_ok=%d/%d "
            "syntax_ok=%d/%d tests_ran=%s tests_passed=%s",
            task.id,
            int(summary.get("n_targets") or 0),
            int(summary.get("n_patches_ok") or 0),
            int(summary.get("n_targets") or 0),
            int(summary.get("n_syntax_ok") or 0),
            int(summary.get("n_targets") or 0),
            bool(summary.get("tests_ran")), bool(summary.get("tests_passed")),
        )
        return
    n_targets = int(summary.get("n_targets") or 0)
    n_p_fail = int(summary.get("n_patches_failed") or 0)
    n_s_fail = int(summary.get("n_syntax_failed") or 0)
    logger.warning(
        "Tracker: codegen self-test FAILED task id=%s targets=%d patches_failed=%d "
        "syntax_failed=%d tests_ran=%s tests_passed=%s",
        task.id, n_targets, n_p_fail, n_s_fail,
        bool(summary.get("tests_ran")), bool(summary.get("tests_passed")),
    )
    if n_targets == 0:
        plan_md = str(out.get("plan_md") or "")
        logger.warning(
            "Tracker: codegen self-test no fenced 'diff path=...' blocks parsed "
            "from plan_md (len=%d, head=%r)", len(plan_md), plan_md[:200],
        )
    for p in (report.get("patches") or [])[:5]:
        if not isinstance(p, dict) or p.get("ok"):
            continue
        logger.warning("Tracker:   patch FAILED file=%r error=%s",
                       p.get("path"), str(p.get("error") or "")[:200])
    for d in (report.get("diagnostics") or [])[:5]:
        if not isinstance(d, dict) or d.get("ok"):
            continue
        logger.warning("Tracker:   syntax FAIL file=%r line=%s error=%s",
                       d.get("path"), d.get("line"), str(d.get("error") or "")[:200])
    tests = report.get("tests")
    if isinstance(tests, dict) and tests.get("ran") and tests.get("returncode") not in (0, None):
        tail = str(tests.get("stdout_tail") or "")[-400:]
        logger.warning("Tracker:   tests FAILED rc=%s tail=%r",
                       tests.get("returncode"), tail)


def _compact_codegen_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return a UI-sized summary of a ``CodegenReport.to_dict()`` payload.

    Surfaces only the failure-diagnosis fields the React panel needs:
    overall verdict, per-patch errors with their rejected-hunk preview,
    syntax diagnostics, and trimmed pytest output. Returns ``{}`` when
    the report wasn't produced (e.g. self_test disabled) or carries an
    error we can't unpack.
    """
    summary = report.get("summary") or {}
    if not summary and not report.get("error"):
        return {}
    compact: Dict[str, Any] = {
        "overall_ok": bool(summary.get("overall_ok")),
        "attempts": int(report.get("attempts") or 0),
    }
    if report.get("error"):
        compact["error"] = str(report["error"])[:240]
        return compact
    counts = {k: int(summary.get(k) or 0) for k in (
        "n_targets", "n_patches_ok", "n_patches_failed",
        "n_syntax_ok", "n_syntax_failed",
    )}
    compact["counts"] = counts
    compact["tests_ran"] = bool(summary.get("tests_ran"))
    compact["tests_passed"] = bool(summary.get("tests_passed"))
    patch_failures: List[Dict[str, Any]] = []
    for p in (report.get("patches") or []):
        if not isinstance(p, dict) or p.get("ok"):
            continue
        rejected = p.get("rejected_hunks") or []
        preview = rejected[0] if rejected else ""
        patch_failures.append({
            "file": str(p.get("path") or ""),
            "error": str(p.get("error") or "patch failed")[:240],
            "rejected_preview": str(preview)[:600],
        })
    if patch_failures:
        compact["patch_failures"] = patch_failures[:10]
    syntax_errors: List[Dict[str, Any]] = []
    for d in (report.get("diagnostics") or []):
        if not isinstance(d, dict) or d.get("ok"):
            continue
        syntax_errors.append({
            "file": str(d.get("path") or ""),
            "language": str(d.get("language") or ""),
            "error": str(d.get("error") or "")[:240],
            "line": d.get("line"),
        })
    if syntax_errors:
        compact["syntax_errors"] = syntax_errors[:10]
    tests = report.get("tests")
    if isinstance(tests, dict):
        compact["tests"] = {
            "ran": bool(tests.get("ran")),
            "returncode": tests.get("returncode"),
            "tests_selected": [str(t) for t in (tests.get("tests_selected") or [])][:10],
            "skipped_reason": tests.get("skipped_reason"),
            "stdout_tail": str(tests.get("stdout_tail") or "")[-2000:],
            "stderr_tail": str(tests.get("stderr_tail") or "")[-1000:],
        }
    return compact


def _extract_display_output(task: Task) -> Dict[str, Any]:
    """Return the user-facing output fields for a completed task.

    Only includes the fields the UI actually renders — keeps the SSE
    payload lean while giving the user something concrete to read.
    """
    out = task.output or {}
    result: Dict[str, Any] = {}
    if task.kind in (TaskKind.PLAN, TaskKind.SCAFFOLD):
        if out.get("plan_md"):
            result["plan_md"] = str(out["plan_md"])
        diffs = out.get("diffs") or []
        if diffs:
            result["diffs"] = [
                {"file": str(d.get("file") or d.get("path") or ""),
                 "patch": str(d.get("patch") or d.get("diff") or "")}
                for d in diffs if isinstance(d, dict)
            ]
        report = out.get("codegen_report")
        if isinstance(report, dict):
            cr = _compact_codegen_report(report)
            if cr:
                result["codegen_report"] = cr
    elif task.kind == TaskKind.SCAFFOLD_MANIFEST:
        if out.get("plan_md"):
            result["plan_md"] = str(out["plan_md"])
        if out.get("layers"):
            result["layers"] = out["layers"]
    elif task.kind == TaskKind.SCAFFOLD_FILE:
        fp = out.get("file")
        if fp:
            result["file"] = str(fp)
        patch = out.get("patch")
        if patch:
            result["diffs"] = [{"file": str(fp or ""), "patch": str(patch)}]
        result["syntax_ok"] = bool(out.get("syntax_ok", True))
        if out.get("syntax_error"):
            result["syntax_error"] = str(out["syntax_error"])
    elif task.kind == TaskKind.FILL_LOGIC:
        if out.get("file_path"):
            result["file_path"] = str(out["file_path"])
        if out.get("function_name"):
            result["function_name"] = str(out["function_name"])
        if out.get("body_code"):
            result["body_code"] = str(out["body_code"])
        result["syntax_ok"] = bool(out.get("syntax_ok", True))
        if out.get("syntax_error"):
            result["syntax_error"] = str(out["syntax_error"])
        if out.get("stitch_error"):
            result["stitch_error"] = str(out["stitch_error"])
    elif task.kind == TaskKind.ASK:
        if out.get("answer_md"):
            result["answer_md"] = str(out["answer_md"])
    elif task.kind == TaskKind.SEARCH:
        top_files = out.get("top_files") or []
        if top_files:
            result["top_files"] = [
                {"file": str(f.get("file") or f.get("path") or ""),
                 "score": float(f.get("score") or f.get("similarity") or 0)}
                for f in top_files[:10] if isinstance(f, dict)
            ]
        hits = out.get("hits") or []
        if hits and not top_files:
            result["top_files"] = [
                {"file": str(h.get("file") or h.get("path") or ""), "score": 0}
                for h in hits[:10] if isinstance(h, dict)
            ]
    elif task.kind == TaskKind.APPLY:
        applied = out.get("applied_files") or []
        failed = out.get("failed_files") or []
        if applied:
            result["applied_files"] = [str(f) for f in applied]
        if failed:
            result["failed_files"] = [
                {"file": str(f.get("file") or f.get("path") or ""),
                 "error": str(f.get("error") or "")}
                for f in failed if isinstance(f, dict)
            ]
        backup = out.get("backup_dir")
        if backup:
            result["backup_dir"] = str(backup)
        project_tree = out.get("project_tree")
        if project_tree:
            result["project_tree"] = str(project_tree)
        diffs = out.get("diffs") or []
        if diffs:
            result["diffs"] = [
                {"file": str(d.get("file") or d.get("path") or ""),
                 "patch": str(d.get("patch") or d.get("diff") or "")}
                for d in diffs if isinstance(d, dict)
            ]
    elif task.kind == TaskKind.VERIFY:
        result["tests_passed"] = bool(out.get("tests_passed"))
        result["ran"] = bool(out.get("ran"))
        if out.get("returncode") is not None:
            result["returncode"] = int(out.get("returncode") or 0)
        if out.get("tests_selected"):
            result["tests_selected"] = [str(t) for t in (out.get("tests_selected") or [])]
        if out.get("skipped_reason"):
            result["skipped_reason"] = str(out.get("skipped_reason"))
        # Trim test output so SSE payloads stay reasonable.
        if out.get("stdout"):
            result["stdout_tail"] = str(out.get("stdout"))[-3000:]
        if out.get("stderr"):
            result["stderr_tail"] = str(out.get("stderr"))[-1500:]
    return result


class Tracker:
    """Execute a :class:`Plan` and emit :class:`AgentEvent`s.

    Use :meth:`stream` for incremental UI updates, or :meth:`run` for a
    blocking call that returns the final plan.
    """

    def __init__(
        self,
        capabilities: Dict[str, Capability],
        judge: Optional[Judge] = None,
        stop_on_fail: bool = True,
        progress_interval: float = _DEFAULT_PROGRESS_INTERVAL,
    ) -> None:
        self.capabilities = capabilities or {}
        self.judge = judge
        self.stop_on_fail = bool(stop_on_fail)
        # ``progress_interval <= 0`` disables threaded dispatch and the
        # task_progress heartbeat entirely (legacy behaviour, used by the
        # unit tests that count emitted events).
        self.progress_interval = float(progress_interval)

    # ------------------------------------------------------------------
    # Streaming execution.
    # ------------------------------------------------------------------
    def stream(self, plan: Plan) -> Iterator[AgentEvent]:
        logger.info("Tracker.stream: starting plan id=%s goal=%r tasks=%d",
                    plan.id, plan.goal[:80], len(plan.tasks))
        yield AgentEvent(type="plan", payload={"plan": plan.to_dict()})
        prior_outputs: List[Dict[str, Any]] = []
        halted = False
        # Index-based loop so that inject_tasks inserted mid-run are visited.
        i = 0
        while i < len(plan.tasks):
            task = plan.tasks[i]
            i += 1

            if halted:
                logger.info("Tracker: skipping task id=%s (plan halted)", task.id)
                task.status = TaskStatus.SKIPPED
                yield AgentEvent(type="task_skipped",
                                 payload={"task_id": task.id})
                continue

            # Skip SCAFFOLD_FILE tasks whose target path is already on
            # disk from a previous attempt. The retry loop carries the
            # prior plan's ``owned_files`` forward, so a stray scaffold
            # for an already-applied file would only overwrite working
            # code. Apply this guard before marking the task RUNNING so
            # the timeline shows it as ⊝ skipped from the start.
            if (
                task.kind == TaskKind.SCAFFOLD_FILE
                and plan.owned_files.get(str(task.inputs.get("path") or "")) == "applied"
            ):
                target = str(task.inputs.get("path") or "")
                task.status = TaskStatus.SKIPPED
                task.started_at = time.time()
                task.ended_at = task.started_at
                reason = (
                    f"already applied on a previous attempt: {target}"
                )
                logger.info(
                    "Tracker: skipping SCAFFOLD_FILE id=%s (%s)",
                    task.id, reason,
                )
                yield AgentEvent(type="task_skipped",
                                 payload={"task_id": task.id,
                                          "kind": task.kind.value,
                                          "name": task.name or task.description,
                                          "reason": reason})
                continue

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            logger.info("Tracker: starting task id=%s kind=%s name=%r",
                        task.id, task.kind.value, (task.name or task.description)[:60])
            yield AgentEvent(type="task_start",
                             payload={"task_id": task.id,
                                      "name": task.name or task.description,
                                      "description": task.description,
                                      "kind": task.kind.value})
            cap = self.capabilities.get(task.kind.value)
            if cap is None:
                task.status = TaskStatus.SKIPPED
                task.ended_at = time.time()
                yield AgentEvent(type="task_skipped",
                                 payload={"task_id": task.id,
                                          "reason": f"no capability for kind={task.kind.value}"})
                continue
            try:
                if self.progress_interval > 0:
                    output_holder: Dict[str, Any] = {}
                    error_holder: List[BaseException] = []

                    def _worker(_cap=cap, _task=task,
                                _prior=prior_outputs) -> None:
                        try:
                            output_holder["out"] = self._dispatch(
                                _cap, _task, _prior)
                        except BaseException as e:  # noqa: BLE001
                            error_holder.append(e)

                    worker = threading.Thread(target=_worker, daemon=True)
                    worker.start()
                    while True:
                        worker.join(self.progress_interval)
                        if not worker.is_alive():
                            break
                        elapsed = time.time() - (task.started_at or time.time())
                        yield AgentEvent(
                            type="task_progress",
                            payload={"task_id": task.id,
                                     "name": task.name or task.description,
                                     "kind": task.kind.value,
                                     "elapsed": elapsed})
                    if error_holder:
                        raise error_holder[0]
                    output = output_holder.get("out", {})
                else:
                    output = self._dispatch(cap, task, prior_outputs)
                task.output = output if isinstance(output, dict) else {"result": output}
                task.status = TaskStatus.DONE
                logger.info("Tracker: task DONE id=%s kind=%s", task.id, task.kind.value)
                _log_codegen_diagnostics(task)
            except Exception as e:
                task.error = f"{type(e).__name__}: {e}"
                task.status = TaskStatus.FAILED
                task.ended_at = time.time()
                logger.warning("Tracker: task FAILED id=%s error=%s", task.id, task.error)
                yield AgentEvent(type="task_failed",
                                 payload={"task_id": task.id, "error": task.error})
                if self.stop_on_fail and task.kind not in _SOFT_FAIL_KINDS:
                    halted = True
                continue

            # Judge step.
            if self.judge is not None:
                verdict = self.judge.judge(task)
                task.judge = verdict.to_dict()
                logger.info("Tracker: judge task id=%s verdict=%s confidence=%.2f",
                            task.id, verdict.verdict, verdict.confidence)
                yield AgentEvent(type="judge", payload={"task_id": task.id,
                                                        **task.judge})
                if not verdict.passed:
                    task.status = TaskStatus.FAILED
                    task.ended_at = time.time()
                    logger.warning("Tracker: judge FAILED task id=%s rationale=%r",
                                   task.id, verdict.rationale[:100])
                    # Pass the capability's output through so the UI can
                    # still render the rejected diff + codegen_report
                    # rather than leaving the user with just an error.
                    yield AgentEvent(type="task_failed",
                                     payload={"task_id": task.id,
                                              "kind": task.kind.value,
                                              "name": task.name or task.description,
                                              "error": f"judge: {verdict.rationale}",
                                              "summary": _summarize_task_output(task),
                                              "output": _extract_display_output(task)})
                    if self.stop_on_fail and task.kind not in _SOFT_FAIL_KINDS:
                        halted = True
                    continue

            task.ended_at = time.time()
            prior_outputs.append(task.output or {})
            # Update the plan's file manifest after every APPLY task so the
            # retry loop knows which files are on disk and which still need fixing.
            if task.kind == TaskKind.APPLY:
                out = task.output or {}
                for fp in (out.get("applied_files") or []):
                    plan.owned_files[str(fp)] = "applied"
                for entry in (out.get("failed_files") or []):
                    fp = entry.get("file") if isinstance(entry, dict) else str(entry)
                    if fp:
                        plan.owned_files[str(fp)] = "failed"

            # Dynamic task injection: if a capability (e.g. scaffold_manifest)
            # returned an "inject_tasks" list, insert those tasks immediately
            # after the current position so they execute before APPLY/VERIFY.
            inject = (task.output or {}).pop("inject_tasks", None)
            if inject and isinstance(inject, list):
                logger.info(
                    "Tracker: injecting %d task(s) after task id=%s",
                    len(inject), task.id)
                for j, new_task in enumerate(inject):
                    plan.tasks.insert(i + j, new_task)
                # Re-emit the updated plan so the UI can render new rows.
                yield AgentEvent(type="plan", payload={"plan": plan.to_dict()})

            elapsed = (task.ended_at - task.started_at) if task.started_at else None
            logger.info("Tracker: task completed id=%s elapsed=%.1fs", task.id, elapsed or 0)
            yield AgentEvent(type="task_done",
                             payload={"task_id": task.id,
                                      "kind": task.kind.value,
                                      "name": task.name or task.description,
                                      "description": task.description,
                                      "output_keys": list((task.output or {}).keys()),
                                      "summary": _summarize_task_output(task),
                                      "elapsed": elapsed,
                                      "output": _extract_display_output(task)})

        completed = sum(1 for t in plan.tasks if t.status == TaskStatus.DONE)
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in plan.tasks if t.status == TaskStatus.SKIPPED)
        logger.info("Tracker.stream: plan complete completed=%d failed=%d skipped=%d",
                    completed, failed, skipped)
        yield AgentEvent(type="summary",
                         payload={"plan": plan.to_dict(),
                                  "completed": completed,
                                  "failed": failed,
                                  "skipped": skipped})

    def run(self, plan: Plan) -> Plan:
        """Drain :meth:`stream` and return the (mutated) plan."""
        for _ in self.stream(plan):
            pass
        return plan

    # ------------------------------------------------------------------
    # Capability dispatch.
    # ------------------------------------------------------------------
    @staticmethod
    def _dispatch(cap: Capability, task: Task,
                  prior_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = dict(task.inputs)
        if task.kind in (TaskKind.SUMMARIZE, TaskKind.APPLY, TaskKind.VERIFY):
            # These kinds consume previous task outputs (the prior PLAN's
            # diffs in particular) rather than a free-text query.
            return cap(prior_outputs, **inputs)
        if task.kind in (TaskKind.SCAFFOLD, TaskKind.SCAFFOLD_MANIFEST, TaskKind.SCAFFOLD_FILE):
            # Forward prior outputs so sibling scaffold tasks share one
            # coherent file tree. The wrapper in loop.py extracts
            # ``_prior_outputs`` before calling the engine.
            return cap(task.description, _prior_outputs=prior_outputs, **inputs)
        # FILL_LOGIC: task description carries the instruction; inputs carry
        # file_path, function_name, and optional skeleton.
        if task.kind == TaskKind.FILL_LOGIC:
            return cap(task.description, **inputs)
        # ASK / PLAN / SEARCH all take the task description as the first arg.
        return cap(task.description, **inputs)

    # ------------------------------------------------------------------
    # Convenience renderer for the UI/CLI.
    # ------------------------------------------------------------------
    @staticmethod
    def render_plan(plan: Plan) -> str:
        return "\n".join(plan.summary_lines())
