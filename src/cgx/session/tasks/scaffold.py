

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
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.pypi_client import PyPIClient
from cgx.session.scaffold_validate import (
    check_contract_compliance,
    cross_check_first_party_imports,
    validate_scaffold_diffs,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)

# Default intra-layer worker count for providers that advertise
# ``parallel_scaffold_capable`` (remote/cloud endpoints) when
# ``CGX_SCAFFOLD_CONCURRENCY`` is unset. Bounded so a burst of sibling
# files never opens an unreasonable number of concurrent HTTP requests.
_CLOUD_SCAFFOLD_CONCURRENCY = 4

# Number of global coherence passes run after the main scaffold loop
# (#2). Each pass re-checks first-party imports across the finished tree
# and regenerates only the importer files that reference a symbol no
# sibling defines. One pass clears the common single-hop mismatch; a hard
# cap stops a stubborn model from looping and keeps the retry bounded.
_COHERENCE_PASS_BUDGET = 1


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

    # Contract-first (P0): the WORK_PLAN carries a ``contracts`` block --
    # the shared interfaces (endpoints, schemas, function signatures,
    # constants) DECOMPOSE declared. Thread it into every per-file
    # generation so cross-file assumptions are honoured, not re-derived.
    # Absent/malformed -> ``None`` so the generator prompt is unchanged.
    contracts = content.get("contracts")
    if not isinstance(contracts, dict) or not contracts:
        contracts = None

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

    diffs: List[Dict[str, str]] = []
    existing_with_content: List[Dict[str, str]] = []
    generated: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    concurrency = _scaffold_concurrency(deps.provider)

    # Targeted regeneration (router failed-files splices): when a prior
    # SCAFFOLD/APPLY dropped specific files, only those paths are
    # re-generated while every prior-good diff is reused verbatim. This
    # keeps the retry proportional to the failure instead of re-running the
    # whole manifest and risking re-breaking files that were fine. Falls
    # back to a whole-tree regenerate when the marker or prior artifact is
    # absent, so the happy path is unchanged.
    regen_set = _resolve_regenerate_set(task, deps)
    if regen_set is not None:
        for reused in _reused_good_files(task, deps, regen_set):
            diffs.append({"file": reused["path"], "patch": reused["patch"]})
            existing_with_content.append(
                {"path": reused["path"], "content": reused["content"]})
            generated.append({
                "file": reused["path"], "layer": "reused",
                "syntax_ok": True, "confidence": None,
                "bytes": len(reused["content"]), "reused": True,
            })

    # Resumable progress (B4): when a prior SCAFFOLD for this same work
    # plan crashed or was killed mid-run, the runner threads the id of its
    # last on-disk checkpoint here. Seed every already-generated file so
    # this attempt regenerates only the remainder instead of discarding
    # completed work. Skipped when a targeted regenerate is active (that
    # path already reuses prior-good diffs) or the pointer is unresolvable.
    resume_done: set = set()
    if regen_set is None:
        for done in _resume_generated_files(task, deps, work_plan_id):
            diffs.append({"file": done["path"], "patch": done["patch"]})
            existing_with_content.append(
                {"path": done["path"], "content": done["content"]})
            generated.append({
                "file": done["path"], "layer": "resumed",
                "syntax_ok": True, "confidence": None,
                "bytes": len(done["content"]), "resumed": True,
            })
            resume_done.add(done["path"])

    # Build the SCAFFOLD_PATCHES artifact up front with a stable id and
    # ``complete=False`` so it can be checkpointed to the store after every
    # layer. ``content`` holds live references to the diffs/generated/failed
    # lists, so each checkpoint save reflects the latest progress; the final
    # save (with pin adjustments + ``complete=True``) upserts the same row.
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
            "pin_adjustments": [],
            "complete": False,
        },
    )

    # Progress denominator: every planned manifest file. ``progress_done``
    # counts files whose content is settled (reused/resumed already, then
    # each freshly generated file) so the UI can render "i / total" and an
    # ETA while a long serial SCAFFOLD grinds through the manifest.
    total_files = sum(
        1 for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or [])
        if isinstance(e, dict) and str(e.get("path") or "").strip())
    progress_done = len(generated)
    # Running count of files whose generation failed. Threaded into every
    # beat so the UI can distinguish a genuine failure from a counter that
    # simply hasn't advanced: on failure ``progress_done`` stays put, so
    # without this the next file's ``start`` reuses the same index and looks
    # like a silent restart.
    progress_failed = 0

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "project").strip()
        # Collect this layer's files (manifest order) after the targeted-
        # regenerate skip, so the parallel and serial paths share one plan.
        pending: List[Tuple[str, str, List[str]]] = []
        for entry in (layer.get("files") or []):
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            description = str(entry.get("description") or path).strip()
            if not path:
                continue
            # Manifest-declared dependency paths for this file, used to
            # scope the generator's context digest (P1.3). Absent/malformed
            # entries degrade to an empty list -> full-context fallback.
            raw_deps = entry.get("depends_on")
            depends_on = [str(d).strip() for d in raw_deps
                          if isinstance(d, str) and str(d).strip()] \
                if isinstance(raw_deps, list) else []
            # Targeted regenerate: skip every file not slated for
            # regeneration -- it was reused above (or was never at fault).
            if regen_set is not None and path not in regen_set:
                continue
            # Resume: skip files already generated by the crashed attempt.
            if path in resume_done:
                continue
            pending.append((path, description, depends_on))
        if not pending:
            continue

        parallel = concurrency > 1 and len(pending) > 1
        if parallel:
            # Bounded fan-out within the layer: every file sees the same
            # frozen cross-layer context (prior layers only), so sibling
            # generations are independent and safe to run concurrently.
            # Results are gathered by manifest index so diff/failure order
            # is deterministic regardless of completion order. The worker
            # count honours the GPU throttle -- it defaults to 1 (serial)
            # so a single local GPU is never over-subscribed.
            context_snapshot = list(existing_with_content)
            layer_results: List[Any] = [None] * len(pending)
            max_workers = min(concurrency, len(pending))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_to_idx = {
                    pool.submit(
                        _generate_one, p, d, layer_name,
                        list(context_snapshot), deps.provider, goal,
                        depends_on=dep, contracts=contracts): i
                    for i, (p, d, dep) in enumerate(pending)
                }
                for fut in as_completed(fut_to_idx):
                    layer_results[fut_to_idx[fut]] = fut.result()
        else:
            # Serial: each file additionally sees its already-generated
            # siblings, maximising intra-layer import resolution (the legacy
            # behaviour, preserved as the default). A start/done progress
            # pair is emitted per file so the UI shows the active file and
            # advancing count instead of a frozen IN_PROGRESS.
            layer_results = []
            for (p, d, dep) in pending:
                _emit_scaffold_progress(
                    deps, task, file=p, layer=layer_name,
                    index=progress_done + 1, total=total_files,
                    status="start", failed_count=progress_failed)
                started = time.time()
                on_token = _make_stream_beat(
                    deps, task, file=p, layer=layer_name,
                    index=progress_done + 1, total=total_files,
                    failed_count=progress_failed)
                ok, fail = _generate_one(
                    p, d, layer_name, list(existing_with_content),
                    deps.provider, goal, on_token=on_token,
                    depends_on=dep, contracts=contracts)
                elapsed_ms = int((time.time() - started) * 1000)
                layer_results.append((ok, fail))
                if ok is not None:
                    existing_with_content.append(
                        {"path": ok["file"], "content": ok["content"]})
                    progress_done += 1
                    _emit_scaffold_progress(
                        deps, task, file=p, layer=layer_name,
                        index=progress_done, total=total_files,
                        status="done", bytes=len(ok["content"]),
                        elapsed_ms=elapsed_ms,
                        failed_count=progress_failed)
                else:
                    progress_failed += 1
                    _emit_scaffold_progress(
                        deps, task, file=p, layer=layer_name,
                        index=progress_done, total=total_files,
                        status="failed", elapsed_ms=elapsed_ms,
                        failed_count=progress_failed)

        # Fold the layer's outcomes into the running state in manifest
        # order. In the parallel path the context snapshot was frozen, so
        # append successes now to feed the next layer.
        for ok, fail in layer_results:
            if fail is not None:
                failed.append(fail)
                if parallel:
                    progress_failed += 1
                    _emit_scaffold_progress(
                        deps, task, file=str(fail.get("file") or ""),
                        layer=layer_name, index=progress_done,
                        total=total_files, status="failed",
                        failed_count=progress_failed)
                continue
            diffs.append({"file": ok["file"], "patch": ok["patch"]})
            generated.append({
                "file": ok["file"],
                "layer": ok["layer"],
                "syntax_ok": ok["syntax_ok"],
                "confidence": ok["confidence"],
                "bytes": len(ok["content"]),
            })
            if parallel:
                existing_with_content.append(
                    {"path": ok["file"], "content": ok["content"]})
                progress_done += 1
                _emit_scaffold_progress(
                    deps, task, file=ok["file"], layer=layer_name,
                    index=progress_done, total=total_files,
                    status="done", bytes=len(ok["content"]),
                    failed_count=progress_failed)

        # Checkpoint after each layer so a crash mid-run is resumable.
        _checkpoint_progress(deps, artifact)

    if not diffs:
        return ExecutorResult(
            failure="SCAFFOLD: every file generation failed")

    # Global coherence pass (#2): the per-file loop resolves imports
    # against whichever siblings existed when each file was generated (and
    # the parallel path freezes a per-layer snapshot), so a file can still
    # reference a symbol another sibling never defined. Before APPLY writes
    # anything, re-check first-party imports across the whole tree and
    # regenerate just the importer files that don't resolve, folding the
    # unresolved symbols in as a constraint so cross-file mismatches
    # self-heal. Best-effort and bounded -- a failure leaves the bundle
    # untouched.
    reconciled_count = 0
    try:
        reconciled_count = _reconcile_import_warnings(
            diffs=diffs, generated=generated,
            existing_with_content=existing_with_content,
            layers=layers, goal=goal, provider=deps.provider,
            contracts=contracts)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: coherence reconciliation raised; skipping")
        reconciled_count = 0
    if reconciled_count:
        _checkpoint_progress(deps, artifact)

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

    # Phase 3.3: static first-party import cross-check. Parse every
    # generated Python file and flag ``from <local> import <name>`` where
    # ``name`` is absent from the referenced module. Best-effort and
    # non-fatal -- any failure degrades to an empty warning list so the
    # scaffold is never blocked by the checker itself.
    import_warnings: List[Dict[str, Any]] = []
    try:
        xcheck_contents = {
            e["path"]: e["content"] for e in existing_with_content
            if e.get("path") and isinstance(e.get("content"), str)}
        import_warnings = cross_check_first_party_imports(xcheck_contents)
    except Exception:  # pragma: no cover - defensive: checker is best-effort
        logger.exception(
            "SCAFFOLD: import cross-check raised; skipping")
        import_warnings = []

    # Contract enforcement gate (#1, deepens P0): statically verify the
    # generated files honour the WORK_PLAN ``contracts`` block (declared
    # endpoints/schemas/functions/constants) so a mismatch surfaces here
    # rather than only when VERIFY runs the suite. Best-effort and
    # non-fatal -- like the import cross-check it only records
    # ``contract_warnings`` the router can turn into a regenerate
    # constraint; a raised checker degrades to an empty list.
    contract_warnings: List[Dict[str, Any]] = []
    try:
        contract_warnings = check_contract_compliance(
            xcheck_contents, contracts)
    except Exception:  # pragma: no cover - defensive: checker is best-effort
        logger.exception(
            "SCAFFOLD: contract compliance check raised; skipping")
        contract_warnings = []

    # Finalise the checkpoint artifact in place: pin validation reassigns
    # ``diffs`` to a new list, so re-point the content at it, attach the
    # adjustment log, and flip ``complete``. Same artifact_id, so the
    # runner's post-return save_artifact upserts the checkpoint row.
    artifact.content["diffs"] = diffs
    artifact.content["generated"] = generated
    artifact.content["failed"] = failed
    artifact.content["pin_adjustments"] = pin_adjustments
    artifact.content["import_warnings"] = import_warnings
    artifact.content["contract_warnings"] = contract_warnings
    artifact.content["reconciled_count"] = reconciled_count
    artifact.content["complete"] = True
    return ExecutorResult(
        outputs={
            "scaffold_artifact_id": artifact.artifact_id,
            "generated_count": len(generated),
            "failed_count": len(failed),
            "failed": failed,
            "pin_adjustments_count": len(pin_adjustments),
            "import_warnings_count": len(import_warnings),
            "contract_warnings_count": len(contract_warnings),
            "contract_warnings": contract_warnings,
            "reconciled_count": reconciled_count,
        },
        artifact=artifact,
    )


def _generate_one(
        path: str, description: str, layer_name: str,
        context: List[Dict[str, str]], provider: Any, goal: str,
        on_token: Optional[Callable[[str], None]] = None,
        depends_on: Optional[List[str]] = None,
        contracts: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Generate one scaffold file. Returns ``(ok_entry, fail_entry)``.

    Exactly one element is non-``None``. ``ok_entry`` carries both the diff
    and the metadata/content the caller folds into the artifact; ``fail_entry``
    is the ``{file, error}`` shape the router turns into a targeted regenerate
    constraint. It reads no shared state, so it is safe to run inside the
    bounded per-layer worker pool. The generator import is resolved on each
    call so monkeypatched stubs (and any hot-swapped prompt templates) take
    effect. ``on_token`` (serial path only) forwards streamed generation
    deltas for live progress. ``depends_on`` scopes the context digest to
    the manifest-declared dependency paths. ``contracts`` is the WORK_PLAN
    shared-interface block, threaded verbatim so every file honours the
    same declared endpoints/schemas/signatures.
    """
    from cgx.answer.engine import generate_single_scaffold_file
    try:
        result = generate_single_scaffold_file(
            path, description, provider,
            layer=layer_name,
            existing_files_with_content=context,
            goal=goal,
            on_token=on_token,
            depends_on=depends_on,
            contracts=contracts,
        )
    except Exception as exc:
        logger.exception(
            "SCAFFOLD: generate_single_scaffold_file crashed for %s", path)
        return None, {"file": path, "error": f"{type(exc).__name__}: {exc}"}

    file_path = str(result.get("file") or path).strip()
    patch = str(result.get("patch") or "")
    file_content = str(result.get("content") or "")
    syntax_error = str(result.get("syntax_error") or "").strip()
    if not file_path or not patch:
        # A cleared/empty patch means a single-file gate (duplicate content,
        # undefined first-party symbol, ...) already rejected the body.
        # Surface its concrete reason when present so the router's
        # regenerate constraint is specific rather than a generic empty-patch.
        return None, {"file": file_path or path,
                      "error": syntax_error or "generator returned empty patch"}
    if not bool(result.get("syntax_ok")):
        # Explicit file-level failure (P2.1): a file that failed the inline
        # syntax gate (Python/TOML/JS/TS/JSX/Vue) or the extension/content
        # mismatch check still carries a non-empty patch here. Shipping it
        # to APPLY only to have APPLY's own syntax gate silently drop it can
        # leave the project without an entry point or its only test (VERIFY
        # then reports a false success). Fail it explicitly instead so the
        # router turns it into a targeted regenerate with the exact error.
        return None, {"file": file_path,
                      "error": syntax_error
                      or "generated file failed syntax validation"}
    return {
        "file": file_path,
        "patch": patch,
        "content": file_content,
        "layer": layer_name,
        "syntax_ok": bool(result.get("syntax_ok")),
        "confidence": result.get("confidence"),
    }, None


def _emit_scaffold_progress(
        deps: ExecutorDeps, task: TaskNode, *, file: str, layer: str,
        index: int, total: int, status: str,
        bytes: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        failed_count: int = 0) -> None:
    """Publish one intra-SCAFFOLD progress beat (best-effort).

    ``status`` is one of ``start`` / ``done`` / ``failed``. A rough ETA is
    derived from the just-finished file's wall time projected over the
    remaining manifest entries -- coarse, but enough for the UI to show a
    shrinking countdown instead of a frozen spinner. ``failed_count`` carries
    the running number of files that failed so far; on failure ``index`` does
    not advance, so this lets the UI show the failure instead of rendering an
    apparent counter reset. Never raises: progress telemetry must not be able
    to fail a generation run.
    """
    store = getattr(deps, "store", None)
    if store is None:
        return
    progress: Dict[str, Any] = {
        "index": index, "total": total, "path": file,
        "layer": layer, "status": status, "failed_count": failed_count,
    }
    if bytes is not None:
        progress["bytes"] = bytes
    if elapsed_ms is not None:
        progress["elapsed_ms"] = elapsed_ms
        remaining = max(total - index, 0)
        progress["eta_seconds"] = round((elapsed_ms / 1000.0) * remaining, 1)
    else:
        progress["eta_seconds"] = None
    try:
        store.emit_task_progress(task.session_id, task.task_id, progress)
    except Exception:  # pragma: no cover - telemetry must never break a run
        logger.debug("SCAFFOLD: progress emit failed", exc_info=True)


# Minimum wall gap between streamed ``stream`` beats. Token deltas arrive
# far faster than the UI needs; coalescing to a few per second keeps the
# progress bar moving without flooding the SSE bridge.
_STREAM_BEAT_MIN_INTERVAL_S = 0.25


def _make_stream_beat(
        deps: ExecutorDeps, task: TaskNode, *, file: str, layer: str,
        index: int, total: int,
        failed_count: int = 0) -> Callable[[str], None]:
    """Return an ``on_token`` callback that emits throttled ``stream`` beats.

    Accumulates the streamed character count for the file being generated
    and publishes a ``status="stream"`` progress beat at most every
    :data:`_STREAM_BEAT_MIN_INTERVAL_S` seconds, so the UI shows the active
    file growing in real time instead of a frozen ``start``. ``failed_count``
    is the running failure tally carried into each beat. Best-effort:
    delegates to :func:`_emit_scaffold_progress`, which never raises.
    """
    state = {"chars": 0, "last": 0.0}

    def _beat(delta: str) -> None:
        state["chars"] += len(delta)
        now = time.time()
        if now - state["last"] < _STREAM_BEAT_MIN_INTERVAL_S:
            return
        state["last"] = now
        _emit_scaffold_progress(
            deps, task, file=file, layer=layer,
            index=index, total=total, status="stream",
            bytes=state["chars"], failed_count=failed_count)

    return _beat


def _scaffold_concurrency(provider: Any = None) -> int:
    """Return the bounded per-layer generation worker count (>= 1).

    Provider-gated: a provider that advertises ``parallel_scaffold_capable``
    (remote/cloud endpoints) fans out to :data:`_CLOUD_SCAFFOLD_CONCURRENCY`
    by default, while a single local GPU (Ollama) stays serial so it is never
    over-subscribed -- the runner's ``_GPU_INFERENCE_SEMAPHORE`` already
    serialises whole LLM tasks, so intra-layer fan-out only pays off when the
    backend can service concurrent requests.

    ``CGX_SCAFFOLD_CONCURRENCY`` overrides the gate in both directions (pin a
    cloud run serial, or opt a local host with GPU headroom into fan-out);
    malformed or sub-1 values clamp to 1.
    """
    raw = os.environ.get("CGX_SCAFFOLD_CONCURRENCY")
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1
    if getattr(provider, "parallel_scaffold_capable", False):
        return _CLOUD_SCAFFOLD_CONCURRENCY
    return 1


def _resolve_pypi_client(deps: ExecutorDeps) -> PyPIClient:
    """Return the injected PyPI client, or build a default."""
    injected = (deps.extra or {}).get("pypi_client")
    if isinstance(injected, PyPIClient):
        return injected
    return PyPIClient()


def _resolve_regenerate_set(
        task: TaskNode, deps: ExecutorDeps) -> Optional[set]:
    """Return the set of paths to regenerate, or ``None`` for whole-tree.

    Targeted regeneration is active only when the router threaded both a
    non-empty ``regenerate_files`` list and a ``prior_scaffold_artifact_id``
    that resolves to a real SCAFFOLD_PATCHES artifact. Any missing piece
    degrades to a whole-tree regenerate (``None``) so a stale or dangling
    marker can never cause files to be silently skipped.
    """
    files = task.inputs.get("regenerate_files")
    prior_id = str(task.inputs.get("prior_scaffold_artifact_id") or "").strip()
    if not isinstance(files, list) or not files or not prior_id:
        return None
    if deps.store is None:
        return None
    prior = deps.store.get_artifact(prior_id)
    if prior is None or prior.kind is not ArtifactKind.SCAFFOLD_PATCHES:
        return None
    regen = {str(p).strip() for p in files if str(p).strip()}
    return regen or None


def _reused_good_files(
        task: TaskNode, deps: ExecutorDeps,
        regen_set: set) -> List[Dict[str, str]]:
    """Return prior-good ``{path, patch, content}`` entries to reuse verbatim.

    Reads the prior SCAFFOLD_PATCHES diffs and returns every file *not* in
    ``regen_set``. ``content`` is reconstructed from the new-file patch so
    the regenerated files still receive the good files as cross-file
    context (imports, symbol inventory). The caller has already verified
    the artifact resolves via :func:`_resolve_regenerate_set`.
    """
    prior_id = str(task.inputs.get("prior_scaffold_artifact_id") or "").strip()
    prior = deps.store.get_artifact(prior_id)
    out: List[Dict[str, str]] = []
    seen: set = set()
    for d in ((prior.content or {}).get("diffs") or []):
        if not isinstance(d, dict):
            continue
        fpath = str(d.get("file") or "").strip()
        patch = str(d.get("patch") or "")
        if not fpath or not patch or fpath in regen_set or fpath in seen:
            continue
        seen.add(fpath)
        out.append({
            "path": fpath, "patch": patch,
            "content": _content_from_new_file_patch(patch),
        })
    return out


def _content_from_new_file_patch(patch: str) -> str:
    """Reconstruct file content from a ``/dev/null`` new-file unified diff.

    Inverse of ``engine._content_to_new_file_patch``: strips the leading
    ``+`` from every body line after the ``@@`` hunk header. Best-effort --
    used only to seed cross-file context, never to write to disk (the
    reused patch itself is what APPLY re-applies).
    """
    body: List[str] = []
    started = False
    for line in patch.splitlines():
        if not started:
            if line.startswith("@@"):
                started = True
            continue
        if line.startswith("+"):
            body.append(line[1:])
    return "\n".join(body)


def _checkpoint_progress(deps: ExecutorDeps, artifact: Artifact) -> None:
    """Persist the in-progress SCAFFOLD_PATCHES artifact after a layer.

    Best-effort: ``save_artifact`` uses ``INSERT OR REPLACE`` keyed by
    ``artifact_id``, so calling it after every layer upserts the same row
    with the latest ``diffs``/``generated``/``failed`` -- a crash or
    timeout mid-run then leaves the completed files on disk for
    :func:`_resume_generated_files` to seed. A checkpoint save must never
    fail the generation task itself, so any store error is swallowed
    (the next layer's checkpoint, or the runner's post-return save, will
    persist the same state).
    """
    store = deps.store
    if store is None:
        return
    try:
        store.save_artifact(artifact)
    except Exception:  # pragma: no cover - defensive: checkpoint is best-effort
        logger.exception("SCAFFOLD: checkpoint save_artifact failed")


def _resume_generated_files(
        task: TaskNode, deps: ExecutorDeps,
        work_plan_id: str) -> List[Dict[str, str]]:
    """Return already-generated ``{path, patch, content}`` from a checkpoint.

    Resume is active only when the router threaded a
    ``resume_scaffold_artifact_id`` that resolves to a real
    SCAFFOLD_PATCHES artifact produced for the *same* work plan -- a
    checkpoint from a different plan (or a dangling id) degrades to an
    empty list so a stale marker can never seed the wrong files. Content
    is reconstructed from each new-file patch (as in
    :func:`_reused_good_files`) so resumed files still serve as cross-file
    context for the remaining generations.
    """
    resume_id = str(task.inputs.get("resume_scaffold_artifact_id") or "").strip()
    if not resume_id or deps.store is None:
        return []
    prior = deps.store.get_artifact(resume_id)
    if prior is None or prior.kind is not ArtifactKind.SCAFFOLD_PATCHES:
        return []
    prior_content = prior.content or {}
    if str(prior_content.get("work_plan_artifact_id") or "").strip() \
            != work_plan_id:
        return []
    out: List[Dict[str, str]] = []
    seen: set = set()
    for d in (prior_content.get("diffs") or []):
        if not isinstance(d, dict):
            continue
        fpath = str(d.get("file") or "").strip()
        patch = str(d.get("patch") or "")
        if not fpath or not patch or fpath in seen:
            continue
        seen.add(fpath)
        out.append({
            "path": fpath, "patch": patch,
            "content": _content_from_new_file_patch(patch),
        })
    return out


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


def _reconcile_import_warnings(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
        goal: str,
        provider: Any,
        contracts: Optional[Dict[str, Any]],
) -> int:
    """Regenerate importer files whose first-party imports don't resolve.

    Re-checks first-party imports across the finished scaffold tree and
    regenerates only the importer files carrying an unresolved symbol,
    folding the missing ``module.name`` list in as a constraint so the
    model aligns each file to what its siblings actually define. Mutates
    ``diffs``/``generated``/``existing_with_content`` in place and returns
    the number of files rewritten. Bounded by
    :data:`_COHERENCE_PASS_BUDGET`; a regeneration that fails or raises
    leaves the original file untouched, so the pass can only improve (or
    no-op) the bundle.
    """
    if provider is None:
        return 0
    # Manifest lookup: path -> (description, layer_name, depends_on).
    meta: Dict[str, Tuple[str, str, List[str]]] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "project").strip()
        for entry in (layer.get("files") or []):
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            raw_deps = entry.get("depends_on")
            dep = [str(d).strip() for d in raw_deps
                   if isinstance(d, str) and str(d).strip()] \
                if isinstance(raw_deps, list) else []
            desc = str(entry.get("description") or path).strip()
            meta[path] = (desc, layer_name, dep)

    def _find(seq: List[Dict[str, Any]], key: str, value: str) -> int:
        for i, e in enumerate(seq):
            if str(e.get(key) or "") == value:
                return i
        return -1

    reconciled = 0
    for _ in range(_COHERENCE_PASS_BUDGET):
        contents = {e["path"]: e["content"] for e in existing_with_content
                    if e.get("path") and isinstance(e.get("content"), str)}
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for w in cross_check_first_party_imports(contents):
            f = str(w.get("file") or "").strip()
            if f and f in meta:
                by_file.setdefault(f, []).append(w)
        if not by_file:
            break
        rewritten = 0
        for path, file_warnings in by_file.items():
            desc, layer_name, dep = meta[path]
            constraints = [{
                "kind": "unresolved import",
                "rationale": (
                    f"`{w.get('name')}` imported from `{w.get('module')}` "
                    "is not defined there; import only names that exist in "
                    "the referenced sibling modules, or define what you use"),
            } for w in file_warnings]
            aug_goal = _augment_goal_with_constraints(
                goal, constraints,
                header="Cross-file import mismatches to fix this pass:")
            context = [e for e in existing_with_content
                       if e.get("path") != path]
            ok, _fail = _generate_one(
                path, desc, layer_name, context, provider, aug_goal,
                depends_on=dep, contracts=contracts)
            if ok is None:
                continue
            di = _find(diffs, "file", path)
            if di >= 0:
                diffs[di] = {"file": ok["file"], "patch": ok["patch"]}
            gi = _find(generated, "file", path)
            if gi >= 0:
                generated[gi] = {
                    "file": ok["file"], "layer": ok["layer"],
                    "syntax_ok": ok["syntax_ok"],
                    "confidence": ok["confidence"],
                    "bytes": len(ok["content"]), "reconciled": True,
                }
            ei = _find(existing_with_content, "path", path)
            if ei >= 0:
                existing_with_content[ei] = {
                    "path": ok["file"], "content": ok["content"]}
            rewritten += 1
        reconciled += rewritten
        if rewritten == 0:
            break
    return reconciled


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
