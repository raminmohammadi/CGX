

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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.pypi_client import PyPIClient
from cgx.session.scaffold_validate import (
    check_client_server_payload_coherence,
    check_contract_compliance,
    cross_check_first_party_imports,
    is_requirements_path,
    validate_scaffold_diffs,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
    session_skills,
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

    # Explicit skill selection (e.g. from an Agent Profile) pinned on the
    # session; ``None`` when unset so the generator falls back to
    # auto-detecting from ``goal`` exactly as before.
    skills = session_skills(task, deps)

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

    # Regenerate that must *add* files (B: missing entry point): a build
    # that cannot resolve an entry module is not fixable by re-authoring
    # the planned files -- the file the toolchain wants was never in the
    # manifest. REPAIR names it in ``additional_files``; fold those
    # entries into the manifest for this attempt (and into the targeted
    # set, so they are actually generated rather than skipped).
    layers, added_paths = _with_additional_files(task, layers)
    if added_paths and regen_set is not None:
        regen_set = regen_set | set(added_paths)

    # Durable env fix: a repair may have re-pinned requirements.txt to a
    # self-consistent, conflict-free set (env_manager conflict re-resolve)
    # and marked it env-locked. A whole-tree regenerate would otherwise
    # re-emit the model's stale manifest and clobber that fix, reviving the
    # exact dependency conflict the repair resolved. Carry every env-locked
    # file forward verbatim off disk and skip regenerating it below (this
    # also takes precedence over the prior-scaffold reuse, whose diff still
    # holds the pre-repair pins). Best-effort: no marker or no file on disk
    # leaves the normal generate path untouched.
    carried: set = set()
    for locked in _carry_forward_locked_files(task, deps):
        diffs.append({"file": locked["path"], "patch": locked["patch"]})
        existing_with_content.append(
            {"path": locked["path"], "content": locked["content"]})
        generated.append({
            "file": locked["path"], "layer": "carried",
            "syntax_ok": True, "confidence": None,
            "bytes": len(locked["content"]), "carried": True,
        })
        carried.add(locked["path"])

    if regen_set is not None:
        for reused in _reused_good_files(task, deps, regen_set):
            if reused["path"] in carried:
                continue
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
            if done["path"] in carried:
                continue
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
            # Env-locked: a repair-resolved file (e.g. requirements.txt)
            # was carried forward verbatim above -- never regenerate it.
            if path in carried:
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
                        depends_on=dep, contracts=contracts, skills=skills): i
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
                    depends_on=dep, contracts=contracts, skills=skills)
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

    # Manifest-coherence synthesis: the planner regularly authors tests
    # and entry points that import a first-party application module it
    # never planned (``from backend.app import app`` with no
    # ``backend/app.py`` anywhere in the manifest or the batch). The
    # import-coherence and phantom gates below would drop those importers
    # -- discarding the very suite that proves the app works. Author the
    # omitted modules against their callers first, so the gates see them as
    # first-party and keep the importers, and so the coherence pass that
    # follows can symbol-check a freshly authored module against its
    # callers. Best-effort and bounded; a failure leaves the bundle
    # untouched.
    synthesized_modules: List[str] = []
    try:
        synthesized_modules = _synthesize_missing_first_party_modules(
            diffs=diffs, generated=generated,
            existing_with_content=existing_with_content, layers=layers,
            goal=goal, provider=deps.provider, contracts=contracts,
            skills=skills, project_root=deps.project_root)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: first-party module synthesis raised; skipping")
        synthesized_modules = []
    if synthesized_modules:
        _checkpoint_progress(deps, artifact)

    # Frontend asset-coherence pass: symmetric with the first-party module
    # synthesis above, for the JS/TS family. A scaffold model routinely
    # writes ``import './index.css'`` in an entry point without authoring
    # the stylesheet, so ``npm run build`` cannot resolve it and the whole
    # tree is unbuildable (live failure: ses_aa99f1fb6914488d, where the
    # ensuing whole-tree regenerate reproduced the identical miss).
    # Synthesize an empty stub for every relative stylesheet import whose
    # target was never generated -- the zero-risk conventional fix -- so the
    # build resolves and the needless regenerate is avoided. Best-effort and
    # bounded; a failure leaves the bundle untouched.
    stylesheets_synthesized: List[str] = []
    try:
        stylesheets_synthesized = _synthesize_missing_frontend_stylesheets(
            diffs=diffs, generated=generated,
            existing_with_content=existing_with_content, layers=layers,
            project_root=deps.project_root)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: frontend stylesheet synthesis raised; skipping")
        stylesheets_synthesized = []
    if stylesheets_synthesized:
        _checkpoint_progress(deps, artifact)

    # JS test-harness coherence (P1a): the scaffold routinely authors React
    # unit tests (``*.test.jsx``) without the toolchain to run them, so
    # VERIFY's NpmRunner finds no ``test`` script and the suite is silently
    # skipped while ``npm run build`` still reports green -- the blind spot
    # that let ses_4cbf963cdc67435a ship a broken app with unrun tests.
    # Deterministically backfill the vite/vitest harness (test script +
    # devDeps + jsdom config + setup) so the tests the model wrote are
    # actually exercised. Best-effort and bounded; a failure leaves the
    # bundle untouched.
    js_harness_synthesized: List[str] = []
    try:
        js_harness_synthesized = _synthesize_js_test_harness(
            diffs=diffs, generated=generated,
            existing_with_content=existing_with_content, layers=layers,
            project_root=deps.project_root)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: JS test-harness synthesis raised; skipping")
        js_harness_synthesized = []
    if js_harness_synthesized:
        _checkpoint_progress(deps, artifact)

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
            contracts=contracts, skills=skills)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: coherence reconciliation raised; skipping")
        reconciled_count = 0
    if reconciled_count:
        _checkpoint_progress(deps, artifact)

    # Import-coherence gate: a generated .py file that imports a module
    # which resolves nowhere -- not to the manifest or the generated
    # batch, not to a file on disk, not to a dependency declared in
    # requirements.txt, and not to a real installed package -- ships a
    # hallucinated first-party import (live failure: ``from core import
    # compute`` with no core module anywhere). Left alone it reaches
    # BOOTSTRAP_ENV, where pip tries to install the fabricated name and
    # the session fails terminally. Fail the importer files here instead
    # so the router's regenerate edge retries them with the concrete
    # unknown-module constraint. Best-effort: a raised checker leaves
    # the bundle untouched.
    try:
        coherence_failed = _import_coherence_failures(
            existing_with_content, layers, deps.project_root)
    except Exception:  # pragma: no cover - defensive: gate is best-effort
        logger.exception("SCAFFOLD: import-coherence gate raised; skipping")
        coherence_failed = []
    if coherence_failed:
        dropped = {f["file"] for f in coherence_failed}
        failed.extend(coherence_failed)
        diffs[:] = [d for d in diffs if d.get("file") not in dropped]
        generated[:] = [g for g in generated
                        if g.get("file") not in dropped]
        existing_with_content[:] = [
            e for e in existing_with_content
            if e.get("path") not in dropped]
        _checkpoint_progress(deps, artifact)

    # Phantom first-party import gate: the coherence gate above only
    # judges an import's root segment, so ``from backend.core import
    # calculate_expression`` passes on the strength of a ``backend/``
    # directory even when no ``backend/core.py`` is planned or generated,
    # and ``from main import app`` passes because ``backend/main.py``
    # contributed the basename. Both die at import time (live failure:
    # pytest collection with "No module named 'main'"). Fail the importer
    # files here with the real module inventory so the regenerate edge
    # retries them against it. Best-effort: a raised checker leaves the
    # bundle untouched.
    try:
        phantom_failed = _phantom_first_party_import_failures(
            existing_with_content, layers)
    except Exception:  # pragma: no cover - defensive: gate is best-effort
        logger.exception(
            "SCAFFOLD: phantom-import gate raised; skipping")
        phantom_failed = []
    if phantom_failed:
        dropped = {f["file"] for f in phantom_failed}
        failed.extend(phantom_failed)
        diffs[:] = [d for d in diffs if d.get("file") not in dropped]
        generated[:] = [g for g in generated
                        if g.get("file") not in dropped]
        existing_with_content[:] = [
            e for e in existing_with_content
            if e.get("path") not in dropped]
        _checkpoint_progress(deps, artifact)

    # Circular-import gate: two (or more) generated first-party modules
    # importing each other survive every per-file check -- each import
    # resolves to a real sibling -- but Python cannot initialise the
    # cycle: pytest collection dies with "cannot import name ... from
    # partially initialized module ..." (live failure: backend/routes.py
    # <-> backend/models.py importing each other). Detect cycles
    # statically over the batch's import graph and fail one file per
    # cycle so the router's regenerate edge breaks it with a concrete
    # constraint. Best-effort: a raised checker leaves the bundle
    # untouched.
    try:
        cycle_failed = _circular_import_failures(existing_with_content)
    except Exception:  # pragma: no cover - defensive: gate is best-effort
        logger.exception("SCAFFOLD: circular-import gate raised; skipping")
        cycle_failed = []
    if cycle_failed:
        dropped = {f["file"] for f in cycle_failed}
        failed.extend(cycle_failed)
        diffs[:] = [d for d in diffs if d.get("file") not in dropped]
        generated[:] = [g for g in generated
                        if g.get("file") not in dropped]
        existing_with_content[:] = [
            e for e in existing_with_content
            if e.get("path") not in dropped]
        _checkpoint_progress(deps, artifact)

    # Frontend script-import coherence gate: symmetric with the Python
    # import-coherence gate above, for the JS/TS family. A generated source
    # that does ``import Foo from './Foo'`` with no matching generated
    # sibling (and no file on disk) ships a tree ``npm run build`` cannot
    # resolve. Unlike a stylesheet -- which the pass above stubs empty -- a
    # script has behaviour, so fail the importer and let the regenerate edge
    # re-author it against the real sibling inventory rather than emitting a
    # meaningless stub. Best-effort: a raised gate leaves the bundle
    # untouched.
    try:
        js_coherence_failed = _js_import_coherence_failures(
            existing_with_content, deps.project_root)
    except Exception:  # pragma: no cover - defensive: gate is best-effort
        logger.exception(
            "SCAFFOLD: JS import-coherence gate raised; skipping")
        js_coherence_failed = []
    if js_coherence_failed:
        dropped = {f["file"] for f in js_coherence_failed}
        failed.extend(js_coherence_failed)
        diffs[:] = [d for d in diffs if d.get("file") not in dropped]
        generated[:] = [g for g in generated
                        if g.get("file") not in dropped]
        existing_with_content[:] = [
            e for e in existing_with_content
            if e.get("path") not in dropped]
        _checkpoint_progress(deps, artifact)

    # Import-smoke synthesis: when the plan authored a test suite but every
    # generated test was unrecoverable and the gates above dropped them all,
    # the tree reaches VERIFY with source modules and no test -- the honest
    # but unsatisfying ``no_tests``. Synthesize a deterministic
    # tests/test_smoke.py that imports every surviving first-party source
    # module, turning that into a real signal: it passes only if the tree
    # loads (no missing dep, syntax error, or circular import) and fails
    # honestly otherwise. Runs after every Python drop gate so it probes the
    # exact surviving set, and never overrides a model-authored test that
    # survived. Best-effort: any failure leaves the bundle untouched.
    smoke_test_synthesized: Optional[str] = None
    try:
        smoke_test_synthesized = _synthesize_import_smoke_test(
            diffs=diffs, generated=generated,
            existing_with_content=existing_with_content, layers=layers)
    except Exception:  # pragma: no cover - defensive: pass is best-effort
        logger.exception(
            "SCAFFOLD: import-smoke synthesis raised; skipping")
        smoke_test_synthesized = None
    if smoke_test_synthesized:
        _checkpoint_progress(deps, artifact)

    # JS runtime-dependency guard: symmetric with the deterministic
    # requirements.txt salvage. A weak model routinely imports a runtime
    # package (e.g. ``axios``) in a component while omitting it from
    # package.json, so the build resolves nothing and VERIFY ends on an
    # unrecoverable red. Cross-check every generated JS/TS file's external
    # imports against package.json and splice the missing ones into
    # ``dependencies`` in place. Best-effort: any failure leaves the bundle
    # untouched.
    js_deps_added: List[str] = []
    try:
        js_deps_added = _reconcile_js_dependencies(
            diffs=diffs, existing_with_content=existing_with_content)
    except Exception:  # pragma: no cover - defensive: gate is best-effort
        logger.exception(
            "SCAFFOLD: JS dependency reconciliation raised; skipping")
        js_deps_added = []
    if js_deps_added:
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

    # Cross-language payload coherence gate (P0b): a JS ``fetch`` body whose
    # keys disagree with the backend handler it targets (e.g. ``operator``
    # vs ``operation``) is invisible to the Python-only contract check above
    # and to a build smoke, yet it makes the app non-functional at runtime.
    # Fold any mismatch into ``contract_warnings`` as a ``payload`` kind so
    # the router regenerates only the offending client file. Best-effort:
    # a raised checker leaves the existing warnings untouched.
    try:
        payload_warnings = check_client_server_payload_coherence(
            xcheck_contents, contracts)
    except Exception:  # pragma: no cover - defensive: checker is best-effort
        logger.exception(
            "SCAFFOLD: payload coherence check raised; skipping")
        payload_warnings = []
    if payload_warnings:
        contract_warnings = list(contract_warnings) + payload_warnings

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
    artifact.content["js_deps_added"] = js_deps_added
    artifact.content["synthesized_modules"] = synthesized_modules
    artifact.content["smoke_test_synthesized"] = smoke_test_synthesized
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
            "js_deps_added": js_deps_added,
            "synthesized_modules": synthesized_modules,
            "smoke_test_synthesized": smoke_test_synthesized,
        },
        artifact=artifact,
    )


def _known_import_root_resolver(
        files_with_content: List[Dict[str, str]],
        layers: List[Any],
        project_root: Optional[str],
) -> Tuple[Callable[[str], bool], set]:
    """Build ``(is_known, known_local)`` for judging first-party import roots.

    ``is_known(root)`` returns True when the top-level segment of an import
    resolves somewhere a build could satisfy it: the ``__future__``
    pseudo-module, the stdlib, a namespace root, a module/directory name
    derivable from the work-plan manifest or the generated batch, a
    distribution declared in the batch's (or on-disk) ``requirements.txt``,
    or a package findable under ``project_root`` / in the running
    environment. ``known_local`` is the set of project-local module and
    directory names the manifest + batch define. Shared by the
    import-coherence gate (which drops an importer of an unknown root as a
    hallucination) and the first-party synthesizer (which reads the same
    verdict the other way: a plainly first-party unknown root names a
    module the planner omitted, to be authored rather than dropped).
    """
    from cgx.codegen.env_manager import (
        _IMPORT_TO_PYPI, _NAMESPACE_ROOTS, _STDLIB_TOP, _is_local_package)

    # Module/package names resolvable inside the project: every ``.py``
    # basename and directory segment from the manifest + generated batch.
    known_local: set = set()
    manifest_paths = [
        str(e.get("path") or "").strip()
        for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or []) if isinstance(e, dict)]
    batch_paths = [str(e.get("path") or "") for e in files_with_content]
    for p in manifest_paths + batch_paths:
        parts = [seg for seg in p.split("/") if seg]
        if not parts:
            continue
        base = parts[-1]
        if base.endswith(".py") and base != "__init__.py":
            known_local.add(base[:-3])
        for seg in parts[:-1]:
            known_local.add(seg)

    # Distributions declared in the batch's requirements.txt, falling
    # back to the on-disk one; normalised for comparison.
    req_text: Optional[str] = None
    for e in files_with_content:
        if is_requirements_path(str(e.get("path") or "")):
            req_text = str(e.get("content") or "")
            break
    if req_text is None and project_root:
        req_file = Path(project_root) / "requirements.txt"
        if req_file.is_file():
            try:
                req_text = req_file.read_text(encoding="utf-8",
                                              errors="ignore")
            except Exception:
                req_text = None
    declared: set = set()
    for line in (req_text or "").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        pkg = re.split(r"[>=<!~;\[ @]", line, maxsplit=1)[0].strip()
        if pkg:
            declared.add(pkg.lower().replace("-", "_"))

    _known_cache: Dict[str, bool] = {}

    def _known(root_name: str) -> bool:
        cached = _known_cache.get(root_name)
        if cached is not None:
            return cached
        _known_cache[root_name] = True  # optimistic; flipped below
        norm = root_name.lower().replace("-", "_")
        if (root_name == "__future__" or norm in _STDLIB_TOP
                or root_name in _NAMESPACE_ROOTS
                or root_name in known_local or norm in declared):
            return True
        pypi = _IMPORT_TO_PYPI.get(root_name)
        if pypi and pypi.lower().replace("-", "_") in declared:
            return True
        if project_root and _is_local_package(root_name, project_root):
            return True
        # Known-real-package probe: findable in the running environment
        # (its site-packages or the stdlib tree) means pip can plausibly
        # satisfy it too. Origins under the server's own cwd are ignored
        # so CGX's own first-party modules never whitelist a name.
        import importlib.util as _ilu
        try:
            spec = _ilu.find_spec(root_name)
        except Exception:
            spec = None
        if spec is not None:
            origin = getattr(spec, "origin", None) or ""
            locations = list(
                getattr(spec, "submodule_search_locations", None) or [])
            stdlib_dir = os.path.dirname(os.__file__)
            for loc in [origin] + locations:
                loc = str(loc or "")
                if origin in ("built-in", "frozen"):
                    return True
                if ("site-packages" in loc or "dist-packages" in loc
                        or loc.startswith(stdlib_dir)):
                    return True
        _known_cache[root_name] = False
        return False

    return _known, known_local


def _import_coherence_failures(
        files_with_content: List[Dict[str, str]],
        layers: List[Any],
        project_root: Optional[str],
) -> List[Dict[str, str]]:
    """Flag generated ``.py`` files whose absolute imports resolve nowhere.

    An import root is *unknown* when it is not stdlib, does not match any
    module or package name derivable from the work-plan manifest or the
    generated batch (``.py`` basenames and directory segments), does not
    exist under ``project_root`` on disk, is not declared in the batch's
    (or project's) requirements.txt, and is not a real package findable
    in the running environment. Such an import can only be a
    hallucination: APPLY would write it and BOOTSTRAP_ENV's preflight
    would then ``pip install`` the fabricated name and fail. Returns
    ``{file, error}`` entries shaped for the router's regenerate splice.
    """
    import ast as _ast

    known, _known_local = _known_import_root_resolver(
        files_with_content, layers, project_root)

    manifest_paths = [
        str(e.get("path") or "").strip()
        for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or []) if isinstance(e, dict)]
    batch_paths = [str(e.get("path") or "") for e in files_with_content]

    # The importable first-party inventory, named in the error so the
    # regenerate edge has something to aim at. Without it a model that
    # invented ``import app`` has no way to learn the module is really
    # ``backend.main`` and simply invents it again on every retry.
    from cgx.session.scaffold_validate import _module_name_for_path
    inventory = sorted({
        mod for p in manifest_paths + batch_paths
        if p.strip().replace("\\", "/").lstrip("./").endswith(".py")
        for mod in (_module_name_for_path(
            p.strip().replace("\\", "/").lstrip("./")),)
        if mod})[:12]

    out: List[Dict[str, str]] = []
    for entry in files_with_content:
        path = str(entry.get("path") or "")
        content = entry.get("content") or ""
        if not path.endswith(".py") or not content:
            continue
        try:
            tree = _ast.parse(content)
        except SyntaxError:
            continue
        unknown: List[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                mods = [a.name or "" for a in node.names]
            elif isinstance(node, _ast.ImportFrom) and not node.level:
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                root_name = mod.split(".")[0]
                if (root_name and root_name not in unknown
                        and not known(root_name)):
                    unknown.append(root_name)
        if unknown:
            out.append({
                "file": path,
                "error": (
                    f"imports unknown module(s) {sorted(unknown)}: not "
                    "defined in the project manifest, not on disk, and "
                    "not declared in requirements.txt -- import only "
                    "from the manifest's modules or its declared "
                    f"dependencies; the project's modules are {inventory}"),
            })
    return out


# Cap on first-party modules synthesized per SCAFFOLD when generated files
# import a module the planner omitted. Bounded so a badly-planned tree (or a
# model that keeps inventing new names) cannot fan the pass out into an
# unbounded generation loop; anything past the cap falls through to the drop
# gates exactly as before.
_SYNTH_MODULE_BUDGET = 8


def _module_to_source_path(dotted: str) -> str:
    """The ``.py`` file path backing a dotted first-party module name."""
    return dotted.replace(".", "/") + ".py"


def _is_test_path(path: str) -> bool:
    """True when ``path`` is a pytest test module or a conftest."""
    norm = (path or "").replace("\\", "/").strip("/")
    base = norm.rsplit("/", 1)[-1]
    return (base == "conftest.py" or base.startswith("test_")
            or base.endswith("_test.py")
            or norm == "tests" or norm.startswith("tests/")
            or "/tests/" in norm)


def _looks_local_module_ref(mod: str, root: str) -> bool:
    """True when a named-symbol import reads as a project-local module.

    Consulted only after the root has resolved nowhere a build could
    satisfy it and is not a known PyPI import alias. A dotted path
    (``pkg.mod``) is a submodule reference and a snake_case compound root
    (``calculation_service``) reads as an application module; a genuinely
    forgotten third-party distribution is imported by its bare single-word
    top name instead, so that shape is left to the requirements repair
    rather than fabricated as an empty first-party module.
    """
    return "." in mod or "_" in root


def _synthesized_module_description(
        mod: str, path: str, symbols: List[str]) -> str:
    """Generation brief for a first-party module the planner omitted.

    Names the module and the file to author and -- crucially -- lists the
    exact top-level symbols its callers already import, so the generator
    writes a public surface that lines up with the rest of the tree instead
    of re-deriving an interface that will not match.
    """
    needs = ""
    if symbols:
        needs = (" It must define these top-level names that other generated "
                 f"files already import from it: {', '.join(symbols)}.")
    return (
        f"First-party module `{mod}` (file `{path}`). Other generated files "
        f"import from this module, but the work plan omitted it, so author "
        f"it now to match how its callers already use it.{needs}")


def _missing_first_party_imports(
        files_with_content: List[Dict[str, str]],
        layers: List[Any],
        project_root: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """First-party modules generated files import but the plan omits.

    The planner routinely authors a test suite (or an entry point) that
    imports the application module it forgot to plan -- ``from backend.app
    import app, compute`` with no ``backend/app.py`` anywhere in the
    manifest or the batch. The importer is correct; the module is simply
    missing. Returns, per omitted module, everything needed to author it:
    the ``.py`` path to create, the union of the symbols siblings import
    from it, and the importer paths (fed back as generation context).

    A reference is a synthesis candidate only when it is unambiguously
    first-party -- never a mistyped dependency:

    * its root segment is a generated project source directory (a dotted
      import into an existing package the plan under-populated); or
    * it is imported *with named symbols* (``from X import a, b``), its root
      resolves nowhere a build could satisfy it (not stdlib, not installed,
      not declared in requirements) and is not a known PyPI import alias,
      *and* either the importer is a test/conftest module (any such
      reference is the module-under-test the plan forgot) or the reference
      reads as project-local -- a dotted submodule path or a snake_case
      compound root (see :func:`_looks_local_module_ref`).

    A bare single-word import (``import X`` with no named symbols, or a
    named import of a plain single-word root from a non-test file) carries
    no first-party signal and is left to the requirements repair rather than
    fabricated here. Self-imports and already-generated modules are skipped;
    parse failures abstain for that file.
    """
    import ast as _ast
    from cgx.codegen.env_manager import _IMPORT_TO_PYPI
    from cgx.session.scaffold_validate import _module_name_for_path

    known, _known_local = _known_import_root_resolver(
        files_with_content, layers, project_root)

    manifest_paths = [
        str(e.get("path") or "") for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or []) if isinstance(e, dict)]
    batch_paths = [str(e.get("path") or "") for e in files_with_content]
    py_paths = [
        p.strip().replace("\\", "/").lstrip("./")
        for p in manifest_paths + batch_paths
        if p.strip().replace("\\", "/").lstrip("./").endswith(".py")]
    existing_paths = set(py_paths)

    modules: set = set()
    prefixes: set = set()
    for p in py_paths:
        mod = _module_name_for_path(p)
        if not mod:
            continue
        modules.add(mod)
        parts = mod.split(".")
        for i in range(1, len(parts)):
            prefixes.add(".".join(parts[:i]))
    fp_roots = {p.split("/")[0] for p in py_paths if "/" in p}

    candidates: Dict[str, Dict[str, Any]] = {}

    def _record(mod: str, importer: str, symbols: List[str]) -> None:
        target = _module_to_source_path(mod)
        if mod in modules or mod in prefixes or target in existing_paths:
            return
        rec = candidates.setdefault(
            mod, {"path": target, "symbols": set(), "importers": []})
        rec["symbols"].update(s for s in symbols if s and s != "*")
        if importer and importer not in rec["importers"]:
            rec["importers"].append(importer)

    for entry in files_with_content:
        path = str(entry.get("path") or "").replace("\\", "/").lstrip("./")
        content = entry.get("content") or ""
        if not path.endswith(".py") or not content:
            continue
        try:
            tree = _ast.parse(content)
        except SyntaxError:
            continue
        importer_mod = _module_name_for_path(path)
        is_test = _is_test_path(path)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom) and not node.level:
                refs = [(node.module or "", [a.name for a in node.names])]
            elif isinstance(node, _ast.Import):
                refs = [(a.name or "", []) for a in node.names]
            else:
                continue
            for mod, symbols in refs:
                if not mod or mod == importer_mod:
                    continue
                root = mod.split(".")[0]
                first_party = False
                if root in fp_roots:
                    first_party = True
                elif (symbols and not known(root)
                        and root not in _IMPORT_TO_PYPI
                        and (is_test or _looks_local_module_ref(mod, root))):
                    first_party = True
                if first_party:
                    _record(mod, path, symbols)
    return candidates


def _synthesize_missing_first_party_modules(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
        goal: str,
        provider: Any,
        contracts: Optional[Dict[str, Any]],
        skills: Optional[List[str]],
        project_root: Optional[str],
) -> List[str]:
    """Author first-party modules generated files import but the plan omits.

    Symmetric with the requirements.txt / package.json salvage: rather than
    letting the import-coherence and phantom gates drop the tests and entry
    points that reference an omitted application module,
    :func:`_missing_first_party_imports` finds those modules and this
    generates each one against its callers -- so the model derives the
    module's contract from the symbols they already use -- then splices the
    results into the diff bundle *and* the in-memory manifest so the
    downstream gates treat them as first-party and leave the importers
    intact. A dotted module in a package with no generated ``__init__.py``
    also gets an empty package marker so pytest can import it. Bounded by
    :data:`_SYNTH_MODULE_BUDGET`; a module whose generation fails is left
    for the drop gates, so the pass can only add resolvable files (or
    no-op). Mutates the passed lists/manifest in place and returns the paths
    authored.
    """
    if provider is None:
        return []
    missing = _missing_first_party_imports(
        existing_with_content, layers, project_root)
    if not missing:
        return []

    from cgx.answer.engine import _content_to_new_file_patch

    existing_paths = {
        str(e.get("path") or "").replace("\\", "/").lstrip("./")
        for e in existing_with_content}
    synth_files: List[Dict[str, Any]] = []
    added: List[str] = []

    def _add(path: str, content: str,
             ok_meta: Optional[Dict[str, Any]]) -> None:
        patch = (ok_meta["patch"] if ok_meta
                 else _content_to_new_file_patch(path, content))
        diffs.append({"file": path, "patch": patch})
        generated.append({
            "file": path, "layer": "synthesized",
            "syntax_ok": bool(ok_meta["syntax_ok"]) if ok_meta else True,
            "confidence": ok_meta.get("confidence") if ok_meta else None,
            "bytes": len(content), "synthesized": True})
        existing_with_content.append({"path": path, "content": content})
        existing_paths.add(path.replace("\\", "/").lstrip("./"))
        synth_files.append({"path": path})
        added.append(path)

    # Deterministic order + hard cap so the pass is reproducible and cannot
    # fan out unboundedly on a badly-planned tree.
    for mod, info in sorted(missing.items(), key=lambda kv: kv[0])[
            :_SYNTH_MODULE_BUDGET]:
        target = str(info["path"])
        if target.replace("\\", "/").lstrip("./") in existing_paths:
            continue
        # Package markers for every intermediate package so ``a.b.c`` stays
        # importable even when the plan never created the package dirs.
        parts = mod.split(".")
        for i in range(1, len(parts)):
            pkg_init = "/".join(parts[:i]) + "/__init__.py"
            if pkg_init not in existing_paths:
                _add(pkg_init, '"""Package marker."""\n', None)
        symbols = sorted(info.get("symbols") or [])
        importer_set = set(info.get("importers") or [])
        context = [e for e in existing_with_content
                   if str(e.get("path") or "") in importer_set] \
            or list(existing_with_content)
        desc = _synthesized_module_description(mod, target, symbols)
        ok, _fail = _generate_one(
            target, desc, "synthesized", context, provider, goal,
            depends_on=None, contracts=contracts, skills=skills)
        if ok is None:
            logger.warning(
                "SCAFFOLD: could not synthesize omitted first-party module "
                "%r (imported by %s); leaving importer(s) for the drop gate",
                mod, sorted(importer_set)[:3])
            continue
        _add(ok["file"], ok["content"], ok)

    if synth_files:
        layers.append({"name": "synthesized", "files": synth_files})
        logger.warning(
            "SCAFFOLD: work plan omitted first-party module(s) imported by "
            "generated files; synthesized %s so the importers resolve", added)
    return added


_SMOKE_TEST_PATH = "tests/test_smoke.py"


def _is_pytest_test_module(path: str) -> bool:
    """True when ``path`` is a runnable pytest test module (not a conftest)."""
    base = (path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py")


def _importable_module_name(path: str) -> Optional[str]:
    """Runtime-importable dotted name for a first-party source ``.py`` path.

    Mirrors the generated root ``conftest.py`` (and the smoke test's own
    ``sys.path`` bootstrap), which prepend ``src/`` to ``sys.path`` for the
    src/ layout: a module under ``src/`` imports by the name *below* ``src/``
    (``src/calc.py`` -> ``calc``); everything else imports by its
    project-root-relative dotted path (``backend/app.py`` -> ``backend.app``).
    Returns ``None`` for non-Python paths and for a bare ``src`` package.
    """
    from cgx.session.scaffold_validate import _module_name_for_path
    mod = _module_name_for_path(path)
    if not mod or mod == "src":
        return None
    if mod.startswith("src."):
        return mod[len("src."):]
    return mod


def _render_import_smoke_test(modules: List[str]) -> str:
    """Render the deterministic import-smoke test body for ``modules``."""
    listed = "\n".join(f"    {m!r}," for m in modules)
    return (
        '"""Auto-generated import smoke test.\n'
        "\n"
        "Imports every first-party source module and fails if any raises at\n"
        "import time (missing dependency, syntax error, circular import). A\n"
        "green run proves the package tree loads; it asserts no behaviour.\n"
        "Synthesized only when the plan produced no usable test module, so\n"
        "VERIFY exercises the code instead of reporting no_tests.\n"
        '"""\n'
        "import importlib\n"
        "import os\n"
        "import sys\n"
        "\n"
        "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
        "_ROOT = os.path.dirname(_HERE)\n"
        '_SRC = os.path.join(_ROOT, "src")\n'
        "for _p in (_ROOT, _SRC):\n"
        "    if os.path.isdir(_p) and _p not in sys.path:\n"
        "        sys.path.insert(0, _p)\n"
        "\n"
        "MODULES = [\n"
        f"{listed}\n"
        "]\n"
        "\n"
        "\n"
        "def test_first_party_modules_import():\n"
        "    errors = []\n"
        "    for _name in MODULES:\n"
        "        try:\n"
        "            importlib.import_module(_name)\n"
        "        except Exception as exc:  # noqa: BLE001\n"
        "            errors.append(\n"
        '                "%s: %s: %s" % (_name, type(exc).__name__, exc))\n'
        "    assert not errors, (\n"
        '        "first-party modules failed to import: " + "; ".join(errors))\n'
    )


def _synthesize_import_smoke_test(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
) -> Optional[str]:
    """Author ``tests/test_smoke.py`` when planned tests were all dropped.

    The weak-model failure mode this closes: the planner authored a test
    suite, but every generated test was unrecoverable (collapsed newlines,
    truncation) and the gates dropped them all, leaving a tree with source
    modules and no test. VERIFY would then report the honest-but-useless
    ``no_tests``. Rather than fake a pass, synthesize a deterministic smoke
    test that ``importlib.import_module()``s every surviving first-party
    source module: it passes only if the tree actually loads (no missing
    dependency, syntax error, or circular import) and fails honestly
    otherwise -- a real signal VERIFY can gate on.

    Kept narrow so it never disturbs a well-formed scaffold: it fires only
    when the *manifest planned* at least one pytest test module, *none*
    survived the gates, and there is at least one importable source module
    to probe. A model-authored test that survived is always left in place.
    Mutates the passed lists/manifest in place; returns the path authored
    (or ``None``).
    """
    from cgx.answer.engine import _content_to_new_file_patch

    manifest_paths = [
        str(e.get("path") or "") for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or []) if isinstance(e, dict)]
    # Only step in for the "planned tests, kept none" shape.
    if not any(_is_pytest_test_module(p) for p in manifest_paths):
        return None
    surviving = [str(e.get("path") or "").replace("\\", "/").lstrip("./")
                 for e in existing_with_content]
    if any(_is_pytest_test_module(p) for p in surviving):
        return None

    modules: List[str] = []
    for path in surviving:
        if not path.endswith(".py"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base == "__init__.py" or _is_test_path(path):
            continue
        name = _importable_module_name(path)
        if name:
            modules.append(name)
    modules = sorted(dict.fromkeys(modules))
    if not modules:
        return None

    content = _render_import_smoke_test(modules)
    patch = _content_to_new_file_patch(_SMOKE_TEST_PATH, content)
    diffs.append({"file": _SMOKE_TEST_PATH, "patch": patch})
    generated.append({
        "file": _SMOKE_TEST_PATH, "layer": "smoke", "syntax_ok": True,
        "confidence": 1.0, "bytes": len(content), "synthesized": True})
    existing_with_content.append(
        {"path": _SMOKE_TEST_PATH, "content": content})
    layers.append({"name": "smoke", "files": [{"path": _SMOKE_TEST_PATH}]})
    logger.warning(
        "SCAFFOLD: the plan's test module(s) were all dropped; synthesized %s "
        "importing %d first-party module(s) so VERIFY runs a real import "
        "smoke test instead of reporting no_tests", _SMOKE_TEST_PATH,
        len(modules))
    return _SMOKE_TEST_PATH


def _phantom_first_party_import_failures(
        files_with_content: List[Dict[str, str]],
        layers: List[Any],
) -> List[Dict[str, str]]:
    """Flag first-party imports that name a module the project never defines.

    :func:`_import_coherence_failures` judges only an import's *root*
    segment, so ``from backend.core import calculate_expression`` passes
    whenever a ``backend/`` directory exists -- even with no
    ``backend/core.py`` anywhere in the manifest or the batch. The
    cross-check in ``scaffold_validate`` abstains on the same import for
    the mirror reason: it has no source to verify the *name* against. The
    tree therefore ships with an import that dies the moment anything
    touches it (live failure: every entry point and the whole test module
    unimportable).

    Two shapes are decidable from the paths alone and caught here:

    * a dotted import whose root is a first-party source directory but
      whose full dotted name is neither a generated module nor one of
      their package prefixes;
    * a bare ``from main import app`` naming a module that only exists
      inside a generated *package* (``backend/main.py``), whose real
      importable name is ``backend.main`` -- pytest otherwise dies with
      ``ModuleNotFoundError: No module named 'main'``.

    Returns ``{file, error}`` entries shaped for the router's regenerate
    splice. Any parse failure abstains for that file.
    """
    import ast as _ast
    from cgx.session.scaffold_validate import _module_name_for_path

    manifest_paths = [
        str(e.get("path") or "") for lay in layers if isinstance(lay, dict)
        for e in (lay.get("files") or []) if isinstance(e, dict)]
    batch_paths = [str(e.get("path") or "") for e in files_with_content]
    py_paths: List[str] = []
    for p in manifest_paths + batch_paths:
        s = p.strip().replace("\\", "/").lstrip("./")
        if s.endswith(".py"):
            py_paths.append(s)

    modules: set = set()
    prefixes: set = set()
    for p in py_paths:
        mod = _module_name_for_path(p)
        if not mod:
            continue
        modules.add(mod)
        parts = mod.split(".")
        for i in range(1, len(parts)):
            prefixes.add(".".join(parts[:i]))
    # Only roots that are first-party source directories are judged: a
    # flat or src/-layout module is imported by bare name via sys.path,
    # so its root says nothing about what should exist on disk.
    fp_roots = {p.split("/")[0] for p in py_paths if "/" in p}
    # A directory is a package only with a generated ``__init__.py``; a
    # module inside one is importable as ``pkg.mod``, never bare.
    pkg_dirs = {p.rsplit("/", 1)[0] for p in py_paths
                if p.endswith("/__init__.py")}
    nested: Dict[str, str] = {}
    for p in py_paths:
        directory, _, base = p.rpartition("/")
        if not directory or directory not in pkg_dirs or base == "__init__.py":
            continue
        bare = base[:-3]
        if bare in modules:
            # Also importable top-level: the bare form is legitimate.
            continue
        nested.setdefault(bare, _module_name_for_path(p) or "")

    inventory = sorted(modules)[:12]
    out: List[Dict[str, str]] = []
    for entry in files_with_content:
        path = str(entry.get("path") or "")
        content = entry.get("content") or ""
        if not path.endswith(".py") or not content:
            continue
        try:
            tree = _ast.parse(content)
        except SyntaxError:
            continue
        phantom: List[str] = []
        misrooted: List[Tuple[str, str]] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                mods = [a.name or "" for a in node.names]
            elif isinstance(node, _ast.ImportFrom) and not node.level:
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                if not mod:
                    continue
                if "." in mod:
                    if (mod.split(".")[0] in fp_roots
                            and mod not in modules and mod not in prefixes
                            and mod not in phantom):
                        phantom.append(mod)
                elif mod in nested and (mod, nested[mod]) not in misrooted:
                    misrooted.append((mod, nested[mod]))
        if not phantom and not misrooted:
            continue
        msgs: List[str] = []
        if phantom:
            msgs.append(
                f"imports first-party module(s) {sorted(phantom)} that no "
                "generated file defines")
        for bare, real in sorted(misrooted):
            msgs.append(
                f"imports {bare!r} as a top-level module, but that module "
                f"is {real!r} inside a generated package")
        out.append({
            "file": path,
            "error": (
                "; ".join(msgs)
                + " -- regenerate importing only these modules: "
                + f"{inventory}"),
        })
    return out


# Basenames that mark a module as "foundational" -- the layer others build
# on. When an import cycle must be broken, the foundational member is the
# one regenerated (models must not import from routes, not vice versa) so
# the retry converges on the conventional one-way layering instead of
# arbitrarily rewriting the higher-level module.
_FOUNDATION_MODULE_NAMES = frozenset({
    "models", "model", "db", "database", "config", "settings",
    "constants", "schemas", "types", "utils", "helpers", "core",
})


def _circular_import_failures(
        files_with_content: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Flag generated ``.py`` files that form a first-party import cycle.

    Builds the directed import graph over the generated batch (module ->
    imported sibling module, covering ``import a.b``, ``from a import b``
    and relative forms) and finds its strongly connected components. Any
    component with more than one module is a cycle Python cannot
    initialise: pytest collection dies with ``ImportError: cannot import
    name ... from partially initialized module ...``. One file per cycle
    -- the most foundational-looking member, falling back to the first in
    module order -- is failed with a constraint naming the modules it
    must stop importing, shaped ``{file, error}`` for the router's
    regenerate splice.
    """
    import ast as _ast

    # Dotted module name + source for every generated .py file.
    mod_by_path: Dict[str, str] = {}
    content_by_path: Dict[str, str] = {}
    for entry in files_with_content:
        path = str(entry.get("path") or "")
        content = entry.get("content")
        if not path.endswith(".py") or not isinstance(content, str):
            continue
        parts = [seg for seg in path.split("/") if seg]
        if not parts:
            continue
        if parts[-1] == "__init__.py":
            dotted = ".".join(parts[:-1])
        else:
            dotted = ".".join(parts)[:-3]
        if dotted:
            mod_by_path[path] = dotted
            content_by_path[path] = content
    path_by_mod = {m: p for p, m in mod_by_path.items()}

    def _resolve(name: str) -> Optional[str]:
        # Longest batch module matching ``name`` or one of its prefixes:
        # ``import backend.models.extra`` depends on ``backend.models``.
        while name:
            if name in path_by_mod:
                return name
            name = name.rpartition(".")[0]
        return None

    edges: Dict[str, set] = {m: set() for m in path_by_mod}
    for path, mod in mod_by_path.items():
        try:
            tree = _ast.parse(content_by_path[path])
        except SyntaxError:
            continue
        pkg_parts = mod.split(".")
        if not path.endswith("__init__.py"):
            pkg_parts = pkg_parts[:-1]
        for node in _ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, _ast.Import):
                targets = [a.name or "" for a in node.names]
            elif isinstance(node, _ast.ImportFrom):
                if node.level:
                    cut = len(pkg_parts) - (node.level - 1)
                    if cut < 0:
                        continue
                    head = pkg_parts[:cut]
                    if node.module:
                        head = head + node.module.split(".")
                    base = ".".join(head)
                else:
                    base = node.module or ""
                if base:
                    # ``from a import b`` may name the submodule a.b.
                    targets = [base] + [f"{base}.{a.name}"
                                        for a in node.names]
            for name in targets:
                dep = _resolve(name)
                if dep and dep != mod:
                    edges[mod].add(dep)

    # Tarjan SCC: every component with >1 member is an import cycle.
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: set = set()
    stack: List[str] = []
    cycles: List[List[str]] = []
    counter = [0]

    def _scc(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(edges.get(v, ())):
            if w not in index:
                _scc(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                cycles.append(sorted(comp))

    for v in sorted(edges):
        if v not in index:
            _scc(v)

    out: List[Dict[str, str]] = []
    for comp in cycles:
        members = set(comp)
        foundational = [m for m in comp
                        if m.rsplit(".", 1)[-1] in _FOUNDATION_MODULE_NAMES]
        target = foundational[0] if foundational else comp[0]
        offenders = sorted(edges[target] & members)
        out.append({
            "file": path_by_mod[target],
            "error": (
                f"circular import among first-party modules {comp}: this "
                f"file imports {offenders}, which import(s) it back, so "
                "Python cannot finish initialising either module -- "
                f"regenerate this file WITHOUT importing {offenders}; "
                "define the needed symbols locally (or accept them as "
                "function parameters) so the dependency is one-way"),
        })
    return out


def _generate_one(
        path: str, description: str, layer_name: str,
        context: List[Dict[str, str]], provider: Any, goal: str,
        on_token: Optional[Callable[[str], None]] = None,
        depends_on: Optional[List[str]] = None,
        contracts: Optional[Dict[str, Any]] = None,
        skills: Optional[List[str]] = None,
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
    same declared endpoints/schemas/signatures. ``skills`` pins the
    generator to an explicit skill list instead of auto-detecting from
    ``goal`` -- ``None`` preserves the existing auto-detect behavior.
    """
    from cgx.answer.engine import generate_single_scaffold_file
    try:
        result = generate_single_scaffold_file(
            path, description, provider,
            layer=layer_name,
            existing_files_with_content=context,
            goal=goal,
            skills=skills,
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
            bytes=int(state["chars"]), failed_count=failed_count)

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


def _with_additional_files(
        task: TaskNode,
        layers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return ``layers`` extended with the regenerate's ``additional_files``.

    A missing entry point (``[UNRESOLVED_ENTRY] Cannot resolve entry
    module index.html``) is a file that does not exist, not a file that
    is wrong: regenerating the manifest verbatim reproduces it forever,
    which is exactly how a repair loop burns its whole budget on one
    build error. REPAIR therefore names the absent paths and SCAFFOLD
    appends them to the last populated layer, so they are generated with
    the rest of the tree as context.

    The work plan artifact is never mutated -- the affected layer is
    shallow-copied. Paths already in the manifest are ignored, so a
    stale marker cannot duplicate an entry. Returns the (possibly
    unchanged) layers and the paths that were added.
    """
    raw = task.inputs.get("additional_files")
    if not isinstance(raw, list) or not raw:
        return layers, []
    have = {str(e.get("path") or "").strip()
            for lay in layers if isinstance(lay, dict)
            for e in (lay.get("files") or []) if isinstance(e, dict)}
    new_entries: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path or path in have:
            continue
        have.add(path)
        new_entries.append(
            {"path": path,
             "description": str(item.get("description") or path).strip()})
    if not new_entries:
        return layers, []
    idx = next((i for i in range(len(layers) - 1, -1, -1)
                if isinstance(layers[i], dict) and layers[i].get("files")),
               None)
    if idx is None:
        return layers, []
    out = list(layers)
    target = dict(out[idx])
    target["files"] = list(target.get("files") or []) + new_entries
    out[idx] = target
    added = [e["path"] for e in new_entries]
    logger.warning(
        "SCAFFOLD: regenerate adds file(s) absent from the manifest: %s",
        added)
    return out, added


# Env-lock files a repair may resolve on disk (e.g. a conflict-free
# requirements.txt) that a whole-tree regenerate must not re-emit from
# the model. Carried forward verbatim so the deterministic env fix
# survives the re-scaffold.
_LOCKED_ENV_FILES: Tuple[str, ...] = ("requirements.txt",)


def _carry_forward_locked_files(
        task: TaskNode, deps: ExecutorDeps) -> List[Dict[str, str]]:
    """Return ``{path, patch, content}`` for env-locked files to reuse verbatim.

    When a repair re-pinned requirements.txt to a self-consistent set
    (env_manager conflict re-resolve), it marks the file env-locked. A
    later whole-tree regenerate would otherwise re-emit the model's stale
    manifest and clobber the fix, reintroducing the resolved conflict.
    Read each locked file off ``deps.project_root`` and hand it back as a
    verbatim new-file diff so the regenerated bundle keeps the resolved
    pins. Best-effort: no project root, no lock marker, or an unreadable /
    empty file yields an empty list so the normal generate path is
    unchanged.
    """
    root = getattr(deps, "project_root", None)
    if not root:
        return []
    try:
        from cgx.codegen.env_manager import requirements_locked
        if not requirements_locked(str(root)):
            return []
    except Exception:  # pragma: no cover - defensive
        return []
    from cgx.answer.engine import _content_to_new_file_patch
    out: List[Dict[str, str]] = []
    for rel in _LOCKED_ENV_FILES:
        path = Path(root) / rel
        try:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - defensive
            continue
        if not content.strip():
            continue
        out.append({
            "path": rel,
            "patch": _content_to_new_file_patch(rel, content),
            "content": content,
        })
    return out


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


# Frontend relative-import coherence. A bundler resolves an extension-less
# specifier by probing these script suffixes (and ``<spec>/index.<ext>``);
# stylesheet specifiers are safe to stub empty (they carry no behaviour),
# while the remaining asset families (images/fonts/json) are neither
# stubbable nor this gate's concern. Kept together so both frontend passes
# classify a relative specifier identically.
_STYLESHEET_EXTS = (".css", ".scss", ".sass", ".less", ".styl")
_JS_SCRIPT_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue")
_JS_ASSET_EXTS = (
    ".json", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".mp3", ".mp4", ".wasm",
    ".txt", ".md")

# Cap on stylesheet stubs synthesized in one pass so a badly-planned tree
# cannot fan the bundle out unboundedly (mirrors _SYNTH_MODULE_BUDGET).
_SYNTH_STYLESHEET_BUDGET = 16

# JS test-harness synthesis (P1a). When the scaffold authored JS/TS test
# files but omitted the toolchain that runs them, VERIFY's NpmRunner has no
# runnable ``test`` script to invoke, so the suite is silently never
# exercised while ``npm run build`` still reports green -- the exact blind
# spot that let ses_4cbf963cdc67435a ship a broken app with unrun React
# tests. Deterministically backfill the vite/vitest harness (test script,
# devDeps, jsdom config, setup) so the tests the model wrote are actually
# runnable. Pinned to the vite/vitest React ecosystem the scaffolder
# targets; caret ranges let npm resolve a compatible release at install
# time. A Vue tree is out of scope -- a react-shaped harness there would be
# wrong -- so it is left untouched.
_JS_TEST_EXTS = (".js", ".jsx", ".ts", ".tsx")
_VITEST_CONFIG_PATH = "vitest.config.js"
_VITEST_SETUP_PATH = "vitest.setup.js"
_JS_TEST_SCRIPT = "vitest run"
_JS_TEST_DEVDEPS = {
    "vitest": "^1.6.0",
    "jsdom": "^24.1.0",
    "@testing-library/react": "^15.0.7",
    "@testing-library/jest-dom": "^6.4.6",
    "@testing-library/user-event": "^14.5.2",
}
_JS_REACT_PLUGIN = ("@vitejs/plugin-react", "^4.3.1")


def _norm_rel_path(path: str) -> str:
    """Forward-slash, ``./``-stripped project-relative form of ``path``."""
    return (path or "").replace("\\", "/").strip().lstrip("./")


def _specifier_extension(spec: str) -> str:
    """Lowercase file extension of an import specifier's final segment."""
    tail = (spec or "").split("?", 1)[0].split("#", 1)[0].rstrip("/")
    tail = tail.rsplit("/", 1)[-1]
    dot = tail.rfind(".")
    return tail[dot:].lower() if dot > 0 else ""


def _resolve_relative_specifier(importer: str, spec: str) -> str:
    """Project-relative path a ``./`` / ``../`` specifier points at.

    Resolves ``spec`` against the importer's directory, collapsing ``.``
    and ``..`` segments and stripping any ``?query``/``#hash`` suffix a
    bundler tolerates. Returns ``""`` when resolution walks off the top of
    the tree with nothing left.
    """
    clean = (spec or "").split("?", 1)[0].split("#", 1)[0]
    imp = (importer or "").replace("\\", "/")
    parts = imp.rsplit("/", 1)[0].split("/") if "/" in imp else []
    parts = [p for p in parts if p]
    for seg in clean.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


def _generated_js_sources(
        existing_with_content: List[Dict[str, str]],
) -> List[Tuple[str, str]]:
    """``(path, content)`` for every generated JS/TS/Vue source in the batch."""
    out: List[Tuple[str, str]] = []
    for e in existing_with_content:
        path = _norm_rel_path(str(e.get("path") or ""))
        content = e.get("content")
        if path and isinstance(content, str) and path.lower().endswith(
                _JS_SCRIPT_EXTS):
            out.append((path, content))
    return out


def _synthesize_missing_frontend_stylesheets(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
        project_root: Optional[str],
) -> List[str]:
    """Author empty stylesheet stubs generated JS imports but the plan omits.

    The frontend analogue of :func:`_synthesize_missing_first_party_modules`:
    a scaffold model routinely writes ``import './index.css'`` in an entry
    point (the Vite React template ships one) without authoring the
    stylesheet, so ``npm run build`` aborts with ``Could not resolve
    "./index.css"`` and the whole tree is unbuildable -- observed live in
    ses_aa99f1fb6914488d, where the ensuing whole-tree regenerate reproduced
    the identical miss. Scan every generated JS/TS source for relative
    stylesheet imports (``.css/.scss/.sass/.less/.styl``) whose resolved
    target is neither generated nor already on disk, and splice a minimal
    empty stub into the diff bundle + in-memory manifest so the importer
    resolves. An empty stylesheet is the zero-risk conventional fix: it
    changes no behaviour and builds cleanly. Bounded by
    :data:`_SYNTH_STYLESHEET_BUDGET`; mutates the passed lists in place and
    returns the paths authored.
    """
    from cgx.answer.engine import (
        _content_to_new_file_patch,
        _js_relative_imports,
    )

    existing_paths = {
        _norm_rel_path(str(e.get("path") or "")) for e in existing_with_content}
    root = Path(project_root) if project_root else None

    missing: List[str] = []
    seen: set = set()
    for path, content in _generated_js_sources(existing_with_content):
        for spec in _js_relative_imports(content):
            if _specifier_extension(spec) not in _STYLESHEET_EXTS:
                continue
            target = _resolve_relative_specifier(path, spec)
            if not target or target in existing_paths or target in seen:
                continue
            if root is not None and (root / target).exists():
                continue
            seen.add(target)
            missing.append(target)
    if not missing:
        return []

    synth_files: List[Dict[str, Any]] = []
    added: List[str] = []
    for target in sorted(missing)[:_SYNTH_STYLESHEET_BUDGET]:
        content = (
            f"/* Auto-generated stylesheet stub for `{target}`.\n"
            "   A generated source imports this file but the work plan\n"
            "   omitted it; an empty stub lets the bundle build. */\n")
        diffs.append({
            "file": target,
            "patch": _content_to_new_file_patch(target, content)})
        generated.append({
            "file": target, "layer": "synthesized", "syntax_ok": True,
            "confidence": None, "bytes": len(content), "synthesized": True})
        existing_with_content.append({"path": target, "content": content})
        existing_paths.add(_norm_rel_path(target))
        synth_files.append({"path": target})
        added.append(target)

    layers.append({"name": "synthesized", "files": synth_files})
    logger.warning(
        "SCAFFOLD: generated frontend source(s) import stylesheet(s) the "
        "work plan omitted; synthesized empty stub(s) %s so the build "
        "resolves", added)
    return added


def _is_js_test_path(path: str) -> bool:
    """True for a JS/TS unit-test file (``*.test.jsx`` / ``__tests__/*``)."""
    p = _norm_rel_path(path).lower()
    ext = _specifier_extension(p)
    if ext not in _JS_TEST_EXTS:
        return False
    if "/__tests__/" in p or p.startswith("__tests__/"):
        return True
    stem = p[: -len(ext)]
    return stem.endswith(".test") or stem.endswith(".spec")


def _project_uses_react(
        existing_with_content: List[Dict[str, str]]) -> bool:
    """True when any generated JS source is JSX/TSX or imports ``react``."""
    for path, content in _generated_js_sources(existing_with_content):
        if path.lower().endswith((".jsx", ".tsx")):
            return True
        if "from 'react'" in content or 'from "react"' in content:
            return True
    return False


def _project_uses_vue(
        existing_with_content: List[Dict[str, str]]) -> bool:
    """True when the generated tree contains a ``.vue`` single-file component."""
    return any(
        _norm_rel_path(str(e.get("path") or "")).lower().endswith(".vue")
        for e in existing_with_content)


def _js_test_config_present(
        existing_with_content: List[Dict[str, str]]) -> bool:
    """True when a vitest config already exists (standalone or in vite config)."""
    for e in existing_with_content:
        base = _norm_rel_path(str(e.get("path") or "")).lower().rsplit(
            "/", 1)[-1]
        content = str(e.get("content") or "")
        if base.startswith("vitest.config."):
            return True
        if base.startswith("vite.config.") and (
                "test:" in content or "test :" in content):
            return True
    return False


def _splice_generated_file(
        path: str, content: str, *,
        diffs: List[Dict[str, str]], generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]]) -> bool:
    """Insert or replace a synthesized file across the scaffold bundles.

    Rewrites the new-file diff, generated-metadata row and content mirror
    for ``path`` in place (adding them when the file is new) so a
    deterministically authored/edited file rides the same APPLY path as a
    model-generated one. Returns ``True`` when ``path`` was newly added
    (so the caller can fold it into the synthesized manifest layer),
    ``False`` when an existing entry was rewritten.
    """
    from cgx.answer.engine import _content_to_new_file_patch
    norm = _norm_rel_path(path)
    patch = _content_to_new_file_patch(path, content)
    for d in diffs:
        if _norm_rel_path(str(d.get("file") or "")) == norm:
            d["patch"] = patch
            break
    else:
        diffs.append({"file": path, "patch": patch})
    for g in generated:
        if _norm_rel_path(str(g.get("file") or "")) == norm:
            g["bytes"] = len(content)
            g["syntax_ok"] = True
            g["synthesized"] = True
            break
    else:
        generated.append({
            "file": path, "layer": "synthesized", "syntax_ok": True,
            "confidence": None, "bytes": len(content), "synthesized": True})
    is_new = True
    for e in existing_with_content:
        if _norm_rel_path(str(e.get("path") or "")) == norm:
            e["content"] = content
            is_new = False
            break
    if is_new:
        existing_with_content.append({"path": path, "content": content})
    return is_new


def _ensure_pkg_test_harness(pkg_data: Dict[str, Any],
                             uses_react: bool) -> bool:
    """Fold a real ``test`` script + harness devDeps into a package.json dict.

    Sets ``scripts.test`` to ``vitest run`` only when absent or the npm
    ``no test specified`` placeholder (never clobbering a real script), and
    adds every missing harness devDependency (skipping any already declared
    in ``dependencies`` or ``devDependencies``). Mutates ``pkg_data`` in
    place; returns ``True`` when anything changed.
    """
    changed = False
    scripts = pkg_data.get("scripts")
    if not isinstance(scripts, dict):
        scripts = {}
        pkg_data["scripts"] = scripts
    cur = str(scripts.get("test") or "")
    if (not cur) or ("no test specified" in cur):
        scripts["test"] = _JS_TEST_SCRIPT
        changed = True
    dev = pkg_data.get("devDependencies")
    if not isinstance(dev, dict):
        dev = {}
        pkg_data["devDependencies"] = dev
    deps = (pkg_data.get("dependencies")
            if isinstance(pkg_data.get("dependencies"), dict) else {})
    wanted = dict(_JS_TEST_DEVDEPS)
    if uses_react:
        wanted[_JS_REACT_PLUGIN[0]] = _JS_REACT_PLUGIN[1]
    for name, ver in wanted.items():
        if name not in dev and name not in deps:
            dev[name] = ver
            changed = True
    return changed


def _vitest_config_content(uses_react: bool) -> str:
    """A minimal jsdom vitest config wired to the setup file (react-aware)."""
    react_import = ("import react from '@vitejs/plugin-react';\n"
                    if uses_react else "")
    plugins = "  plugins: [react()],\n" if uses_react else ""
    return (
        "import { defineConfig } from 'vitest/config';\n"
        f"{react_import}"
        "\n"
        "export default defineConfig({\n"
        f"{plugins}"
        "  test: {\n"
        "    environment: 'jsdom',\n"
        "    globals: true,\n"
        f"    setupFiles: './{_VITEST_SETUP_PATH}',\n"
        "  },\n"
        "});\n")


def _synthesize_js_test_harness(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
        project_root: Optional[str],
) -> List[str]:
    """Backfill the vite/vitest harness so scaffolded JS tests are runnable.

    When the scaffold authored ``*.test.jsx`` / ``*.spec.ts`` files (or a
    ``__tests__`` module) but the plan omitted the toolchain to run them,
    VERIFY's NpmRunner finds no ``test`` script and the suite is silently
    skipped while ``npm run build`` still passes (the ses_4cbf963cdc67435a
    blind spot). Deterministically ensure package.json carries a ``vitest
    run`` test script plus the harness devDeps, and synthesize a jsdom
    ``vitest.config.js`` + a ``@testing-library/jest-dom`` setup file when
    none exists. Mutates the passed bundles in place; returns the touched
    paths. Vue trees are skipped (out of scope). Best-effort: any parse
    failure abstains, leaving the bundle untouched.
    """
    import json as _json

    test_files = [p for p, _ in _generated_js_sources(existing_with_content)
                  if _is_js_test_path(p)]
    if not test_files:
        return []
    if _project_uses_vue(existing_with_content):
        logger.debug("SCAFFOLD: JS test-harness synthesis skipped "
                     "(Vue tree out of scope)")
        return []

    uses_react = _project_uses_react(existing_with_content)
    existing_paths = {
        _norm_rel_path(str(e.get("path") or "")): e
        for e in existing_with_content}
    touched: List[str] = []
    new_files: List[str] = []

    # 1) package.json: ensure a runnable test script + harness devDeps.
    pkg_entry = existing_paths.get("package.json")
    if pkg_entry is not None:
        try:
            pkg_data = _json.loads(str(pkg_entry.get("content") or "{}"))
        except Exception:
            logger.debug("SCAFFOLD: package.json is not valid JSON; skipping "
                         "test-harness backfill")
            pkg_data = None
        if isinstance(pkg_data, dict):
            if _ensure_pkg_test_harness(pkg_data, uses_react):
                content = _json.dumps(pkg_data, indent=2) + "\n"
                _splice_generated_file(
                    "package.json", content, diffs=diffs,
                    generated=generated,
                    existing_with_content=existing_with_content)
                touched.append("package.json")
    else:
        pkg_data = {"name": "app", "private": True, "version": "0.0.0",
                    "type": "module", "scripts": {}, "devDependencies": {}}
        _ensure_pkg_test_harness(pkg_data, uses_react)
        content = _json.dumps(pkg_data, indent=2) + "\n"
        if _splice_generated_file(
                "package.json", content, diffs=diffs, generated=generated,
                existing_with_content=existing_with_content):
            new_files.append("package.json")
        touched.append("package.json")

    # 2) vitest config + setup: only when the plan wired neither, so a
    # model-authored config (or a vite.config test block) is never clobbered.
    if not _js_test_config_present(existing_with_content):
        cfg = _vitest_config_content(uses_react)
        if _splice_generated_file(
                _VITEST_CONFIG_PATH, cfg, diffs=diffs, generated=generated,
                existing_with_content=existing_with_content):
            new_files.append(_VITEST_CONFIG_PATH)
        touched.append(_VITEST_CONFIG_PATH)
        if _VITEST_SETUP_PATH not in existing_paths:
            setup = "import '@testing-library/jest-dom';\n"
            if _splice_generated_file(
                    _VITEST_SETUP_PATH, setup, diffs=diffs,
                    generated=generated,
                    existing_with_content=existing_with_content):
                new_files.append(_VITEST_SETUP_PATH)
            touched.append(_VITEST_SETUP_PATH)

    if new_files:
        layers.append({"name": "synthesized",
                       "files": [{"path": p} for p in new_files]})
    if touched:
        logger.warning(
            "SCAFFOLD: scaffolded JS test file(s) but no runnable harness; "
            "backfilled %s so VERIFY can actually run the suite", touched)
    return touched


def _js_import_coherence_failures(
        existing_with_content: List[Dict[str, str]],
        project_root: Optional[str],
) -> List[Dict[str, str]]:
    """Flag JS/TS files with a relative *script* import that resolves nowhere.

    The frontend analogue of :func:`_import_coherence_failures`: a generated
    ``.jsx`` that does ``import Foo from './Foo'`` (or ``'./Foo.jsx'``) with
    no matching generated sibling and no file on disk ships a tree ``npm run
    build`` cannot resolve. A stub would be wrong here -- a script carries
    behaviour, unlike the stylesheets the pass above stubs empty -- so fail
    the importer and let the router's regenerate edge re-author it against
    the real sibling inventory. Stylesheet specifiers (handled by the stub
    backfill) and non-script asset specifiers (images/fonts/json) are
    skipped. Returns ``{"file", "error"}`` records; regex-based and
    best-effort.
    """
    from cgx.answer.engine import _js_relative_imports

    existing_paths = {
        _norm_rel_path(str(e.get("path") or "")) for e in existing_with_content}
    root = Path(project_root) if project_root else None

    def _resolves(target: str) -> bool:
        candidates = [target]
        if _specifier_extension(target) not in _JS_SCRIPT_EXTS:
            candidates += [target + ext for ext in _JS_SCRIPT_EXTS]
            candidates += [f"{target}/index{ext}" for ext in _JS_SCRIPT_EXTS]
        for cand in candidates:
            if _norm_rel_path(cand) in existing_paths:
                return True
            if root is not None and (root / cand).exists():
                return True
        return False

    failures: List[Dict[str, str]] = []
    for path, content in _generated_js_sources(existing_with_content):
        unresolved: List[str] = []
        for spec in _js_relative_imports(content):
            ext = _specifier_extension(spec)
            if ext in _STYLESHEET_EXTS or ext in _JS_ASSET_EXTS:
                continue
            target = _resolve_relative_specifier(path, spec)
            if target and not _resolves(target) and spec not in unresolved:
                unresolved.append(spec)
        if unresolved:
            specs = ", ".join(repr(s) for s in unresolved)
            failures.append({
                "file": path,
                "error": (
                    f"relative script import(s) {specs} do not resolve to a "
                    "generated module or a file on disk; import only existing "
                    "siblings or author the referenced module"),
            })
    return failures


def _reconcile_js_dependencies(
        *,
        diffs: List[Dict[str, str]],
        existing_with_content: List[Dict[str, str]],
) -> List[str]:
    """Splice missing runtime deps into package.json in place.

    Locates the ``package.json`` diff, reconstructs its content, and hands
    it to :func:`cgx.answer.engine._deterministic_package_json_repair`
    together with every generated JS/TS file so any bare import the manifest
    omits is added under ``dependencies``. Rewrites the diff (and the
    matching ``existing_with_content`` entry) with the repaired JSON using
    the same new-file unified-diff helpers APPLY expects. No-ops (returns
    ``[]``) when there is no package.json or nothing to add. Returns the
    list of added package names for the artifact/telemetry.
    """
    from cgx.answer.engine import (
        _content_to_new_file_patch,
        _deterministic_package_json_repair,
    )

    pkg_idx = -1
    for i, d in enumerate(diffs):
        if str(d.get("file") or "").strip().rsplit("/", 1)[-1] == "package.json":
            pkg_idx = i
            break
    if pkg_idx < 0:
        return []
    pkg_path = str(diffs[pkg_idx].get("file") or "").strip()

    ei = -1
    for i, e in enumerate(existing_with_content):
        if str(e.get("path") or "").strip() == pkg_path:
            ei = i
            break
    if ei < 0:
        return []
    before = str(existing_with_content[ei].get("content") or "")

    import json as _json
    try:
        before_deps = set((_json.loads(before) or {}).get("dependencies") or {})
    except Exception:
        before_deps = set()

    repaired = _deterministic_package_json_repair(
        before, list(existing_with_content))
    if repaired is None:
        return []
    try:
        after_deps = set((_json.loads(repaired) or {}).get("dependencies") or {})
    except Exception:
        return []
    added = sorted(after_deps - before_deps)
    if not added:
        return []

    diffs[pkg_idx] = {
        "file": pkg_path,
        "patch": _content_to_new_file_patch(pkg_path, repaired),
    }
    existing_with_content[ei] = {"path": pkg_path, "content": repaired}
    logger.warning(
        "SCAFFOLD: package.json omitted imported runtime dep(s) %s; "
        "adding them so the project can build", added)
    return added


def _reconcile_import_warnings(
        *,
        diffs: List[Dict[str, str]],
        generated: List[Dict[str, Any]],
        existing_with_content: List[Dict[str, str]],
        layers: List[Any],
        goal: str,
        provider: Any,
        contracts: Optional[Dict[str, Any]],
        skills: Optional[List[str]] = None,
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
                depends_on=dep, contracts=contracts, skills=skills)
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
