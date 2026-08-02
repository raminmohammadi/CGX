


"""REPAIR executor: turn a failed VERIFY into a targeted patch.

Reads the upstream ``VERIFY_REPORT``, classifies the failure via
:mod:`cgx.session.repair.classify`, and -- for the classifications we
have a deterministic locator + proposer for -- emits a typed
``REPAIR_PLAN`` artifact whose ``diffs`` list is shaped exactly like a
``CODE_CHANGE_PLAN``. The shared APPLY executor consumes the plan in
the next router step; APPLY's own backup mirror keeps the rewrite
recoverable.

Every deterministic classification has a mechanical fix. When
classification returns ``unknown`` (an ordinary logic/assertion
failure with no deterministic locator), the executor attempts a
bounded LLM-driven repair (:func:`_propose_llm_logic_repair`): it
feeds the captured failure output plus the on-disk source/test files
to the provider and turns any accepted rewrite into a unified diff.
That path is a no-op without a provider or once the per-session
repair budget is spent, in which case the executor falls back to an
empty plan and the router escalates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cgx.session.budget import LoopBudget
from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.classify import (
    circular_import_modules,
    classify_runtime_report,
    classify_verify_report,
    failure_signature,
    missing_module_names,
    required_package_names,
    runtime_failure_text,
    third_party_import_breaks,
    traceback_source_files,
    undefined_names,
    unresolved_entry_paths,
)
from cgx.session.repair.locate import (
    MissingFixtureLocation,
    MissingPythonpathLocation,
    StyleMixLocation,
    locate_missing_fixture,
    locate_missing_module_pythonpath,
    locate_unittest_pytest_mix,
)
from cgx.session.repair.propose import (
    propose_missing_fixture,
    propose_missing_module_pythonpath,
    propose_third_party_pin,
    propose_unittest_pytest_mix,
)
from cgx.session.repair.pypi_client import PyPIClient
from cgx.session.scaffold_validate import stack_entry_description
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)

# Beyond this diff count the patch is large enough that a regenerate
# pass (with the failure recorded as a constraint) is likely cheaper
# than reviewing the per-file rewrites. The router still requires a
# SCAFFOLD ancestor + remaining regenerate budget to actually take the
# regenerate branch; otherwise the patch is applied as-is.
_PATCH_DIFF_LIMIT = 5

# Bounded LLM logic-repair caps. The router already gates the overall
# loop via ``REPAIR_BUDGET`` + the progress-aware failing-test-count
# trend (P2) + flap detection on ``failure_signature``; these caps keep a
# single REPAIR call's LLM cost and blast radius proportional to the
# failure. LLM repair is attempted on every repair attempt the router
# still funds (#4: it used to give up after 2 shots even while the router
# was willing to keep going on a genuinely-shrinking failure) -- the
# progress gate, not this executor, decides when to stop iterating. At
# most ``_LLM_REPAIR_MAX_FILES`` files are shown to the provider.
_LLM_REPAIR_MAX_ATTEMPT = 4
_LLM_REPAIR_MAX_FILES = 8

# Retrieval-fed repair (#6). When an index is wired into ``deps``
# (existing-repo / explore mode -- greenfield sessions have none), any
# file-slot the failure-localized candidates leave unused is filled with
# the source files hybrid retrieval judges most relevant to the failure,
# so a fix that must touch a symbol APPLY never wrote this attempt is not
# invisible to the provider. Retrieval widens ``top_k_per_view`` by this
# slack so path-resolution / de-dup drops still leave enough hits to fill
# the remaining budget.
_LLM_REPAIR_RETRIEVAL_SLACK = 4

# Classifications whose only realistic fix is to re-author the offending
# scaffold layer rather than mechanically patch what was already
# written. Used by :func:`_select_repair_strategy` when no diff was
# produced.
_REGENERATE_CLASSES = frozenset({
    "circular_import",
    "third_party_import_break",
    "relative_import_error",
    "smoke_import_failure",
    "api_check_failure",
    "runtime_failure",
    "empty_test_suite",
    "missing_fixture",
    "missing_module_pythonpath",
    "undefined_name",
    "unknown",
})


@register_executor(TaskKind.REPAIR)
def run_repair(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate a targeted REPAIR_PLAN for the upstream failure.

    Accepts three upstream artifact kinds:

    * ``VERIFY_REPORT`` -- the classic post-pytest repair loop. The
      classifier picks a deterministic locator + proposer.
    * ``SMOKE_REPORT`` -- a third-party import broke under the
      bootstrapped venv before VERIFY could even run. v1 has no
      deterministic proposer for this class (Phase 3.2 will add the
      dependency-aware proposer), so REPAIR records the structured
      failure with classification ``smoke_import_failure`` and lets
      the router escalate to ASK_USER.
    * ``API_CHECK_REPORT`` -- a hallucinated third-party symbol was
      caught before SMOKE/VERIFY. Same v1 contract as SMOKE: structured
      rationale + ``can_apply=False`` so the router escalates to
      ASK_USER until Phase 3.2's dependency-aware proposer lands.
    * ``RUNTIME_REPORT`` -- the unit suite passed but the app failed to
      boot under RUNTIME_VERIFY (bottleneck #3). There is no mechanical
      locator for a boot failure, so REPAIR records the captured
      import/create_app traceback with classification ``runtime_failure``
      and ``strategy='regenerate'`` so the router re-authors the failing
      entry module(s) via the nearest SCAFFOLD ancestor.
    """
    if not deps.project_root:
        return ExecutorResult(failure="REPAIR requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(failure="REPAIR requires a session store in deps")

    runtime_artifact_id = str(
        task.inputs.get("runtime_artifact_id") or "").strip()
    if runtime_artifact_id:
        return _run_runtime_repair(task, deps, runtime_artifact_id)

    api_check_artifact_id = str(
        task.inputs.get("api_check_artifact_id") or "").strip()
    if api_check_artifact_id:
        return _run_api_check_repair(task, deps, api_check_artifact_id)

    smoke_artifact_id = str(
        task.inputs.get("smoke_artifact_id") or "").strip()
    if smoke_artifact_id:
        return _run_smoke_repair(task, deps, smoke_artifact_id)

    verify_artifact_id = str(
        task.inputs.get("verify_artifact_id") or "").strip()
    if not verify_artifact_id:
        return ExecutorResult(
            failure="REPAIR missing verify_artifact_id, smoke_artifact_id, "
                    "or api_check_artifact_id input")
    verify_artifact = deps.store.get_artifact(verify_artifact_id)
    if (verify_artifact is None
            or verify_artifact.kind is not ArtifactKind.VERIFY_REPORT):
        return ExecutorResult(
            failure=f"REPAIR: artifact {verify_artifact_id!r} missing or "
                    "wrong kind (need VERIFY_REPORT)")

    content = dict(verify_artifact.content or {})
    classification = classify_verify_report(content)
    signature = failure_signature(content)
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1

    if classification == "missing_dependency":
        # The failure names the exact pip package it needs (e.g.
        # starlette's TestClient guard for httpx). Route straight to
        # install_deps -- no source rewrite can install a package.
        return _run_verify_missing_dependency_repair(
            task, verify_artifact_id, content,
            list(required_package_names(content)), signature)

    diffs: List[Dict[str, str]] = []
    rationale = ""
    locations_payload: List[Dict[str, Any]] = []
    extra_plan_fields: Dict[str, Any] = {}
    if classification == "unittest_pytest_mix":
        candidate_files = _candidate_test_files(content)
        locations = locate_unittest_pytest_mix(
            Path(deps.project_root), candidate_files)
        diffs = propose_unittest_pytest_mix(
            Path(deps.project_root), locations)
        rationale = _unittest_rationale(locations)
        locations_payload = [_loc_to_dict(loc) for loc in locations]
    elif classification == "missing_module_pythonpath":
        pp_locations = locate_missing_module_pythonpath(
            Path(deps.project_root), content)
        diffs = propose_missing_module_pythonpath(
            Path(deps.project_root), pp_locations)
        if not diffs and not pp_locations:
            # No project file claims the missing name at all: this is a
            # package absent from the venv (e.g. a transitive test-client
            # extra), not an authoring gap a regenerate could fill.
            pip_roots = _pip_installable_roots(
                Path(deps.project_root), content)
            if pip_roots:
                return _run_verify_missing_dependency_repair(
                    task, verify_artifact_id, content, pip_roots, signature)
        rationale = _pythonpath_rationale(pp_locations, bool(diffs))
        locations_payload = [_pp_loc_to_dict(loc) for loc in pp_locations]
    elif classification == "missing_fixture":
        fx_locations = locate_missing_fixture(
            Path(deps.project_root), content)
        diffs = propose_missing_fixture(
            Path(deps.project_root), fx_locations)
        rationale = _fixture_rationale(content, fx_locations, bool(diffs))
        locations_payload = [_fx_loc_to_dict(loc) for loc in fx_locations]
    elif classification == "third_party_import_break":
        pairs = third_party_import_breaks(content)
        installed = _installed_packages_from_build(deps, content)
        pypi_client = _resolve_pypi_client(deps)
        diffs, decisions = propose_third_party_pin(
            Path(deps.project_root), content,
            pairs=pairs, installed_packages=installed,
            pypi_client=pypi_client,
        )
        rationale = _third_party_rationale(pairs, decisions, bool(diffs))
        extra_plan_fields["import_breaks"] = [
            {"symbol": s, "package": p} for s, p in pairs
        ]
        extra_plan_fields["pin_decisions"] = decisions
    elif classification == "empty_test_suite":
        # pytest exit 5: the selected test file(s) exist but collected
        # zero tests -- almost always ``def test_*`` nested inside a
        # fixture / another function, or fixtures misnamed ``test_*``.
        # There is no mechanical patch; re-scaffold with an explicit
        # constraint so the regenerated tests are collectable.
        rationale = (
            "pytest collected 0 tests from the selected test file(s) "
            "(exit code 5). Test functions must be defined at module "
            "top level with names starting with 'test_' -- not nested "
            "inside a @pytest.fixture or any other function -- and "
            "fixtures must not be named 'test_*'. Rewrite the test "
            "module(s) so every test is a top-level 'def test_*'.")
    elif classification == "relative_import_error":
        # An "attempted relative import beyond top-level package" (or with no
        # known parent package): the scaffold authored a relative import that
        # cannot resolve -- a phantom sibling module or a level that walks
        # above the package root. There is no mechanical patch; re-scaffold
        # with an explicit constraint so the regenerated module imports only
        # real first-party modules (preferring absolute imports).
        rationale = (
            "A relative import could not be resolved (Python raised "
            "'attempted relative import beyond top-level package' / 'with no "
            "known parent package'). Re-author the offending module(s) so "
            "every first-party import targets a module that actually exists "
            "in the project -- prefer absolute imports rooted at the top-level "
            "package, and never import a module that was not generated.")
    elif classification == "circular_import":
        # First-party modules import each other in a cycle, so Python could
        # not finish initialising them ("partially initialized module"). No
        # single-file patch can decide which import to break or where the
        # shared symbols belong; re-scaffold with the cycle folded in as a
        # constraint so the regenerated module(s) keep the dependency
        # one-way.
        cycle_mods = circular_import_modules(content)
        extra_plan_fields["circular_modules"] = list(cycle_mods)
        mod_list = (", ".join(cycle_mods) if cycle_mods
                    else "the modules named in the traceback")
        rationale = (
            "Test collection failed on a circular import: "
            f"{mod_list} could not finish initialising because first-party "
            "modules import each other. Re-author the offending module(s) "
            "so the dependency is strictly one-way -- move shared symbols "
            "into the lower-level module (or a new shared module) and never "
            "import back from a module that imports this one.")
    elif classification == "undefined_name":
        # A generated module uses a name it never binds, so it explodes
        # at import time (``NameError: name 'enum' is not defined``) and
        # takes the whole collection down with it. The missing binding
        # could be an import, an assignment or a definition, so there is
        # no mechanical patch; re-scaffold with the unbound names folded
        # in as a constraint.
        names = undefined_names(content)
        extra_plan_fields["undefined_names"] = list(names)
        name_list = (", ".join(repr(n) for n in names) if names
                     else "the name(s) reported in the traceback")
        rationale = (
            f"Test collection failed with NameError: {name_list} used but "
            "never defined in the module that referenced it. Re-author the "
            "offending module(s) so every name is bound before use -- add "
            "the missing import, assignment or definition -- and make sure "
            "every module, class and constant the file references is either "
            "imported at the top of the file or defined in it.")
    else:
        diffs = _propose_llm_logic_repair(task, deps, content)
        if diffs:
            rationale = (
                "Bounded LLM repair proposed a targeted patch for the "
                "failing test(s) from the captured failure output and the "
                "current source; applying and re-verifying.")
        else:
            rationale = (
                "No deterministic repair available for this failure class; "
                "escalating to ASK_USER.")

    strategy, extra_constraints = _select_repair_strategy(
        classification=classification, diffs=diffs,
        rationale=rationale, extra_plan_fields=extra_plan_fields,
        locations_payload=locations_payload,
    )

    plan_content: Dict[str, Any] = {
        "verify_artifact_id": verify_artifact_id,
        "build_artifact_id": content.get("build_artifact_id"),
        "apply_artifact_id": content.get("apply_artifact_id"),
        "classification": classification,
        "failure_signature": signature,
        "repair_attempt": attempt,
        "rationale": rationale,
        "locations": locations_payload,
        "diffs": diffs,
        "strategy": strategy,
        "extra_constraints": extra_constraints,
        "mode": content.get("mode") or task.inputs.get("mode"),
    }
    plan_content.update(extra_plan_fields)
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content=plan_content,
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": artifact.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": len(diffs),
            "can_apply": bool(diffs),
            "strategy": strategy,
            "extra_constraints": extra_constraints,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _run_smoke_repair(task: TaskNode, deps: ExecutorDeps,
                      smoke_artifact_id: str) -> ExecutorResult:
    """Emit a REPAIR_PLAN from a SMOKE_REPORT.

    Phase 2.1 has no deterministic proposer for third-party import
    failures (Phase 3.2 will add the dependency-aware repair). The
    plan therefore carries ``can_apply=False`` so the router escalates
    to ASK_USER with the structured rationale -- which is still strictly
    better than the pre-SMOKE world where the same break surfaced as
    an opaque pytest collection trace.
    """
    smoke_artifact = deps.store.get_artifact(smoke_artifact_id)
    if (smoke_artifact is None
            or smoke_artifact.kind is not ArtifactKind.SMOKE_REPORT):
        return ExecutorResult(
            failure=f"REPAIR: artifact {smoke_artifact_id!r} missing or "
                    "wrong kind (need SMOKE_REPORT)")
    content = dict(smoke_artifact.content or {})
    failed = [str(m) for m in content.get("failed_modules") or []]
    build_smoke = content.get("build_smoke")
    build_broke = isinstance(build_smoke, dict) and not build_smoke.get("ok")
    signature = str(content.get("failure_signature") or "").strip()
    if not signature:
        signature = "smoke_import|" + ",".join(sorted(failed))
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    classification = "smoke_import_failure"
    build_stderr = ""
    missing_entries: Tuple[str, ...] = ()
    if failed:
        rationale = (
            f"Third-party import(s) {', '.join(failed)} failed under the "
            "bootstrapped venv. No deterministic repair is available in "
            "v1 -- the most likely fixes are a missing dependency in "
            "requirements.txt or a transitive version conflict. Phase "
            "3.2 will add a dependency-aware proposer; for now the "
            "router escalates to ASK_USER.")
    elif build_broke:
        # JS/TS build-smoke break: the frontend does not build. There is
        # no dependency to pin; re-authoring the offending files is the
        # only fix, so fold the build error into the regenerate feedback.
        build_stderr = str(build_smoke.get("stderr_tail") or "")
        label = str(build_smoke.get("label") or "npm run build")
        # ... unless the bundler could not resolve its *entry module*, in
        # which case no amount of re-authoring helps: the file is absent
        # from the manifest, so the regenerate has to add it.
        missing_entries = unresolved_entry_paths(build_stderr)
        if missing_entries:
            rationale = (
                f"The JS/TS build-smoke (`{label}`) could not resolve its "
                f"entry module(s) {', '.join(missing_entries)}: the "
                "file(s) were never generated. Add them to the tree.")
        else:
            rationale = (
                f"The JS/TS build-smoke (`{label}`) failed: the applied "
                "frontend does not build. Re-author the failing file(s) to "
                "fix the reported build error.")
    else:
        rationale = (
            "SMOKE reported a failure but no failed modules were "
            "recorded; escalating to ASK_USER.")
    extra_constraints: Dict[str, Any] = {
        "kind": "smoke_import_failure",
        "failed_modules": failed,
        "rationale": rationale,
    }
    if build_broke:
        extra_constraints["kind"] = "invalid_build_smoke"
        extra_constraints["build_error"] = build_stderr
    if missing_entries:
        # Mechanical fix for the one build failure regeneration cannot
        # reach: the router threads these into the new SCAFFOLD's
        # ``additional_files`` so the manifest grows the missing entry.
        extra_constraints["kind"] = "missing_entry_module"
        extra_constraints["missing_files"] = [
            {"path": p, "description": stack_entry_description(p)}
            for p in missing_entries]
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "smoke_artifact_id": smoke_artifact_id,
            "build_artifact_id": content.get("build_artifact_id"),
            "apply_artifact_id": content.get("apply_artifact_id"),
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "rationale": rationale,
            "failed_modules": failed,
            "diffs": [],
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
            "mode": task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": artifact.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": 0,
            "can_apply": False,
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
        },
        artifact=artifact,
    )


def _run_runtime_repair(task: TaskNode, deps: ExecutorDeps,
                        runtime_artifact_id: str) -> ExecutorResult:
    """Emit a REPAIR_PLAN from a RUNTIME_REPORT boot failure (#3).

    The unit suite is green but the app did not boot, so there is no
    failing test to patch and no mechanical locator to run. The plan
    therefore carries ``strategy='regenerate'`` with the captured
    import/``create_app`` traceback folded into ``extra_constraints`` so
    the router re-authors the failing entry module(s) -- and whatever they
    import -- via the nearest SCAFFOLD ancestor. ``can_apply=False`` keeps
    the empty-diff plan off the patch path.
    """
    artifact = deps.store.get_artifact(runtime_artifact_id)
    if (artifact is None
            or artifact.kind is not ArtifactKind.RUNTIME_REPORT):
        return ExecutorResult(
            failure=f"REPAIR: artifact {runtime_artifact_id!r} missing or "
                    "wrong kind (need RUNTIME_REPORT)")
    content = dict(artifact.content or {})
    classification = classify_runtime_report(content)
    failed = [str(f).strip() for f in content.get("failed_entries") or []
              if str(f).strip()]
    outcome = str(content.get("outcome") or "").strip()
    signature = str(content.get("failure_signature") or "").strip()
    if not signature:
        signature = "runtime_boot|" + ",".join(sorted(failed))
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    error_text = runtime_failure_text(content)
    if failed:
        rationale = (
            f"The application failed to boot ({outcome}): entry module(s) "
            f"{', '.join(failed)} raised at import or create_app() time. The "
            "unit tests pass but the app itself does not run. Re-author the "
            "failing entry module(s) and any first-party module they import "
            "so the module imports cleanly and, when a create_app() factory "
            "is present, it returns without raising. Boot error:\n"
            + (error_text or "(no captured output)"))
    else:
        rationale = (
            "RUNTIME_VERIFY reported a boot failure but no failing entry "
            "files were recorded; escalating to ASK_USER.")
    extra_constraints: Dict[str, Any] = {
        "kind": "runtime_failure",
        "failed_entries": failed,
        "outcome": outcome,
        "runtime_error": error_text,
        "rationale": rationale,
    }
    plan = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "runtime_artifact_id": runtime_artifact_id,
            "build_artifact_id": content.get("build_artifact_id"),
            "apply_artifact_id": content.get("apply_artifact_id"),
            "scaffold_artifact_id": content.get("scaffold_artifact_id"),
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "rationale": rationale,
            "failed_entries": failed,
            "diffs": [],
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
            "mode": task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": plan.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": 0,
            "can_apply": False,
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
        },
        artifact=plan,
    )


def _run_api_check_repair(task: TaskNode, deps: ExecutorDeps,
                          api_check_artifact_id: str) -> ExecutorResult:
    """Emit a REPAIR_PLAN from an API_CHECK_REPORT.

    Two failure shapes are handled:

    * ``missing_dependency`` -- the API_CHECK report lists
      ``missing_modules`` (a whole top-level package is absent from the
      venv). The symbol is valid; the fix is to (re)install the
      package, not to rewrite code. The plan carries
      ``strategy='install_deps'`` so the router re-runs BOOTSTRAP_ENV
      (whose preflight installs the undeclared imports and syncs
      requirements.txt) and then re-probes via API_CHECK.
    * hallucinated symbols -- a valid module is installed but the
      referenced attribute does not exist. Mirrors
      :func:`_run_smoke_repair`: v1 has no deterministic proposer, so
      the plan carries ``strategy='regenerate'`` / ``can_apply=False``.
    """
    artifact = deps.store.get_artifact(api_check_artifact_id)
    if (artifact is None
            or artifact.kind is not ArtifactKind.API_CHECK_REPORT):
        return ExecutorResult(
            failure=f"REPAIR: artifact {api_check_artifact_id!r} missing or "
                    "wrong kind (need API_CHECK_REPORT)")
    content = dict(artifact.content or {})
    missing_modules = [str(m).strip()
                       for m in content.get("missing_modules") or []
                       if str(m).strip()]
    if missing_modules:
        return _run_missing_dependency_repair(
            task, api_check_artifact_id, content, missing_modules)
    failed = [dict(r) for r in content.get("failed_references") or []
              if isinstance(r, dict)]
    signature = str(content.get("failure_signature") or "").strip()
    if not signature:
        parts = sorted(
            f"{r.get('module')}.{r.get('name')}" for r in failed)
        signature = "api_check|" + ",".join(parts)
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    classification = "api_check_failure"
    if failed:
        # Split "module could not be imported at all" (a wrong-path or
        # invented first-party import -- ``from app import x`` when the
        # module lives at ``backend/app.py``) from "module resolved but the
        # attribute is absent" (a hallucinated/outdated third-party symbol).
        # The two need opposite fixes -- correct the import path vs remove
        # the symbol -- so the regenerate constraint must not conflate them.
        unresolved = [r for r in failed
                      if "No module named" in str(r.get("error") or "")]
        absent = [r for r in failed if r not in unresolved]
        parts: List[str] = []
        if unresolved:
            mods = ", ".join(sorted({str(r.get("module"))
                                     for r in unresolved}))
            parts.append(
                f"The module(s) {mods} could NOT be imported (No module "
                "named ...). They are not installable third-party packages. "
                "If a module is defined elsewhere in this project, import it "
                "by its correct in-project path (e.g. `from backend.app "
                "import app`), matching the actual file layout; do not "
                "invent module names. If a module is not real, REMOVE every "
                "reference to it.")
        if absent:
            names = ", ".join(
                f"{r.get('module')}.{r.get('name')}" for r in absent[:5])
            if len(absent) > 5:
                names += f", ... (+{len(absent) - 5} more)"
            parts.append(
                f"The symbol(s) {names} do NOT exist in the installed "
                "package version (API_CHECK resolved the module but the "
                "attribute is absent -- a hallucinated or outdated import). "
                "When regenerating, REMOVE every reference to those exact "
                "symbols and use the correct, currently-supported API for "
                "the installed version instead. In particular, do not import "
                "a test client from werkzeug -- a Flask test uses "
                "`app.test_client()` obtained from the app object.")
        rationale = " ".join(parts)
    else:
        rationale = (
            "API_CHECK reported a failure but no failed references were "
            "recorded.")
    extra_constraints = {
        "kind": "api_check_failure",
        "failed_references": failed,
        "rationale": rationale,
    }
    plan = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "api_check_artifact_id": api_check_artifact_id,
            "build_artifact_id": content.get("build_artifact_id"),
            "apply_artifact_id": content.get("apply_artifact_id"),
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "rationale": rationale,
            "failed_references": failed,
            "diffs": [],
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
            "mode": task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": plan.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": 0,
            "can_apply": False,
            "strategy": "regenerate",
            "extra_constraints": extra_constraints,
        },
        artifact=plan,
    )


def _run_missing_dependency_repair(
        task: TaskNode,
        api_check_artifact_id: str,
        content: Dict[str, Any],
        missing_modules: List[str]) -> ExecutorResult:
    """Emit a REPAIR_PLAN that reinstalls absent third-party packages.

    A ``missing_dependency`` failure means a whole top-level package the
    scaffold imports is not installed in the venv -- a bootstrap/install
    problem, not a hallucinated API. Regenerating the code cannot fix it
    (the code is correct); the router consumes ``strategy='install_deps'``
    to re-run BOOTSTRAP_ENV, whose preflight installs the undeclared
    imports and syncs requirements.txt, then re-probes via API_CHECK.
    """
    mods = sorted(dict.fromkeys(missing_modules))
    signature = str(content.get("failure_signature") or "").strip()
    if not signature:
        signature = "api_check|missing:" + ",".join(mods)
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    classification = "missing_dependency"
    rationale = (
        f"Missing third-party dependency(ies): {', '.join(mods)}. The "
        "referenced symbol(s) are valid but the package(s) are not "
        "installed in the project venv. The correct fix is to install "
        "the package(s) (adding them to requirements.txt) and re-probe, "
        "not to regenerate code that references valid APIs.")
    extra_constraints = {
        "kind": "missing_dependency",
        "missing_modules": mods,
        "rationale": rationale,
    }
    plan = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "api_check_artifact_id": api_check_artifact_id,
            "build_artifact_id": content.get("build_artifact_id"),
            "apply_artifact_id": content.get("apply_artifact_id"),
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "rationale": rationale,
            "missing_modules": mods,
            "diffs": [],
            "strategy": "install_deps",
            "extra_constraints": extra_constraints,
            "mode": task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": plan.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": 0,
            "can_apply": False,
            "strategy": "install_deps",
            "missing_modules": mods,
            "extra_constraints": extra_constraints,
        },
        artifact=plan,
    )


def _pip_installable_roots(project_root: Path,
                           content: Dict[str, Any]) -> List[str]:
    """Missing-import roots that are pip problems, not authoring problems.

    A ModuleNotFoundError root whose top-level name has no ``<top>.py``
    file or ``<top>/`` directory under ``project_root`` cannot be fixed
    by regenerating source -- nothing on disk claims the name, so the
    realistic fix is a package install. Roots that do exist on disk (a
    missing *leaf* of a real package) stay with the regenerate path.
    Order-preserving and de-duplicated on the top-level name.
    """
    roots: List[str] = []
    for dotted in missing_module_names(content):
        top = dotted.split(".", 1)[0]
        if not top or top in roots:
            continue
        if (project_root / f"{top}.py").exists() \
                or (project_root / top).is_dir():
            continue
        roots.append(top)
    return roots


def _run_verify_missing_dependency_repair(
        task: TaskNode,
        verify_artifact_id: str,
        content: Dict[str, Any],
        missing_modules: List[str],
        signature: str) -> ExecutorResult:
    """Emit an install-deps REPAIR_PLAN for a VERIFY-time missing package.

    Mirrors :func:`_run_missing_dependency_repair` for the VERIFY path:
    pytest collection died because a package is absent from the venv --
    typically a transitive extra no first-party file imports directly
    (e.g. the fastapi/starlette TestClient's ``httpx``), so neither the
    bootstrap preflight nor a source regenerate can ever satisfy it.
    The plan carries ``strategy='install_deps'`` with the explicit
    module list so the router re-runs BOOTSTRAP_ENV, which installs the
    package(s) and flows back to VERIFY through API_CHECK/SMOKE.
    """
    mods = sorted(dict.fromkeys(m for m in missing_modules if m))
    if not signature:
        signature = "verify|missing:" + ",".join(mods)
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    classification = "missing_dependency"
    rationale = (
        f"Missing third-party dependency(ies): {', '.join(mods)}. Test "
        "collection failed because the package(s) are not installed in "
        "the project venv -- typically a transitive extra (such as the "
        "fastapi/starlette TestClient's httpx) that no first-party file "
        "imports directly. The correct fix is to install the package(s) "
        "(adding them to requirements.txt) and re-verify, not to "
        "regenerate source code that never imports them.")
    extra_constraints = {
        "kind": classification,
        "missing_modules": mods,
        "rationale": rationale,
    }
    plan = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "verify_artifact_id": verify_artifact_id,
            "build_artifact_id": content.get("build_artifact_id"),
            "apply_artifact_id": content.get("apply_artifact_id"),
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "rationale": rationale,
            "missing_modules": mods,
            "diffs": [],
            "strategy": "install_deps",
            "extra_constraints": extra_constraints,
            "mode": content.get("mode") or task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": plan.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": 0,
            "can_apply": False,
            "strategy": "install_deps",
            "missing_modules": mods,
            "extra_constraints": extra_constraints,
        },
        artifact=plan,
    )


def _select_repair_strategy(
        *, classification: str,
        diffs: List[Dict[str, str]],
        rationale: str,
        extra_plan_fields: Dict[str, Any],
        locations_payload: List[Dict[str, Any]]) -> tuple:
    """Pick ``patch`` vs ``regenerate`` and build the constraint payload.

    Heuristic (matches the Phase 6.1 spec):

    * ``patch`` when the proposer produced diffs *and* the diff count
      fits inside :data:`_PATCH_DIFF_LIMIT`. The router writes the
      diffs as it always has.
    * ``regenerate`` when (a) no diff was produced, *or* (b) the diff
      count exceeds :data:`_PATCH_DIFF_LIMIT` -- in both cases the
      router walks up to the nearest SCAFFOLD ancestor and re-queues a
      fresh SCAFFOLD with the constraint payload folded into its
      inputs. The router still falls back to ASK_USER when no SCAFFOLD
      ancestor exists or the regenerate budget is exhausted.

    The constraint payload is shaped per classification so SCAFFOLD's
    prompt builder (Phase 7) can pattern-match on ``kind`` rather than
    re-parsing free-text rationales.
    """
    diff_count = len(diffs)
    if diff_count and diff_count <= _PATCH_DIFF_LIMIT:
        return "patch", {}
    if (not diff_count
            and classification not in _REGENERATE_CLASSES):
        return "patch", {}
    constraints: Dict[str, Any] = {
        "kind": classification,
        "rationale": rationale,
    }
    if classification == "third_party_import_break":
        constraints["failures"] = list(
            extra_plan_fields.get("import_breaks") or [])
        constraints["attempted_pins"] = list(
            extra_plan_fields.get("pin_decisions") or [])
    elif classification == "circular_import":
        constraints["modules"] = list(
            extra_plan_fields.get("circular_modules") or [])
    elif classification == "undefined_name":
        constraints["undefined_names"] = list(
            extra_plan_fields.get("undefined_names") or [])
    elif classification == "unittest_pytest_mix":
        constraints["affected_classes"] = sorted({
            entry.get("class_name")
            for entry in locations_payload
            if entry.get("class_name")})
    if diff_count > _PATCH_DIFF_LIMIT:
        constraints["oversized_patch"] = {
            "diff_count": diff_count,
            "limit": _PATCH_DIFF_LIMIT,
        }
    return "regenerate", constraints


def _propose_llm_logic_repair(
        task: TaskNode, deps: ExecutorDeps,
        content: Dict[str, Any]) -> List[Dict[str, str]]:
    """Bounded LLM repair for an ``unknown`` logic/assertion failure.

    Reads the failing-test output plus the on-disk files most relevant to
    the failure and asks the provider for corrected complete file
    contents, turning each accepted rewrite into a unified diff shaped
    like every other ``propose_*`` result.

    Candidate files are localized from the failure itself (#4): the source
    files named in the traceback frames come *first* (that is where the
    error actually flowed, and it may be a source file APPLY never touched
    this attempt), followed by the files APPLY wrote / selected
    (``changed_files`` / ``tests_selected``). Any file-slot those leave
    unused is then filled by hybrid retrieval over the project index (#6)
    so a fix that must reach a symbol in an existing file neither the
    traceback nor APPLY named is still in scope -- a no-op in greenfield
    (no index). The traceback-referenced subset is passed to the generator
    so the prompt can point the model at the failing frames instead of
    asking it to re-derive the culprit.

    Returns an empty list -- so the caller falls back to the regenerate
    path -- when there is no provider, the repair attempt budget is spent,
    no candidate file is readable, or the model declined / produced
    nothing that parses.
    """
    if deps.provider is None or not deps.project_root:
        return []
    attempt = LoopBudget.from_inputs(task.inputs).repair_attempt or 1
    if attempt > _LLM_REPAIR_MAX_ATTEMPT:
        return []
    from cgx.answer.engine import generate_repair_files
    from cgx.session.repair.classify import failure_text
    from cgx.session.repair.propose import _unified_diff
    root = Path(deps.project_root)
    localized = _localized_source_files(content, root)
    localized_set = set(localized)
    goal = str(task.inputs.get("prior_goal") or content.get("goal") or "").strip()
    blob = failure_text(content)
    candidates = _repair_candidate_files(content, root)
    if len(candidates) < _LLM_REPAIR_MAX_FILES:
        query = _repair_retrieval_query(goal, blob)
        limit = _LLM_REPAIR_MAX_FILES - len(candidates)
        for rel in _retrieval_relevant_files(deps, query, root, limit):
            if rel not in candidates:
                candidates.append(rel)
    files: List[Dict[str, str]] = []
    originals: Dict[str, str] = {}
    shown_localized: List[str] = []
    for rel in candidates[:_LLM_REPAIR_MAX_FILES]:
        try:
            text = (root / rel).resolve().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        originals[rel] = text
        files.append({"path": rel, "content": text})
        if rel in localized_set:
            shown_localized.append(rel)
    if not files:
        return []
    try:
        fixed = generate_repair_files(
            deps.provider,
            goal=goal,
            failure_text=blob,
            files=files,
            max_files=_LLM_REPAIR_MAX_FILES,
            localized_files=shown_localized,
        )
    except Exception:  # pragma: no cover - defensive: provider hiccup
        logger.exception("REPAIR: bounded LLM logic repair crashed")
        return []
    diffs: List[Dict[str, str]] = []
    for rel, new_content in fixed.items():
        original = originals.get(rel)
        if original is None:
            continue
        patch = _unified_diff(rel, original, new_content)
        if patch:
            diffs.append({"file": rel, "patch": patch})
    return diffs


def _candidate_test_files(content: Dict[str, Any]) -> List[str]:
    """Return the files the locator should scan for an unittest/pytest mix.

    Prefers the explicit ``changed_files`` from the VERIFY_REPORT (the
    set of files the upstream APPLY wrote). Falls back to
    ``tests_selected`` so a collection error in an existing test file
    still gets scanned. Both lists are repo-relative paths.
    """
    out: List[str] = []
    for key in ("changed_files", "tests_selected"):
        v = content.get(key)
        if isinstance(v, list):
            for entry in v:
                s = str(entry).strip()
                if s and s not in out:
                    out.append(s)
    return out


def _localized_source_files(content: Dict[str, Any], root: Path) -> List[str]:
    """Resolve the traceback-named ``.py`` frames to on-disk repo paths.

    Each raw frame path (which may be runner-relative or absolute) is
    resolved against ``root``; only paths that resolve to an existing file
    inside the project are kept, order-preserving and de-duplicated.
    Anything outside the tree (stdlib / site-packages frames) is dropped
    so the repair context stays scoped to first-party code.
    """
    out: List[str] = []
    for raw in traceback_source_files(content):
        rel = _resolve_repo_relative(root, raw)
        if rel and rel not in out:
            out.append(rel)
    return out


def _resolve_repo_relative(root: Path, raw: str) -> Optional[str]:
    """Return ``raw`` as a repo-relative path when it names a project file.

    Handles both an absolute frame path under ``root`` and an already-
    relative one (optionally ``./``-prefixed). Returns ``None`` when the
    path escapes the tree or does not exist on disk.
    """
    candidate = raw[2:] if raw.startswith("./") else raw
    p = Path(candidate)
    if p.is_absolute():
        try:
            candidate = str(p.resolve().relative_to(root.resolve()))
        except (ValueError, OSError):
            return None
    if ".." in Path(candidate).parts:
        return None
    return candidate if (root / candidate).is_file() else None


def _repair_candidate_files(content: Dict[str, Any], root: Path) -> List[str]:
    """Order the files shown to the LLM repair, traceback frames first.

    The traceback-localized source files lead (that is where the failure
    actually flowed, and the culprit is often a source file APPLY did not
    touch this attempt), followed by the APPLY-written / selected files
    from :func:`_candidate_test_files`. De-duplicated, order-preserving.
    """
    out: List[str] = list(_localized_source_files(content, root))
    for rel in _candidate_test_files(content):
        if rel not in out:
            out.append(rel)
    return out


def _repair_retrieval_query(goal: str, failure_blob: str) -> str:
    """Compose the retrieval query for the #6 candidate-fill step.

    Pairs the original goal (what the code is supposed to do) with the
    first exception / pytest error line from the failure blob (what broke),
    which together steer hybrid retrieval at the symbols the fix is most
    likely to touch. Falls back to whichever half is present.
    """
    first = ""
    for raw in failure_blob.splitlines():
        line = raw.strip()
        if (line.startswith("E ") or line.startswith("E\t")
                or "Error:" in line or "Exception:" in line):
            first = line
            break
    return " ".join(p for p in (goal, first) if p).strip()


def _retrieval_relevant_files(
        deps: ExecutorDeps, query: str, root: Path, limit: int) -> List[str]:
    """Return up to ``limit`` repo-relative source files hybrid retrieval
    judges most relevant to ``query`` (#6).

    Best-effort and self-disabling: returns ``[]`` when no index is wired
    into ``deps`` (every greenfield session), when the query is empty, or
    when retrieval raises -- the caller then keeps its failure-localized
    candidates unchanged. Retrieval ``top_files`` are resolved against the
    project root and de-duplicated the same way traceback frames are, so
    only existing first-party files are ever handed to the provider.
    """
    if limit <= 0 or not query or not deps.index_dir or not deps.records_path:
        return []
    try:
        from cgx.pipeline.auto import run_query_auto
        out = run_query_auto(
            index_dir=deps.index_dir,
            records_path=deps.records_path,
            query=query,
            embedder=(deps.extra.get("embedder") if deps.extra else None),
            top_k_per_view=limit + _LLM_REPAIR_RETRIEVAL_SLACK,
        )
    except Exception:  # pragma: no cover - defensive: retrieval hiccup
        logger.exception("REPAIR: retrieval-fed candidate lookup crashed")
        return []
    files: List[str] = []
    for entry in (out or {}).get("top_files") or []:
        raw = (str((entry or {}).get("file") or "").strip()
               if isinstance(entry, dict) else str(entry).strip())
        if not raw:
            continue
        rel = _resolve_repo_relative(root, raw)
        if rel and rel not in files:
            files.append(rel)
        if len(files) >= limit:
            break
    return files


def _unittest_rationale(locations: List[StyleMixLocation]) -> str:
    """Compose a human-readable rationale for the UI / decision log."""
    if not locations:
        return ("Detected an AttributeError on a unittest helper, but no "
                "matching class was found to rewrite.")
    classes = ", ".join(sorted({loc.class_name for loc in locations}))
    helpers = sorted({h for loc in locations for h in loc.helpers})
    helper_str = ", ".join(helpers[:5]) + ("..." if len(helpers) > 5 else "")
    return (f"Added unittest.TestCase inheritance to {classes} so "
            f"self.{helper_str} calls resolve at runtime.")


def _loc_to_dict(loc: StyleMixLocation) -> Dict[str, Any]:
    return {
        "file": loc.rel_path,
        "class_name": loc.class_name,
        "lineno": loc.class_lineno,
        "helpers": sorted(loc.helpers),
    }


def _pythonpath_rationale(
    locations: List[MissingPythonpathLocation],
    has_diff: bool,
) -> str:
    """Compose a human-readable rationale for the pythonpath repair."""
    if not locations:
        return ("A test imported a module that does not exist as a project "
                "file on disk (ModuleNotFoundError during collection). No "
                "conftest sys.path entry can create a module that was never "
                "authored -- regenerate so every imported first-party module "
                "and symbol actually exists, and imports reference real "
                "modules only.")
    modules = ", ".join(sorted({loc.module_name for loc in locations}))
    if not has_diff:
        return (f"Project module(s) {modules} resolved on disk, but "
                "conftest.py already carries the sys.path fix from a "
                "previous repair attempt -- no further deterministic "
                "action is available.")
    return (f"Added project root to sys.path via conftest.py so pytest can "
            f"import {modules}.")


def _pp_loc_to_dict(loc: MissingPythonpathLocation) -> Dict[str, Any]:
    return {
        "file": loc.resolved_path,
        "module_name": loc.module_name,
        "top_level": loc.top_level,
    }


def _fixture_rationale(
    content: Dict[str, Any],
    locations: List[MissingFixtureLocation],
    has_diff: bool,
) -> str:
    """Compose a rationale for the missing_fixture repair."""
    from cgx.session.repair.classify import missing_fixture_names
    wanted = missing_fixture_names(content)
    if not locations:
        names = ", ".join(wanted) if wanted else "(unknown)"
        return (f"Pytest reported missing fixture(s) {names}, but no "
                "matching @pytest.fixture definition was found anywhere "
                "in the project; the fixture must be authored before "
                "the failure can be auto-repaired.")
    targets = sorted({loc.target_rel_path for loc in locations})
    names = ", ".join(loc.fixture_name for loc in locations)
    if not has_diff:
        return (f"Located @pytest.fixture definitions for {names}, but "
                f"{', '.join(targets)} already carries the hoist marker "
                "from a previous repair attempt -- no further "
                "deterministic action is available.")
    return (f"Hoisted @pytest.fixture {names} into {', '.join(targets)} "
            "so pytest can discover the fixture(s) during collection.")


def _fx_loc_to_dict(loc: MissingFixtureLocation) -> Dict[str, Any]:
    return {
        "fixture_name": loc.fixture_name,
        "file": loc.source_rel_path,
        "lineno": loc.source_lineno,
        "target": loc.target_rel_path,
    }


def _installed_packages_from_build(
        deps: ExecutorDeps,
        verify_content: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{normalised_name: version}`` from the upstream BUILD_REPORT.

    Reads ``resolved_packages`` (the ``pip freeze --all`` snapshot
    surfaced by Phase 1.1) off the BUILD_REPORT artifact referenced by
    the VERIFY_REPORT. Returns an empty mapping when no BUILD_REPORT is
    available (explore mode, or a stub session) -- the proposer treats
    that as "cannot resolve" and emits no diff.
    """
    build_artifact_id = str(verify_content.get("build_artifact_id") or "").strip()
    if not build_artifact_id or deps.store is None:
        return {}
    artifact = deps.store.get_artifact(build_artifact_id)
    if artifact is None or artifact.kind is not ArtifactKind.BUILD_REPORT:
        return {}
    out: Dict[str, str] = {}
    for entry in (artifact.content or {}).get("resolved_packages") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip().lower().replace("_", "-")
        version = str(entry.get("version") or "").strip()
        if name and version:
            out[name] = version
    return out


def _resolve_pypi_client(deps: ExecutorDeps) -> PyPIClient:
    """Return the injected PyPI client, or build a default."""
    injected = (deps.extra or {}).get("pypi_client")
    if isinstance(injected, PyPIClient):
        return injected
    return PyPIClient()


def _third_party_rationale(
        pairs: tuple,
        decisions: List[Dict[str, Any]],
        has_diff: bool) -> str:
    """Compose a human-readable rationale for the dependency-pin repair."""
    if not pairs:
        return ("Detected an ImportError on a third-party symbol, but no "
                "(symbol, package) pairs could be extracted from the "
                "failure record; escalating to ASK_USER.")
    pretty = ", ".join(f"{pkg}.{sym}" for sym, pkg in pairs)
    if not has_diff:
        why = "; ".join(d.get("reason", "") for d in decisions if d.get("reason"))
        return (f"Third-party symbol(s) {pretty} are missing under the "
                f"current install. No corrective pin could be derived: "
                f"{why or 'no candidates found'}; escalating to ASK_USER.")
    chosen = [d for d in decisions if d.get("pin")]
    pin_str = ", ".join(
        f"{d.get('broken_pkg')} <- {d.get('pin')} "
        f"(consumer={d.get('consumer')}=={d.get('consumer_version')})"
        for d in chosen
    )
    return (f"Detected third-party API break {pretty}; pinning "
            f"requirements.txt: {pin_str}.")


def _normalize_diffs(raw: Any) -> List[Dict[str, str]]:
    """Re-shape diffs in case a caller persisted mixed-key entries."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        file = str(entry.get("file") or entry.get("path") or "").strip()
        patch = str(entry.get("patch") or entry.get("diff") or "")
        if not file or not patch:
            continue
        out.append({"file": file, "patch": patch})
    return out


_get_repair_diffs: Optional[Any] = _normalize_diffs  # exported alias
