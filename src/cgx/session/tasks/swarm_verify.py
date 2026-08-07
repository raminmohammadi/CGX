"""SWARM_VERIFY executor: the tree-level verification ladder.

The Developer generates and gates each file in isolation; this terminal stage
verifies the *whole* tree once every file has been attempted, mirroring the
graded checks the greenfield chain runs after SCAFFOLD:

1. **Skeleton coverage** -- every planned path exists on disk and (for ``.py``)
   parses; a missing or unparseable file is a coverage gap. A planned pytest
   module (``test_*.py`` / ``*_test.py``) that parses but defines no
   collectible test is also a gap -- an empty test file is a promised test the
   tree does not deliver, not a satisfied one.
2. **Import coherence** -- :func:`cross_check_first_party_imports` flags a
   ``from <first-party> import <name>`` whose ``name`` no generated file
   defines (a phantom/misrooted import).
3. **Contract compliance** -- :func:`check_contract_compliance` flags a
   declared WORK_PLAN interface no file satisfies.

Coverage gaps and import breaks name a concrete file, so a bounded
regeneration loop re-runs the :mod:`swarm_generate` ladder on just those files
(``_MAX_VERIFY_ROUNDS`` rounds) before giving up. Finally an environment
dry-run installs any missing imports (:func:`preflight_install`) and runs the
project's tests (:func:`run_tests_on_disk`) so a red suite is visible. When the
tree is structurally clean but the suite is nonetheless red -- a defect the
static gates cannot see -- the failure output is fed back through
:func:`generate_repair_files` (failure-driven repair) so the model corrects the
specific error rather than blindly re-emitting the same broken file. The merged
result is persisted as a ``SWARM_VERIFY_REPORT`` artifact and the router ends
the session COMPLETED only when the tree is structurally clean.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from cgx.session.import_audit import resolve_first_party_imports
from cgx.session.models import Artifact, ArtifactKind, TaskKind
from cgx.session.scaffold_validate import (
    _module_name_for_path, check_contract_compliance,
    cross_check_first_party_imports)
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.swarm_generate import generate_file
from cgx.session.tasks.swarm_ground import _safe_read
from cgx.session.tasks.swarm_log import swarm_beat
from cgx.session.tasks.swarm_plan import plan_specs
from cgx.session.tasks.swarm_tools import edit_file

_MAX_VERIFY_ROUNDS = 2
# Failure-driven repair re-runs against the *current* red suite each round, so
# a defect unmasked only after an earlier fix (a logic bug hidden behind an
# import error that aborted collection) earns its own targeted repair rather
# than being stranded by a single pass.
_MAX_DYNAMIC_REPAIR_ROUNDS = 3


def _load_plan(deps: ExecutorDeps, artifact_id: str) -> Dict[str, Any]:
    """The WORK_PLAN artifact content, or ``{}`` when unavailable."""
    if not artifact_id or deps.store is None:
        return {}
    art = deps.store.get_artifact(artifact_id)
    return dict(art.content) if art and art.content else {}


def _collect_contents(paths: List[str], root: str) -> Dict[str, str]:
    """On-disk text of every planned path that is readable."""
    out: Dict[str, str] = {}
    for p in paths:
        src = _safe_read(p, root)
        if src is not None:
            out[p] = src
    return out


def _is_pytest_test_path(path: str) -> bool:
    """True when ``path`` is a ``.py`` file pytest collects as a test module.

    Matches pytest's default naming (``test_*.py`` / ``*_test.py``); a helper
    or fixtures module under ``tests/`` that does not follow the convention is
    not collected, so it is excluded here too.
    """
    p = (path or "").strip().lower()
    if not p.endswith(".py"):
        return False
    base = p.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py")


def _has_collectible_test(src: str) -> bool:
    """True when a parsed test module defines at least one pytest test.

    A ``test_*`` function at module level, or a ``Test*`` class containing a
    ``test_*`` method -- pytest's default collection. Assumes ``src`` parses
    (the caller flags a syntax error as a gap first), so a green ``ast.parse``
    on an empty ``test_*.py`` no longer masks a file with nothing to run.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:  # pragma: no cover - caller already flagged this
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                return True
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name.startswith("test"):
                    return True
    return False


def _coverage_gaps(paths: List[str], contents: Dict[str, str]) -> List[str]:
    """Planned paths missing from disk, unparsable ``.py``, or empty tests.

    A planned pytest module that parses but defines no collectible test is a
    coverage hole -- the plan promised a test the tree does not deliver -- so
    it is named as a gap and drives a targeted regenerate, rather than being
    silently accepted because the (empty) file happened to parse.
    """
    gaps: List[str] = []
    for p in paths:
        src = contents.get(p)
        if src is None:
            gaps.append(p)
            continue
        if p.endswith(".py"):
            try:
                ast.parse(src)
            except SyntaxError:
                gaps.append(p)
                continue
            if _is_pytest_test_path(p) and not _has_collectible_test(src):
                gaps.append(p)
    return gaps


def _merge_import_warnings(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union import warnings from every gate, de-duplicated by (file, module)."""
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for group in groups:
        for w in group or []:
            key = (str(w.get("file") or ""), str(w.get("module") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(w)
    return merged

def _check_phantom_third_party_imports(paths: List[str], contents: Dict[str, str], allowed: List[str], root: str) -> List[Dict[str, Any]]:
    from cgx.codegen.env_manager import _extract_imports_python, _is_local_package, _STDLIB_TOP
    warnings = []
    allowed_set = set(allowed)
    for p in paths:
        if not p.endswith(".py"): continue
        src = contents.get(p)
        if not src: continue
        imports = _extract_imports_python(src)
        for imp in imports:
            if imp in _STDLIB_TOP or _is_local_package(root, imp):
                continue
            if imp not in allowed_set:
                warnings.append({
                    "kind": "phantom_third_party",
                    "file": p,
                    "module": imp,
                    "reason": f"Imported third-party module '{imp}' is not in the plan's allowed third_party_dependencies."
                })
    return warnings


def _structural_scan(
        paths: List[str], contents: Dict[str, str],
        contracts: Dict[str, Any], root: str) -> Tuple[List[str], List[Dict[str, Any]],
                                            List[Dict[str, Any]]]:
    """Run the three structural checks; never raises (each gate is defensive).

    Import coherence is the union of two complementary gates: the symbol-level
    ``cross_check_first_party_imports`` (a ``from X import name`` naming a
    symbol no file defines) and the path-level ``resolve_first_party_imports``
    (a first-party module that resolves against neither the project root nor
    ``root/src`` -- the misrooting class the basename-blind check abstained on).
    """
    gaps = _coverage_gaps(paths, contents)
    try:
        symbol_w = cross_check_first_party_imports(contents)
    except Exception:  # pragma: no cover - the gate is best-effort
        symbol_w = []
    try:
        resolve_w = resolve_first_party_imports(contents, paths)
    except Exception:  # pragma: no cover - the gate is best-effort
        resolve_w = []
    try:
        phantom_3p_w = _check_phantom_third_party_imports(paths, contents, contracts.get("third_party_dependencies") or [], root)
    except Exception:
        phantom_3p_w = []
    imports = _merge_import_warnings(symbol_w, resolve_w, phantom_3p_w)
    try:
        contract = check_contract_compliance(contents, contracts)
    except Exception:  # pragma: no cover - the gate is best-effort
        contract = []
    return gaps, imports, contract


def _regen_targets(gaps: List[str],
                   import_warnings: List[Dict[str, Any]]) -> List[str]:
    """The precise files a structural failure implicates, de-duplicated."""
    targets: List[str] = list(gaps)
    for w in import_warnings:
        f = str(w.get("file") or "")
        if f and f not in targets:
            targets.append(f)
    return targets


def _regenerate(targets: List[str], specs: Dict[str, Any],
                contracts: Dict[str, Any], goal: str, root: str,
                provider: Any, all_paths: List[str]) -> List[str]:
    """Re-run the generation ladder on ``targets``; return the still-failing."""
    still_bad: List[str] = []
    for path in targets:
        spec = specs.get(path, {})
        outcome = generate_file(
            path=path, description=str(spec.get("description") or ""),
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=root, provider=provider, layer=path,
            manifest_paths=all_paths, log_root=root)
        if outcome.ok:
            edit_file(path, outcome.content, root)
        else:
            still_bad.append(path)
    return still_bad


def _run_env_dryrun(paths: List[str], root: str) -> Dict[str, Any]:
    """Install missing imports then run the project's tests (best-effort)."""
    py = [p for p in paths if p.endswith(".py")]
    report: Dict[str, Any] = {"ran": False, "outcome": "skipped"}
    try:
        from cgx.codegen.env_manager import preflight_install
        missing, results = preflight_install(py, root)
        report["missing_installed"] = missing
        report["install_results"] = results
    except Exception as exc:  # pragma: no cover - install is best-effort
        report["install_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from cgx.codegen.test_runner import run_tests_on_disk
        outcome = run_tests_on_disk(root, py)
        report["ran"] = bool(outcome.ran)
        report["returncode"] = outcome.returncode
        report["skipped_reason"] = outcome.skipped_reason
        report["tests_selected"] = list(outcome.tests_selected)
        # Retained so a red suite can be parsed for the specific import that
        # broke and mapped back to the source file to regenerate (6d).
        report["output"] = ((outcome.stdout or "") + "\n"
                             + (outcome.stderr or "")).strip()
        report["outcome"] = (
            "skipped" if not outcome.ran
            else "passed" if outcome.returncode == 0 else "failed")
    except Exception as exc:  # pragma: no cover - runner is best-effort
        report["test_error"] = f"{type(exc).__name__}: {exc}"
    return report


_MODULE_ERR_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):[^\n]*?['\"]([\w.]+)['\"]")


def _repair_context_paths(localized: List[str], paths: List[str],
                          specs: Dict[str, Any]) -> List[str]:
    """The focused ``.py`` set to offer the repairer, localized files first.

    A weak local model declines (returns ``{"files": []}``) when handed the
    whole planned tree as context -- the signal drowns in unrelated files, so a
    one-line ``import`` fix in a single module is never made. When the
    traceback localized one or more files, offer *only* those plus their direct
    ``depends_on`` (the sibling API the failing frame calls), so the prompt is
    small and centred on the defect. With no localization (a runtime failure
    naming no file) fall back to every planned ``.py`` -- there is no better
    hint and small trees still fit. Order is localized-first so
    ``generate_repair_files``' own ``max_files`` cap never drops a target.
    """
    py = [p for p in paths if p.endswith(".py")]
    if not localized:
        return py
    keep: List[str] = []
    for t in localized:
        if t in py and t not in keep:
            keep.append(t)
        for dep in (specs.get(t, {}) or {}).get("depends_on") or []:
            dep = str(dep)
            if dep in py and dep not in keep:
                keep.append(dep)
    return keep or py


def _dynamic_repair(env: Dict[str, Any], localized: List[str],
                    contents: Dict[str, str], goal: str, root: str,
                    provider: Any, paths: List[str],
                    specs: Dict[str, Any]) -> List[str]:
    """Failure-driven repair of a red-but-structurally-clean tree.

    Blind regeneration re-asks with the same description and contracts, so a
    weak model reproduces the same runtime defect (a misused pytest API, an
    off-by-one assertion). This instead threads the red suite's own output and
    the implicated file bodies into :func:`generate_repair_files`, which
    diagnoses the concrete failure and returns corrected content -- the step
    that turns "detected the failure" into "produced working code". The context
    is focused via :func:`_repair_context_paths` (the localized files plus
    their dependencies, not the whole tree) so a weak model does not decline on
    an over-large prompt; ``localized`` is flagged so it starts at the failing
    frames. Only files the repairer actually rewrote (and that pass its own
    source validation) are written back. Returns the paths it changed.
    """
    from cgx.answer.engine import generate_repair_files
    failure_text = str(env.get("output") or "")
    if not failure_text:
        return []
    context_paths = _repair_context_paths(localized, paths, specs)
    files = [{"path": p, "content": contents[p]}
             for p in context_paths if p in contents]
    if not files:
        return []
    try:
        repaired = generate_repair_files(
            provider, goal=goal, failure_text=failure_text,
            files=files, localized_files=localized)
    except Exception:  # pragma: no cover - repair is best-effort
        return []
    written: List[str] = []
    for path, content in (repaired or {}).items():
        edit_file(path, content, root)
        written.append(path)
    return written


def _dynamic_regen_targets(env: Dict[str, Any], paths: List[str]) -> List[str]:
    """Planned source files implicated by a red suite's import failures.

    Two complementary signals, both bounded to *planned* paths so a stray
    traceback frame in a dependency can never widen the blast radius:

    * any planned path named verbatim in the pytest output (the importing
      frame pytest prints for a collection error), and
    * any planned module whose dotted name (or ``src``-stripped variant)
      matches a ``ModuleNotFoundError`` / ``ImportError`` target -- the file
      that was expected to *provide* the missing module.
    """
    text = str(env.get("output") or "")
    if not text:
        return []
    targets: List[str] = []
    for p in paths:
        if p and p in text and p not in targets:
            targets.append(p)
    wanted = set(_MODULE_ERR_RE.findall(text))
    if wanted:
        for p in paths:
            mod = _module_name_for_path(p)
            if not mod:
                continue
            variants = {mod}
            if mod.startswith("src."):
                variants.add(mod[len("src."):])
            if (variants & wanted) and p not in targets:
                targets.append(p)
    return targets


@register_executor(TaskKind.SWARM_VERIFY)
def swarm_verify(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Verify the generated tree, regenerate broken files, dry-run the env."""
    if deps.provider is None:
        return ExecutorResult(
            failure="No provider configured for Swarm mode.", retryable=False)

    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "")
    failed_paths: List[str] = list(task.inputs.get("failed_paths") or [])
    plan = _load_plan(deps, work_plan_id)
    paths: List[str] = list(plan.get("paths") or [])
    contracts = plan.get("contracts") or {}
    specs = plan_specs({"layers": plan.get("layers") or []})
    goal = str(task.inputs.get("goal") or plan.get("goal") or "")
    project_root = (task.inputs.get("project_root")
                    or plan.get("project_root")
                    or deps.project_root or ".")

    swarm_beat(project_root, "verify", "structural", total=len(paths))
    rounds = 0
    while True:
        contents = _collect_contents(paths, project_root)
        gaps, import_w, contract_w = _structural_scan(paths, contents,
                                                      contracts, project_root)
        targets = _regen_targets(gaps, import_w)
        if not targets or rounds >= _MAX_VERIFY_ROUNDS:
            break
        rounds += 1
        swarm_beat(project_root, "verify", "regenerate", round=rounds,
                   targets=targets)
        _regenerate(targets, specs, contracts, goal, project_root,
                    deps.provider, paths)

    env = _run_env_dryrun(paths, project_root)
    structural_ok = not (gaps or import_w or contract_w)
    # Contract warnings are *soft*: they are advisory (a declared interface a
    # file does not obviously satisfy) and, unlike coverage gaps and import
    # breaks, name no file the regeneration loop can act on. They must not gate
    # the pytest-driven repair -- a red suite on a tree with only contract
    # warnings still deserves failure-driven repair, so key that loop on the
    # *hard* structural signals alone.
    hard_structural_ok = not (gaps or import_w)

    # A structurally-clean tree whose suite is nonetheless red is exactly the
    # case the static gates cannot see: an import that resolves on paper but
    # breaks at runtime. Parse the failure, regenerate only the implicated
    # source files, and dry-run once more before the stage goes terminal.
    dyn_rounds = 0
    while (hard_structural_ok and env.get("outcome") == "failed"
           and dyn_rounds < _MAX_DYNAMIC_REPAIR_ROUNDS):
        # Re-localize against the *current* failure each round: a red suite is
        # exactly the defect the static gates cannot see, and a bug exposed
        # only once an earlier import error is cleared (collection aborts at the
        # first broken module, masking the rest) must get its own targeted pass.
        # Import-style failures localize to the file expected to provide the
        # missing module; other red suites (a wrong assertion, a runtime
        # TypeError) name no such file but are still fair game -- the import
        # targets, when present, merely seed the localization hint.
        prev_output = str(env.get("output") or "")
        dyn_targets = _dynamic_regen_targets(env, paths)
        dyn_rounds += 1
        swarm_beat(project_root, "verify", "dynamic_regenerate",
                   round=dyn_rounds, targets=dyn_targets)
        # Prefer failure-driven repair (fed the red suite's output); fall back
        # to blind regeneration of the import targets only when the repairer
        # declines, so a missing-module case still gets its provider
        # regenerated even if the repair pass produced nothing.
        repaired = _dynamic_repair(env, dyn_targets, contents, goal,
                                   project_root, deps.provider, paths, specs)
        if not repaired and dyn_targets:
            _regenerate(dyn_targets, specs, contracts, goal, project_root,
                        deps.provider, paths)
        contents = _collect_contents(paths, project_root)
        gaps, import_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root)
        structural_ok = not (gaps or import_w or contract_w)
        hard_structural_ok = not (gaps or import_w)
        env = _run_env_dryrun(paths, project_root)
        # Stop early when a round cannot make progress: nothing to act on (no
        # repair and no regen target), or the failure is byte-for-byte
        # unchanged -- another identical pass would only burn a round.
        if not repaired and not dyn_targets:
            break
        if str(env.get("output") or "") == prev_output:
            break

    tests_red = env.get("outcome") == "failed"
    verify_ok = structural_ok and not failed_paths and not tests_red

    content = {
        "work_plan_artifact_id": work_plan_id,
        "project_root": project_root,
        "paths": paths,
        "coverage_gaps": gaps,
        "import_warnings": import_w,
        "contract_warnings": contract_w,
        "regen_rounds": rounds,
        "dynamic_regen_rounds": dyn_rounds,
        "failed_paths": failed_paths,
        "env": env,
        "structural_ok": structural_ok,
        "verify_ok": verify_ok,
    }
    artifact = Artifact.new(
        session_id=task.session_id, produced_by_task_id=task.task_id,
        kind=ArtifactKind.SWARM_VERIFY_REPORT, content=content)
    swarm_beat(project_root, "verify", "report", ok=verify_ok,
               gaps=len(gaps), imports=len(import_w),
               contracts=len(contract_w), tests=env.get("outcome"))
    return ExecutorResult(
        artifact=artifact,
        outputs={"verify_ok": verify_ok,
                 "verify_report_artifact_id": artifact.artifact_id,
                 "coverage_gaps": gaps,
                 "failed_paths": failed_paths,
                 "project_root": project_root})
