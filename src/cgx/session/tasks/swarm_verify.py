"""SWARM_VERIFY executor: the tree-level verification ladder.

The Developer generates and gates each file in isolation; this terminal stage
verifies the *whole* tree once every file has been attempted, mirroring the
graded checks the greenfield chain runs after SCAFFOLD:

1. **Skeleton coverage** -- every planned path exists on disk and (for ``.py``)
   parses; a missing or unparseable file is a coverage gap.
2. **Import coherence** -- :func:`cross_check_first_party_imports` flags a
   ``from <first-party> import <name>`` whose ``name`` no generated file
   defines (a phantom/misrooted import).
3. **Contract compliance** -- :func:`check_contract_compliance` flags a
   declared WORK_PLAN interface no file satisfies.

Coverage gaps and import breaks name a concrete file, so a bounded
regeneration loop re-runs the :mod:`swarm_generate` ladder on just those files
(``_MAX_VERIFY_ROUNDS`` rounds) before giving up. Finally an environment
dry-run installs any missing imports (:func:`preflight_install`) and runs the
project's tests (:func:`run_tests_on_disk`) so a red suite is visible. The
merged result is persisted as a ``SWARM_VERIFY_REPORT`` artifact and the router
ends the session COMPLETED only when the tree is structurally clean.
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


def _coverage_gaps(paths: List[str], contents: Dict[str, str]) -> List[str]:
    """Planned paths missing from disk, or ``.py`` files that will not parse."""
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


def _structural_scan(
        paths: List[str], contents: Dict[str, str],
        contracts: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]],
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
    imports = _merge_import_warnings(symbol_w, resolve_w)
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
                                                      contracts)
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

    # A structurally-clean tree whose suite is nonetheless red is exactly the
    # case the static gates cannot see: an import that resolves on paper but
    # breaks at runtime. Parse the failure, regenerate only the implicated
    # source files, and dry-run once more before the stage goes terminal.
    dyn_rounds = 0
    if structural_ok and env.get("outcome") == "failed":
        dyn_targets = _dynamic_regen_targets(env, paths)
        if dyn_targets:
            dyn_rounds = 1
            swarm_beat(project_root, "verify", "dynamic_regenerate",
                       targets=dyn_targets)
            _regenerate(dyn_targets, specs, contracts, goal, project_root,
                        deps.provider, paths)
            contents = _collect_contents(paths, project_root)
            gaps, import_w, contract_w = _structural_scan(
                paths, contents, contracts)
            structural_ok = not (gaps or import_w or contract_w)
            env = _run_env_dryrun(paths, project_root)

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
