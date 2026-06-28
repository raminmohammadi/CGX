


"""REPAIR executor: turn a failed VERIFY into a targeted patch.

Reads the upstream ``VERIFY_REPORT``, classifies the failure via
:mod:`cgx.session.repair.classify`, and -- for the classifications we
have a deterministic locator + proposer for -- emits a typed
``REPAIR_PLAN`` artifact whose ``diffs`` list is shaped exactly like a
``CODE_CHANGE_PLAN``. The shared APPLY executor consumes the plan in
the next router step; APPLY's own backup mirror keeps the rewrite
recoverable.

The executor is intentionally LLM-free in v1: every supported
classification has a deterministic fix. If classification returns
``unknown`` the executor emits a plan with empty ``diffs`` and an
explanatory ``rationale`` -- the router then routes to ASK_USER
instead of looping with no progress.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.repair.classify import (
    classify_verify_report,
    failure_signature,
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
    propose_unittest_pytest_mix,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.REPAIR)
def run_repair(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate a targeted REPAIR_PLAN for the upstream VERIFY failure."""
    if not deps.project_root:
        return ExecutorResult(failure="REPAIR requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(failure="REPAIR requires a session store in deps")

    verify_artifact_id = str(
        task.inputs.get("verify_artifact_id") or "").strip()
    if not verify_artifact_id:
        return ExecutorResult(failure="REPAIR missing verify_artifact_id input")
    verify_artifact = deps.store.get_artifact(verify_artifact_id)
    if (verify_artifact is None
            or verify_artifact.kind is not ArtifactKind.VERIFY_REPORT):
        return ExecutorResult(
            failure=f"REPAIR: artifact {verify_artifact_id!r} missing or "
                    "wrong kind (need VERIFY_REPORT)")

    content = dict(verify_artifact.content or {})
    classification = classify_verify_report(content)
    signature = failure_signature(content)
    attempt = int(task.inputs.get("repair_attempt") or 1)

    diffs: List[Dict[str, str]] = []
    rationale = ""
    locations_payload: List[Dict[str, Any]] = []
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
        rationale = _pythonpath_rationale(pp_locations, bool(diffs))
        locations_payload = [_pp_loc_to_dict(loc) for loc in pp_locations]
    elif classification == "missing_fixture":
        fx_locations = locate_missing_fixture(
            Path(deps.project_root), content)
        diffs = propose_missing_fixture(
            Path(deps.project_root), fx_locations)
        rationale = _fixture_rationale(content, fx_locations, bool(diffs))
        locations_payload = [_fx_loc_to_dict(loc) for loc in fx_locations]
    else:
        rationale = (
            "No deterministic repair available for this failure class; "
            "escalating to ASK_USER.")

    artifact = Artifact.new(
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
            "locations": locations_payload,
            "diffs": diffs,
            "mode": content.get("mode") or task.inputs.get("mode"),
        },
    )
    return ExecutorResult(
        outputs={
            "repair_artifact_id": artifact.artifact_id,
            "classification": classification,
            "failure_signature": signature,
            "repair_attempt": attempt,
            "diff_count": len(diffs),
            "can_apply": bool(diffs),
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

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
        return ("Detected ModuleNotFoundError during collection, but none of "
                "the missing modules map to a project file; the package is "
                "likely third-party and belongs to BOOTSTRAP_ENV.")
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
