

"""BOOTSTRAP_ENV executor: provision a project-local runtime environment.

Sits between APPLY and VERIFY in the greenfield loop. Its job is to
make ``project_root`` runnable in isolation so a subsequent VERIFY can
exit code-1 (assertion failure) means the tests are wrong, not "Flask
wasn't installed". Concretely:

* Detect project type (currently ``python`` is the only kind handled --
  unknown stacks short-circuit with ``outcome=skipped``).
* Ensure ``.venv/bin/python`` exists and ``requirements.txt`` is
  installed into it (via :func:`cgx.codegen.test_runner.ensure_project_venv`).
* Scan the just-applied files for top-level imports and pip-install any
  packages missing from ``requirements.txt`` (via
  :func:`cgx.codegen.env_manager.preflight_install`); successful adds
  are appended to ``requirements.txt`` so the manifest stays in sync.

Emits a :data:`ArtifactKind.BUILD_REPORT` artifact carrying the venv
path, the manifests installed from, the dynamically-installed packages,
any failures, and a tail of pip's stderr for the UI to surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.BOOTSTRAP_ENV)
def run_bootstrap_env(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Provision the project's runtime environment for VERIFY."""
    if not deps.project_root:
        return ExecutorResult(
            failure="BOOTSTRAP_ENV requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="BOOTSTRAP_ENV requires a session store in deps")

    root = Path(deps.project_root).resolve()
    if not root.is_dir():
        return ExecutorResult(
            failure=f"BOOTSTRAP_ENV: project_root not a directory: {root}")

    project_type = _detect_project_type(root)
    applied_files = _resolve_applied_files(task, deps)
    timeout = float(task.inputs.get("timeout_seconds") or 300.0)

    if project_type != "python":
        artifact = _build_artifact(
            task, project_type=project_type, venv_path=None,
            python_exe=None, installed_from=[], installed_packages=[],
            failed_installs={}, outcome="skipped",
            pip_log_tail="", applied_files=applied_files,
            style_issues=[],
            note="non-python project: no venv provisioned",
        )
        return ExecutorResult(
            outputs={"build_artifact_id": artifact.artifact_id,
                     "outcome": "skipped",
                     "project_type": project_type,
                     "style_issue_count": 0},
            artifact=artifact)

    # Lazy imports: pull subprocess + env_manager only when needed.
    from cgx.codegen.test_runner import ensure_project_venv
    from cgx.codegen.env_manager import preflight_install, update_requirements

    try:
        python_exe = ensure_project_venv(str(root), timeout=timeout)
    except Exception as exc:
        logger.exception("BOOTSTRAP_ENV: ensure_project_venv crashed")
        return ExecutorResult(
            failure=f"venv provisioning failed: {type(exc).__name__}: {exc}")

    venv_path = _detect_venv_path(root, python_exe)
    installed_from = _installed_from_manifests(root)

    # Preflight: install undeclared imports into the project's venv. The
    # files arg accepts absolute paths; map applied_files (which are
    # repo-relative) to absolute paths under root.
    abs_files = [str(root / p) for p in applied_files if p]
    installed_packages: List[str] = []
    failed_installs: Dict[str, bool] = {}
    pip_log_tail = ""
    try:
        missing, results = preflight_install(abs_files, str(root),
                                             python=python_exe)
        for pkg, ok in (results or {}).items():
            if ok:
                installed_packages.append(pkg)
            else:
                failed_installs[pkg] = False
        if installed_packages:
            update_requirements(str(root), installed_packages)
    except Exception as exc:
        logger.warning(
            "BOOTSTRAP_ENV: preflight_install raised %s", exc)
        pip_log_tail = f"{type(exc).__name__}: {exc}"

    style_issues = _preflight_test_style_lint(root, applied_files)

    outcome = _classify_outcome(
        venv_path=venv_path, failed_installs=failed_installs)
    artifact = _build_artifact(
        task, project_type=project_type, venv_path=venv_path,
        python_exe=python_exe, installed_from=installed_from,
        installed_packages=installed_packages,
        failed_installs=failed_installs, outcome=outcome,
        pip_log_tail=pip_log_tail, applied_files=applied_files,
        style_issues=style_issues, note=None,
    )
    failure: Optional[str] = None
    if outcome == "failed":
        failure = (f"bootstrap failed: {len(failed_installs)} package(s) "
                   f"could not be installed: {sorted(failed_installs)}")
    return ExecutorResult(
        outputs={
            "build_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "project_type": project_type,
            "venv_path": venv_path,
            "python_exe": python_exe,
            "installed_count": len(installed_packages),
            "failed_count": len(failed_installs),
            "style_issue_count": len(style_issues),
        },
        artifact=artifact,
        failure=failure,
    )


# --------------------- helpers ---------------------

def _detect_project_type(root: Path) -> str:
    """Return ``python`` when a Python manifest is present, else ``unknown``.

    Looks for ``requirements.txt`` / ``pyproject.toml`` / ``setup.py`` /
    ``setup.cfg`` at the project root. Greenfield scaffolds always emit
    one of these for Python projects via ``_inject_required_manifest_files``,
    so the check is reliable for our supported templates.
    """
    for name in ("requirements.txt", "pyproject.toml",
                 "setup.py", "setup.cfg"):
        if (root / name).is_file():
            return "python"
    return "unknown"


def _resolve_applied_files(task: TaskNode,
                           deps: ExecutorDeps) -> List[str]:
    """Return the list of files applied upstream, for preflight scanning.

    Prefers ``apply_artifact_id`` (the immediate upstream from APPLY in
    the greenfield loop). Falls back to a ``scaffold_artifact_id`` if
    present, then to an explicit ``applied_files`` input.
    """
    explicit = task.inputs.get("applied_files")
    if isinstance(explicit, list) and explicit:
        return [str(p) for p in explicit if str(p).strip()]
    for key in ("apply_artifact_id", "scaffold_artifact_id"):
        art_id = str(task.inputs.get(key) or "").strip()
        if not art_id:
            continue
        art = deps.store.get_artifact(art_id)
        if art is None:
            continue
        applied = (art.content or {}).get("applied_files")
        if isinstance(applied, list) and applied:
            return [str(p) for p in applied if str(p).strip()]
        # SCAFFOLD_PATCHES carries a ``generated`` list instead.
        generated = (art.content or {}).get("generated")
        if isinstance(generated, list) and generated:
            return [str(e.get("file") or "") for e in generated
                    if isinstance(e, dict) and e.get("file")]
    return []


def _installed_from_manifests(root: Path) -> List[str]:
    """Return the manifest filenames the venv was provisioned from."""
    out: List[str] = []
    for name in ("requirements.txt", "requirements-dev.txt",
                 "requirements-test.txt", "pyproject.toml"):
        if (root / name).is_file():
            out.append(name)
    return out


def _detect_venv_path(root: Path, python_exe: str) -> Optional[str]:
    """Return the ``.venv`` (or ``venv``) directory matching ``python_exe``."""
    for name in (".venv", "venv"):
        candidate = root / name / "bin" / "python"
        if str(candidate) == python_exe and candidate.is_file():
            return str(root / name)
    return None


def _classify_outcome(*, venv_path: Optional[str],
                      failed_installs: Dict[str, bool]) -> str:
    """Reduce the bootstrap result to a single token for the UI.

    ``no_venv`` means ``ensure_project_venv`` fell back to the host
    interpreter (offline / missing venv module). ``failed`` means at
    least one preflight install failed. ``succeeded`` is the happy
    path; ``partial`` is reserved for future per-stack outcomes.
    """
    if venv_path is None:
        return "no_venv"
    if failed_installs:
        return "failed"
    return "succeeded"


def _build_artifact(
    task: TaskNode, *,
    project_type: str,
    venv_path: Optional[str],
    python_exe: Optional[str],
    installed_from: List[str],
    installed_packages: List[str],
    failed_installs: Dict[str, bool],
    outcome: str,
    pip_log_tail: str,
    applied_files: List[str],
    style_issues: List[Dict[str, Any]],
    note: Optional[str],
) -> Artifact:
    """Construct the ``BUILD_REPORT`` artifact for this bootstrap run."""
    content: Dict[str, Any] = {
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "project_type": project_type,
        "venv_path": venv_path,
        "python_exe": python_exe,
        "installed_from": list(installed_from),
        "installed_packages": list(installed_packages),
        "failed_installs": sorted(failed_installs.keys()),
        "outcome": outcome,
        "pip_log_tail": pip_log_tail or "",
        "applied_files": list(applied_files),
        "style_issues": list(style_issues),
    }
    if note:
        content["note"] = note
    return Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.BUILD_REPORT,
        content=content,
    )


def _preflight_test_style_lint(
    root: Path, applied_files: List[str],
) -> List[Dict[str, Any]]:
    """Scan applied test files for unittest/pytest-mix style issues.

    Only inspects files that look like test modules (``tests/`` prefix
    or ``test_*.py`` basename) to keep the AST walk proportional to
    the change set. Wraps :func:`cgx.session.repair.locate.lint_test_style`
    so the linter and the REPAIR locator stay in lock-step on what
    "violation" means.
    """
    if not applied_files:
        return []
    test_files = [
        p for p in applied_files
        if str(p).endswith(".py") and (
            str(p).startswith("tests/")
            or str(p).startswith("test/")
            or Path(p).name.startswith("test_"))]
    if not test_files:
        return []
    try:
        from cgx.session.repair.locate import lint_test_style
        return lint_test_style(root, test_files)
    except Exception as exc:
        logger.warning(
            "BOOTSTRAP_ENV: test-style lint raised %s", exc)
        return []


# --------------------- typing helpers ---------------------

# Exported for the tests so they can assert against the same tuple.
BOOTSTRAP_OUTCOMES: Tuple[str, ...] = (
    "succeeded", "failed", "no_venv", "skipped", "partial",
)

