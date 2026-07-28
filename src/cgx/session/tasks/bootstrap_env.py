

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
* Snapshot the resolved venv contents via ``pip freeze --all`` so
  downstream REPAIR can reason about *resolved* dependency versions
  (e.g. detect a Flask 2.1 + Werkzeug 3 mismatch) rather than guess
  from an ImportError traceback alone.

Emits a :data:`ArtifactKind.BUILD_REPORT` artifact carrying the venv
path, the manifests installed from, the dynamically-installed packages,
the resolved-package snapshot, any failures, and a tail of pip's
stderr for the UI to surface.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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

    if project_type == "node":
        return _bootstrap_node(task, root, applied_files, timeout)

    if project_type != "python":
        artifact = _build_artifact(
            task, project_type=project_type, venv_path=None,
            python_exe=None, installed_from=[], installed_packages=[],
            failed_installs={}, outcome="skipped",
            pip_log_tail="", applied_files=applied_files,
            style_issues=[],
            resolved_packages=[], pip_freeze_text="",
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

    # Snapshot the fully-resolved venv contents so REPAIR can diagnose
    # transitive-dep failures without re-running pip. Best-effort: a
    # freeze failure must not fail BOOTSTRAP, so we record empty fields
    # and move on. Only runs when we actually have a project venv --
    # ``no_venv`` falls back to host interpreter and freezing that
    # would leak unrelated packages into the report.
    resolved_packages: List[Dict[str, str]] = []
    pip_freeze_text = ""
    if venv_path is not None and python_exe:
        resolved_packages, pip_freeze_text = _capture_pip_freeze(python_exe)

    outcome = _classify_outcome(
        venv_path=venv_path, failed_installs=failed_installs)
    # Honesty gate: even when nothing failed to *install*, the scaffold's
    # own third-party imports must actually resolve in the venv. If they
    # don't (a malformed/unresolvable requirements line that aborted the
    # batch install, a bad import→PyPI mapping, ...), the app cannot run
    # and BOOTSTRAP must not report success.
    missing_runtime: List[str] = []
    if outcome == "succeeded":
        missing_runtime = _verify_runtime_imports(
            root, applied_files, python_exe)
        if missing_runtime:
            outcome = "failed"
    artifact = _build_artifact(
        task, project_type=project_type, venv_path=venv_path,
        python_exe=python_exe, installed_from=installed_from,
        installed_packages=installed_packages,
        failed_installs=failed_installs, outcome=outcome,
        pip_log_tail=pip_log_tail, applied_files=applied_files,
        style_issues=style_issues,
        resolved_packages=resolved_packages,
        pip_freeze_text=pip_freeze_text,
        missing_imports=missing_runtime,
        note=None,
    )
    failure: Optional[str] = None
    if outcome == "failed":
        reasons: List[str] = []
        if failed_installs:
            reasons.append(
                f"{len(failed_installs)} package(s) could not be "
                f"installed: {sorted(failed_installs)}")
        if missing_runtime:
            reasons.append(
                f"{len(missing_runtime)} runtime import(s) not importable "
                f"in venv: {missing_runtime}")
        failure = "bootstrap failed: " + "; ".join(reasons)
    return ExecutorResult(
        outputs={
            "build_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "project_type": project_type,
            "venv_path": venv_path,
            "python_exe": python_exe,
            "installed_count": len(installed_packages),
            "failed_count": len(failed_installs),
            "missing_import_count": len(missing_runtime),
            "style_issue_count": len(style_issues),
        },
        artifact=artifact,
        failure=failure,
    )


# --------------------- helpers ---------------------

def _detect_project_type(root: Path) -> str:
    """Return ``python`` / ``node`` for a known stack, else ``unknown``.

    Python markers (``requirements.txt`` / ``pyproject.toml`` /
    ``setup.py`` / ``setup.cfg``) take priority over ``package.json``:
    a polyglot repo still gets its richer venv provisioning + preflight
    path, and VERIFY's ``NpmRunner`` best-effort installs ``node_modules``
    on its own. A ``package.json``-only project resolves to ``node`` so
    BOOTSTRAP_ENV can provision ``node_modules`` before VERIFY runs the
    build/test smoke -- otherwise the JS signal would be a false success.
    """
    for name in ("requirements.txt", "pyproject.toml",
                 "setup.py", "setup.cfg"):
        if (root / name).is_file():
            return "python"
    if (root / "package.json").is_file():
        return "node"
    return "unknown"


def _bootstrap_node(task: TaskNode, root: Path,
                    applied_files: List[str],
                    timeout: float) -> ExecutorResult:
    """Provision ``node_modules`` for a JS/TS project (best-effort).

    Runs a bounded ``npm install`` so VERIFY's ``NpmRunner`` build/test
    smoke has its dependencies. Provisioning is deliberately non-fatal:
    an offline box (no ``npm``, or an install that cannot reach the
    registry) degrades to ``skipped`` rather than failing the session --
    the real build/test signal is produced downstream by VERIFY, and
    hard-failing here would deny the loop that signal entirely. Emits a
    ``BUILD_REPORT`` with ``project_type=node`` and no ``python_exe`` so
    the Python-only API_CHECK / SMOKE gates skip cleanly.
    """
    node_modules = root / "node_modules"
    pip_log_tail = ""
    note: Optional[str]
    if shutil.which("npm") is None:
        outcome = "skipped"
        note = "npm not installed"
    elif node_modules.is_dir():
        outcome = "succeeded"
        note = "node_modules already present"
    else:
        rc, pip_log_tail = _run_npm_install(root, timeout)
        if node_modules.is_dir():
            outcome = "succeeded"
            note = None
        else:
            outcome = "skipped"
            note = f"npm install did not provision node_modules (rc={rc})"
    artifact = _build_artifact(
        task, project_type="node", venv_path=None, python_exe=None,
        installed_from=[], installed_packages=[], failed_installs={},
        outcome=outcome, pip_log_tail=pip_log_tail,
        applied_files=applied_files, style_issues=[],
        resolved_packages=[], pip_freeze_text="", note=note,
    )
    return ExecutorResult(
        outputs={
            "build_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "project_type": "node",
            "venv_path": None,
            "python_exe": None,
            "installed_count": 0,
            "failed_count": 0,
            "missing_import_count": 0,
            "style_issue_count": 0,
        },
        artifact=artifact,
    )


def _run_npm_install(root: Path, timeout: float) -> Tuple[int, str]:
    """Run a bounded ``npm install``; return ``(returncode, stderr_tail)``.

    Best-effort by design -- any timeout or spawn error collapses to a
    non-zero code and a short diagnostic tail so the caller degrades to
    ``skipped`` instead of raising.
    """
    try:
        proc = subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund"],
            cwd=str(root), capture_output=True, text=True,
            timeout=min(timeout, 300.0),
        )
    except subprocess.TimeoutExpired as exc:
        return 124, ((exc.stderr or "") + "\n[timeout]")[-800:]
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stderr or proc.stdout or "")[-800:]


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


def _verify_runtime_imports(
        root: Path,
        applied_files: List[str],
        python_exe: Optional[str]) -> List[str]:
    """Return third-party import roots the scaffold uses but can't import.

    Scans the applied ``.py`` files for top-level import roots, drops
    stdlib / first-party / bare-namespace roots, then probes the
    remaining roots in the provisioned venv. A non-empty result means
    the app cannot import its own dependencies even though provisioning
    did not raise -- the caller degrades the outcome to ``failed``.
    Best-effort: any import/probe error yields ``[]`` so a transient
    hiccup never fabricates a bootstrap failure.
    """
    if not python_exe or not applied_files:
        return []
    try:
        from cgx.codegen.env_manager import (
            scan_imports, _probe_importable, _is_local_package,
            _STDLIB_TOP, _NAMESPACE_ROOTS,
        )
    except Exception:
        return []
    abs_files = [str(root / p) for p in applied_files
                 if str(p).endswith(".py")]
    if not abs_files:
        return []
    try:
        roots = {r.split(".")[0] for r in scan_imports(abs_files)}
        candidates = sorted(
            r for r in roots
            if r
            and r.lower().replace("-", "_") not in _STDLIB_TOP
            and r not in _NAMESPACE_ROOTS
            and not _is_local_package(r, str(root)))
        if not candidates:
            return []
        importable = _probe_importable(candidates, python_exe)
    except Exception as exc:
        logger.warning(
            "BOOTSTRAP_ENV: runtime-import verification raised %s", exc)
        return []
    return [r for r in candidates if r not in importable]


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
    resolved_packages: List[Dict[str, str]],
    pip_freeze_text: str,
    note: Optional[str],
    missing_imports: Optional[List[str]] = None,
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
        "missing_imports": list(missing_imports or []),
        "outcome": outcome,
        "pip_log_tail": pip_log_tail or "",
        "applied_files": list(applied_files),
        "style_issues": list(style_issues),
        "resolved_packages": list(resolved_packages),
        "pip_freeze_text": pip_freeze_text or "",
    }
    if note:
        content["note"] = note
    return Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.BUILD_REPORT,
        content=content,
    )


# --------------------- pip freeze capture ---------------------

def _capture_pip_freeze(
    python_exe: str,
    *,
    timeout: float = 30.0,
) -> Tuple[List[Dict[str, str]], str]:
    """Run ``pip freeze --all`` and return ``(parsed, raw_text)``.

    Best-effort by design: any subprocess failure, non-zero return
    code, decode error, or parse error collapses to ``([], "")`` so a
    busted venv never fails the BOOTSTRAP step. The ``--all`` flag
    includes pip / setuptools / wheel which a downstream classifier
    needs to spot e.g. setuptools-version-driven import breaks.

    Parsed entries are ``{"name": <canonical>, "version": <str>}`` --
    name lowercased and PEP 503-normalised so REPAIR can match by
    distribution key without worrying about case or ``_`` vs ``-``.
    Editable / VCS / URL / file installs (lines without ``==``) are
    skipped from the parsed list but kept verbatim in ``raw_text``.
    """
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pip", "freeze", "--all"],
            capture_output=True, timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "BOOTSTRAP_ENV: pip freeze raised %s: %s",
            type(exc).__name__, exc)
        return [], ""
    if proc.returncode != 0:
        logger.warning(
            "BOOTSTRAP_ENV: pip freeze exited rc=%d", proc.returncode)
        return [], ""
    try:
        raw_text = (proc.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return [], ""
    return _parse_pip_freeze(raw_text), raw_text


def _parse_pip_freeze(text: str) -> List[Dict[str, str]]:
    """Parse ``pip freeze`` output into ``[{name, version}, ...]``.

    Only handles canonical ``name==version`` lines; ``-e ...``,
    ``pkg @ url``, and comment lines are skipped. Names are
    normalised per PEP 503 (lowercased, ``_`` and ``.`` collapsed to
    ``-``) so lookups match what PyPI exposes at
    ``/pypi/{name}/{version}/json``.
    """
    out: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.strip()
        version = version.strip().split(" ")[0]
        if not name or not version:
            continue
        canonical = _canonicalise_name(name)
        out.append({"name": canonical, "version": version})
    return out


def _canonicalise_name(name: str) -> str:
    """PEP 503-style name normalisation: lowercase + ``_.`` -> ``-``."""
    lowered = name.lower()
    buf = []
    for ch in lowered:
        if ch in ("_", "."):
            buf.append("-")
        else:
            buf.append(ch)
    # Collapse runs of ``-`` so e.g. ``Foo._bar`` -> ``foo-bar``.
    out = []
    prev_dash = False
    for ch in buf:
        if ch == "-":
            if prev_dash:
                continue
            prev_dash = True
        else:
            prev_dash = False
        out.append(ch)
    return "".join(out).strip("-")


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

