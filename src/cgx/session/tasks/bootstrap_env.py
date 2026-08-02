

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
* Install any ``missing_modules`` threaded through ``task.inputs`` by an
  ``install_deps`` repair verdict (import roots the API_CHECK probe found
  absent from the venv), syncing ``requirements.txt`` likewise.
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
import re
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

    # Deterministic guard: a scaffold that pins an old framework without
    # capping its transitive deps (e.g. ``flask==2.2.2`` while Werkzeug
    # floats to 3.x, which deleted ``url_quote``) installs a combination
    # that cannot import. Cap the known-incompatible transitives in
    # requirements.txt *before* the venv installs so the fix is durable
    # (survives re-bootstrap) and model-independent.
    _pin_transitive_constraints(root)

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

    # Repair-driven installs: an ``install_deps`` verdict threads the
    # API_CHECK report's ``missing_modules`` (import roots that failed to
    # resolve in the venv) through task.inputs. The preflight above only
    # covers imports scanned from applied files, so consume the explicit
    # list too -- otherwise the install_deps strategy is a no-op whenever
    # the file scan disagrees with the probe.
    requested = [str(m).strip()
                 for m in (task.inputs.get("missing_modules") or [])
                 if str(m).strip()]
    if requested:
        from cgx.codegen.env_manager import (
            find_missing_python_packages, install_packages)
        try:
            to_install = find_missing_python_packages(
                set(requested), str(root), python=python_exe)
            results = install_packages(to_install, python=python_exe) \
                if to_install else {}
            newly: List[str] = []
            for pkg, ok in (results or {}).items():
                if ok:
                    newly.append(pkg)
                    if pkg not in installed_packages:
                        installed_packages.append(pkg)
                else:
                    failed_installs[pkg] = False
            if newly:
                update_requirements(str(root), newly)
        except Exception as exc:
            logger.warning(
                "BOOTSTRAP_ENV: missing_modules install raised %s", exc)
            if not pip_log_tail:
                pip_log_tail = f"{type(exc).__name__}: {exc}"

    # Transitive test-client extra: fastapi/starlette's TestClient needs
    # httpx at import time, but no first-party file imports httpx
    # directly, so neither the file-scan preflight above nor
    # requirements.txt covers it and VERIFY would die at collection.
    # Detect TestClient usage in the applied files and install the extra
    # up front.
    extras = _testclient_extra_roots(root, applied_files)
    if extras:
        from cgx.codegen.env_manager import (
            find_missing_python_packages, install_packages)
        try:
            to_install = find_missing_python_packages(
                set(extras), str(root), python=python_exe)
            results = install_packages(to_install, python=python_exe) \
                if to_install else {}
            newly = []
            for pkg, ok in (results or {}).items():
                if ok:
                    newly.append(pkg)
                    if pkg not in installed_packages:
                        installed_packages.append(pkg)
                else:
                    failed_installs[pkg] = False
            if newly:
                update_requirements(str(root), newly)
        except Exception as exc:
            logger.warning(
                "BOOTSTRAP_ENV: test-client extra install raised %s", exc)
            if not pip_log_tail:
                pip_log_tail = f"{type(exc).__name__}: {exc}"

    # Repair-driven re-resolve: a ``resolve_deps`` verdict threads the
    # API_CHECK report's ``conflict_packages`` (the consumer whose stale
    # exact pin dragged in an incompatible peer, plus that peer) through
    # task.inputs. The package is installed but its own import chain is
    # broken, so no install/regenerate can help; force-upgrade the
    # implicated distributions to a self-consistent set and re-pin
    # requirements.txt reproducibly, then let the runtime-import gate and
    # API_CHECK re-probe confirm the fix.
    resolve_requested = [str(p).strip()
                         for p in (task.inputs.get("resolve_packages") or [])
                         if str(p).strip()]
    if resolve_requested:
        from cgx.codegen.env_manager import resolve_dependency_conflict
        try:
            resolve_dependency_conflict(
                str(root), resolve_requested, python=python_exe)
        except Exception as exc:
            logger.warning(
                "BOOTSTRAP_ENV: dependency conflict re-resolve raised %s",
                exc)
            if not pip_log_tail:
                pip_log_tail = f"{type(exc).__name__}: {exc}"

    # Defense-in-depth: only a *declared* dependency that fails to
    # install is a fatal environment problem. A scan-discovered import
    # that pip cannot satisfy (typically a hallucinated first-party-
    # looking name like ``core``, or an unresolvable guess) is a code
    # problem: record it as ``uninstallable`` and proceed so API_CHECK
    # probes honestly and routes the failure to a regenerate, instead of
    # ending the session in a terminal bootstrap failure.
    from cgx.codegen.env_manager import _read_requirements
    declared_names = _read_requirements(str(root))
    fatal_installs = {
        pkg: ok for pkg, ok in failed_installs.items()
        if pkg.lower().replace("-", "_") in declared_names}
    uninstallable = sorted(p for p in failed_installs
                           if p not in fatal_installs)
    if uninstallable:
        logger.warning(
            "BOOTSTRAP_ENV: %d undeclared import(s) could not be "
            "installed (non-fatal, deferred to API_CHECK): %s",
            len(uninstallable), uninstallable)

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
        venv_path=venv_path, failed_installs=fatal_installs)
    # Honesty gate: even when nothing failed to *install*, the scaffold's
    # own third-party imports must actually resolve in the venv. If they
    # don't (a malformed/unresolvable requirements line that aborted the
    # batch install, a bad import→PyPI mapping, ...), the app cannot run
    # and BOOTSTRAP must not report success. Roots already recorded as
    # ``uninstallable`` are excluded -- proceeding past those is the
    # deliberate decision above, and API_CHECK owns their diagnosis.
    missing_runtime: List[str] = []
    if outcome == "succeeded":
        missing_runtime = _verify_runtime_imports(
            root, applied_files, python_exe,
            skip_roots=_import_roots_for(uninstallable))
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
        uninstallable=uninstallable,
        note=None,
    )
    failure: Optional[str] = None
    if outcome == "failed":
        reasons: List[str] = []
        if fatal_installs:
            reasons.append(
                f"{len(fatal_installs)} declared package(s) could not be "
                f"installed: {sorted(fatal_installs)}")
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
            "failed_count": len(fatal_installs),
            "uninstallable_count": len(uninstallable),
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


# Known transitive incompatibilities a scaffold routinely pins without a
# matching cap. Each entry maps a *declared* package + version predicate
# to the transitive constraint that must be present for the declared pin
# to import. Keeping this a tiny, explicit table (rather than a resolver)
# keeps the guard deterministic and auditable: we only touch a line when
# we are certain the pinned combination is broken.
#
# Flask 2.2/2.1 import ``url_quote`` from ``werkzeug.urls``, which
# Werkzeug 3.0 removed; an uncapped Werkzeug therefore floats to 3.x and
# every ``import flask`` raises ``ImportError: cannot import name
# 'url_quote'``. Capping Werkzeug < 2.3 restores a compatible pair.
_TRANSITIVE_CAPS: Tuple[Tuple[str, str, str, str], ...] = (
    ("flask", r"^2\.[12](\.|$)", "werkzeug", "werkzeug<2.3"),
)


def _pin_transitive_constraints(root: Path) -> List[str]:
    """Cap known-incompatible transitive deps in ``requirements.txt``.

    Scans the project's ``requirements.txt`` for any declared pin listed
    in :data:`_TRANSITIVE_CAPS` and, when the pinned version matches the
    predicate and the required transitive is not already constrained,
    appends the cap. Idempotent (a cap already present is left alone) and
    best-effort (a missing/unreadable file is a no-op). Returns the list
    of constraint lines added, for logging/observability.
    """
    req_path = root / "requirements.txt"
    if not req_path.is_file():
        return []
    try:
        text = req_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # pragma: no cover - defensive
        return []
    lines = text.splitlines()
    declared: Dict[str, str] = {}
    for raw in lines:
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[>=<!~;\[ ]", line, maxsplit=1)[0].strip()
        if name:
            declared[name.lower().replace("-", "_")] = line

    added: List[str] = []
    for pkg, version_re, transitive, cap in _TRANSITIVE_CAPS:
        pkg_key = pkg.lower().replace("-", "_")
        trans_key = transitive.lower().replace("-", "_")
        spec = declared.get(pkg_key)
        if not spec or trans_key in declared:
            continue
        version = spec[len(pkg):].lstrip(" =<>!~").strip()
        if not re.match(version_re, version):
            continue
        added.append(cap)

    if not added:
        return []
    tail = "\n" if text and not text.endswith("\n") else ""
    try:
        req_path.write_text(text + tail + "\n".join(added) + "\n",
                            encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        return []
    logger.info("BOOTSTRAP_ENV: pinned %d transitive constraint(s): %s",
                len(added), added)
    return added


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


def _import_roots_for(pypi_names: List[str]) -> frozenset:
    """Map PyPI distribution names back to plausible import roots.

    The normalised distribution name itself is usually the import root
    (``core`` -> ``core``, ``flask-cors`` -> ``flask_cors``); known
    divergent pairs are recovered by reversing
    :data:`cgx.codegen.env_manager._IMPORT_TO_PYPI` (``PyYAML`` ->
    ``yaml``). Used to exclude already-recorded uninstallable packages
    from the runtime-import honesty gate.
    """
    if not pypi_names:
        return frozenset()
    try:
        from cgx.codegen.env_manager import _IMPORT_TO_PYPI
    except Exception:  # pragma: no cover - defensive
        _IMPORT_TO_PYPI = {}
    roots = set()
    for pkg in pypi_names:
        norm = str(pkg).lower().replace("-", "_")
        roots.add(norm)
        for imp, dist in _IMPORT_TO_PYPI.items():
            if dist.lower().replace("-", "_") == norm:
                roots.add(imp.lower().replace("-", "_"))
    return frozenset(roots)


# Substrings that mark a file as using the fastapi/starlette test
# client, whose import pulls in the optional ``httpx`` extra at runtime.
_TESTCLIENT_MARKERS = ("fastapi.testclient", "starlette.testclient")


def _testclient_extra_roots(root: Path,
                            applied_files: List[str]) -> List[str]:
    """Return the import roots the TestClient transitively requires.

    Scans the applied ``.py`` files for a fastapi/starlette
    ``testclient`` reference and returns ``["httpx"]`` when found --
    the optional extra ``starlette.testclient`` imports at module load
    but which no first-party file imports directly. Best-effort: read
    errors skip the file so a probe hiccup never fails bootstrap.
    """
    for rel in applied_files:
        if not str(rel).endswith(".py"):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if any(marker in text for marker in _TESTCLIENT_MARKERS):
            return ["httpx"]
    return []


def _verify_runtime_imports(
        root: Path,
        applied_files: List[str],
        python_exe: Optional[str],
        skip_roots: frozenset = frozenset()) -> List[str]:
    """Return third-party import roots the scaffold uses but can't import.

    Scans the applied ``.py`` files for top-level import roots, drops
    stdlib / first-party / bare-namespace roots (and any root in
    ``skip_roots`` -- packages already recorded as uninstallable, whose
    diagnosis belongs to API_CHECK), then probes the remaining roots in
    the provisioned venv. A non-empty result means the app cannot import
    its own dependencies even though provisioning did not raise -- the
    caller degrades the outcome to ``failed``.
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
            and r.lower().replace("-", "_") not in skip_roots
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
    least one *declared* preflight install failed (the caller passes the
    declared-only partition; undeclared scan-install failures are
    recorded as ``uninstallable`` and deferred to API_CHECK).
    ``succeeded`` is the happy path; ``partial`` is reserved for future
    per-stack outcomes.
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
    uninstallable: Optional[List[str]] = None,
) -> Artifact:
    """Construct the ``BUILD_REPORT`` artifact for this bootstrap run.

    ``uninstallable`` lists scan-discovered (undeclared) packages whose
    install failed -- recorded non-fatally so API_CHECK can classify the
    corresponding import failures as hallucinated modules rather than
    missing dependencies.
    """
    content: Dict[str, Any] = {
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "project_type": project_type,
        "venv_path": venv_path,
        "python_exe": python_exe,
        "installed_from": list(installed_from),
        "installed_packages": list(installed_packages),
        "failed_installs": sorted(failed_installs.keys()),
        "uninstallable": list(uninstallable or []),
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

