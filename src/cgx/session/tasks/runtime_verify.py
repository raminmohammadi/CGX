

"""RUNTIME_VERIFY executor: boot the scaffolded app after a green VERIFY.

Sits after a *passing* VERIFY in greenfield mode. A unit suite the model
wrote can pass while the app never actually boots (an import-time
NameError, a bad ``create_app`` wiring, a config read that throws at
module load). RUNTIME_VERIFY closes that gap: for each detected entry
module (``app.py`` / ``main.py`` / a file that constructs a Flask /
FastAPI app or defines ``create_app``), it runs an import-and-call smoke
under the project's bootstrapped venv -- importing the module and, when
present, invoking the ``create_app`` factory -- so "the tests pass"
becomes "the app actually runs".

Emits :data:`ArtifactKind.RUNTIME_REPORT` whose ``probes`` list pairs
each entry file with ``ok`` / ``kind`` / ``stderr_tail``. The artifact's
``outcome`` token (``passed`` / ``failed`` / ``timeout`` / ``error`` /
``skipped``) drives the terminal router branch in
:func:`cgx.session.router._runtime_verify_terminal_session_actions`.
Like SMOKE it never returns ``ExecutorResult.failure`` for a boot
failure -- the structured report is always persisted so a later
runtime-repair pass (bottleneck #3) has something to classify.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

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

_DEFAULT_BOOT_TIMEOUT = 20.0
_STDERR_TAIL_CHARS = 1200

# Basenames that conventionally hold an application entry point. A file
# matching one of these is probed even when it does not statically look
# like it constructs an app object.
_ENTRY_BASENAMES = frozenset({
    "app.py", "main.py", "wsgi.py", "asgi.py",
    "manage.py", "server.py", "run.py", "__main__.py",
})

# Substrings that mark a module as constructing a web app worth booting.
_APP_MARKERS = ("Flask(", "FastAPI(", "create_app")

# Directories the whole-tree entry scan never descends into -- vendored
# deps, build output, VCS/tooling caches. Keeps the scan bounded and stops
# a dependency's own ``app.py`` from being mistaken for the project entry.
_TREE_SCAN_SKIP_DIRS = frozenset({
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__",
    "build", "dist", ".cgx", ".cgx-backups", ".next", ".nuxt", "out",
    "coverage", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
})
# Cap on entry files probed, so a pathological tree cannot make the boot
# gate run unboundedly many subprocesses.
_MAX_ENTRY_CANDIDATES = 20

# The subprocess probe: put the project root, its ``src`` dir, and the
# entry file's own directory on ``sys.path`` (so both ``src`` layouts and
# flat sibling imports resolve), load the file by location, and -- when a
# ``create_app`` factory is present -- call it. Any exception at import or
# factory time is a boot failure (exit 1); a clean load is exit 0.
_PROBE_SCRIPT = r"""
import importlib.util, sys, traceback
root, file_abs, file_dir = sys.argv[1], sys.argv[2], sys.argv[3]
for p in (root, root + "/src", file_dir):
    if p and p not in sys.path:
        sys.path.insert(0, p)
try:
    spec = importlib.util.spec_from_file_location("_cgx_runtime_probe", file_abs)
    if spec is None or spec.loader is None:
        sys.stderr.write("cannot load module spec\n")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_app", None)
    if callable(factory):
        factory()
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
"""


@register_executor(TaskKind.RUNTIME_VERIFY)
def run_runtime_verify(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Boot each detected entry module under the bootstrapped venv."""
    if not deps.project_root:
        return ExecutorResult(
            failure="RUNTIME_VERIFY requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="RUNTIME_VERIFY requires a session store in deps")

    root = Path(deps.project_root).resolve()
    applied_files = _resolve_applied_files(task, deps)
    python_exe = _resolve_python_exe(task, deps)
    timeout = float(task.inputs.get("boot_timeout") or _DEFAULT_BOOT_TIMEOUT)

    candidates = _entry_candidates(root, applied_files)
    probes: List[Dict[str, Any]] = []
    if not python_exe or not candidates:
        outcome = "skipped"
    else:
        for rel in candidates:
            abs_path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
            res = _probe_boot(python_exe, str(root), abs_path, timeout)
            probes.append({"file": rel, **res})
        outcome = _combine_outcome(probes)

    failed = [p["file"] for p in probes if not p["ok"]]
    signature = (_signature(failed)
                 if outcome in ("failed", "timeout", "error") else "")
    content: Dict[str, Any] = {
        "verify_artifact_id": task.inputs.get("verify_artifact_id"),
        "build_artifact_id": task.inputs.get("build_artifact_id"),
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "python_exe": python_exe,
        "entry_files": list(candidates),
        "probes": probes,
        "outcome": outcome,
        "failed_entries": failed,
        "failure_signature": signature,
    }
    # Pre-compute the classification token here (as VERIFY does) so the
    # pure router can gate a hard boot failure to the DIAGNOSE reasoning
    # rung vs a mechanical REPAIR by membership test alone (design 12.4)
    # -- it never runs the classifier itself.
    from cgx.session.repair.classify import classify_runtime_report
    classification = classify_runtime_report(content)
    content["classification"] = classification
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.RUNTIME_REPORT,
        content=content,
    )
    return ExecutorResult(
        outputs={
            "runtime_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "tested_count": len(probes),
            "failed_count": len(failed),
            "failure_signature": signature,
            "classification": classification,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _resolve_applied_files(task: TaskNode, deps: ExecutorDeps) -> List[str]:
    """Return the files to consider as boot candidates.

    Prefers an explicit ``applied_files`` input; otherwise reads the
    upstream APPLIED_CHANGES (or SCAFFOLD_PATCHES) artifact, mirroring
    :func:`cgx.session.tasks.smoke._resolve_applied_files`.
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
        content = art.content or {}
        applied = content.get("applied_files")
        if isinstance(applied, list) and applied:
            return [str(p) for p in applied if str(p).strip()]
        generated = content.get("generated")
        if isinstance(generated, list) and generated:
            paths: List[str] = []
            for entry in generated:
                if isinstance(entry, dict):
                    p = str(entry.get("file") or "").strip()
                    if p:
                        paths.append(p)
            if paths:
                return paths
    return []


def _resolve_python_exe(task: TaskNode,
                        deps: ExecutorDeps) -> Optional[str]:
    """Return the bootstrapped venv python from the upstream BUILD_REPORT."""
    explicit = str(task.inputs.get("python_exe") or "").strip()
    if explicit:
        return explicit
    build_id = str(task.inputs.get("build_artifact_id") or "").strip()
    if not build_id:
        return None
    art = deps.store.get_artifact(build_id)
    if art is None or art.kind is not ArtifactKind.BUILD_REPORT:
        return None
    py = (art.content or {}).get("python_exe")
    return str(py) if isinstance(py, str) and py else None


def _entry_candidates(root: Path, applied_files: List[str]) -> List[str]:
    """Return ``.py`` files across the tree that look like an app entry.

    A file qualifies when its basename is a conventional entry point
    (``app.py`` / ``main.py`` / ...) or its source statically references
    a web-app marker (``Flask(`` / ``FastAPI(`` / ``create_app``). Files
    named in the last APPLY come first (fast path, order-stable), then a
    bounded whole-tree scan backfills any qualifying module the last APPLY
    did not touch -- so a nested ``backend/app.py`` scaffolded in an
    earlier chain is still probed rather than letting the boot gate skip
    (the ses_4cbf963cdc67435a blind spot: a real server never booted
    because it was not in the final applied-files list). The result is
    de-duplicated, order-stable, and capped at ``_MAX_ENTRY_CANDIDATES``.
    """
    out: List[str] = []
    seen: set[str] = set()

    def _consider(rel: str) -> None:
        rel = rel.replace("\\", "/")
        if not rel.endswith(".py") or rel in seen:
            return
        base = Path(rel).name
        qualifies = base in _ENTRY_BASENAMES
        if not qualifies:
            abs_path = ((root / rel) if not Path(rel).is_absolute()
                        else Path(rel))
            try:
                src = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return
            qualifies = any(marker in src for marker in _APP_MARKERS)
        if qualifies:
            seen.add(rel)
            out.append(rel)

    for rel in applied_files:
        _consider(rel)
    # Whole-tree backfill: qualifying modules the last APPLY did not list.
    for rel in _scan_tree_for_entries(root):
        if len(out) >= _MAX_ENTRY_CANDIDATES:
            break
        _consider(rel)
    return out[:_MAX_ENTRY_CANDIDATES]


def _scan_tree_for_entries(root: Path) -> List[str]:
    """Walk the project tree for ``.py`` files that look like an app entry.

    Prunes vendored/build/cache dirs (``_TREE_SCAN_SKIP_DIRS``) so the
    scan is bounded and a dependency's own ``app.py`` is never mistaken
    for the project's. Returns root-relative, forward-slash paths sorted
    for a stable probe order; qualification (basename / marker) is decided
    by the caller so the two candidate sources share one rule.
    """
    hits: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _TREE_SCAN_SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            abs_path = Path(dirpath) / name
            try:
                rel = abs_path.relative_to(root).as_posix()
            except ValueError:
                continue
            if name in _ENTRY_BASENAMES:
                hits.append(rel)
                continue
            try:
                src = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if any(marker in src for marker in _APP_MARKERS):
                hits.append(rel)
    return sorted(hits)


def _probe_boot(python_exe: str, root: str, file_abs: Path,
                timeout: float) -> Dict[str, Any]:
    """Import ``file_abs`` (and call ``create_app`` if any) in a subprocess.

    Returns ``{"ok", "kind", "stderr_tail"}`` where ``kind`` is ``ok`` on
    a clean boot, ``import_error`` on a non-zero exit, ``timeout`` when
    the boot hangs past ``timeout``, or ``launch_error`` when the probe
    interpreter could not be started at all.
    """
    file_dir = str(file_abs.parent)
    try:
        proc = subprocess.run(
            [python_exe, "-c", _PROBE_SCRIPT, root, str(file_abs), file_dir],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        tail = ((exc.stderr or "") + "\n[timeout]")[-_STDERR_TAIL_CHARS:]
        return {"ok": False, "kind": "timeout", "stderr_tail": tail}
    except (OSError, FileNotFoundError) as exc:
        return {"ok": False, "kind": "launch_error",
                "stderr_tail": f"{type(exc).__name__}: {exc}"}
    if proc.returncode == 0:
        return {"ok": True, "kind": "ok", "stderr_tail": ""}
    tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
    return {"ok": False, "kind": "import_error", "stderr_tail": tail}


def _combine_outcome(probes: List[Dict[str, Any]]) -> str:
    """Fold per-entry probe results into one RUNTIME_REPORT outcome token."""
    if not probes:
        return "skipped"
    if all(p["ok"] for p in probes):
        return "passed"
    kinds = {p["kind"] for p in probes if not p["ok"]}
    if "timeout" in kinds:
        return "timeout"
    if "import_error" in kinds:
        return "failed"
    return "error"


def _signature(failed_entries: List[str]) -> str:
    """A stable string keyed on the failing entry files for loop detection."""
    return "runtime_boot|" + ",".join(sorted(failed_entries))
