

"""SMOKE executor: catch import-class failures before pytest collection.

Sits between BOOTSTRAP_ENV (greenfield) and VERIFY. For each top-level
third-party module the applied files import, runs
``<venv>/bin/python -c "import <pkg>"`` with a short timeout. A single
import error here means VERIFY is guaranteed to fail at collection
time -- catching it now gives the operator (and Phase 3.2 REPAIR
proposer) a clean, 200 ms signal instead of a multi-second pytest
trace full of fixture noise.

Emits :data:`ArtifactKind.SMOKE_REPORT` whose ``modules`` list pairs
each tested package with ``ok`` / ``stderr_tail``. The artifact's
``outcome`` token (``passed`` / ``failed`` / ``skipped``) drives the
router branch in :func:`cgx.session.router._smoke_to_verify_or_repair`.
SMOKE never returns ``ExecutorResult.failure`` so the report is always
persisted even when an import broke -- REPAIR needs the structured
content to plan a remediation.
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
import subprocess
import sys
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

_DEFAULT_TIMEOUT_PER_MODULE = 5.0
_DEFAULT_BUILD_SMOKE_TIMEOUT = 180.0
_STDERR_TAIL_CHARS = 800
# JS bundlers (Vite/rolldown, esbuild) print the actionable cause -- e.g.
# ``[UNRESOLVED_ENTRY] Cannot resolve entry module index.html`` -- at the
# HEAD of stderr, then a long, generic async stack trace. A tail-only clip
# keeps only the useless stack, so REPAIR's ``build_error`` constraint is
# non-actionable. Keep the head (the real diagnostic) and the tail (any
# summary line) so both survive into the regenerate constraint.
_STDERR_HEAD_CHARS = 1200


def _clip_output(text: str) -> str:
    """Clip build output to a head+tail window, preserving both ends.

    The head carries the bundler's primary error; the tail carries any
    trailing summary. When the text fits within the combined budget it is
    returned verbatim; otherwise the elided middle is marked.
    """
    if not text:
        return ""
    if len(text) <= _STDERR_HEAD_CHARS + _STDERR_TAIL_CHARS:
        return text
    return (text[:_STDERR_HEAD_CHARS]
            + "\n...[truncated]...\n"
            + text[-_STDERR_TAIL_CHARS:])


@register_executor(TaskKind.SMOKE)
def run_smoke(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Probe each third-party top-level import declared by applied files."""
    if not deps.project_root:
        return ExecutorResult(failure="SMOKE requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(failure="SMOKE requires a session store in deps")

    root = Path(deps.project_root).resolve()
    applied_files = _resolve_applied_files(task, deps)
    python_exe = _resolve_python_exe(task, deps)

    candidates = _collect_third_party_imports(root, applied_files)
    modules: List[Dict[str, Any]] = []
    if not python_exe or not candidates:
        py_outcome = "skipped"
    else:
        timeout = float(task.inputs.get("timeout_per_module")
                        or _DEFAULT_TIMEOUT_PER_MODULE)
        for pkg in candidates:
            ok, tail = _probe_import(python_exe, pkg, timeout, root)
            modules.append({"name": pkg, "ok": ok, "stderr_tail": tail})
        py_outcome = "passed" if all(m["ok"] for m in modules) else "failed"

    # JS/TS build-smoke: a non-building frontend must fail honestly here
    # rather than skip through to a false-green VERIFY. Runs ``npm run
    # build`` (buildability only); VERIFY's ``NpmRunner`` then runs the
    # ``test`` script. Gated on a provisioned ``node_modules`` (see
    # BOOTSTRAP_ENV) so an offline box that never installed deps skips
    # rather than fabricating a build failure.
    build_smoke = _npm_build_smoke(root, task)

    failed = [m["name"] for m in modules if not m["ok"]]
    part_outcomes = [py_outcome]
    if build_smoke is not None:
        part_outcomes.append("passed" if build_smoke["ok"] else "failed")
    if "failed" in part_outcomes:
        outcome = "failed"
    elif "passed" in part_outcomes:
        outcome = "passed"
    else:
        outcome = "skipped"

    sig_parts = list(failed)
    if build_smoke is not None and not build_smoke["ok"]:
        sig_parts.append(build_smoke["label"])
    signature = _signature(sig_parts) if outcome == "failed" else ""
    content: Dict[str, Any] = {
        "build_artifact_id": task.inputs.get("build_artifact_id"),
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "python_exe": python_exe,
        "applied_files": list(applied_files),
        "modules": modules,
        "build_smoke": build_smoke,
        "outcome": outcome,
        "failed_modules": failed,
        "failure_signature": signature,
    }
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.SMOKE_REPORT,
        content=content,
    )
    return ExecutorResult(
        outputs={
            "smoke_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "failed_count": len(failed),
            "tested_count": len(modules),
            "failure_signature": signature,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _resolve_applied_files(task: TaskNode, deps: ExecutorDeps) -> List[str]:
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


def _collect_third_party_imports(root: Path,
                                applied_files: List[str]) -> List[str]:
    """Return deduplicated top-level package names worth smoke-testing.

    Drops:
      * stdlib modules (anything in ``sys.stdlib_module_names``);
      * relative imports (``from .foo import x``);
      * dotted imports whose top segment resolves to a file or package
        directory under ``root`` (first-party code, no install needed).
    """
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    out: List[str] = []
    seen: set[str] = set()
    for rel in applied_files:
        if not rel.endswith(".py"):
            continue
        abs_path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        try:
            src = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src, filename=str(abs_path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = (alias.name or "").split(".")[0]
                    _maybe_add(pkg, root, stdlib, seen, out)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # relative import
                mod = (node.module or "").split(".")[0]
                _maybe_add(mod, root, stdlib, seen, out)
    return out


def _maybe_add(pkg: str, root: Path, stdlib: frozenset,
               seen: set, out: List[str]) -> None:
    if not pkg or pkg in seen:
        return
    if pkg in stdlib or pkg == "__future__":
        seen.add(pkg)
        return
    if _is_first_party(root, pkg):
        seen.add(pkg)
        return
    seen.add(pkg)
    out.append(pkg)


def _is_first_party(root: Path, pkg: str) -> bool:
    """True when ``pkg`` resolves to a file or package directory under ``root``."""
    for base in (root, root / "src"):
        if (base / f"{pkg}.py").is_file():
            return True
        if (base / pkg / "__init__.py").is_file():
            return True
        if (base / pkg).is_dir():
            return True
    return False


def _probe_import(python_exe: str, pkg: str,
                  timeout: float, root: Path) -> Tuple[bool, str]:
    """Run ``python -I -c "import <pkg>"`` in ``root``; report ok + stderr tail.

    ``cwd=root`` plus ``-I`` (isolated mode) keep the probe pinned to
    the project venv's import universe -- without them the subprocess
    inherits the CGX server's working directory, which ``-c`` puts on
    ``sys.path``, letting the server's own files shadow the probed
    module and produce a false verdict.
    """
    try:
        proc = subprocess.run(
            [python_exe, "-I", "-c", f"import {pkg}"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired as e:
        tail = (e.stderr or "")[-_STDERR_TAIL_CHARS:] + "\n[timeout]"
        return False, tail
    except (OSError, FileNotFoundError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
    return False, tail


def _signature(failed_modules: List[str]) -> str:
    """A stable string keyed on the failing modules for loop detection."""
    return "smoke_import|" + ",".join(sorted(failed_modules))


def _npm_build_command(root: Path) -> Optional[List[str]]:
    """Return the ``npm run build`` invocation, or ``None`` when absent.

    The smoke deliberately probes buildability only (the ``build``
    script); the ``test`` script is VERIFY's job via ``NpmRunner``, so
    we never run it twice.
    """
    try:
        data = json.loads(
            (root / "package.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    scripts = data.get("scripts") if isinstance(data.get("scripts"), dict) else {}
    if str(scripts.get("build") or ""):
        return ["npm", "run", "build", "--silent"]
    return None


def _npm_build_smoke(root: Path, task: TaskNode) -> Optional[Dict[str, Any]]:
    """Run ``npm run build`` as a fast buildability gate for JS/TS projects.

    Returns ``None`` (skip) when there is no ``package.json``, no ``npm``
    binary, no ``build`` script, or no provisioned ``node_modules`` --
    the last guard keeps an offline box (deps never installed) from
    fabricating a build failure. Otherwise returns
    ``{"label", "ok", "stderr_tail"}`` so a broken build surfaces as a
    real ``failed`` outcome that routes into REPAIR/REGENERATE.
    """
    if not (root / "package.json").is_file():
        return None
    if shutil.which("npm") is None:
        return None
    if not (root / "node_modules").is_dir():
        return None
    cmd = _npm_build_command(root)
    if cmd is None:
        return None
    timeout = float(task.inputs.get("build_smoke_timeout")
                    or _DEFAULT_BUILD_SMOKE_TIMEOUT)
    label = " ".join(cmd)
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        tail = _clip_output((exc.stderr or "") + "\n[timeout]")
        return {"label": label, "ok": False, "stderr_tail": tail}
    except Exception as exc:
        return {"label": label, "ok": False,
                "stderr_tail": f"{type(exc).__name__}: {exc}"}
    ok = proc.returncode == 0
    tail = "" if ok else _clip_output(proc.stderr or proc.stdout or "")
    return {"label": label, "ok": ok, "stderr_tail": tail}

