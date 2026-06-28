

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
import logging
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
_STDERR_TAIL_CHARS = 800


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
        outcome = "skipped"
    else:
        timeout = float(task.inputs.get("timeout_per_module")
                        or _DEFAULT_TIMEOUT_PER_MODULE)
        for pkg in candidates:
            ok, tail = _probe_import(python_exe, pkg, timeout)
            modules.append({"name": pkg, "ok": ok, "stderr_tail": tail})
        outcome = "passed" if all(m["ok"] for m in modules) else "failed"

    failed = [m["name"] for m in modules if not m["ok"]]
    signature = _signature(failed) if outcome == "failed" else ""
    content: Dict[str, Any] = {
        "build_artifact_id": task.inputs.get("build_artifact_id"),
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "python_exe": python_exe,
        "applied_files": list(applied_files),
        "modules": modules,
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
                  timeout: float) -> Tuple[bool, str]:
    """Run ``python -c "import <pkg>"`` and report ok + stderr tail."""
    try:
        proc = subprocess.run(
            [python_exe, "-c", f"import {pkg}"],
            capture_output=True, text=True, timeout=timeout,
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

