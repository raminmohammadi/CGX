

"""API_CHECK executor: catch hallucinated symbols before VERIFY collects.

Sits between BOOTSTRAP_ENV and SMOKE in the greenfield chain. For each
applied ``.py`` file, statically walks the AST to collect references
into third-party packages -- both ``from <pkg.sub> import <name>`` and
top-level ``import <pkg> as alias`` followed by ``alias.<attr>`` -- and
resolves each (module, name) pair under the bootstrapped venv via
``importlib.import_module`` + ``hasattr``. Unresolved names surface as
structured rows in :data:`ArtifactKind.API_CHECK_REPORT` so REPAIR
(Phase 3.2) can propose a typed fix (rename, version-aware pin, or
escalate to ASK_USER).

The check is intentionally narrower than SMOKE: SMOKE only asks "can
the package import at all?", which catches transitive dependency
breaks like the Flask 2.1 / Werkzeug 3 ``url_quote`` case. API_CHECK
asks "does ``pkg.fn`` exist?" -- the most common shape of an LLM
hallucination. Both run before VERIFY so a broken scaffold fails in
hundreds of milliseconds instead of multi-second pytest collection.
"""

from __future__ import annotations

import ast
import json
import logging
import re
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
from cgx.session.tasks.smoke import (
    _is_first_party,
    _resolve_applied_files,
    _resolve_python_exe,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROBE_TIMEOUT = 15.0
_STDERR_TAIL_CHARS = 800


@register_executor(TaskKind.API_CHECK)
def run_api_check(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Resolve every third-party symbol the applied files reference."""
    if not deps.project_root:
        return ExecutorResult(
            failure="API_CHECK requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="API_CHECK requires a session store in deps")

    root = Path(deps.project_root).resolve()
    applied_files = _resolve_applied_files(task, deps)
    python_exe = _resolve_python_exe(task, deps)

    specs, references = _collect_third_party_references(root, applied_files)
    rows: List[Dict[str, Any]] = []
    probe_error: Optional[str] = None
    if not python_exe or not specs:
        outcome = "skipped"
    else:
        timeout = float(task.inputs.get("probe_timeout")
                        or _DEFAULT_PROBE_TIMEOUT)
        rows, probe_error = _probe_references(python_exe, specs, timeout,
                                              root)
        if probe_error:
            outcome = "skipped"
        else:
            outcome = "passed" if all(r["ok"] for r in rows) else "failed"

    first_party_mods = _applied_first_party_modules(applied_files)
    uninstallable_roots = _uninstallable_import_roots(task, deps)
    rows = _attach_references(rows, references)
    for r in rows:
        if not r.get("ok"):
            r["category"] = _row_category(r, first_party_mods,
                                          uninstallable_roots)
    failed = [r for r in rows if not r["ok"]]
    # A top-level package that is simply absent from the venv is a
    # bootstrap/install problem, not a hallucinated API. Split those out
    # so REPAIR can install the package instead of regenerating code that
    # references a perfectly valid symbol.
    missing_dep_rows = [r for r in failed
                        if r.get("category") == "missing_dependency"]
    # An installed package whose own import chain broke on an incompatible
    # peer is a dependency-resolution fault: re-resolve the env, never
    # regenerate the (valid) code that references it.
    conflict_rows = [r for r in failed
                     if r.get("category") == "dependency_conflict"]
    hallucinated = [r for r in failed
                    if r.get("category") not in ("missing_dependency",
                                                 "dependency_conflict")]
    missing_modules = sorted({_missing_module_name(r)
                              for r in missing_dep_rows})
    conflict_packages = _conflict_packages(conflict_rows)
    signature = _signature(failed) if outcome == "failed" else ""
    content: Dict[str, Any] = {
        "build_artifact_id": task.inputs.get("build_artifact_id"),
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "python_exe": python_exe,
        "applied_files": list(applied_files),
        "references": rows,
        "outcome": outcome,
        "failed_references": [{"module": r["module"], "name": r["name"],
                               "error": r.get("error", ""),
                               "category": r.get("category", "")}
                              for r in failed],
        "missing_modules": missing_modules,
        "conflict_packages": conflict_packages,
        "conflict_references": [{"module": r["module"], "name": r["name"],
                                 "error": r.get("error", "")}
                                for r in conflict_rows],
        "hallucinated_references": [{"module": r["module"], "name": r["name"]}
                                    for r in hallucinated],
        "failure_signature": signature,
        "probe_error": probe_error,
    }
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.API_CHECK_REPORT,
        content=content,
    )
    return ExecutorResult(
        outputs={
            "api_check_artifact_id": artifact.artifact_id,
            "outcome": outcome,
            "failed_count": len(failed),
            "checked_count": len(rows),
            "missing_module_count": len(missing_modules),
            "conflict_count": len(conflict_rows),
            "conflict_packages": conflict_packages,
            "hallucinated_count": len(hallucinated),
            "failure_signature": signature,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

_Reference = Tuple[str, str, str, int]  # (module, name, file, lineno)


def _collect_third_party_references(
        root: Path,
        applied_files: List[str],
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """Return ([(module, name)], {(module, name): [{file, lineno}]}).

    Both ``from`` imports and qualified attribute access against
    aliased imports are collected. Per-file alias tables are built so
    ``import numpy as np`` followed by ``np.zeros`` resolves to
    ``("numpy", "zeros")``. Relative / stdlib / first-party imports
    are dropped (delegating the third-party check to :mod:`smoke`).
    """
    import sys
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    seen: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    order: List[Tuple[str, str]] = []

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
        _walk_module(tree, rel, root, stdlib, seen, order)
    return order, seen


def _walk_module(tree: ast.AST, rel: str, root: Path,
                 stdlib: frozenset,
                 seen: Dict[Tuple[str, str], List[Dict[str, Any]]],
                 order: List[Tuple[str, str]]) -> None:
    """Single-pass: collect ImportFrom names + alias-qualified attrs.

    Aliases are tracked at module scope only -- the heuristic skips
    function-local imports rather than carrying a full scope stack,
    keeping the executor small and predictable. The trade-off is
    benign: a locally-shadowed alias just produces a false negative
    (we don't probe it), never a false positive.
    """
    aliases: Dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                full = a.name or ""
                top = full.split(".")[0]
                if not top or top in stdlib or top == "__future__":
                    continue
                if _is_first_party(root, top):
                    continue
                key = a.asname or top
                aliases[key] = full
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            module = node.module or ""
            top = module.split(".")[0]
            if not top or top in stdlib or top == "__future__":
                continue
            if _is_first_party(root, top):
                continue
            for a in node.names:
                name = a.name or ""
                if not name or name == "*":
                    continue
                _record(order, seen, module, name, rel, node.lineno)
        elif isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name):
            base = node.value.id
            target = aliases.get(base)
            if not target:
                continue
            _record(order, seen, target, node.attr, rel, node.lineno)


def _record(order: List[Tuple[str, str]],
            seen: Dict[Tuple[str, str], List[Dict[str, Any]]],
            module: str, name: str, file: str, lineno: int) -> None:
    key = (module, name)
    refs = seen.setdefault(key, [])
    refs.append({"file": file, "lineno": int(lineno)})
    if len(refs) == 1:
        order.append(key)


_PROBE_SCRIPT = r"""
import importlib, json, sys
specs = json.loads(sys.stdin.read())
out = []
for module_path, name in specs:
    row = {"module": module_path, "name": name, "ok": False, "error": ""}
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        row["error"] = "{}: {}".format(type(exc).__name__, exc)
        out.append(row)
        continue
    if not name:
        row["ok"] = True
    elif hasattr(mod, name):
        row["ok"] = True
    else:
        # A submodule the package binds lazily (``from jose import jwt``
        # works, yet ``hasattr(jose, 'jwt')`` is False until something
        # imports it). Importing it is exactly what the referencing code
        # does, so resolve it the same way before calling it missing.
        try:
            importlib.import_module(module_path + "." + name)
            row["ok"] = True
        except Exception:
            row["error"] = "AttributeError: module '{}' has no attribute '{}'".format(
                module_path, name)
    out.append(row)
print(json.dumps(out))
"""


def _probe_references(python_exe: str,
                      specs: List[Tuple[str, str]],
                      timeout: float,
                      root: Path,
                      ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Run all (module, name) probes in a single subprocess.

    Returns (rows, probe_error). When ``probe_error`` is non-empty
    the run never completed -- callers should treat the result as
    ``skipped`` and surface the message for diagnostics.

    The probe runs with ``cwd=root`` and ``-I`` (isolated mode) so it
    sees exactly the project venv's import universe. Without this the
    subprocess inherits the CGX server's working directory, which ``-c``
    puts on ``sys.path``, letting the server's own files (e.g. a root
    ``app.py`` launcher) shadow the probed modules and produce false
    verdicts.
    """
    payload = json.dumps([list(s) for s in specs])
    try:
        proc = subprocess.run(
            [python_exe, "-I", "-c", _PROBE_SCRIPT],
            input=payload, capture_output=True, text=True, timeout=timeout,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired:
        return [], "[timeout]"
    except (OSError, FileNotFoundError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
        return [], tail or f"probe exited rc={proc.returncode}"
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return [], f"probe output not JSON: {exc}"
    return rows, None


def _attach_references(rows: List[Dict[str, Any]],
                       references: Dict[Tuple[str, str],
                                        List[Dict[str, Any]]],
                       ) -> List[Dict[str, Any]]:
    """Annotate each probe row with its source file/lineno occurrences."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (row.get("module", ""), row.get("name", ""))
        copy = dict(row)
        copy["references"] = references.get(key, [])
        out.append(copy)
    return out


_NO_MODULE_RE = re.compile(r"No module named '([^']+)'")

# A transitive version conflict: importing the referenced package failed
# *inside its own dependency chain* -- ``ImportError: cannot import name
# '<sym>' from '<peer>'`` -- because a stale pin dragged in an incompatible
# peer major (the Flask 2.0.1 / Werkzeug 3 ``url_quote`` break). Distinct
# from a hallucinated symbol (the probe imports the module fine and reports
# an ``AttributeError`` on ``hasattr``) and from an absent package (a
# dotless ``No module named``). Regenerating source can never fix it; the
# fix is to re-resolve the environment. The captured group is the peer
# whose incompatible version needs re-resolving.
_CANNOT_IMPORT_NAME_RE = re.compile(
    r"cannot import name\s+'[^']+'\s+from\s+"
    r"'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)'")


def _applied_first_party_modules(applied_files: List[str]) -> frozenset:
    """Top-level module names that resolve to a first-party file on disk.

    Covers a module referenced by the *wrong import path* -- e.g. a test
    that does ``from app import x`` while the app actually lives at
    ``backend/app.py``. The probe imports bare ``app``, which raises
    ``ModuleNotFoundError: No module named 'app'`` even though ``app`` is
    a perfectly real first-party module. Such a name must never be treated
    as a third-party package to install (pip cannot satisfy it); the fix is
    to regenerate the offending import. Each applied ``.py`` file
    contributes its module stem, and a package ``__init__.py`` contributes
    its directory name.
    """
    mods: set = set()
    for rel in applied_files:
        p = Path(rel)
        if p.suffix != ".py":
            continue
        if p.stem == "__init__":
            if p.parent.name:
                mods.add(p.parent.name)
        elif p.stem:
            mods.add(p.stem)
    return frozenset(mods)


def _uninstallable_import_roots(task: TaskNode,
                                deps: ExecutorDeps) -> frozenset:
    """Import roots BOOTSTRAP_ENV already failed to pip-install.

    Reads the ``uninstallable`` list from the upstream BUILD_REPORT --
    scan-discovered (undeclared) packages whose install failed, recorded
    non-fatally so this task owns their diagnosis -- and maps the PyPI
    names back to plausible import roots. A probe failure on such a root
    must not be classified ``missing_dependency``: pip has already
    proven it cannot satisfy the name, so an ``install_deps`` repair
    would loop. Best-effort: any lookup problem yields the empty set.
    """
    build_id = str(task.inputs.get("build_artifact_id") or "").strip()
    if not build_id or deps.store is None:
        return frozenset()
    art = deps.store.get_artifact(build_id)
    if art is None or art.kind is not ArtifactKind.BUILD_REPORT:
        return frozenset()
    names = (art.content or {}).get("uninstallable")
    if not isinstance(names, list) or not names:
        return frozenset()
    try:
        from cgx.session.tasks.bootstrap_env import _import_roots_for
        return _import_roots_for([str(n) for n in names])
    except Exception:  # pragma: no cover - defensive
        return frozenset(str(n).lower().replace("-", "_") for n in names)


def _row_category(row: Dict[str, Any],
                  first_party_mods: frozenset,
                  uninstallable_roots: frozenset = frozenset()) -> str:
    """Classify a failed probe row as missing-dependency vs hallucination.

    ``missing_dependency`` when a whole top-level package is absent from
    the venv -- i.e. a ``ModuleNotFoundError`` for a *dotless* module
    name (``No module named 'flask'``) that is **not** a first-party
    module present on disk. That is a bootstrap/install problem: the
    referenced symbol may be perfectly valid. Everything else -- an
    ``AttributeError`` on an installed module, a missing *submodule* of an
    installed package (``No module named 'flask.foo'``), a dotless name
    that matches a first-party module reached by the wrong import path
    (``app`` when the file is ``backend/app.py``), or a name BOOTSTRAP_ENV
    already failed to pip-install (``uninstallable_roots`` -- typically a
    hallucinated module like ``core``) -- is a genuine
    ``api_check_failure`` (hallucinated / mis-routed import) that pip
    cannot fix, so it routes to a regenerate instead of a no-op
    ``install_deps`` loop.
    """
    err = str(row.get("error") or "")
    m = _NO_MODULE_RE.search(err)
    if m and "." not in m.group(1):
        root_name = m.group(1)
        if root_name in first_party_mods:
            return "api_check_failure"
        if root_name.lower().replace("-", "_") in uninstallable_roots:
            return "api_check_failure"
        return "missing_dependency"
    # The module is installed but its own import chain broke on an
    # incompatible peer -- a dependency-resolution fault, not authoring.
    if _CANNOT_IMPORT_NAME_RE.search(err):
        return "dependency_conflict"
    return "api_check_failure"


def _conflict_packages(rows: List[Dict[str, Any]]) -> List[str]:
    """Top-level distributions implicated in an import-time version conflict.

    For each ``dependency_conflict`` row, the consumer (the referenced
    module's root, whose stale pin drags in an incompatible peer) and the
    peer named in the ``cannot import name ... from '<peer>'`` error. Both
    are handed to the resolve-deps repair so pip re-resolves a consistent
    set. Order-preserving and de-duplicated.
    """
    out: List[str] = []
    for r in rows:
        consumer = str(r.get("module") or "").split(".")[0]
        if consumer and consumer not in out:
            out.append(consumer)
        m = _CANNOT_IMPORT_NAME_RE.search(str(r.get("error") or ""))
        if m:
            peer = m.group(1).split(".")[0]
            if peer and peer not in out:
                out.append(peer)
    return out


def _missing_module_name(row: Dict[str, Any]) -> str:
    """Return the absent top-level package name for a missing-dep row."""
    m = _NO_MODULE_RE.search(str(row.get("error") or ""))
    if m:
        return m.group(1).split(".")[0]
    return str(row.get("module") or "").split(".")[0]


def _signature(failed: List[Dict[str, Any]]) -> str:
    """Stable string keyed on the failing (module, name) pairs."""
    parts = sorted(f"{r.get('module')}.{r.get('name')}" for r in failed)
    return "api_check|" + ",".join(parts)
