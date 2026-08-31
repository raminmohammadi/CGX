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
import os
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
# than being stranded by a single pass. Set to 5 (was 3): a weaker local model
# often needs a couple more targeted passes -- with the temperature ramp -- to
# converge on a genuine logic/validation bug before the stage goes terminal.
_MAX_DYNAMIC_REPAIR_ROUNDS = 5


def _repair_temperature(dyn_rounds: int) -> float:
    """Temperature ramp for repair rounds: 0.2, +0.2/round, capped at 0.8.

    A first repair is near-deterministic; if the same defect survives, later
    rounds warm up to escape a repeated wrong fix. Shared by the AST logic-fix
    and the failure-driven repair so the schedule stays in one place.
    """
    return min(0.2 + (dyn_rounds * 0.2), 0.8)


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
    allowed_set.add("pytest")
    allowed_set.add("typing")
    for p in paths:
        if not p.endswith(".py"): continue
        src = contents.get(p)
        if not src: continue
        imports = _extract_imports_python(src)
        for imp in imports:
            if imp in _STDLIB_TOP or _is_local_package(imp, root):
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
        contracts: Dict[str, Any], root: str) -> Tuple[
            List[str], List[Dict[str, Any]], List[Dict[str, Any]],
            List[Dict[str, Any]]]:
    """Run the structural checks; never raises (each gate is defensive).

    Returns ``(gaps, import_breaks, advisory, contract)``:

    * ``import_breaks`` -- **hard** first-party coherence failures: a
      ``from X import name`` naming a symbol no file defines
      (``cross_check_first_party_imports``) or a first-party module that
      resolves nowhere (``resolve_first_party_imports``). These are genuine
      bugs a test run may not even reach, so they gate.
    * ``advisory`` -- **soft** ``phantom_third_party`` warnings (an import not
      listed in the plan's ``third_party_dependencies``). Since verify now
      reconciles + installs every real import, a legitimate dependency the
      model simply forgot to *declare* (e.g. ``uvicorn``) must not fail a build
      whose tests actually pass; a genuinely hallucinated package fails the real
      install/build instead. Reported, not gating.
    * ``contract`` -- soft contract-compliance advisories (also non-gating).
    """
    gaps = _coverage_gaps(paths, contents)
    try:
        symbol_w = cross_check_first_party_imports(contents)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="symbol_imports",
                   error=repr(e))
        symbol_w = []
    try:
        resolve_w = resolve_first_party_imports(contents, paths)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="resolve_imports",
                   error=repr(e))
        resolve_w = []
    try:
        phantom_3p_w = _check_phantom_third_party_imports(paths, contents, contracts.get("third_party_dependencies") or [], root)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="phantom_third_party",
                   error=repr(e))
        phantom_3p_w = []
    import_breaks = _merge_import_warnings(symbol_w, resolve_w)
    try:
        contract = check_contract_compliance(contents, contracts)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="contract_compliance",
                   error=repr(e))
        contract = []
    return gaps, import_breaks, phantom_3p_w, contract


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
                provider: Any, all_paths: List[str],
                skills: Optional[List[str]] = None) -> List[str]:
    """Re-run the generation ladder on ``targets``; return the still-failing."""
    still_bad: List[str] = []
    for path in targets:
        spec = specs.get(path, {})
        outcome = generate_file(
            path=path, description=str(spec.get("description") or ""),
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=root, provider=provider, layer=path,
            manifest_paths=all_paths, log_root=root, skills=skills)
        if outcome.ok:
            edit_file(path, outcome.content, root)
        else:
            still_bad.append(path)
    return still_bad


def _run_env_dryrun(paths: List[str], root: str) -> Dict[str, Any]:
    """Install deps then run the project's tests/build across all stacks.

    Polyglot: Python imports are pip-installed here so pytest can import them,
    and the shared :func:`run_project_tests` then detects and runs *every*
    applicable stack (pytest + a ``package.json`` test/build for a JS/TS
    component), merging them into one pass/fail signal -- so a React frontend
    gets a real ``npm run build`` gate instead of being silently skipped. The
    npm runner installs its own node_modules.
    """
    py = [p for p in paths if p.endswith(".py")]
    py_abs = [os.path.join(root, p) for p in py]
    report: Dict[str, Any] = {"ran": False, "outcome": "skipped"}
    # Proactively reconcile each component's manifest against its source imports
    # *before* building/testing, so a directly-imported-but-undeclared package
    # (Python or npm) never fails the run -- one mechanism for any package,
    # rather than reacting to each missing-dependency error.
    _reconcile_manifests(paths, root)
    if py_abs:
        try:
            from cgx.codegen.env_manager import preflight_install
            missing, results = preflight_install(py_abs, root)
            report["missing_installed"] = missing
            report["install_results"] = results
        except Exception as exc:  # pragma: no cover - install is best-effort
            report["install_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from cgx.codegen.test_runners import run_project_tests
        outcome = run_project_tests(root, py or paths)
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


# Source extensions the repairer can rewrite (Python + the JS/TS family). A
# build/test failure in either ecosystem should offer the implicated source to
# the repair model, not just ``.py`` -- otherwise a broken ``.jsx``/``.ts`` is
# never handed to the repairer and a red frontend build can never self-heal.
_REPAIRABLE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue",
                   ".json")


def _repair_context_paths(localized: List[str], paths: List[str],
                          specs: Dict[str, Any]) -> List[str]:
    """The focused source set to offer the repairer, localized files first.

    A weak local model declines (returns ``{"files": []}``) when handed the
    whole planned tree as context -- the signal drowns in unrelated files, so a
    one-line fix in a single module is never made. When the failure localized
    one or more files, offer *only* those plus their direct ``depends_on`` (the
    sibling API the failing frame calls), so the prompt is small and centred on
    the defect. With no localization fall back to every planned source file --
    there is no better hint and small trees still fit. Language-aware: Python
    *and* JS/TS sources are eligible, so a red ``npm run build`` can repair the
    implicated frontend file. Order is localized-first so
    ``generate_repair_files``' own ``max_files`` cap never drops a target.
    """
    src = [p for p in paths if p.endswith(_REPAIRABLE_EXT)]
    if not localized:
        return src
    keep: List[str] = []
    for t in localized:
        if t in src and t not in keep:
            keep.append(t)
        for dep in (specs.get(t, {}) or {}).get("depends_on") or []:
            dep = str(dep)
            if dep in src and dep not in keep:
                keep.append(dep)
    return keep or src


def _reconcile_manifests(paths: List[str], root: str) -> None:
    """Make each component's manifest declare (and install) what its source
    imports -- the systemic fix for 'code imports X but the manifest lacks X'.

    Rather than react to each build/test error for a specific missing package,
    this scans the generated source once, per ecosystem, and closes the
    manifest gap generically for *any* package:

    * **Python** -- every third-party import root (not stdlib, not first-party)
      is mapped to its PyPI distribution, installed, and pinned to
      ``requirements.txt``. FastAPI/Starlette additionally imply ``httpx`` (its
      ``TestClient`` needs it but never imports it) -- a bounded, framework-level
      known-dep, not a per-package rule.
    * **Node** -- for each ``package.json`` component, every bare import in its
      JS/TS source that isn't already a (dev)dependency is ``npm install``ed
      (which also writes it to ``dependencies``).

    Best-effort and idempotent: a package already declared/installed is skipped,
    so this never loops. Never raises.
    """
    try:
        _reconcile_python_requirements(paths, root)
    except Exception:  # pragma: no cover - reconciliation is best-effort
        pass
    try:
        _reconcile_node_dependencies(root)
    except Exception:  # pragma: no cover
        pass


# FastAPI/Starlette TestClient needs httpx but the app never imports it, so an
# import scan can't see it. This is the *only* implied-dependency mapping --
# framework-level and bounded, deliberately not a growing per-package list.
_PY_IMPLIED_DEPS = {"fastapi": ["httpx"], "starlette": ["httpx"]}


def _reconcile_python_requirements(paths: List[str], root: str) -> None:
    from cgx.codegen.env_manager import (
        _STDLIB_TOP, _import_root_to_pypi, _is_local_package, _read_requirements,
        install_packages, scan_imports, update_requirements)
    py_abs = [os.path.join(root, p) for p in paths if p.endswith(".py")]
    if not py_abs:
        return
    imports = scan_imports(py_abs)
    if not imports:
        return
    have = _read_requirements(root)  # normalized existing requirement names
    dists: set = set()
    for imp in imports:
        rootmod = imp.split(".")[0]
        if not rootmod or rootmod.lower() in _STDLIB_TOP \
                or _is_local_package(rootmod, root):
            continue
        for extra in _PY_IMPLIED_DEPS.get(rootmod.lower(), []):
            dists.add(extra)
        dists.add(_import_root_to_pypi(rootmod) or rootmod)
    needed = sorted(d for d in dists
                    if d.lower().replace("-", "_") not in have)
    if not needed:
        return
    results = install_packages(needed)
    installed = [d for d in needed if results.get(d)]
    if installed:
        update_requirements(root, installed)
        swarm_beat(root, "verify", "deps_reconciled", ecosystem="python",
                   packages=installed)


def _reconcile_node_dependencies(root: str) -> None:
    import json as _json
    import shutil
    import subprocess
    if shutil.which("npm") is None:
        return
    from cgx.codegen.env_manager import scan_file_imports
    from cgx.codegen.test_runners import _find_package_json_dirs
    for d in _find_package_json_dirs(root):
        pj = os.path.join(d, "package.json")
        try:
            data = _json.loads(open(pj, encoding="utf-8").read())
        except Exception:
            continue
        have = {k.lower() for k in (data.get("dependencies") or {})}
        have |= {k.lower() for k in (data.get("devDependencies") or {})}
        # Scan the component's own JS/TS source for bare imports.
        imports: set = set()
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x != "node_modules"]
            for fn in filenames:
                if fn.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
                    imports |= scan_file_imports(os.path.join(dirpath, fn))
        needed = sorted(p for p in imports if p.lower() not in have)
        if not needed:
            continue
        try:
            proc = subprocess.run(
                ["npm", "install", "--no-audit", "--no-fund",
                 "--legacy-peer-deps", *needed],
                cwd=d, capture_output=True, text=True, timeout=180.0)
        except Exception:  # pragma: no cover - install is best-effort
            continue
        if proc.returncode == 0:
            swarm_beat(root, "verify", "deps_reconciled", ecosystem="node",
                       dir=os.path.relpath(d, root), packages=needed)


def _auto_fix_missing_imports(env: Dict[str, Any], root: str, provider: Any) -> bool:
    """Parse output for NameError and safely inject missing imports using AST."""
    output = str(env.get("output") or "")
    match = re.search(r"NameError: name '(\w+)' is not defined", output)
    if not match:
        return False
    
    missing_name = match.group(1)
    file_match = re.search(r"^([^:\n]+):[0-9]+: NameError", output, re.MULTILINE)
    if not file_match:
        return False
    
    failed_file = file_match.group(1).strip()
    if not failed_file.endswith(".py"):
        return False
        
    prompt = (f"The name '{missing_name}' is undefined in a Python project. "
              f"What is the standard import statement for this? "
              f"Reply with ONLY the single import line, e.g. 'from fastapi import {missing_name}'. "
              "Do not use markdown fences.")
    
    try:
        reply = provider.chat([{"role": "user", "content": prompt}], force_json=False).get("content", "").strip()
    except Exception as e:
        swarm_beat(root, "verify", "auto_fix_error", fix="missing_imports",
                   error=repr(e))
        return False

    reply = reply.replace("```python", "").replace("```", "").strip()
    if not (reply.startswith("import ") or reply.startswith("from ")):
        return False
        
    path = os.path.join(root, failed_file)
    if not os.path.exists(path):
        return False
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if reply in content:
        return False
        
    new_content = reply + "\n" + content
    from cgx.session.tasks.swarm_tools import edit_file
    edit_file(failed_file, new_content, root)
    return True


def _auto_fix_function_logic(env: Dict[str, Any], root: str, provider: Any, dyn_rounds: int = 0) -> bool:
    """Parse output for logic errors, extract the broken function via AST, and ask LLM to rewrite it."""
    import ast
    output = str(env.get("output") or "")
    
    matches = list(re.finditer(r"^([^:\n]+):([0-9]+): ([a-zA-Z0-9_]+Error)", output, re.MULTILINE))
    if not matches:
        return False
        
    last_match = matches[-1]
    failed_file = last_match.group(1).strip()
    line_number = int(last_match.group(2).strip())
    error_type = last_match.group(3).strip()
    
    if not failed_file.endswith(".py"):
        return False
        
    error_msg_match = re.search(r"E\s+(.*?\n)\n" + re.escape(last_match.group(0)), output)
    if not error_msg_match:
        error_msg_match = re.search(r"E\s+(.*?)\n", output[output.rfind('E '):])
        
    error_message = error_msg_match.group(1).strip() if error_msg_match else error_type
    
    path = os.path.join(root, failed_file)
    if not os.path.exists(path):
        return False
        
    with open(path, "r", encoding="utf-8") as f:
        source_code = f.read()
        
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return False
        
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= line_number <= node.end_lineno:
                    # In python 3.8+, decorators are included in lineno, but let's be safe
                    target_node = node
                    break
                    
    if not target_node:
        return False
        
    lines = source_code.split("\n")
    start_idx = target_node.lineno - 1
    if hasattr(target_node, "decorator_list") and target_node.decorator_list:
        start_idx = min(start_idx, target_node.decorator_list[0].lineno - 1)
        
    end_idx = target_node.end_lineno
    old_function_code = "\n".join(lines[start_idx:end_idx])
    
    prompt = (
        f"The following Python code failed with this error:\n"
        f"ERROR: {error_message}\n\n"
        f"CODE:\n```python\n{old_function_code}\n```\n\n"
        f"Rewrite this specific code block to fix the logic error. "
        f"Return ONLY the raw rewritten code inside a markdown block. Do not include JSON."
    )
    
    repair_temp = _repair_temperature(dyn_rounds)

    try:
        reply = provider.chat([{"role": "user", "content": prompt}], force_json=False, temperature=repair_temp).get("content", "").strip()
    except Exception as e:
        swarm_beat(root, "verify", "auto_fix_error", fix="function_logic",
                   error=repr(e))
        return False
        
    import re as re_mod
    code_match = re_mod.search(r"```python\s*(.*?)\s*```", reply, re_mod.DOTALL)
    if code_match:
        new_function_code = code_match.group(1).strip()
    else:
        new_function_code = reply.replace("```python", "").replace("```", "").strip()
        
    if not new_function_code:
        return False
        
    # verify it parses as valid python
    try:
        ast.parse(new_function_code)
    except SyntaxError:
        return False
        
    new_content = "\n".join(lines[:start_idx]) + "\n" + new_function_code + "\n" + "\n".join(lines[end_idx:])
    
    from cgx.session.tasks.swarm_tools import edit_file
    edit_file(failed_file, new_content, root)
    return True
def _dynamic_repair(env: Dict[str, Any], localized: List[str],
                    contents: Dict[str, str], goal: str, root: str,
                    provider: Any, paths: List[str],
                    specs: Dict[str, Any], dyn_rounds: int = 0) -> List[str]:
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
    from cgx.session.tasks.swarm_generate import ToolWrapper

    failure_text = str(env.get("output") or "")
    if not failure_text:
        return []
    context_paths = _repair_context_paths(localized, paths, specs)
    files = [{"path": p, "content": contents[p]}
             for p in context_paths if p in contents]
    if not files:
        return []
    try:
        wrapped_provider = ToolWrapper(provider, root)
        repaired = generate_repair_files(
            wrapped_provider, goal=goal, failure_text=failure_text,
            files=files, localized_files=localized)
    except Exception as e:  # pragma: no cover - repair is best-effort
        swarm_beat(root, "verify", "dynamic_repair_error", error=repr(e))
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
        rel_p = p
        if "/src/" in p:
            rel_p = "src/" + p.split("/src/", 1)[-1]
        elif "/tests/" in p:
            rel_p = "tests/" + p.split("/tests/", 1)[-1]
        
        if (p and p in text) or (rel_p and rel_p in text):
            if p not in targets:
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
    skills = list(plan.get("skills") or [])  # stack for skill-guided regen
    goal = str(task.inputs.get("goal") or plan.get("goal") or "")
    project_root = (task.inputs.get("project_root")
                    or plan.get("project_root")
                    or deps.project_root or ".")

    swarm_beat(project_root, "verify", "structural", total=len(paths))
    rounds = 0
    while True:
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root)
        targets = _regen_targets(gaps, import_w)
        if not targets or rounds >= _MAX_VERIFY_ROUNDS:
            break
        rounds += 1
        swarm_beat(project_root, "verify", "regenerate", round=rounds,
                   targets=targets)
        _regenerate(targets, specs, contracts, goal, project_root,
                    deps.provider, paths, skills)

    env = _run_env_dryrun(paths, project_root)
    # Only *hard* signals gate the build: missing/unparseable files (gaps) and
    # first-party import breaks. Phantom-third-party and contract warnings are
    # advisory -- a build whose tests + real build actually pass must not be
    # failed because the model under-declared a (working, installed) dependency
    # like ``uvicorn``; a truly hallucinated package fails the real install.
    structural_ok = not (gaps or import_w)

    # A structurally-clean tree whose suite is nonetheless red is exactly the
    # case the static gates cannot see: an import that resolves on paper but
    # breaks at runtime. Parse the failure, regenerate only the implicated
    # source files, and dry-run once more before the stage goes terminal.
    # UPDATE: We also allow dynamic repair even if there are static import warnings,
    # because static warnings are sometimes false positives (e.g. valid relative imports)
    # and the test traceback is a much stronger repair signal.
    dyn_rounds = 0
    while (env.get("outcome") == "failed"
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
        
        # Lightweight AST-based auto-fix for missing imports first.
        if _auto_fix_missing_imports(env, project_root, deps.provider):
            repaired = {"auto_fixed": True}
        # Then AST-based function-logic repair.
        elif _auto_fix_function_logic(env, project_root, deps.provider, dyn_rounds):
            repaired = {"auto_fixed_logic": True}
        # Finally, failure-driven repair fed the red suite's output.
        else:
            repaired = _dynamic_repair(env, dyn_targets, contents, goal,
                                       project_root, deps.provider, paths, specs, dyn_rounds)
        if not repaired and dyn_targets:
            _regenerate(dyn_targets, specs, contracts, goal, project_root,
                        deps.provider, paths, skills)
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root)
        structural_ok = not (gaps or import_w)
        env = _run_env_dryrun(paths, project_root)
        # Stop early when a round cannot make progress: nothing to act on (no
        # repair and no regen target), or the failure is byte-for-byte
        # unchanged -- another identical pass would only burn a round.
        if not repaired and not dyn_targets:
            break
        if str(env.get("output") or "") == prev_output:
            break

    tests_red = env.get("outcome") == "failed"
    # ``failed_paths`` recorded during the Developer chain is advisory only:
    # Verify's regeneration loop may have successfully rebuilt a file that
    # failed generation. Reconcile against the *final* structural scan -- a
    # path is only still failed if it remains a coverage gap (missing from
    # disk or unparseable). Threading the raw, never-cleared list into the
    # verdict is what previously sank a fully-repaired tree.
    still_failed = [p for p in failed_paths if p in gaps]
    verify_ok = structural_ok and not still_failed and not tests_red

    content = {
        "work_plan_artifact_id": work_plan_id,
        "project_root": project_root,
        "paths": paths,
        "coverage_gaps": gaps,
        "import_warnings": import_w,
        "phantom_third_party": phantom_w,   # advisory (non-gating)
        "contract_warnings": contract_w,    # advisory (non-gating)
        "regen_rounds": rounds,
        "dynamic_regen_rounds": dyn_rounds,
        "failed_paths": still_failed,
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
                 "failed_paths": still_failed,
                 "project_root": project_root})
