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

from cgx.session import repair_engine
from cgx.session.import_audit import resolve_first_party_imports
from cgx.session.models import Artifact, ArtifactKind, TaskKind
from cgx.session.scaffold_validate import (
    check_contract_compliance, cross_check_first_party_imports)
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.swarm_generate import generate_file
from cgx.session.tasks.swarm_ground import _safe_read
from cgx.session.tasks.swarm_log import swarm_beat
from cgx.session.tasks.swarm_plan import plan_specs
from cgx.session.tasks.swarm_tools import edit_file

_MAX_VERIFY_ROUNDS = 3
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
        contracts: Dict[str, Any], root: str,
        skills: Optional[List[str]] = None, goal: str = "") -> Tuple[
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
    # JS/TS coverage: a frontend file that does not parse, or whose body is a
    # leaked <call_tool> tag, is as missing as an unparseable Python module --
    # previously only the real build saw it, and the repair loop then had to
    # localise the diagnostic. Treat it as a coverage gap -> regenerate.
    try:
        js_broken = _js_broken_files(contents)
        if js_broken:
            swarm_beat(root, "verify", "js_broken", files=js_broken[:8])
            gaps = list(gaps) + [p for p in js_broken if p not in gaps]
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="js_broken", error=repr(e))
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
    # Cross-file module resolution for the JS/TS family (and any future
    # ecosystem with a spec row): a *relative* import that resolves to no
    # generated file is an integration break Python's resolver above cannot see
    # for non-Python source. ``cross_ref`` abstains on Python (handled above) and
    # on assets/aliases, so this only adds high-precision, un-gameable signals.
    try:
        from cgx.session.cross_ref import unresolved_references
        crossref_w = unresolved_references(contents, paths, root)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="cross_ref",
                   error=repr(e))
        crossref_w = []
    # JS/TS symbol gate (the Python first-party-symbol gate's twin): a named /
    # default import no dependency exports, a React or router identifier used
    # but never imported, a Router nested inside the entry's BrowserRouter.
    # Each row names the importer to regenerate and carries the real exports
    # as an additive directive.
    try:
        from cgx.session.js_symbols import js_symbol_breaks
        js_w = js_symbol_breaks(contents, paths)
        if js_w:
            swarm_beat(root, "verify", "js_symbols",
                       files=sorted({w["file"] for w in js_w})[:8],
                       reasons=[w["reason"][:120] for w in js_w[:4]])
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="js_symbols", error=repr(e))
        js_w = []
    try:
        phantom_3p_w = _check_phantom_third_party_imports(paths, contents, contracts.get("third_party_dependencies") or [], root)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="phantom_third_party",
                   error=repr(e))
        phantom_3p_w = []
    # Cross-file CIRCULAR-IMPORT gate (the case-study failure): reuse the AST/SCC
    # detector so a routes<->app / model<->app cycle -- which pytest cannot even
    # collect -- gates HARD and names a file to regenerate, instead of burning
    # dynamic-repair rounds one file at a time (which can never break a cycle).
    try:
        from cgx.session.tasks.scaffold import _circular_import_failures
        py_files = [{"path": p, "content": c} for p, c in contents.items()
                    if p.endswith(".py")]
        cycle_w = [{"file": e.get("file", ""), "reason": e.get("error", ""),
                    "kind": "circular_import"}
                   for e in (_circular_import_failures(py_files) or [])]
        if cycle_w:
            swarm_beat(root, "verify", "circular_import",
                       files=[w["file"] for w in cycle_w])
    except Exception as e:  # pragma: no cover - gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="circular_import",
                   error=repr(e))
        cycle_w = []
    # Skill structural validators now run on the SWARM path too (previously only
    # greenfield): a stack-specific FATAL verdict (e.g. Flask's circular-import /
    # undeclared-dep / no-real-app checks) gates with an actionable message.
    skill_w: List[Dict[str, Any]] = []
    try:
        if skills:
            import skills as _sk
            diffs = [{"path": p, "patch": c} for p, c in contents.items()]
            fatal = _sk.validate_scaffold(_sk.skills_by_names(skills), diffs,
                                          goal=goal)
            if fatal is not None and not fatal.passed:
                # Carry the skill's *targeted* fix hints: ``regen_file`` names the
                # file to regenerate (often NOT the file that tripped the check --
                # e.g. define Order in models.py, not the test that queries it),
                # and ``directive`` is the additive instruction for that regen.
                skill_w = [{"file": getattr(fatal, "regen_file", "") or "",
                            "define_in": getattr(fatal, "regen_file", "") or "",
                            "directive": getattr(fatal, "directive", "") or "",
                            "reason": fatal.rationale,
                            "kind": "skill_verdict",
                            "skill": getattr(fatal, "skill", "") or ""}]
                swarm_beat(root, "verify", "skill_verdict",
                           skill=skill_w[0]["skill"],
                           regen_file=skill_w[0]["file"],
                           reason=fatal.rationale[:200])
    except Exception as e:  # pragma: no cover - gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="skill_validate",
                   error=repr(e))
    import_breaks = _merge_import_warnings(symbol_w, resolve_w, crossref_w,
                                           js_w, cycle_w, skill_w)
    try:
        contract = check_contract_compliance(contents, contracts)
    except Exception as e:  # pragma: no cover - the gate is best-effort
        swarm_beat(root, "verify", "gate_error", gate="contract_compliance",
                   error=repr(e))
        contract = []
    return gaps, import_breaks, phantom_3p_w, contract


def _gating(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The import rows that may fail the verdict (``gating`` defaults to True)."""
    return [w for w in rows if w.get("gating", True)]


def _js_broken_files(contents: Dict[str, str]) -> List[str]:
    """JS/TS files that cannot be the real file: unparseable or a tool-call leak."""
    out: List[str] = []
    try:
        from cgx.answer.engine import _JS_TS_GRAMMAR_BY_EXT
        from cgx.codegen.validate import validate_js_ts_source
    except ImportError:  # pragma: no cover - grammar optional
        _JS_TS_GRAMMAR_BY_EXT, validate_js_ts_source = {}, None  # type: ignore[assignment]
    for path, content in (contents or {}).items():
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext not in ("js", "jsx", "ts", "tsx", "mjs", "cjs", "vue"):
            continue
        body = content or ""
        if "<call_tool" in body or "<tool_response" in body or not body.strip():
            out.append(path)
            continue
        if validate_js_ts_source is not None and ext in _JS_TS_GRAMMAR_BY_EXT:
            try:
                if not validate_js_ts_source(path, body, _JS_TS_GRAMMAR_BY_EXT[ext]).ok:
                    out.append(path)
            except Exception:  # pragma: no cover
                continue
    return out


def _regen_targets(gaps: List[str],
                   import_warnings: List[Dict[str, Any]],
                   known_paths: Optional[List[str]] = None) -> List[str]:
    """The precise files a structural failure implicates, de-duplicated.

    Prefers ``define_in`` -- the file that should DEFINE a referenced-but-missing
    symbol -- over ``file`` (where the broken reference merely appears). The
    coffee-shop lesson: an undefined ``Order`` is regenerated by growing
    ``models.py`` (the definer), not by stripping the test that queries it.

    When ``known_paths`` is given, a target is only accepted if it is a PLANNED
    file: a skill may synthesise a conventional ``models.py`` path that the plan
    never declared, and regenerating THAT would write an orphan file nothing
    imports. So prefer ``define_in`` when planned, else fall back to the
    (planned) importer ``file``, else skip -- never invent a path.
    """
    known = set(known_paths or [])
    targets: List[str] = list(gaps)
    for w in import_warnings:
        define_in = str(w.get("define_in") or "")
        f_ref = str(w.get("file") or "")
        if define_in and (not known or define_in in known):
            f = define_in
        elif f_ref and (not known or f_ref in known):
            f = f_ref
        else:
            f = ""  # neither is a planned file -> do not create an orphan
        if f and f not in targets:
            targets.append(f)
    return targets


def _regen_directives(import_warnings: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map each regen target to the additive instruction that should steer it.

    A skill (or the generic import check) can attach a ``directive`` naming
    exactly what a file must gain -- e.g. "DEFINE Order(db.Model) with columns
    ...". Threaded into the file's regen prompt so the pass is additive and
    precise, never a blind re-ask that reproduces the same gap.
    """
    out: Dict[str, str] = {}
    for w in import_warnings:
        directive = str(w.get("directive") or "").strip()
        target = str(w.get("define_in") or w.get("file") or "")
        if not target:
            continue
        # Fall back to the plain reason so even an un-enriched break gives the
        # regen a concrete "this symbol is missing" hint.
        text = directive or str(w.get("reason") or "").strip()
        if not text:
            continue
        out[target] = (out[target] + " " + text).strip() if target in out \
            else text
    return out


def _regenerate(targets: List[str], specs: Dict[str, Any],
                contracts: Dict[str, Any], goal: str, root: str,
                provider: Any, all_paths: List[str],
                skills: Optional[List[str]] = None,
                directives: Optional[Dict[str, str]] = None,
                design: Optional[Dict[str, Any]] = None) -> List[str]:
    """Re-run the generation ladder on ``targets``; return the still-failing.

    ``directives`` optionally appends a targeted, additive instruction to a
    file's description (from a skill verdict, import break, design check or
    design critic), so a regen ADDS the missing definition instead of
    re-emitting the same shape. ``design`` is the build's design brief so a
    regenerated UI file keeps its product/theme guidance.
    """
    still_bad: List[str] = []
    for path in targets:
        spec = specs.get(path, {})
        desc = str(spec.get("description") or "")
        extra = (directives or {}).get(path, "")
        if extra:
            desc = (desc + "\n\nREQUIRED FIX (additive -- do not remove existing "
                    "code): " + extra).strip()
        outcome = generate_file(
            path=path, description=desc,
            depends_on=list(spec.get("depends_on") or []), contracts=contracts,
            goal=goal, root=root, provider=provider, layer=path,
            manifest_paths=all_paths, log_root=root, skills=skills,
            design=design)
        if outcome.ok:
            edit_file(path, outcome.content, root)
        else:
            still_bad.append(path)
    return still_bad


def _apply_deterministic_repairs(paths: List[str], root: str,
                                 skills: Optional[List[str]]) -> List[str]:
    """Apply every active skill's deterministic ``repair_scaffold`` fixes.

    The Layer-1 fast-path: mechanical, 100%-decidable bugs (Flask double-prefix,
    ...) are fixed in place with NO model call, before any LLM repair round is
    spent. Returns the paths changed (for logging); writes fixes to disk.
    """
    if not skills:
        return []
    try:
        import skills as _sk
        contents = _collect_contents(paths, root)
        files = [{"path": p, "content": c} for p, c in contents.items()]
        fixes = _sk.apply_scaffold_repairs(_sk.skills_by_names(skills), files)
    except Exception as e:  # pragma: no cover - repair pass is best-effort
        swarm_beat(root, "verify", "autofix_error", error=repr(e))
        return []
    changed: List[str] = []
    for path, new_content in (fixes or {}).items():
        if path in contents and new_content != contents[path]:
            edit_file(path, new_content, root)
            changed.append(path)
            swarm_beat(root, "verify", "autofix", file=path, method="deterministic")
    return changed


# A red build's *stable* failure fingerprint: the "hard" defect identity (error
# classes, missing modules/symbols, and toolchain error codes) plus the "soft"
# set of failing test names -- everything that identifies the DEFECT, with the
# temperature-driven prose (file offsets, object reprs, timings) that jiggles
# between otherwise-identical rounds removed.
_ERRCLASS_RE = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b")
_MISSING_RE = re.compile(
    r"(?:No module named|cannot import name|has no attribute|no such table)"
    r"\s+['\"]?(\w+)['\"]?", re.I)
# Non-Python toolchains print no *Error class; key off their error CODES so a
# red tsc / rustc / MSVC build has a real signature too (else it looks empty ->
# spuriously "stalled"). e.g. ``error TS2322``, ``error[E0308]``, ``error C2065``.
_TOOLCODE_RE = re.compile(r"error\s*(TS\d+|\[?[A-Z]\d{3,}\]?|C\d{3,})", re.I)


def _hard_signature(output: str) -> frozenset:
    """The *hard* defect identity: error classes / codes / missing symbols.

    These are the failures a repeated, unchanged occurrence proves are STUCK
    (an import that won't resolve, a table with no model). A soft test-assertion
    failure is deliberately excluded -- iterating on an assertion across the
    repair-temperature ramp is legitimate progress, not a stall.
    """
    out = output or ""
    sig: set = set()
    sig |= {f"err:{m}" for m in _ERRCLASS_RE.findall(out)}
    sig |= {f"miss:{m}" for m in _MISSING_RE.findall(out)}
    sig |= {f"code:{m}" for m in _TOOLCODE_RE.findall(out)}
    return frozenset(sig)


def _failure_signature(output: str) -> frozenset:
    """Full fingerprint: the hard identity plus the failing-test-name set."""
    out = output or ""
    sig = set(_hard_signature(out))
    sig |= {f"test:{n}" for n in _failing_test_names(out)}
    return frozenset(sig)


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


# pytest prints one ``____ test_name ____`` banner per failing test in the
# FAILURES section, and ``FAILED path::[Class::]test_name`` in the short
# summary. These names drive two consumers: the human-readable summary
# ("tests failed: <names>") and the failing-test-file localizer below.
_FAIL_BANNER_RE = re.compile(r"^_{3,}\s+([\w.]+)\s+_{3,}\s*$", re.M)
_FAIL_SUMMARY_RE = re.compile(r"^FAILED\s+\S+?::(?:\w+::)?(\w+)", re.M)


def _failing_test_names(output: str) -> set:
    """Names of failing pytest tests, from the FAILURES banners + summary."""
    names: set = set()
    for m in _FAIL_BANNER_RE.finditer(output or ""):
        names.add(m.group(1).split(".")[-1])  # Class.method -> method
    for m in _FAIL_SUMMARY_RE.finditer(output or ""):
        names.add(m.group(1))
    return names


def _failing_test_files(output: str, paths: List[str], root: str) -> List[str]:
    """Planned pytest files that define a failing test named in ``output``.

    A red suite whose diagnostic prints no source PATH (a plain ``assert`` diff
    names only the test) is still localizable by its failing test NAME:
    :func:`repair_engine.neighbor_context` then pulls in each test's
    ``depends_on`` impl, so the repairer sees BOTH sides of a test<->impl
    disagreement. pytest-shaped; a runner that prints no such names simply
    contributes nothing here (the general path-appearance localizer still runs).
    """
    names = _failing_test_names(output or "")
    if not names:
        return []
    targets: List[str] = []
    for p in paths:
        if not _is_pytest_test_path(p):
            continue
        src = _safe_read(p, root)
        if not src:
            continue
        if any(re.search(rf"def\s+{re.escape(n)}\s*\(", src) for n in names):
            targets.append(p)
    return targets


def _localize_failure(output: str, paths: List[str], root: str) -> List[str]:
    """Planned files a red build implicates, from three general signals.

    De-duplicated (order preserved): files named verbatim in the diagnostic
    (:func:`repair_engine.localize_paths`, boundary-guarded so ``a.py`` never
    matches inside ``data.py``); the planned module a ``ModuleNotFoundError``
    points at (:func:`repair_engine.python_module_targets`); and any planned
    pytest file that defines a failing test (:func:`_failing_test_files`).
    """
    seen: List[str] = []
    for group in (repair_engine.localize_paths(output, paths),
                  repair_engine.python_module_targets(output, paths),
                  _failing_test_files(output, paths, root)):
        for p in group:
            if p not in seen:
                seen.append(p)
    return seen


# Source extensions the repairer can rewrite (Python + the JS/TS family). A
# build/test failure in either ecosystem should offer the implicated source to
# the repair model, not just ``.py`` -- otherwise a broken ``.jsx``/``.ts`` is
# never handed to the repairer and a red frontend build can never self-heal.
_REPAIRABLE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue",
                   ".json")


def _repair_context_paths(localized: List[str], paths: List[str],
                          specs: Dict[str, Any]) -> List[str]:
    """The focused source set to offer the repairer, localized files first.

    Delegates to :func:`repair_engine.neighbor_context`: the localized files,
    their direct ``depends_on``, and their reverse dependencies, filtered to
    repairable source extensions (Python *and* the JS/TS family, so a red
    ``npm run build`` can repair the implicated frontend file). A weak local
    model declines when handed the whole tree, so this small, defect-centred
    context is what turns "detected the failure" into "produced working code";
    with no localization it falls back to every source file. Order is
    localized-first so ``generate_repair_files``' ``max_files`` cap never drops
    a target.
    """
    return repair_engine.neighbor_context(localized, paths, specs,
                                          _REPAIRABLE_EXT)


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
    is focused via :func:`_repair_context_paths` (the localized files plus their
    dependencies and reverse dependencies, not the whole tree) so a weak model
    does not decline on an over-large prompt yet still sees BOTH sides of a
    test<->impl disagreement; ``localized`` is flagged so it starts at the
    failing frames. The "fix the wrong side, never weaken a correct test"
    guardrail lives in :data:`~cgx.answer.engine._LOGIC_REPAIR_SYSTEM`, so a
    single repair path serves every red signal (import, assertion, or build).
    Only files the repairer actually rewrote (and that pass its own source
    validation) are written back. Returns the paths it changed.
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


def _resolve_design(plan: Dict[str, Any], goal: str, skills: List[str],
                    paths: List[str]) -> Optional[Dict[str, Any]]:
    """The build's design brief: the persisted one, else derived from the goal.

    ``None`` for a non-UI build (or when the design layer is unavailable), so
    every design step below is a no-op for backend-only projects.
    """
    design = plan.get("design")
    if isinstance(design, dict) and design.get("applies"):
        return design
    if not goal:
        return None
    try:
        from cgx.design import build_brief
        brief = build_brief(goal, skills, plan_paths=paths)
        return brief.to_dict() if brief.applies else None
    except Exception:  # pragma: no cover - design layer must never break verify
        return None


def _ensure_design_tokens(design: Optional[Dict[str, Any]], paths: List[str],
                          root: str) -> Optional[str]:
    """Write the deterministic tokens stylesheet if the tree lacks it.

    The design system is a guarantee: if the plan dropped the tokens file (or
    the Developer failed it), CGX writes it itself -- no model call -- and
    adds it to the verified path set so the checks and the summary see it.
    Returns the path written, else ``None``.
    """
    if not design or not design.get("applies"):
        return None
    tokens_path = str(design.get("tokens_path") or "")
    if not tokens_path:
        return None
    full = os.path.join(root, tokens_path)
    if os.path.isfile(full) and os.path.getsize(full) > 0:
        if tokens_path not in paths:
            paths.append(tokens_path)
        return None
    try:
        from cgx.design import tokens_content
        edit_file(tokens_path, tokens_content(design), root)
    except Exception as e:  # pragma: no cover - best-effort
        swarm_beat(root, "verify", "design_tokens_error", error=repr(e))
        return None
    if tokens_path not in paths:
        paths.append(tokens_path)
    swarm_beat(root, "verify", "autofix", file=tokens_path, method="design-tokens")
    return tokens_path


def _design_scan(design: Optional[Dict[str, Any]], contents: Dict[str, str],
                 root: str) -> List[Any]:
    """Run the deterministic UI-quality checks; ``[]`` when not a UI build."""
    if not design or not design.get("applies"):
        return []
    try:
        from cgx.design import check_tree
        issues = check_tree(design, contents)
    except Exception as e:  # pragma: no cover - checks are best-effort
        swarm_beat(root, "verify", "gate_error", gate="design", error=repr(e))
        return []
    if issues:
        swarm_beat(root, "verify", "design_check",
                   fatal=sum(1 for i in issues if i.severity == "fatal"),
                   warnings=sum(1 for i in issues if i.severity != "fatal"),
                   ids=sorted({i.id for i in issues})[:12])
    return issues


def _design_rows(issues: List[Any]) -> List[Dict[str, Any]]:
    """Fatal design issues in the regen-target row shape (``define_in`` + ``directive``)."""
    try:
        from cgx.design import fatal_issues, issues_as_warnings
        return issues_as_warnings(fatal_issues(issues))
    except Exception:  # pragma: no cover
        return []


def _run_design_critic(design: Optional[Dict[str, Any]], paths: List[str],
                       root: str, provider: Any) -> Dict[str, str]:
    """One bounded LLM design review -> ``{planned path: fix directive}``."""
    if not design or not design.get("applies") or provider is None:
        return {}
    try:
        from cgx.design import critic_enabled, critique_ui
        if not critic_enabled():
            return {}
        contents = _collect_contents(paths, root)
        fixes = critique_ui(design, contents, provider)
    except Exception as e:  # pragma: no cover - the critic is best-effort
        swarm_beat(root, "verify", "design_critic_error", error=repr(e))
        return {}
    # Never regenerate a path the plan does not own.
    fixes = {p: f for p, f in (fixes or {}).items() if p in paths}
    if fixes:
        swarm_beat(root, "verify", "design_critic",
                   fixes={p: f[:160] for p, f in fixes.items()})
    else:
        swarm_beat(root, "verify", "design_critic", fixes={}, verdict="pass")
    return fixes


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
    # The design brief (UI builds only): drives the deterministic UI-quality
    # checks, the tokens-file guarantee, the design critic and the design gate.
    design = _resolve_design(plan, goal, skills, paths)

    swarm_beat(project_root, "verify", "structural", total=len(paths))
    # Every defect the run encountered, for the Layer-2 post-mortem to learn
    # from (prevention lessons are valuable even for defects that WERE fixed).
    incidents: List[Dict[str, Any]] = []
    tokens_written = _ensure_design_tokens(design, paths, project_root)
    if tokens_written:
        incidents.append({
            "phase": "autofix", "fixed": True, "files": [tokens_written],
            "detail": "Design tokens stylesheet was missing; CGX wrote it deterministically."})
    # Layer-1 fast-path FIRST: deterministically fix the mechanical, decidable
    # bugs (Flask double-prefix, ...) with no model call, so the structural scan
    # below sees a cleaner tree and the LLM repair budget is spent only on what
    # genuinely needs a model.
    autofixed = _apply_deterministic_repairs(paths, project_root, skills)
    if autofixed:
        incidents.append({
            "phase": "autofix", "fixed": True, "files": autofixed,
            "detail": ("Deterministically fixed a mechanical defect (e.g. a "
                       "blueprint route repeating its own url_prefix) in: "
                       + ", ".join(autofixed))})
    rounds = 0
    first_scan = True
    design_issues: List[Any] = []
    while True:
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root, skills, goal)
        # Design-quality checks run beside the structural scan: a FATAL design
        # defect (no stylesheet import, an unstyled component, a hard-coded
        # endpoint, no persistence ...) names the file to regenerate and the
        # additive fix, exactly like an import break does.
        design_issues = _design_scan(design, contents, project_root)
        d_rows = _design_rows(design_issues)
        if first_scan:
            first_scan = False
            for w in import_w:
                incidents.append({
                    "phase": "structural", "fixed": True,  # repaired below (best-effort)
                    "files": [w.get("define_in") or w.get("file") or ""],
                    "detail": str(w.get("reason") or "structural break")})
            for w in d_rows:
                incidents.append({
                    "phase": "design", "fixed": True,  # repaired below (best-effort)
                    "files": [w.get("define_in") or w.get("file") or ""],
                    "detail": f"[{w.get('id')}] {w.get('reason') or 'design defect'}"})
        targets = _regen_targets(gaps, import_w + d_rows, paths)
        if not targets or rounds >= _MAX_VERIFY_ROUNDS:
            break
        rounds += 1
        # A break carries WHERE to fix (define_in) and WHAT to add (directive);
        # regenerate the DEFINER additively (grow models.py to define Order),
        # never the file that merely referenced the missing symbol.
        directives = _regen_directives(import_w + d_rows)
        swarm_beat(project_root, "verify", "regenerate", round=rounds,
                   targets=targets,
                   design=sorted({str(w.get("id")) for w in d_rows})[:8])
        _regenerate(targets, specs, contracts, goal, project_root,
                    deps.provider, paths, skills, directives=directives,
                    design=design)
        # A regenerated file can carry a certain, mechanical defect (a hook
        # used without its import, a dangling stylesheet import): fix those
        # without a model call before the next scan so the bounded rounds are
        # spent on defects only a model can fix.
        _apply_deterministic_repairs(paths, project_root, skills)

    # Design critic (UI builds): one bounded review of the UI files against the
    # brief -> targeted, additive regenerations. The deterministic checks catch
    # what is mechanically wrong; the critic catches what is merely weak (a
    # settings page that is one input, a chat with no composer, no hierarchy).
    # Runs BEFORE the build/test dry-run so the final build validates its output.
    critic_fixes = _run_design_critic(design, paths, project_root, deps.provider)
    if critic_fixes:
        rounds += 1
        swarm_beat(project_root, "verify", "regenerate", round=rounds,
                   targets=list(critic_fixes), source="design_critic")
        _regenerate(list(critic_fixes), specs, contracts, goal, project_root,
                    deps.provider, paths, skills, directives=critic_fixes,
                    design=design)
        _apply_deterministic_repairs(paths, project_root, skills)
        # A regenerated file must still clear the structural + design gates;
        # one more targeted round if the critic's rewrite introduced a defect.
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root, skills, goal)
        design_issues = _design_scan(design, contents, project_root)
        d_rows = _design_rows(design_issues)
        post = _regen_targets(gaps, import_w + d_rows, paths)
        if post:
            rounds += 1
            swarm_beat(project_root, "verify", "regenerate", round=rounds,
                       targets=post, source="post_critic")
            _regenerate(post, specs, contracts, goal, project_root,
                        deps.provider, paths, skills,
                        directives=_regen_directives(import_w + d_rows),
                        design=design)
            contents = _collect_contents(paths, project_root)
            gaps, import_w, phantom_w, contract_w = _structural_scan(
                paths, contents, contracts, project_root, skills, goal)
            design_issues = _design_scan(design, contents, project_root)

    env = _run_env_dryrun(paths, project_root)
    # Only *hard* signals gate the build: missing/unparseable files (gaps) and
    # first-party import breaks. Phantom-third-party and contract warnings are
    # advisory -- a build whose tests + real build actually pass must not be
    # failed because the model under-declared a (working, installed) dependency
    # like ``uvicorn``; a truly hallucinated package fails the real install.
    # Prop-contract drift rows (``gating: False``) steer regeneration rounds but
    # never fail a build the real build + runtime smoke accept.
    structural_ok = not (gaps or _gating(import_w))

    # A structurally-clean tree whose suite is nonetheless red is exactly the
    # case the static gates cannot see: an import that resolves on paper but
    # breaks at runtime. Parse the failure, regenerate only the implicated
    # source files, and dry-run once more before the stage goes terminal.
    # UPDATE: We also allow dynamic repair even if there are static import warnings,
    # because static warnings are sometimes false positives (e.g. valid relative imports)
    # and the test traceback is a much stronger repair signal.
    dyn_rounds = 0
    stall = 0                     # times a HARD defect has recurred (stuck)
    seen_hard: set = set()        # every hard signature seen, for A->B->A cycles
    while (env.get("outcome") == "failed"
           and dyn_rounds < _MAX_DYNAMIC_REPAIR_ROUNDS):
        # Re-localize against the *current* failure each round: a red suite is
        # exactly the defect the static gates cannot see, and a bug exposed
        # only once an earlier import error is cleared (collection aborts at the
        # first broken module, masking the rest) must get its own targeted pass.
        # ``_localize_failure`` is toolchain-agnostic (planned paths named in the
        # diagnostic + a missing-module resolver + failing-test files), so a red
        # pytest, tsc, vite, or cargo build all localize the same way.
        prev_output = str(env.get("output") or "")
        prev_hard = _hard_signature(prev_output)
        dyn_targets = _localize_failure(prev_output, paths, project_root)
        dyn_rounds += 1
        swarm_beat(project_root, "verify", "dynamic_regenerate",
                   round=dyn_rounds, targets=dyn_targets)
        # CONVERGE-THEN-ESCALATE: a HARD defect (missing symbol/table, import or
        # toolchain error) that recurs means repairing the localized (usually
        # test) files is not working. On the first recurrence, escalate the
        # STRATEGY -- re-run the structural scan and regenerate the DEFINING
        # files with their additive directives (grow models.py to define the
        # missing model) -- instead of burning more identical rounds on the
        # symptom. Soft (assertion) failures are NOT counted as a stall: iterating
        # on an assertion across the temperature ramp is legitimate progress.
        if stall >= 1:
            _apply_deterministic_repairs(paths, project_root, skills)
            g2, iw2, _p2, _c2 = _structural_scan(
                paths, _collect_contents(paths, project_root), contracts,
                project_root, skills, goal)
            st_targets = _regen_targets(g2, iw2, paths)
            if st_targets:
                swarm_beat(project_root, "verify", "escalate_structural",
                           round=dyn_rounds, targets=st_targets)
                _regenerate(st_targets, specs, contracts, goal, project_root,
                            deps.provider, paths, skills,
                            directives=_regen_directives(iw2), design=design)
        # Re-read AFTER any escalation write so the repairer sees the current
        # tree (never the pre-escalation bodies -- that would clobber the fix).
        contents = _collect_contents(paths, project_root)
        # One failure-driven repair path for every red signal: thread the
        # runner output + the focused file bodies (both sides of a test<->impl
        # disagreement) into generate_repair_files, whose prompt carries the
        # "fix the wrong side, never weaken a correct test" guardrail. Only if
        # the repairer declines AND we localized a file do we fall back to a
        # blind regenerate of that file.
        repaired = _dynamic_repair(env, dyn_targets, contents, goal,
                                   project_root, deps.provider, paths,
                                   specs, dyn_rounds)
        if not repaired and dyn_targets:
            _regenerate(dyn_targets, specs, contracts, goal, project_root,
                        deps.provider, paths, skills, design=design)
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root, skills, goal)
        structural_ok = not (gaps or _gating(import_w))
        env = _run_env_dryrun(paths, project_root)
        # Stop early when a round cannot make progress: nothing to act on (no
        # repair and no regen target), or the failure is byte-for-byte
        # unchanged -- another identical pass would only burn a round.
        if not repaired and not dyn_targets:
            break
        if str(env.get("output") or "") == prev_output:
            break
        # HARD-signal convergence: a hard defect that is UNCHANGED from last
        # round, or that RECURS after we thought we'd moved past it (A->B->A
        # oscillation), means we are stuck on it. Escalate once (stall==1) then
        # bail (stall>=2). An EMPTY hard signature (e.g. a pure assertion diff, or
        # a toolchain we don't fingerprint) never counts as a stall -- those get
        # the full repair budget + byte-equality guard above.
        new_hard = _hard_signature(str(env.get("output") or ""))
        stuck = bool(new_hard) and (new_hard == prev_hard
                                    or new_hard in seen_hard)
        if prev_hard:
            seen_hard.add(prev_hard)
        if stuck:
            stall += 1
            if stall >= 2:
                swarm_beat(project_root, "verify", "converged_stalled",
                           round=dyn_rounds, signature=sorted(new_hard)[:8])
                break
        else:
            stall = 0

    # Runtime smoke (UI builds, optional browser): a green build whose routes
    # render a blank page is NOT a working app. Serve the bundle, load every
    # route headlessly, and turn an empty root / uncaught error into one
    # bounded repair round against the route's files before the verdict.
    smoke = _browser_smoke_and_repair(design, paths, project_root, goal, deps.provider,
                                      specs, env, skills=skills)
    if smoke is not None and smoke.get("rebuilt"):
        env = smoke["env"]
        contents = _collect_contents(paths, project_root)
        gaps, import_w, phantom_w, contract_w = _structural_scan(
            paths, contents, contracts, project_root, skills, goal)
        structural_ok = not (gaps or _gating(import_w))

    tests_red = env.get("outcome") == "failed"
    built = [p for p in paths if p not in gaps]
    # ``failed_paths`` recorded during the Developer chain is advisory only:
    # Verify's regeneration loop may have successfully rebuilt a file that
    # failed generation. Reconcile against the *final* structural scan -- a
    # path is only still failed if it remains a coverage gap (missing from
    # disk or unparseable). Threading the raw, never-cleared list into the
    # verdict is what previously sank a fully-repaired tree.
    still_failed = [p for p in failed_paths if p in gaps]
    structurally_green = structural_ok and not still_failed and not tests_red

    # Independent, objective-grounded behavioural acceptance: only when the tree
    # is otherwise green (no point booting a broken tree) AND the plan declared
    # a run command. Its checks come from the declared endpoints, not the model
    # -- so a stub cannot also weaken them. A genuine route error gates and earns
    # ONE impl-only repair; every environmental case (no command, no bind,
    # toolchain missing) is an advisory SKIP that never blocks.
    accept = _run_acceptance(contracts, project_root) if structurally_green \
        else None
    if accept is not None and accept.outcome == "failed":
        impl_paths = [p for p in paths if not _is_pytest_test_path(p)]
        failure_text = (
            "ACCEPTANCE FAILURE: the running application errored on declared "
            "routes (this is a real behavioural defect a unit test missed):\n"
            + "\n".join(accept.failures)
            + "\n\nFix the IMPLEMENTATION so these routes respond without a "
            "server error. Do NOT edit tests.")
        contents = _collect_contents(paths, project_root)
        swarm_beat(project_root, "verify", "acceptance_repair",
                   failures=accept.failures)
        _dynamic_repair({"output": failure_text}, [], contents, goal,
                        project_root, deps.provider, impl_paths, specs)
        accept = _run_acceptance(contracts, project_root)
    acceptance_ok = accept.ok if accept is not None else True

    # Design gate (UI builds): judge the FINAL tree -- the dynamic repair loop
    # may have rewritten UI files -- and refuse to call an app "verified" while
    # a fatal design defect remains (an unstyled screen, no stylesheet import,
    # a hard-coded endpoint). Advisory-only with CGX_DESIGN_GATE=0.
    design_issues = _design_scan(design, _collect_contents(paths, project_root),
                                 project_root)
    design_fatal = [i for i in design_issues if getattr(i, "severity", "") == "fatal"]
    design_warn = [i for i in design_issues if getattr(i, "severity", "") != "fatal"]
    try:
        from cgx.design import gate_enabled as _design_gate_enabled
        design_gated = _design_gate_enabled()
    except Exception:  # pragma: no cover
        design_gated = True
    design_ok = not design_fatal or not design_gated

    runtime_ok = not (smoke is not None and smoke.get("ran") and not smoke.get("ok"))
    verify_ok = structurally_green and acceptance_ok and design_ok and runtime_ok

    # A human-readable one-liner so a not-green run explains itself concretely
    # ("built N/M files; tests failed: <names>") instead of the UI / CLI showing
    # a bare "session failed". Surfaced on the terminal task by the router
    # (see _swarm_terminal_session_actions) and shown in the dashboard.
    summary = _verify_summary(paths, built, gaps, still_failed, env, verify_ok,
                              accept, design_fatal=design_fatal,
                              design_warnings=design_warn,
                              design_applies=bool(design), runtime=smoke)

    if smoke is not None and smoke.get("vision_model"):
        design_critic_note = {"vision_model": smoke.get("vision_model"),
                              "vision_fixes": smoke.get("vision_fixes") or {}}
    else:
        design_critic_note = None
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
        "acceptance": ({"outcome": accept.outcome, "reason": accept.reason,
                        "failures": accept.failures}
                       if accept is not None else None),
        "design": ({"applies": True,
                    "archetype": design.get("archetype"),
                    "theme": design.get("theme"),
                    "framework": design.get("framework"),
                    "tokens_written": tokens_written,
                    "critic_fixes": critic_fixes,
                    "issues": [i.as_dict() for i in design_issues],
                    "fatal_remaining": len(design_fatal),
                    "gated": design_gated}
                   if design else {"applies": False}),
        "runtime": (smoke.get("result") if smoke else None),
        "vision": design_critic_note,
        "verify_ok": verify_ok,
        "summary": summary,
    }
    artifact = Artifact.new(
        session_id=task.session_id, produced_by_task_id=task.task_id,
        kind=ArtifactKind.SWARM_VERIFY_REPORT, content=content)
    swarm_beat(project_root, "verify", "report", ok=verify_ok,
               gaps=len(gaps), imports=len(import_w),
               contracts=len(contract_w), tests=env.get("outcome"),
               design_fatal=len(design_fatal), design_warnings=len(design_warn))

    # LAYER 2 -- learn from this run. Record what stayed broken as an unresolved
    # incident, then run ONE bounded post-mortem (gated by CGX_SWARM_LEARNING)
    # that distils generalised lessons and auto-appends them to the skill/project
    # stores, so the next run avoids the mistake at generation time. Best-effort:
    # learning never changes this run's verdict.
    if not verify_ok:
        incidents.append({
            "phase": "unresolved", "fixed": False, "files": still_failed,
            "signature": sorted(_failure_signature(str(env.get("output") or ""))),
            "detail": "Run did not reach green: " + summary})
    for i in design_fatal:
        incidents.append({
            "phase": "design", "fixed": False, "files": [i.regen_file or i.file],
            "detail": f"[{i.id}] {i.message}"})
    if smoke is not None and smoke.get("ran") and not smoke.get("ok"):
        incidents.append({
            "phase": "runtime", "fixed": False, "files": list(smoke.get("targets") or []),
            "detail": "Runtime smoke: " + "; ".join(smoke.get("failures") or [])[:400]})
    if incidents:
        try:
            from cgx.learning.postmortem import run_postmortem
            recorded = run_postmortem(
                deps.provider, goal=goal, skills=skills, root=project_root,
                incidents=incidents, session_id=task.session_id,
                task_id=task.task_id)
            for r in recorded:
                swarm_beat(project_root, "verify", "lesson_recorded",
                           scope=r["scope"], skill=r.get("skill", ""),
                           lesson=r["lesson"][:160])
        except Exception as e:  # pragma: no cover - learning is best-effort
            swarm_beat(project_root, "verify", "learn_error", error=repr(e))
    return ExecutorResult(
        artifact=artifact,
        outputs={"verify_ok": verify_ok,
                 "verify_report_artifact_id": artifact.artifact_id,
                 "coverage_gaps": gaps,
                 "failed_paths": still_failed,
                 "summary": summary,
                 "project_root": project_root})


def _route_component_files(route: str, contents: Dict[str, str],
                           paths: List[str]) -> List[str]:
    """Planned files that implement ``route``: the routed component + its imports."""
    from cgx.session.js_symbols import js_imports, resolve_relative
    targets: List[str] = []
    rx = re.compile(r"""<Route\b[^>]*\bpath\s*=\s*['"]""" + re.escape(route)
                    + r"""['"][^>]*\belement\s*=\s*\{\s*<([A-Z]\w*)""", re.S)
    for path, src in contents.items():
        for m in rx.finditer(src or ""):
            comp = m.group(1)
            for stmt in js_imports(src):
                if stmt.default == comp or any(l == comp for _, l in stmt.named):
                    hit = resolve_relative(stmt.spec, path, paths)
                    if hit and hit not in targets:
                        targets.append(hit)
    # the component's own first-party imports are where a wrong call usually lives
    for t in list(targets):
        for stmt in js_imports(contents.get(t, "")):
            hit = resolve_relative(stmt.spec, t, paths)
            if hit and hit not in targets and hit.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
                targets.append(hit)
    return targets


def _vision_round(design: Dict[str, Any], smoke: Any, paths: List[str], root: str,
                  goal: str, provider: Any, specs: Dict[str, Any],
                  skills: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """One bounded screenshot review -> targeted regenerations -> rebuild -> re-smoke."""
    try:
        from cgx.codegen.browser_smoke import run_browser_smoke
        from cgx.design.prompts import file_kind
        from cgx.design.vision import critique_screenshots, resolve_vision_model
    except Exception:  # pragma: no cover
        return None
    model = resolve_vision_model(provider)
    if not model:
        return None
    shots = {r.route: r.screenshot for r in smoke.results if r.screenshot}
    ui_files = [p for p in paths if file_kind(design, p) in ("component", "app", "stylesheet", "html")]
    fixes = critique_screenshots(design, shots, ui_files, provider, model=model)
    fixes = {p: f for p, f in fixes.items() if p in paths}
    swarm_beat(root, "verify", "vision_critic", model=model[1],
               fixes={p: f[:160] for p, f in fixes.items()} or {}, verdict="pass" if not fixes else "fixes")
    if not fixes:
        return {"vision_model": model[1], "vision_fixes": {}}
    _regenerate(list(fixes), specs, {}, goal, root, provider, paths, skills,
                directives=fixes, design=design)
    _apply_deterministic_repairs(paths, root, skills or [str(design.get("framework") or "react")])
    env2 = _run_env_dryrun(paths, root)
    result: Dict[str, Any] = {"vision_model": model[1], "vision_fixes": fixes, "rebuilt": True, "env": env2}
    if env2.get("outcome") == "passed":
        try:
            smoke2 = run_browser_smoke(root, [r.route for r in smoke.results],
                                       ui_root=str(design.get("app_root") or ""))
        except Exception:  # pragma: no cover
            smoke2 = None
        if smoke2 is not None and smoke2.ran:
            failures2 = smoke2.failures()
            swarm_beat(root, "verify", "runtime_smoke", outcome="passed" if not failures2 else "failed",
                       routes=[r.route for r in smoke2.results], after="vision",
                       blank=[r.route for r in failures2 if not r.mounted],
                       errors=[e for r in failures2 for e in r.errors][:4])
            result["result"] = smoke2.as_dict()
            result["ok"] = not failures2
            result["failures"] = [f"{r.route}: {'blank' if not r.mounted else 'errors'}" for r in failures2]
            return result
    result["ok"] = False
    result["failures"] = ["build or smoke failed after the vision round"]
    return result


def _browser_smoke_and_repair(design: Optional[Dict[str, Any]], paths: List[str],
                              root: str, goal: str, provider: Any,
                              specs: Dict[str, Any], env: Dict[str, Any],
                              skills: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Load the built app's routes in a headless browser; repair a blank route once.

    Returns ``None`` when not applicable (non-UI build, red build, no bundle),
    else ``{ran, ok, result, failures, targets, rebuilt, env}``.
    """
    if not design or not design.get("applies") or design.get("framework") not in ("react", "vue"):
        return None
    if env.get("outcome") != "passed":
        return None
    try:
        from cgx.codegen.browser_smoke import (routes_from_sources, run_browser_smoke,
                                               smoke_enabled)
    except Exception:  # pragma: no cover
        return None
    if not smoke_enabled():
        return None
    contents = _collect_contents(paths, root)
    routes = routes_from_sources(contents)
    ui_root = str(design.get("app_root") or "")
    out: Dict[str, Any] = {"ran": False, "ok": True, "result": None, "failures": [],
                           "targets": [], "rebuilt": False, "env": env}
    try:
        smoke = run_browser_smoke(root, routes, ui_root=ui_root)
    except Exception as e:  # pragma: no cover - smoke is best-effort
        swarm_beat(root, "verify", "runtime_smoke_error", error=repr(e))
        return None
    out["result"] = smoke.as_dict()
    if not smoke.ran:
        swarm_beat(root, "verify", "runtime_smoke", outcome="skipped",
                   reason=smoke.skipped_reason)
        return out
    out["ran"] = True
    failures = smoke.failures()
    swarm_beat(root, "verify", "runtime_smoke", outcome="passed" if not failures else "failed",
               routes=[r.route for r in smoke.results],
               blank=[r.route for r in failures if not r.mounted],
               errors=[e for r in failures for e in r.errors][:4])
    if not failures:
        # Vision critic (optional local vision model): LOOK at the rendered
        # screens, regenerate what it flags, rebuild, and re-check. Taste never
        # gates the verdict; the smoke re-check does.
        vision = _vision_round(design, smoke, paths, root, goal, provider, specs, skills=skills)
        if vision:
            out.update(vision)
        return out
    # ONE bounded repair round: the failing routes' component files + their
    # first-party imports, with the console error as the diagnostic.
    lines = []
    targets: List[str] = []
    for r in failures:
        why = ("rendered an EMPTY page (the app crashed during render)" if not r.mounted
               else "logged uncaught errors")
        lines.append(f"- route {r.route}: {why}" + (": " + " | ".join(r.errors[:3]) if r.errors else ""))
        for t in _route_component_files(r.route, contents, paths):
            if t not in targets:
                targets.append(t)
    # Every route blank = a GLOBAL failure (no router provider, a crashing shell,
    # a bad mount): the entry, the shell and the HTML entry are the suspects,
    # not the individual pages.
    if len(failures) == len(smoke.results) and all(not r.mounted for r in failures):
        for key in ("entry_path", "app_path", "html_path"):
            p = str(design.get(key) or "")
            if p in paths and p not in targets:
                targets.insert(0, p)
    if not targets:
        entry = str(design.get("app_path") or "")
        targets = [p for p in (entry,) if p in paths]
    # Keep the round focused: a weak repairer handed ten files rewrites them
    # all badly; four (global suspects first, then the failing route's files)
    # is what it can actually fix.
    targets = targets[:4]
    out["failures"] = [ln.lstrip("- ") for ln in lines]
    out["targets"] = targets
    failure_text = (
        "RUNTIME SMOKE FAILURE: the production build compiled, but loading it in a browser showed:\n"
        + "\n".join(lines)
        + "\n\nA blank page means a component threw during render (a wrong helper call returning "
        "undefined, a missing import, a prop contract mismatch, a nested <Router>). Fix the "
        "IMPLEMENTATION in the implicated files so every route renders; keep the design system "
        "classes and the existing behaviour.")
    swarm_beat(root, "verify", "runtime_repair", targets=targets)
    repaired = _dynamic_repair({"output": failure_text}, targets, contents, goal, root,
                               provider, paths, specs)
    if not repaired and targets:
        _regenerate(targets, specs, {}, goal, root, provider, paths, skills,
                    directives={t: failure_text for t in targets}, design=design)
    _apply_deterministic_repairs(paths, root, skills or [str(design.get("framework") or "react")])
    env2 = _run_env_dryrun(paths, root)
    out["rebuilt"] = True
    out["env"] = env2
    if env2.get("outcome") == "passed":
        try:
            smoke2 = run_browser_smoke(root, routes, ui_root=ui_root)
        except Exception:  # pragma: no cover
            smoke2 = None
        if smoke2 is not None and smoke2.ran:
            out["result"] = smoke2.as_dict()
            failures2 = smoke2.failures()
            swarm_beat(root, "verify", "runtime_smoke", outcome="passed" if not failures2 else "failed",
                       routes=[r.route for r in smoke2.results], after="repair",
                       blank=[r.route for r in failures2 if not r.mounted],
                       errors=[e for r in failures2 for e in r.errors][:4])
            out["ok"] = not failures2
            out["failures"] = [f"{r.route}: {'blank' if not r.mounted else 'errors'}" for r in failures2]
            return out
    out["ok"] = False
    return out


def _run_acceptance(contracts: Dict[str, Any], root: str):
    """Behavioural acceptance against a plan-declared server; never raises.

    Advisory-degrading: returns a SKIPPED result (which does not block) unless a
    declared route genuinely errors. Kept in the executor (not the pure module)
    so it can resolve the project interpreter and emit a telemetry beat.
    """
    try:
        from pathlib import Path
        from cgx.codegen.test_runner import _project_python_exe
        from cgx.session.tasks.acceptance import run_acceptance
        result = run_acceptance(contracts, root,
                                python_exe=_project_python_exe(Path(root)))
        swarm_beat(root, "verify", "acceptance", outcome=result.outcome,
                   reason=result.reason, checks=result.checks_run)
        return result
    except Exception as e:  # pragma: no cover - acceptance is best-effort
        swarm_beat(root, "verify", "acceptance_error", error=repr(e))
        return None


def _verify_summary(paths: List[str], built: List[str], gaps: List[str],
                    still_failed: List[str], env: Dict[str, Any],
                    verify_ok: bool, accept: Any = None, *,
                    design_fatal: Optional[List[Any]] = None,
                    design_warnings: Optional[List[Any]] = None,
                    design_applies: bool = False,
                    runtime: Optional[Dict[str, Any]] = None) -> str:
    """A concise, human-readable outcome line for a swarm run.

    Distinguishes a green build from a *partial* one (most files built, one
    gate red) so a not-green terminal reports what actually happened and what to
    look at next, instead of an opaque "session failed".
    """
    n = len(paths)
    parts = [f"built {len(built)}/{n} files"]
    if gaps:
        parts.append(f"{len(gaps)} missing/unparseable")
    outcome = str(env.get("outcome") or "skipped")
    if outcome == "passed":
        parts.append("tests passed")
    elif outcome == "failed":
        failing = sorted(_failing_test_names(str(env.get("output") or "")))
        if failing:
            shown = ", ".join(failing[:5])
            more = f" (+{len(failing) - 5} more)" if len(failing) > 5 else ""
            parts.append(f"tests failed: {shown}{more}")
        else:
            parts.append("tests/build failed")
    else:
        parts.append("tests not run")
    # Only surface acceptance when it actually ran a verdict (passed/failed);
    # an advisory skip stays quiet so it never implies a behavioural guarantee
    # that was not checked.
    if accept is not None and getattr(accept, "outcome", "skipped") == "failed":
        parts.append(f"acceptance failed: {accept.reason}")
    elif accept is not None and getattr(accept, "outcome", "") == "passed":
        parts.append("acceptance passed")
    # Design verdict (UI builds only): name the remaining fatal defects by id so
    # the user sees WHY a styled-looking build was not called verified.
    if design_applies:
        if design_fatal:
            ids = ", ".join(sorted({getattr(i, "id", "?") for i in design_fatal})[:4])
            parts.append(f"design: {len(design_fatal)} fatal issue(s) ({ids})")
        else:
            n_warn = len(design_warnings or [])
            parts.append("design ok" + (f" ({n_warn} advisory)" if n_warn else ""))
    if runtime and runtime.get("ran"):
        res = runtime.get("result") or {}
        routes = res.get("routes") or []
        if runtime.get("ok"):
            parts.append(f"runtime ok ({len(routes)} route{'s' if len(routes) != 1 else ''} render)")
        else:
            bad = [r.get("route") for r in routes if not (r.get("mounted") and not r.get("errors"))]
            parts.append("runtime: " + ", ".join(str(b) for b in bad[:3]) + " blank/crashed")
    head = "verified" if verify_ok else "partial build"
    return f"{head}: " + "; ".join(parts)
