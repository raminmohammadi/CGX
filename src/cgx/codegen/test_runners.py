"""Pluggable multi-language test-runner registry.

Parallels the parser registry (``_PARSER_REGISTRY`` in
``cgx.parser.parse_codebase``): each stack registers a :class:`TestRunner`
that (a) *detects* whether it applies to a project via marker files and
(b) *runs* its tests (or a build-smoke) against the live working tree,
returning a :class:`~cgx.codegen.test_runner.TestRunOutcome`.

Today two runners ship: Python via pytest (wrapping the existing on-disk
runner) and JS/TS via the ``package.json`` ``test``/``build`` scripts.
Register a new stack by appending to :data:`_TEST_RUNNER_REGISTRY`. This
restores the agent's self-correction loop for non-Python projects: without
an executable test/build signal there is nothing for the Judge to gate on.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cgx.codegen.test_runner import TestRunOutcome, run_tests_on_disk

logger = logging.getLogger(__name__)


class TestRunner:
    """A per-stack runner. Subclasses set ``name`` and implement both methods."""

    # Tell pytest not to collect this (and its subclasses) as a test class:
    # the ``Test`` prefix is a naming coincidence, not a test case.
    __test__ = False

    name: str = "runner"

    def detect(self, project_root: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def run(
        self, project_root: str, changed_files: Sequence[str], *,
        timeout_seconds: float = 180.0, python_exe: Optional[str] = None,
    ) -> TestRunOutcome:  # pragma: no cover - interface
        raise NotImplementedError


class PytestRunner(TestRunner):
    """Python stack: delegates to the existing on-disk pytest runner."""

    name = "pytest"
    _MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")

    def detect(self, project_root: str) -> bool:
        root = Path(project_root)
        if any((root / m).is_file() for m in self._MARKERS):
            return True
        # Marker-less projects still qualify if any test module exists.
        try:
            next(root.rglob("test_*.py"))
            return True
        except StopIteration:
            return False

    def run(
        self, project_root: str, changed_files: Sequence[str], *,
        timeout_seconds: float = 180.0, python_exe: Optional[str] = None,
    ) -> TestRunOutcome:
        return run_tests_on_disk(
            project_root, list(changed_files),
            timeout_seconds=timeout_seconds, python_exe=python_exe,
        )


# JS/TS unit-test files (mirrors scaffold._is_js_test_path): ``*.test.*`` /
# ``*.spec.*`` / anything under a ``__tests__`` dir. Kept in sync so the
# harness SCAFFOLD backfills is the same suite VERIFY looks for.
_JS_TEST_EXTS = (".js", ".jsx", ".ts", ".tsx")
# Directories that never hold first-party tests -- skip them so a
# dependency's bundled ``*.test.js`` never reads as the project's suite.
_JS_TEST_SKIP_DIRS = frozenset({
    "node_modules", ".git", ".cgx", ".cgx-backups", "dist", "build",
    "coverage", ".venv", "venv", "__pycache__", ".next", ".nuxt", "out",
})


def _is_js_test_file(rel_path: str) -> bool:
    """True for a JS/TS unit-test file (``*.test.jsx`` / ``__tests__/*``)."""
    p = rel_path.replace("\\", "/").strip().lstrip("./").lower()
    dot = p.rfind(".")
    ext = p[dot:] if dot > 0 else ""
    if ext not in _JS_TEST_EXTS:
        return False
    if "/__tests__/" in p or p.startswith("__tests__/"):
        return True
    stem = p[: -len(ext)]
    return stem.endswith(".test") or stem.endswith(".spec")


def _has_js_test_files(project_root: str) -> bool:
    """Whether the working tree carries any first-party JS/TS test file.

    Walks the tree pruning vendored/build dirs so a dependency's bundled
    ``*.test.js`` never counts. Lets VERIFY tell a genuinely test-free JS
    project (honest ``no_tests``) apart from one whose scaffolded suite was
    never wired to run -- the coverage blind spot P2 fails closed on.
    """
    root = Path(project_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _JS_TEST_SKIP_DIRS]
        for name in filenames:
            if _is_js_test_file(name):
                return True
    return False


def _load_package_json(project_root: str) -> Optional[Dict[str, Any]]:
    try:
        with open(Path(project_root) / "package.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _npm_script_command(project_root: str) -> Optional[List[str]]:
    """Choose an npm invocation: prefer a real ``test`` script, else ``build``.

    npm's init placeholder (``echo "Error: no test specified"``) is treated
    as *absent* so a JS project that never wired up tests still gets a
    build-smoke signal rather than an automatic failure. Returns ``None``
    when neither a usable test nor a build script exists.
    """
    pkg = _load_package_json(project_root)
    if not pkg:
        return None
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    test_script = str(scripts.get("test") or "")
    if test_script and "no test specified" not in test_script:
        return ["npm", "test", "--silent"]
    if str(scripts.get("build") or ""):
        return ["npm", "run", "build", "--silent"]
    return None


class NpmRunner(TestRunner):
    """JS/TS stack: runs the package.json ``test`` script, else a ``build`` smoke."""

    name = "npm"

    def detect(self, project_root: str) -> bool:
        return (Path(project_root) / "package.json").is_file()

    def run(
        self, project_root: str, changed_files: Sequence[str], *,
        timeout_seconds: float = 180.0, python_exe: Optional[str] = None,
    ) -> TestRunOutcome:
        # Report JS test-file presence on every outcome (even the early
        # skips) so P2 can fail closed on a scaffolded-but-never-run suite
        # regardless of *why* it did not run (npm missing, no test script).
        tp = _has_js_test_files(project_root)
        if shutil.which("npm") is None:
            return TestRunOutcome(
                ran=False, skipped_reason="npm not installed", tests_present=tp)
        cmd = _npm_script_command(project_root)
        if cmd is None:
            return TestRunOutcome(
                ran=False, skipped_reason="no npm test or build script",
                tests_present=tp)
        root = Path(project_root).resolve()
        # Best-effort dependency install so the smoke can run; bounded and
        # non-fatal -- an offline box simply runs the script as-is.
        if not (root / "node_modules").is_dir():
            try:
                subprocess.run(
                    ["npm", "install", "--no-audit", "--no-fund"], cwd=root,
                    capture_output=True, text=True,
                    timeout=min(timeout_seconds, 180.0),
                )
            except Exception as e:
                logger.debug("npm install skipped: %s", e)
        label = " ".join(cmd)
        # ``npm test`` runs a real suite; the ``npm run build`` fallback is a
        # buildability smoke only, so a passing build is not a passing suite.
        ran_tests = cmd[:2] == ["npm", "test"]
        try:
            proc = subprocess.run(
                cmd, cwd=root, capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return TestRunOutcome(
                ran=True, returncode=124, stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\n[timeout]", tests_selected=[label],
                ran_tests=ran_tests, tests_present=tp)
        except Exception as e:
            return TestRunOutcome(
                ran=False, skipped_reason=f"{type(e).__name__}: {e}",
                tests_present=tp)
        return TestRunOutcome(
            ran=True, returncode=proc.returncode, stdout=proc.stdout,
            stderr=proc.stderr, tests_selected=[label], ran_tests=ran_tests,
            tests_present=tp)


# Registry of default runners, checked in order. Append new stacks here.
_TEST_RUNNER_REGISTRY: List[TestRunner] = [PytestRunner(), NpmRunner()]


def detect_test_runners(project_root: str) -> List[TestRunner]:
    """Return every registered runner whose stack markers match the project.

    A polyglot repo (e.g. a Python backend beside a JS frontend) can match
    more than one runner; all matches run and their outcomes are merged.
    """
    out: List[TestRunner] = []
    for r in _TEST_RUNNER_REGISTRY:
        try:
            if r.detect(project_root):
                out.append(r)
        except Exception as e:
            logger.debug("test runner %s detect failed: %s", r.name, e)
    return out


def _aggregate_outcomes(
    pairs: Sequence["tuple[str, TestRunOutcome]"],
) -> TestRunOutcome:
    """Merge per-runner outcomes into one, worst-case-wins on the exit code."""
    ran_any = any(o.ran for _, o in pairs)
    selected: List[str] = []
    out_chunks: List[str] = []
    err_chunks: List[str] = []
    skips: List[str] = []
    returncode = 0
    for name, o in pairs:
        selected.extend(o.tests_selected)
        if o.stdout:
            out_chunks.append(f"== {name} ==\n{o.stdout}")
        if o.stderr:
            err_chunks.append(f"== {name} ==\n{o.stderr}")
        if o.ran and o.returncode != 0 and returncode == 0:
            returncode = o.returncode
        if not o.ran and o.skipped_reason:
            skips.append(f"{name}: {o.skipped_reason}")
    return TestRunOutcome(
        ran=ran_any,
        returncode=returncode,
        stdout="\n\n".join(out_chunks),
        stderr="\n\n".join(err_chunks),
        tests_selected=selected,
        skipped_reason=None if ran_any else ("; ".join(skips) or "no test runner detected"),
    )


def run_project_tests(
    project_root: str, changed_files: Sequence[str], *,
    timeout_seconds: float = 180.0, python_exe: Optional[str] = None,
    runners: Optional[Sequence[TestRunner]] = None,
) -> TestRunOutcome:
    """Detect the project's stack(s) and run each matching runner.

    Outcomes are merged so the caller sees a single pass/fail signal. When
    no runner matches, returns a non-fatal skipped outcome so the agent
    degrades gracefully rather than hard-failing an unknown stack.
    """
    chosen = list(runners) if runners is not None else detect_test_runners(project_root)
    if not chosen:
        return TestRunOutcome(ran=False, skipped_reason="no test runner detected")
    pairs: List["tuple[str, TestRunOutcome]"] = []
    for r in chosen:
        try:
            outcome = r.run(
                project_root, changed_files,
                timeout_seconds=timeout_seconds, python_exe=python_exe,
            )
        except Exception as e:
            logger.warning("test runner %s crashed: %s", r.name, e)
            outcome = TestRunOutcome(
                ran=False, skipped_reason=f"{r.name}: {type(e).__name__}: {e}")
        pairs.append((r.name, outcome))
    return _aggregate_outcomes(pairs)


__all__ = [
    "TestRunner",
    "PytestRunner",
    "NpmRunner",
    "detect_test_runners",
    "run_project_tests",
]
