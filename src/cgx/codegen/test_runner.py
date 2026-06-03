

"""Run impacted tests in a sandbox copy of the project.

We do NOT execute tests in the user's working tree. Instead:

1. Copy the project into a temporary directory.
2. Apply the proposed patches to the copy.
3. Locate test files that look impacted (heuristic: tests whose module names
   appear in any changed file's imports, or tests under ``tests/`` matching
   the changed module's stem).
4. Run ``pytest -q --no-header`` on those tests with a hard timeout.

This module deliberately uses the standard ``subprocess`` runner so we do not
pollute the host interpreter. If ``pytest`` is not installed in the sandbox's
runtime, the runner reports that cleanly rather than raising.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from cgx.codegen.diff_apply import PatchResult

logger = logging.getLogger(__name__)


@dataclass
class TestRunOutcome:
    """Result of running impacted tests against a patched sandbox."""
    ran: bool
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    tests_selected: List[str] = field(default_factory=list)
    sandbox_dir: Optional[str] = None
    skipped_reason: Optional[str] = None


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)


def _module_candidates(rel_path: str) -> List[str]:
    """Return plausible Python module names for ``rel_path``."""
    p = rel_path.replace(os.sep, "/")
    if not p.endswith(".py"):
        return []
    p = p[: -len(".py")]
    parts = [seg for seg in p.split("/") if seg and seg != "__init__"]
    if not parts:
        return []
    # Try the full dotted path AND just the leaf name, since many tests do
    # ``from module_name import X`` regardless of layout.
    cands = [".".join(parts), parts[-1]]
    # Strip a leading 'src.' if present.
    if cands[0].startswith("src."):
        cands.append(cands[0][len("src."):])
    return list(dict.fromkeys(cands))


_SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules",
              "dist", "build", ".tox", ".mypy_cache", ".ruff_cache"}


def discover_all_tests(
    project_root: str,
    *,
    tests_dirs: Sequence[str] = ("tests", "test"),
) -> List[str]:
    """Return absolute paths to every ``test_*.py`` in the project.

    First looks in named ``tests_dirs`` at the root; if none are found,
    falls back to a full recursive scan (skipping venvs / caches) so
    that projects that keep tests under ``src/tests/`` or elsewhere are
    still picked up.
    """
    root = Path(project_root).resolve()
    selected: List[str] = []
    for td in tests_dirs:
        tp = root / td
        if not tp.is_dir():
            continue
        for f in tp.rglob("test_*.py"):
            if f.is_file():
                selected.append(str(f))
    if not selected:
        for f in root.rglob("test_*.py"):
            if f.is_file() and not any(p in _SKIP_DIRS for p in f.parts):
                selected.append(str(f))
    return list(dict.fromkeys(selected))


def find_impacted_tests(
    project_root: str,
    changed_files: Sequence[str],
    *,
    tests_dirs: Sequence[str] = ("tests", "test"),
) -> List[str]:
    """Heuristically locate test files impacted by ``changed_files``.

    Returns absolute paths to test files (relative to ``project_root``) that
    either (a) live in ``tests_dirs`` and import a candidate module name, or
    (b) sit next to a changed module as ``test_<stem>.py``.
    """
    root = Path(project_root).resolve()
    selected: List[str] = []
    candidates: List[str] = []
    for ch in changed_files:
        candidates.extend(_module_candidates(ch))
    candidates = list(dict.fromkeys(candidates))

    # (a) Scan all test files: named dirs first, then full recursive fallback
    candidate_test_files: List[Path] = []
    for td in tests_dirs:
        tp = root / td
        if tp.is_dir():
            candidate_test_files.extend(tp.rglob("test_*.py"))
    # Always do a full recursive scan so tests under src/tests/ etc. are found
    for f in root.rglob("test_*.py"):
        if f.is_file() and not any(p in _SKIP_DIRS for p in f.parts):
            candidate_test_files.append(f)

    for f in list(dict.fromkeys(candidate_test_files)):
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        imported: List[str] = []
        for m in _IMPORT_RE.finditer(src):
            imported.append(m.group(1) or m.group(2) or "")
        if any(c and any(c == imp or imp.endswith("." + c) or imp.startswith(c + ".")
                         for imp in imported) for c in candidates):
            selected.append(str(f))

    # (b) sibling test_<stem>.py
    for ch in changed_files:
        if not ch.endswith(".py"):
            continue
        stem = Path(ch).stem
        sibling = root / Path(ch).parent / f"test_{stem}.py"
        if sibling.is_file():
            selected.append(str(sibling))
    return list(dict.fromkeys(selected))


def _project_python_exe(project_root: Path) -> str:
    """Return the venv python if one exists inside the project, else sys.executable."""
    for candidate in (
        project_root / ".venv" / "bin" / "python",
        project_root / "venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def ensure_project_venv(
    project_root: str,
    *,
    timeout: float = 300.0,
) -> str:
    """Ensure ``project_root`` has a ``.venv`` with pytest + requirements installed.

    Idempotent: when ``.venv`` or ``venv`` already exists, this still runs
    ``pip install -r requirements.txt`` so newly-declared dependencies are
    picked up; pip is a no-op when everything is already up to date. When
    no venv exists yet, one is created at ``.venv`` first.

    Returns the path to the venv's python interpreter; falls back to
    ``sys.executable`` if creation fails (offline, missing ``venv`` module,
    …) so the caller can still attempt to run tests.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        return sys.executable

    existing = _project_python_exe(root)
    if existing != sys.executable:
        python_exe = existing
    else:
        venv_dir = root / ".venv"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                cwd=str(root), capture_output=True, timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "codegen.test_runner: venv creation raised %s; "
                "falling back to sys.executable", exc,
            )
            return sys.executable
        candidate = venv_dir / "bin" / "python"
        if result.returncode != 0 or not candidate.is_file():
            logger.warning(
                "codegen.test_runner: venv creation failed (rc=%d); "
                "falling back to sys.executable", result.returncode,
            )
            return sys.executable
        python_exe = str(candidate)
        logger.info("codegen.test_runner: created project venv at %s", venv_dir)

    pip_base = [python_exe, "-m", "pip", "install", "--quiet", "--no-input"]
    try:
        subprocess.run(pip_base + ["pytest"], cwd=str(root),
                       capture_output=True, timeout=timeout)
    except Exception as exc:
        logger.debug("codegen.test_runner: pytest install raised %s", exc)
    req_path = root / "requirements.txt"
    if req_path.is_file():
        logger.info(
            "codegen.test_runner: installing requirements.txt into project venv"
        )
        try:
            proc = subprocess.run(
                pip_base + ["-r", str(req_path)],
                cwd=str(root), capture_output=True, timeout=timeout,
            )
            if proc.returncode != 0:
                logger.warning(
                    "codegen.test_runner: pip install -r requirements.txt "
                    "failed (rc=%d): %s",
                    proc.returncode,
                    (proc.stderr or b"").decode("utf-8", "ignore")[:300]
                    if isinstance(proc.stderr, bytes) else (proc.stderr or "")[:300],
                )
        except Exception as exc:
            logger.warning(
                "codegen.test_runner: pip install -r requirements.txt raised %s",
                exc,
            )
    return python_exe


def _pytest_env(project_root: Path) -> Dict[str, str]:
    """Return ``os.environ`` plus a ``PYTHONPATH`` that includes ``project_root``.

    Freshly-scaffolded projects often lay code out as ``backend/`` + ``tests/``
    with no ``conftest.py``, ``pyproject.toml`` or ``setup.py`` declaring the
    package roots — so pytest's automatic ``rootdir`` insertion isn't enough
    for first-party imports like ``from backend.calculator import …``. We
    prepend the project root (and ``project_root/src`` when present) to any
    existing ``PYTHONPATH`` so those imports resolve regardless of layout.
    """
    env = dict(os.environ)
    parts: List[str] = [str(project_root)]
    src_dir = project_root / "src"
    if src_dir.is_dir():
        parts.append(str(src_dir))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _ensure_sandbox_venv(sandbox: Path, setup_timeout: float = 120.0) -> str:
    """Create a venv in *sandbox*, install deps, and return the python executable path.

    Falls back to sys.executable if venv creation fails.
    """
    venv_dir = sandbox / ".venv"
    python_exe = str(venv_dir / "bin" / "python")

    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=str(sandbox), capture_output=True, timeout=setup_timeout,
    )
    if result.returncode != 0 or not Path(python_exe).is_file():
        return sys.executable

    pip_base = [python_exe, "-m", "pip", "install", "--quiet"]

    # Always ensure pytest is available.
    subprocess.run(pip_base + ["pytest"], cwd=str(sandbox),
                   capture_output=True, timeout=setup_timeout)

    # Install project requirements.
    for req in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
        req_path = sandbox / req
        if req_path.is_file():
            subprocess.run(pip_base + ["-r", str(req_path)], cwd=str(sandbox),
                           capture_output=True, timeout=setup_timeout)

    # Editable install if the project is a package.
    if (sandbox / "pyproject.toml").is_file() or (sandbox / "setup.py").is_file():
        subprocess.run(pip_base + ["-e", "."], cwd=str(sandbox),
                       capture_output=True, timeout=setup_timeout)

    return python_exe


def _materialize_patches(sandbox: Path, results: Sequence[PatchResult]) -> List[str]:
    written: List[str] = []
    for r in results:
        if not r.ok or r.new_content is None:
            continue
        dest = sandbox / r.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(r.new_content)
        written.append(str(dest))
    return written


def run_impacted_tests(
    project_root: str,
    results: Sequence[PatchResult],
    *,
    timeout_seconds: float = 120.0,
    extra_pytest_args: Iterable[str] = ("-q", "--no-header"),
    copy_filter: Optional[Iterable[str]] = None,
) -> TestRunOutcome:
    """Copy the project, apply patches, and run impacted tests under pytest."""
    logger.info("codegen.test_runner: run_impacted_tests root=%s patches=%d timeout=%.0fs",
                project_root, len(list(results)), timeout_seconds)
    src = Path(project_root).resolve()
    if not src.is_dir():
        logger.warning("codegen.test_runner: project_root not a directory: %s", src)
        return TestRunOutcome(ran=False, skipped_reason=f"project_root not a directory: {src}")

    tmp = Path(tempfile.mkdtemp(prefix="cgx_sandbox_"))
    try:
        ignore = shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            "*.pyc", ".mypy_cache", ".ruff_cache", "cgx_index", "dist", "build",
        )
        shutil.copytree(src, tmp / src.name, ignore=ignore, symlinks=False)
        sandbox = tmp / src.name
        _materialize_patches(sandbox, results)
        python_exe = _ensure_sandbox_venv(sandbox, setup_timeout=min(timeout_seconds, 120.0))
        changed = [r.path for r in results if r.ok]
        tests = find_impacted_tests(str(sandbox), changed)
        if not tests:
            return TestRunOutcome(
                ran=False, sandbox_dir=str(sandbox),
                skipped_reason="no impacted tests located",
            )
        cmd = [python_exe, "-m", "pytest", *list(extra_pytest_args), *tests]
        try:
            proc = subprocess.run(
                cmd, cwd=sandbox, capture_output=True, text=True,
                timeout=timeout_seconds, env=_pytest_env(sandbox),
            )
        except FileNotFoundError:
            return TestRunOutcome(ran=False, sandbox_dir=str(sandbox), skipped_reason="pytest not installed")
        except subprocess.TimeoutExpired as e:
            return TestRunOutcome(
                ran=True, returncode=124,
                stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[timeout]",
                tests_selected=tests, sandbox_dir=str(sandbox),
            )
        logger.info("codegen.test_runner: pytest done rc=%d tests=%d",
                    proc.returncode, len(tests))
        return TestRunOutcome(
            ran=True, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
            tests_selected=tests, sandbox_dir=str(sandbox),
        )
    except Exception as e:
        logger.warning("codegen.test_runner: sandbox run failed: %s: %s",
                       type(e).__name__, e)
        return TestRunOutcome(ran=False, skipped_reason=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_pytest_paths(
    project_root: str,
    test_paths: Sequence[str],
    *,
    timeout_seconds: float = 180.0,
    extra_pytest_args: Iterable[str] = ("-q", "--no-header"),
) -> TestRunOutcome:
    """Run pytest on an explicit list of test files against ``project_root``."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        return TestRunOutcome(ran=False, skipped_reason=f"project_root not a directory: {root}")
    tests = list(test_paths)
    if not tests:
        return TestRunOutcome(ran=False, skipped_reason="no tests located")
    python_exe = _project_python_exe(root)
    cmd = [python_exe, "-m", "pytest", *list(extra_pytest_args), *tests]
    try:
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=timeout_seconds,
            env=_pytest_env(root),
        )
    except FileNotFoundError:
        return TestRunOutcome(ran=False, skipped_reason="pytest not installed")
    except subprocess.TimeoutExpired as e:
        return TestRunOutcome(
            ran=True, returncode=124,
            stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[timeout]",
            tests_selected=tests,
        )
    return TestRunOutcome(
        ran=True, returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr, tests_selected=tests,
    )


def run_tests_on_disk(
    project_root: str,
    changed_files: Sequence[str],
    *,
    timeout_seconds: float = 180.0,
    extra_pytest_args: Iterable[str] = ("-q", "--no-header"),
) -> TestRunOutcome:
    """Run impacted tests directly against ``project_root`` (no sandbox copy).

    Used by the agent's ``verify`` capability after a real-disk write so
    the user gets feedback grounded in the actual working tree.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        return TestRunOutcome(ran=False, skipped_reason=f"project_root not a directory: {root}")
    tests = find_impacted_tests(str(root), list(changed_files))
    if not tests:
        # Fall back to all discovered tests so a freshly scaffolded project
        # with tests under src/tests/ or similar is still exercised.
        tests = discover_all_tests(str(root))
    if not tests:
        return TestRunOutcome(ran=False, skipped_reason="no tests located")
    return run_pytest_paths(
        str(root), tests,
        timeout_seconds=timeout_seconds, extra_pytest_args=extra_pytest_args,
    )
