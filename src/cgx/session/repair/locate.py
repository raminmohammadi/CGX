


"""AST-based locators for repair classifications.

Each ``locate_*`` helper returns a list of typed location records that
the proposer module turns into diffs. Helpers are pure functions over
disk paths so they're trivial to unit-test on a temporary tree.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from cgx.session.repair.classify import (
    missing_fixture_names,
    missing_module_names,
)
from cgx.trace import traced


@dataclass(frozen=True)
class StyleMixLocation:
    """One class that mixes ``unittest`` helpers into pytest-style code.

    ``rel_path`` is project-relative; ``class_name`` is the bare
    identifier of the offending class; ``class_lineno`` points at the
    ``class`` keyword so the diff generator can rewrite the header
    without ambiguity. ``helpers`` is the set of ``self.assert*`` /
    ``self.fail*`` names referenced inside the class body -- carried
    forward for the UI to show "what the fix unlocks".
    """
    rel_path: str
    class_name: str
    class_lineno: int
    helpers: frozenset


@traced("repair.locate")
def locate_unittest_pytest_mix(
    project_root: Path,
    candidate_files: Iterable[str],
) -> List[StyleMixLocation]:
    """Find classes that use ``self.assert*`` without inheriting TestCase.

    Walks each candidate file's AST once. For every ``ClassDef`` whose
    bases do not include a ``TestCase`` reference (bare or
    ``unittest.TestCase``), collects ``self.<helper>`` attribute
    accesses where ``<helper>`` is in :data:`_UNITTEST_HELPERS`. A
    class with one or more such calls becomes a location record.

    Returns the empty list when nothing matches -- callers must treat
    that as "no repair available" rather than "all clear".
    """
    root = Path(project_root).resolve()
    out: List[StyleMixLocation] = []
    for rel in candidate_files:
        rel_clean = str(rel).strip()
        if not rel_clean or not rel_clean.endswith(".py"):
            continue
        abs_path = (root / rel_clean).resolve()
        try:
            abs_path.relative_to(root)
        except ValueError:
            continue
        if not abs_path.is_file():
            continue
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8"),
                             filename=str(abs_path))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if _inherits_testcase(node):
                continue
            helpers = _unittest_helpers_in_class(node)
            if not helpers:
                continue
            out.append(StyleMixLocation(
                rel_path=rel_clean,
                class_name=node.name,
                class_lineno=node.lineno,
                helpers=frozenset(helpers),
            ))
    return out


@dataclass(frozen=True)
class MissingPythonpathLocation:
    """A top-level module pytest can't import that exists on disk.

    ``module_name`` is the dotted name pytest named (e.g. ``app`` or
    ``project.api``); ``top_level`` is the first component that maps to
    a project-root entry; ``resolved_path`` is project-relative and
    points either at a sibling ``.py`` file or at a directory under
    ``project_root``. Only modules that map to a real on-disk entry
    are returned -- third-party packages are BOOTSTRAP_ENV's problem.
    """
    module_name: str
    top_level: str
    resolved_path: str


@traced("repair.locate")
def lint_test_style(
    project_root: Path,
    candidate_files: Iterable[str],
) -> List[Dict[str, Any]]:
    """Run the unittest/pytest-mix locator as a preflight test-style lint.

    Thin wrapper over :func:`locate_unittest_pytest_mix` that returns a
    JSON-ready list of issue dicts (file, class_name, lineno, helpers).
    Called from BOOTSTRAP_ENV so the BUILD_REPORT surfaces "X classes
    mix self.assert* into pytest-style code" before VERIFY runs --
    naming the problem ahead of time even though REPAIR will still
    auto-fix it on the next VERIFY pass.
    """
    issues: List[Dict[str, Any]] = []
    for loc in locate_unittest_pytest_mix(project_root, candidate_files):
        issues.append({
            "kind": "unittest_pytest_mix",
            "file": loc.rel_path,
            "class_name": loc.class_name,
            "lineno": loc.class_lineno,
            "helpers": sorted(loc.helpers),
        })
    return issues


@dataclass(frozen=True)
class MissingFixtureLocation:
    """A pytest fixture name with its source definition + hoist target.

    ``fixture_name`` is the bare name pytest could not resolve;
    ``source_rel_path`` / ``source_lineno`` / ``source_end_lineno``
    point at the ``@pytest.fixture``-decorated function the proposer
    will copy; ``target_rel_path`` is where the proposer should hoist
    it (``tests/conftest.py`` when a ``tests/`` directory exists at
    project root, else ``conftest.py``). All paths are project-relative.
    """
    fixture_name: str
    source_rel_path: str
    source_lineno: int
    source_end_lineno: int
    target_rel_path: str


@traced("repair.locate")
def locate_missing_fixture(
    project_root: Path,
    content: Dict[str, Any],
) -> List[MissingFixtureLocation]:
    """Find @pytest.fixture definitions for the names pytest could not resolve.

    Walks every ``.py`` file under ``project_root`` (skipping ``.venv``
    / dotfile / cache dirs), parses each one, and records the first
    top-level ``@pytest.fixture``-decorated ``FunctionDef`` /
    ``AsyncFunctionDef`` whose name matches a missing fixture. The
    decorator check accepts the bare form (``@pytest.fixture``,
    ``@pytest.fixture()``) and the imported form (``@fixture`` /
    ``@fixture(...)``). Files are visited in sorted order so the
    locator is deterministic across runs.

    Empty list means no on-disk definition matches -- the proposer
    will yield no diffs and the router escalates to ASK_USER.
    """
    root = Path(project_root).resolve()
    wanted = list(missing_fixture_names(content))
    if not wanted:
        return []
    target = _conftest_target(root)
    found: Dict[str, MissingFixtureLocation] = {}
    for abs_path in _iter_python_files(root):
        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8"),
                             filename=str(abs_path))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        try:
            rel = str(abs_path.relative_to(root))
        except ValueError:
            continue
        if rel == target:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in wanted or node.name in found:
                continue
            if not _is_pytest_fixture(node):
                continue
            end = getattr(node, "end_lineno", None) or node.lineno
            deco_start = (node.decorator_list[0].lineno
                          if node.decorator_list else node.lineno)
            found[node.name] = MissingFixtureLocation(
                fixture_name=node.name,
                source_rel_path=rel,
                source_lineno=deco_start,
                source_end_lineno=int(end),
                target_rel_path=target,
            )
    return [found[name] for name in wanted if name in found]


@traced("repair.locate")
def locate_missing_module_pythonpath(
    project_root: Path,
    content: Dict[str, Any],
) -> List[MissingPythonpathLocation]:
    """Find ModuleNotFoundError targets that already live in project_root.

    Walks the names from
    :func:`cgx.session.repair.classify.missing_module_names` and keeps
    only those whose *full* dotted path resolves to a file or package
    inside ``project_root``. Skips ``.venv`` / dotfile dirs so a
    vendored ``.venv/foo`` doesn't masquerade as the project module.

    A missing *leaf* (e.g. ``tests.auth`` where ``tests/`` exists but
    ``tests/auth.py`` does not) is deliberately not resolved here: no
    ``conftest`` sys.path entry can create a module that was never
    authored, so that case is left for the REPAIR regenerate path
    rather than being papered over with a pythonpath patch.
    """
    root = Path(project_root).resolve()
    out: List[MissingPythonpathLocation] = []
    seen_top: Set[str] = set()
    for dotted in missing_module_names(content):
        top = dotted.split(".", 1)[0]
        if not top or top in seen_top:
            continue
        if not _dotted_path_resolves(root, dotted):
            continue
        seen_top.add(top)
        candidates = [root / f"{top}.py", root / top]
        for cand in candidates:
            try:
                cand_resolved = cand.resolve()
                cand_resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if not cand.exists():
                continue
            if cand.is_dir() and not _is_python_dir(cand):
                continue
            rel = cand.name
            out.append(MissingPythonpathLocation(
                module_name=dotted, top_level=top, resolved_path=rel))
            break
    return out


# --------------------- helpers ---------------------

def _dotted_path_resolves(root: Path, dotted: str) -> bool:
    """True when every component of ``dotted`` resolves under ``root``.

    Non-leaf components must be package directories; the leaf may be
    either a ``<name>.py`` module or a package directory. This tells a
    genuine sys.path gap (the whole dotted path exists on disk but isn't
    importable from pytest) apart from a missing leaf module -- an
    authoring error the regenerate path owns, not a pythonpath fix.
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return False
    cur = root
    for i, part in enumerate(parts):
        is_leaf = i == len(parts) - 1
        as_dir = cur / part
        if is_leaf:
            if (cur / f"{part}.py").is_file():
                return True
            return as_dir.is_dir() and _is_python_dir(as_dir)
        if not as_dir.is_dir():
            return False
        cur = as_dir
    return False


def _is_python_dir(path: Path) -> bool:
    """True when ``path`` is a non-hidden directory containing ``.py`` files.

    Mirrors the heuristic the locator uses to decide whether a
    ``ModuleNotFoundError`` target is a real project package rather
    than an arbitrary directory like ``data/`` or ``.venv/``.
    """
    name = path.name
    if name.startswith(".") or name in {"__pycache__", "node_modules"}:
        return False
    try:
        for child in path.iterdir():
            if child.suffix == ".py" or child.name == "__init__.py":
                return True
    except OSError:
        return False
    return False


# Subset of unittest.TestCase helpers that show up in generated tests.
# Conservative on purpose: only names that are unambiguous TestCase
# methods (not e.g. ``assert_called_with`` which is a Mock method).
_UNITTEST_HELPERS: Set[str] = {
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn", "assertGreater", "assertGreaterEqual",
    "assertLess", "assertLessEqual", "assertAlmostEqual",
    "assertNotAlmostEqual", "assertRaises", "assertRaisesRegex",
    "assertWarns", "assertWarnsRegex", "assertLogs", "assertNoLogs",
    "assertCountEqual", "assertDictEqual", "assertListEqual",
    "assertSetEqual", "assertTupleEqual", "assertMultiLineEqual",
    "assertRegex", "assertNotRegex", "fail", "failureException",
}


def _inherits_testcase(node: ast.ClassDef) -> bool:
    """Return True when the class header names ``TestCase`` as a base.

    Accepts the bare form (``class Foo(TestCase):``) and the dotted
    form (``class Foo(unittest.TestCase):``). Does not resolve
    inheritance transitively -- a class inheriting from a custom base
    that itself extends TestCase is treated as not-inheriting, which
    is intentional: we only fire the repair when the textual signal
    is unambiguous.
    """
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "TestCase":
            return True
        if (isinstance(base, ast.Attribute)
                and base.attr == "TestCase"
                and isinstance(base.value, ast.Name)
                and base.value.id == "unittest"):
            return True
    return False


def _unittest_helpers_in_class(node: ast.ClassDef) -> Set[str]:
    """Return ``self.<helper>`` names referenced inside the class body."""
    found: Set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Attribute):
            continue
        if (isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
                and sub.attr in _UNITTEST_HELPERS):
            found.add(sub.attr)
    return found


# Directories the fixture scan must never descend into: vendored
# venvs, build artefacts, caches, and VCS metadata. Keeping the list
# narrow matches the heuristic used by ``_is_python_dir`` above.
_FIXTURE_SCAN_SKIP_DIRS: Set[str] = {
    ".venv", "venv", "__pycache__", "node_modules",
    ".git", ".hg", ".tox", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".eggs",
}


def _iter_python_files(root: Path) -> List[Path]:
    """Return every ``.py`` file under ``root`` in deterministic order.

    Skips hidden directories and the well-known build/venv subtrees
    listed in :data:`_FIXTURE_SCAN_SKIP_DIRS` so the fixture locator
    doesn't pick up an installed pytest plugin's fixture as if it were
    a local definition.
    """
    out: List[Path] = []
    for path in sorted(root.rglob("*.py")):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if any(p.startswith(".") or p in _FIXTURE_SCAN_SKIP_DIRS
               for p in parts[:-1]):
            continue
        out.append(path)
    return out


def _is_pytest_fixture(
    node,  # type: ast.FunctionDef | ast.AsyncFunctionDef
) -> bool:
    """Return True when ``node`` carries a ``@pytest.fixture`` decorator.

    Accepts the bare attribute form (``@pytest.fixture``), the call
    form (``@pytest.fixture(scope="module")``), and the imported-name
    forms (``@fixture`` / ``@fixture(...)``). Other decorators do not
    affect the result so a fixture stacked with ``@pytest.mark.*``
    helpers still matches.
    """
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if (isinstance(target, ast.Attribute)
                and target.attr == "fixture"
                and isinstance(target.value, ast.Name)
                and target.value.id == "pytest"):
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _conftest_target(root: Path) -> str:
    """Return the project-relative conftest path the proposer should write.

    Prefers ``tests/conftest.py`` when a ``tests/`` directory exists at
    the project root (the convention scaffolders emit); otherwise
    falls back to ``conftest.py`` at the root. Either path lets pytest
    discover the fixture before test collection.
    """
    if (root / "tests").is_dir():
        return "tests/conftest.py"
    return "conftest.py"
