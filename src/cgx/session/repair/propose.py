


"""Per-classification diff generators.

Each ``propose_*`` helper returns a list of
``{"file": rel_path, "patch": unified_diff_text}`` entries shaped for
:func:`cgx.codegen.disk_apply.apply_diffs_to_disk`. The returned diffs
are kept as small as possible: one hunk per offending class header,
plus at most one hunk per file to add ``import unittest``. That keeps
the blast radius proportional to the failure and makes the diff easy
to review in the UI.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional

from cgx.session.repair.locate import (
    MissingFixtureLocation,
    MissingPythonpathLocation,
    StyleMixLocation,
)


def propose_unittest_pytest_mix(
    project_root: Path,
    locations: List[StyleMixLocation],
) -> List[Dict[str, str]]:
    """Generate diffs that add ``unittest.TestCase`` inheritance.

    Strategy: for each offending class, rewrite the ``class Foo:`` /
    ``class Foo(...):`` header to inherit ``unittest.TestCase``. If
    the source file does not already import ``unittest``, insert
    ``import unittest`` after the leading docstring + future imports.
    Existing bases are preserved (``class Foo(Bar):`` becomes
    ``class Foo(Bar, unittest.TestCase):``).

    Returns an empty list if no on-disk file actually changed -- the
    caller treats that as "nothing to repair" and escalates.
    """
    root = Path(project_root).resolve()
    by_file: Dict[str, List[StyleMixLocation]] = {}
    for loc in locations:
        by_file.setdefault(loc.rel_path, []).append(loc)

    diffs: List[Dict[str, str]] = []
    for rel_path, file_locs in by_file.items():
        abs_path = (root / rel_path).resolve()
        try:
            original = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        modified = _rewrite_file(original, file_locs)
        if modified == original:
            continue
        patch = _unified_diff(rel_path, original, modified)
        if patch:
            diffs.append({"file": rel_path, "patch": patch})
    return diffs


# Marker line written into the conftest body so we can detect a
# previously-applied fix and refuse to write it twice on a retry.
_PYTHONPATH_MARKER = "# cgx-repair: missing_module_pythonpath"

_CONFTEST_SNIPPET = (
    _PYTHONPATH_MARKER + "\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "_HERE = Path(__file__).resolve().parent\n"
    "if str(_HERE) not in sys.path:\n"
    "    sys.path.insert(0, str(_HERE))\n"
)


# Marker line written into the conftest body so a re-run of the
# proposer (e.g. when pytest still emits the same error because the
# user's fixture body raises) doesn't keep prepending duplicate hoists.
_FIXTURE_MARKER_PREFIX = "# cgx-repair: missing_fixture"


def propose_missing_fixture(
    project_root: Path,
    locations: List[MissingFixtureLocation],
) -> List[Dict[str, str]]:
    """Hoist fixture definitions into the appropriate conftest.py.

    For each located fixture, copies the source span (decorators +
    def + body) verbatim into the target conftest (``tests/conftest.py``
    or root ``conftest.py`` as decided by the locator). Adds
    ``import pytest`` if missing. Each hoisted fixture is wrapped in a
    marker comment (``# cgx-repair: missing_fixture <name>``) so a
    second run is a no-op for fixtures already moved.

    Returns an empty list when no fixture would actually be hoisted --
    the router treats that as "no progress" and escalates.
    """
    if not locations:
        return []
    root = Path(project_root).resolve()
    by_target: Dict[str, List[MissingFixtureLocation]] = {}
    for loc in locations:
        by_target.setdefault(loc.target_rel_path, []).append(loc)

    diffs: List[Dict[str, str]] = []
    for rel_path, file_locs in by_target.items():
        target_abs = (root / rel_path).resolve()
        try:
            target_abs.relative_to(root)
        except ValueError:
            continue
        original = (target_abs.read_text(encoding="utf-8")
                    if target_abs.exists() else "")
        modified = _hoist_fixtures(root, original, file_locs)
        if modified == original:
            continue
        patch = _unified_diff(rel_path, original, modified)
        if patch:
            diffs.append({"file": rel_path, "patch": patch})
    return diffs


def _hoist_fixtures(
    root: Path,
    original: str,
    locations: List[MissingFixtureLocation],
) -> str:
    """Return ``original`` with fixture defs appended (deduped by marker)."""
    appended: List[str] = []
    needs_pytest_import = not _has_pytest_import(original)
    for loc in locations:
        marker = f"{_FIXTURE_MARKER_PREFIX} {loc.fixture_name}"
        if marker in original or any(marker in chunk for chunk in appended):
            continue
        snippet = _extract_fixture_source(root, loc)
        if not snippet:
            continue
        appended.append(f"\n\n{marker}\n{snippet.rstrip()}\n")
    if not appended:
        return original
    base = original
    if needs_pytest_import:
        prefix = "import pytest\n"
        if base and not base.endswith("\n"):
            base += "\n"
        base = prefix + base
    if base and not base.endswith("\n"):
        base += "\n"
    return base + "".join(appended)


def _extract_fixture_source(
    root: Path,
    loc: MissingFixtureLocation,
) -> str:
    """Return the verbatim source lines for the located fixture def."""
    abs_path = (root / loc.source_rel_path).resolve()
    try:
        text = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    lines = text.splitlines(keepends=True)
    start = max(0, loc.source_lineno - 1)
    end = min(len(lines), loc.source_end_lineno)
    if start >= end:
        return ""
    return "".join(lines[start:end])


def _has_pytest_import(source: str) -> bool:
    """Return True when ``pytest`` is already imported at module level."""
    for raw in source.splitlines():
        line = raw.strip()
        if line == "import pytest":
            return True
        if line.startswith("import pytest ") or line.startswith("import pytest\t"):
            return True
        if line.startswith("from pytest "):
            return True
    return False


def propose_missing_module_pythonpath(
    project_root: Path,
    locations: List[MissingPythonpathLocation],
) -> List[Dict[str, str]]:
    """Generate a diff that adds the project root to ``sys.path``.

    The fix is concentrated in one file: ``<project_root>/conftest.py``.
    When the file does not exist, the diff creates it. When it exists
    and does not already carry the repair marker, the snippet is
    prepended. When it exists and the marker is present, returns an
    empty list -- the repair was already attempted and re-running it
    would be a no-op (the router treats that as "no progress" and
    escalates).
    """
    if not locations:
        return []
    root = Path(project_root).resolve()
    conftest = root / "conftest.py"
    rel_path = "conftest.py"
    if conftest.exists():
        try:
            original = conftest.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        if _PYTHONPATH_MARKER in original:
            return []
        modified = _CONFTEST_SNIPPET + "\n" + original
    else:
        original = ""
        modified = _CONFTEST_SNIPPET
    patch = _unified_diff(rel_path, original, modified)
    if not patch:
        return []
    return [{"file": rel_path, "patch": patch}]


# --------------------- helpers ---------------------

# ``class Foo:`` or ``class Foo(Bar, Baz):`` -- captures the indent,
# the name, the optional base list, and the trailing colon (with any
# inline comment / trailing whitespace). Multi-line class headers
# (PEP-8 hanging-indent style) are deliberately not handled: they're
# vanishingly rare in scaffold output and bailing out is safer than
# producing a wrong rewrite.
_CLASS_HEADER_RE = re.compile(
    r"^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<bases>[^)]*)\))?\s*:\s*(?P<trail>#.*)?$"
)


def _rewrite_file(source: str, locations: List[StyleMixLocation]) -> str:
    """Apply every per-class rewrite then add ``import unittest`` if needed."""
    lines = source.splitlines(keepends=True)
    wanted = {loc.class_name for loc in locations}
    rewrote_any = False
    for idx, line in enumerate(lines):
        match = _CLASS_HEADER_RE.match(line.rstrip("\r\n"))
        if not match or match.group("name") not in wanted:
            continue
        new_line = _rewrite_header(line, match)
        if new_line != line:
            lines[idx] = new_line
            rewrote_any = True
    if not rewrote_any:
        return source
    out = "".join(lines)
    if not _has_unittest_import(out):
        out = _insert_unittest_import(out)
    return out


def _rewrite_header(line: str, match: "re.Match[str]") -> str:
    """Return ``line`` with ``unittest.TestCase`` appended to the bases."""
    indent = match.group("indent")
    name = match.group("name")
    bases_raw = (match.group("bases") or "").strip()
    trail = match.group("trail") or ""
    if bases_raw:
        bases = [b.strip() for b in bases_raw.split(",") if b.strip()]
        if "unittest.TestCase" in bases or "TestCase" in bases:
            return line
        bases.append("unittest.TestCase")
        new_bases = ", ".join(bases)
    else:
        new_bases = "unittest.TestCase"
    suffix = f"  {trail}" if trail else ""
    newline = "\n" if line.endswith("\n") else ""
    return f"{indent}class {name}({new_bases}):{suffix}{newline}"


def _has_unittest_import(source: str) -> bool:
    """Return True when ``import unittest`` is already present at module level."""
    for raw in source.splitlines():
        line = raw.strip()
        if line == "import unittest":
            return True
        if line.startswith("import unittest ") or line.startswith("import unittest\t"):
            return True
    return False


def _insert_unittest_import(source: str) -> str:
    """Insert ``import unittest`` after the leading docstring/future imports.

    Conservative placement: scan past the module docstring (if any) and
    any ``from __future__ import ...`` lines, then insert. This keeps
    the new line out of the way of automated tooling that special-cases
    those leading constructs.
    """
    lines = source.splitlines(keepends=True)
    insert_at = _find_import_insertion_point(lines)
    new = "import unittest\n"
    if insert_at < len(lines) and lines[insert_at].strip():
        new += "\n"
    lines.insert(insert_at, new)
    return "".join(lines)


def _find_import_insertion_point(lines: List[str]) -> int:
    """Return the line index after docstring + ``__future__`` imports."""
    i = 0
    n = len(lines)
    # Skip leading blank lines and comments.
    while i < n and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    # Skip a module docstring (single- or triple-quoted).
    if i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            # Single-line docstring closing on the same line.
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                i += 1
            else:
                i += 1
                while i < n and quote not in lines[i]:
                    i += 1
                if i < n:
                    i += 1
    # Skip __future__ imports.
    while i < n and lines[i].lstrip().startswith("from __future__"):
        i += 1
    return i


def _unified_diff(rel_path: str, before: str, after: str) -> Optional[str]:
    """Return a git-style unified diff (or ``None`` if identical)."""
    if before == after:
        return None
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=3,
    )
    return "".join(diff) or None
