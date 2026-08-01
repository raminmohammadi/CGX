


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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cgx.session.budget import LoopBudget
from cgx.session.models import TaskKind, TaskNode
from cgx.session.repair.locate import (
    MissingFixtureLocation,
    MissingPythonpathLocation,
    StyleMixLocation,
)
from cgx.session.repair.pypi_client import PyPIClient
from cgx.trace import traced


@traced("repair.propose")
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


@traced("repair.propose")
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


@traced("repair.propose")
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


# Window we treat as "contemporaneous" when picking a peer version by
# release date. Most upstream / peer projects ship coordinated minor
# bumps within ~two months, and stretching wider risks proposing a peer
# that didn't exist yet when the consumer's release was tagged.
_PEER_RELEASE_WINDOW = timedelta(days=60)


@traced("repair.propose")
def propose_third_party_pin(
    project_root: Path,
    content: Dict[str, Any],
    *,
    pairs: Sequence[Tuple[str, str]],
    installed_packages: Dict[str, str],
    pypi_client: PyPIClient,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Emit a ``requirements.txt`` diff that pins broken third-party peers.

    For each ``(symbol, broken_pkg)`` pair, walk the failure traceback
    to find the *consumer* package (whichever installed distribution's
    ``site-packages`` dir contains the topmost frame outside the test
    code). Then query PyPI for the consumer's ``requires_dist`` and any
    declared upper bound on ``broken_pkg``. When no declared upper
    bound exists, fall back to picking the highest ``broken_pkg``
    version whose ``upload_time`` is within
    :data:`_PEER_RELEASE_WINDOW` of the consumer's release.

    Returns ``(diffs, decisions)`` where ``decisions`` is a list of
    structured ``{symbol, broken_pkg, consumer, reason, pin}`` records
    surfaced on the REPAIR_PLAN so the user can see *why* a version
    was chosen. ``diffs`` is empty when no pair could be resolved.
    """
    if not pairs or not installed_packages:
        return [], []
    decisions: List[Dict[str, Any]] = []
    new_pins: Dict[str, str] = {}
    seen_brokens: set = set()
    for symbol, broken_pkg in pairs:
        broken_key = broken_pkg.lower().replace("_", "-")
        if broken_key in seen_brokens:
            continue
        seen_brokens.add(broken_key)
        if broken_key not in installed_packages:
            decisions.append({
                "symbol": symbol, "broken_pkg": broken_pkg,
                "reason": "broken package not in BUILD_REPORT; skipped",
            })
            continue
        consumer = _detect_consumer(content, broken_key, installed_packages)
        if not consumer:
            decisions.append({
                "symbol": symbol, "broken_pkg": broken_pkg,
                "reason": "consumer package not detected in traceback",
            })
            continue
        consumer_version = installed_packages.get(consumer.lower())
        pin, reason = _resolve_peer_pin(
            pypi_client, consumer, consumer_version or "",
            broken_pkg, broken_key, installed_packages,
        )
        decisions.append({
            "symbol": symbol, "broken_pkg": broken_pkg,
            "consumer": consumer, "consumer_version": consumer_version,
            "reason": reason, "pin": pin,
        })
        if pin:
            new_pins[broken_key] = pin
    if not new_pins:
        return [], decisions
    diffs = _build_requirements_diff(project_root, new_pins)
    return diffs, decisions


def _detect_consumer(content: Dict[str, Any], broken_key: str,
                     installed_packages: Dict[str, str]) -> Optional[str]:
    """Return the installed package whose code triggered the ImportError.

    Scans every ``failures[].traceback`` blob for ``site-packages/<x>/``
    references, picks the first ``<x>`` that's installed and isn't the
    broken package itself. Returns ``None`` when no traceback is
    available or none of the candidates are tracked in BUILD_REPORT.
    """
    failures = content.get("failures") or []
    if not isinstance(failures, list):
        return None
    pattern = re.compile(r"site-packages/([A-Za-z_][A-Za-z0-9_]*)/")
    for fail in failures:
        if not isinstance(fail, dict):
            continue
        tb = fail.get("traceback") or ""
        if not isinstance(tb, str):
            continue
        for m in pattern.finditer(tb):
            name = m.group(1).lower().replace("_", "-")
            if name == broken_key:
                continue
            if name in installed_packages:
                return name
    return None


def _resolve_peer_pin(
    pypi: PyPIClient, consumer: str, consumer_version: str,
    broken_pkg: str, broken_key: str,
    installed_packages: Dict[str, str],
) -> Tuple[Optional[str], str]:
    """Return ``(pin_string, reason)`` for the broken peer.

    Strategy, in priority order:

    1. Look at the consumer's ``info.requires_dist``; if it names the
       peer with any version constraint, reuse that constraint verbatim.
    2. Otherwise, find the highest peer version whose upload time is
       within :data:`_PEER_RELEASE_WINDOW` of the consumer's release.
    3. If neither path produces a candidate, return ``(None, reason)``.
    """
    if not consumer_version:
        return None, f"no installed version of consumer {consumer!r}"
    consumer_meta = pypi.get_release(consumer, consumer_version)
    if consumer_meta:
        constraint = _peer_constraint_from_requires_dist(
            consumer_meta.get("info") or {}, broken_key)
        if constraint:
            return constraint, (
                f"reused declared peer constraint from "
                f"{consumer}=={consumer_version}.requires_dist")
    consumer_release_at = _earliest_upload_time(
        (consumer_meta or {}).get("urls") or [])
    if consumer_release_at is None:
        return None, (f"PyPI metadata for {consumer}=={consumer_version} "
                      "lacked upload timestamps")
    peer_meta = pypi.get_package(broken_pkg)
    if peer_meta is None:
        return None, f"could not fetch PyPI metadata for {broken_pkg!r}"
    best = _pick_contemporary_version(
        peer_meta.get("releases") or {}, consumer_release_at)
    if not best:
        return None, (f"no {broken_pkg} version released within "
                      f"{_PEER_RELEASE_WINDOW.days}d of "
                      f"{consumer}=={consumer_version}")
    return (f"{broken_pkg}=={best}",
            f"highest {broken_pkg} release within "
            f"{_PEER_RELEASE_WINDOW.days}d of "
            f"{consumer}=={consumer_version}")


def _peer_constraint_from_requires_dist(
        info: Dict[str, Any], broken_key: str) -> Optional[str]:
    """Pull a peer pin out of a release's ``requires_dist`` list."""
    reqs = info.get("requires_dist") or []
    if not isinstance(reqs, list):
        return None
    for raw in reqs:
        if not isinstance(raw, str):
            continue
        head = raw.split(";", 1)[0].strip()
        if not head:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9_.\-]*)\s*(.*)$", head)
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        if name != broken_key:
            continue
        rest = m.group(2).strip()
        if not rest:
            continue
        return f"{m.group(1)}{rest}"
    return None


def _earliest_upload_time(urls: Iterable[Any]) -> Optional[datetime]:
    """Return the earliest ``upload_time_iso_8601`` across release files."""
    best: Optional[datetime] = None
    for entry in urls:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("upload_time_iso_8601") or entry.get("upload_time")
        parsed = _parse_pypi_timestamp(ts) if isinstance(ts, str) else None
        if parsed is None:
            continue
        if best is None or parsed < best:
            best = parsed
    return best


def _pick_contemporary_version(
        releases: Dict[str, Any], reference: datetime) -> Optional[str]:
    """Return the highest version whose upload was within the window."""
    candidates: List[Tuple[Tuple[int, ...], str]] = []
    lower = reference - _PEER_RELEASE_WINDOW
    upper = reference + _PEER_RELEASE_WINDOW
    for version, files in releases.items():
        if not isinstance(files, list) or not files:
            continue
        upload = _earliest_upload_time(files)
        if upload is None or not (lower <= upload <= upper):
            continue
        parsed = _parse_version_tuple(version)
        if parsed is None:
            continue
        candidates.append((parsed, version))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _parse_pypi_timestamp(ts: str) -> Optional[datetime]:
    """Parse PyPI's ISO-8601 timestamps (with or without trailing ``Z``)."""
    if not ts:
        return None
    raw = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_version_tuple(version: str) -> Optional[Tuple[int, ...]]:
    """Best-effort numeric-tuple parse for ordering PyPI versions.

    Skips pre/dev/rc releases (anything with a non-numeric component
    after the first non-numeric segment) to keep the proposer
    conservative -- pinning a release candidate would surprise users.
    """
    if not version or any(c in version for c in ("a", "b", "rc", "dev")):
        return None
    parts: List[int] = []
    for chunk in version.split("."):
        if not chunk.isdigit():
            return None
        parts.append(int(chunk))
    return tuple(parts) if parts else None


def _build_requirements_diff(
        project_root: Path,
        new_pins: Dict[str, str]) -> List[Dict[str, str]]:
    """Return a unified diff against ``requirements.txt`` adding pins.

    Existing lines for the same package (case-insensitive) are replaced;
    missing pins are appended. The file is created if it doesn't exist.
    """
    rel_path = "requirements.txt"
    req_path = Path(project_root).resolve() / rel_path
    try:
        original = req_path.read_text(encoding="utf-8") if req_path.exists() else ""
    except (OSError, UnicodeDecodeError):
        original = ""
    lines = original.splitlines(keepends=True)
    remaining = dict(new_pins)
    pkg_line_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.\-]*)")
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = pkg_line_re.match(stripped)
        if not m:
            continue
        key = m.group(1).lower().replace("_", "-")
        if key in remaining:
            replacement = remaining.pop(key) + "\n"
            lines[idx] = replacement
    if remaining:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        for key in sorted(remaining):
            lines.append(remaining[key] + "\n")
    modified = "".join(lines)
    patch = _unified_diff(rel_path, original, modified)
    if not patch:
        return []
    return [{"file": rel_path, "patch": patch}]


@traced("repair.propose")
def propose_regenerate(
    scaffold_task: TaskNode,
    new_constraints: Dict[str, Any],
    *,
    regenerate_files: Optional[Sequence[str]] = None,
    prior_scaffold_artifact_id: Optional[str] = None,
    resume_scaffold_artifact_id: Optional[str] = None,
    prior_failure_signatures: Optional[Sequence[str]] = None,
) -> TaskNode:
    """Return a sibling SCAFFOLD task with ``new_constraints`` folded in.

    Mirrors ``scaffold_task``'s session / parent / description so the
    new SCAFFOLD slots into the same chain the previous one occupied
    after the router marks the abandoned subtree. The original
    ``inputs`` are copied verbatim, then three keys are added or
    extended:

    * ``regenerate_constraints`` -- the running list of structured
      constraint dicts collected across regenerate attempts. Each entry
      typically carries ``{kind, rationale, ...}`` shaped by the
      classifier (e.g. ``third_party_pin_conflict`` with the failing
      pairs and the candidate pins that didn't work). SCAFFOLD's prompt
      builder reads this list to inject prior-constraint warnings.
    * ``regenerate_attempt`` -- incremented monotonically so the router
      can cap the loop via :data:`cgx.session.budget.REGENERATE_BUDGET`.
    * ``regenerated_from_task_id`` -- back-pointer to the SCAFFOLD that
      was abandoned, useful for the UI and for cross-session learning
      (Phase 7).

    When ``regenerate_files`` and ``prior_scaffold_artifact_id`` are both
    supplied (the failed-files router splices), two more keys direct
    SCAFFOLD to a *targeted* regeneration: only the named paths are
    re-generated while every prior-good diff from
    ``prior_scaffold_artifact_id`` is reused verbatim. This keeps the
    blast radius proportional to the failure instead of re-running the
    whole manifest (which also risks re-breaking files that were fine):

    * ``regenerate_files`` -- the concrete paths to re-generate.
    * ``prior_scaffold_artifact_id`` -- the SCAFFOLD_PATCHES artifact
      whose good diffs are reused for every other file.

    Absent those (a whole-tree regenerate, e.g. a REPAIR-classified
    logic failure), any stale targeted markers copied from the prior
    inputs are cleared so the next attempt regenerates the full tree.

    When ``resume_scaffold_artifact_id`` is supplied (the crash-resume
    router path for a SCAFFOLD that died mid-run), it is stamped into
    ``inputs`` so the fresh SCAFFOLD seeds every file the crashed attempt
    already checkpointed and regenerates only the remainder. It is
    orthogonal to the targeted-regenerate markers and cleared when absent
    so a stale pointer cannot resurrect a wrong checkpoint.

    The function is intentionally pure: it does no I/O and returns a
    fresh :class:`TaskNode` the router can wrap in a ``CreateTask``
    alongside the matching ``UpdateTaskStatus(ABANDONED)`` actions for
    the descendants.
    """
    inputs = dict(scaffold_task.inputs)
    prior_constraints = list(inputs.get("regenerate_constraints") or [])
    if isinstance(new_constraints, dict) and new_constraints:
        prior_constraints.append(dict(new_constraints))
    inputs["regenerate_constraints"] = prior_constraints
    inputs["regenerate_attempt"] = (
        LoopBudget.from_inputs(inputs).spend_regenerate().regenerate_attempt)
    inputs["regenerated_from_task_id"] = scaffold_task.task_id
    targeted = [str(p).strip() for p in (regenerate_files or [])
                if str(p).strip()]
    if targeted and prior_scaffold_artifact_id:
        inputs["regenerate_files"] = targeted
        inputs["prior_scaffold_artifact_id"] = str(prior_scaffold_artifact_id)
    else:
        # Whole-tree regenerate: drop any targeted markers copied from the
        # prior inputs so a later attempt cannot silently skip files.
        inputs.pop("regenerate_files", None)
        inputs.pop("prior_scaffold_artifact_id", None)
    if resume_scaffold_artifact_id:
        inputs["resume_scaffold_artifact_id"] = str(resume_scaffold_artifact_id)
    else:
        inputs.pop("resume_scaffold_artifact_id", None)
    # Fold the failed chain's flap ledger into the new SCAFFOLD so the
    # regenerated APPLY -> ... -> VERIFY chain is not amnesiac: a
    # regenerate that reproduces the identical failure signature is
    # stopped by the router's ``budget.seen()`` backstop instead of
    # burning the whole regenerate budget on a fix that cannot work.
    if prior_failure_signatures:
        merged = [str(s) for s in
                  (inputs.get("prior_failure_signatures") or [])]
        for sig in prior_failure_signatures:
            s = str(sig).strip()
            if s and s not in merged:
                merged.append(s)
        inputs["prior_failure_signatures"] = merged
    return TaskNode.new(
        session_id=scaffold_task.session_id,
        kind=TaskKind.SCAFFOLD,
        name=f"{scaffold_task.name} (regenerated)",
        description=scaffold_task.description,
        parent_task_id=scaffold_task.parent_task_id,
        inputs=inputs,
    )


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
