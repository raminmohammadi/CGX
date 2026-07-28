


"""Classify a failed VERIFY into a repair-friendly token.

The classifier is deterministic and conservative: it returns ``unknown``
whenever the failure does not match a pattern we know how to fix. That
matters because the router only spawns REPAIR when classification is
non-``unknown`` -- a wrong "fixable" verdict would burn the retry
budget on a fix that can't possibly help.

Each classification is paired with:

* a regex/string match against the pytest stdout/stderr captured in
  the ``VERIFY_REPORT.content`` -- cheap to evaluate, no AST walk yet.
* a ``failure_signature`` derived from the same content, used by the
  router as a no-progress guard (if a second VERIFY in the same repair
  chain produces an identical signature, we stop).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from cgx.trace import traced


REPAIR_CLASSIFICATIONS: Tuple[str, ...] = (
    "third_party_import_break",
    "unittest_pytest_mix",
    "missing_module_pythonpath",
    "missing_fixture",
    "empty_test_suite",
    "unknown",
)

RepairClassification = str  # one of REPAIR_CLASSIFICATIONS


# Pytest renders AttributeError tracebacks with the offending
# attribute name in quotes. ``assertLogs`` is the canonical case from
# the screenshot, but the same shape covers every ``self.assert*``
# helper that lives only on ``unittest.TestCase``.
_UNITTEST_HELPER_RE = re.compile(
    r"AttributeError:\s+'[^']+'\s+object has no attribute\s+"
    r"'(assert[A-Za-z]+|assertLogs|assertRaises|assertEqual|"
    r"assertTrue|assertFalse|assertIn|assertIs|assertIsNone|"
    r"assertIsNotNone|assertNotEqual|assertRaisesRegex|"
    r"assertWarns|assertGreater|assertLess|assertAlmostEqual|"
    r"failUnless|fail)'"
)

# ModuleNotFoundError during test collection points at a missing import
# path. When the named module exists on disk inside ``project_root``
# but isn't reachable from pytest, the fix is to extend ``sys.path``
# via a project-root ``conftest.py``; that decision is made later by
# the locator, classify only reports the candidate.
_MODULE_NOT_FOUND_RE = re.compile(
    r"ModuleNotFoundError:\s+No module named\s+"
    r"'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)'"
)

# Pytest emits ``fixture '<name>' not found`` (with a leading ``E`` in
# the captured traceback) whenever a test function declares an argument
# that no @pytest.fixture / conftest in scope provides. The fix is to
# locate the fixture definition elsewhere in the tree and hoist it to a
# conftest.py the tests can see.
_FIXTURE_NOT_FOUND_RE = re.compile(
    r"fixture\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+not found"
)

# Third-party API break: ``ImportError: cannot import name '<sym>' from
# '<pkg>'`` is emitted when the named module exists (the install
# succeeded) but the requested attribute is missing -- the canonical
# shape of a version mismatch between a consumer and one of its peers.
# We pull both the symbol and the package out so the dependency-aware
# proposer can recompute the correct pin via PyPI metadata.
_CANNOT_IMPORT_NAME_RE = re.compile(
    r"ImportError:\s+cannot import name\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+"
    r"from\s+'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)'"
)


# --------------------- registry ---------------------

# Each entry is ``(token, predicate)`` where ``predicate(content)``
# returns ``True`` if the failure text matches the classifier. Order
# matters: the first matching entry wins, so the more-specific patterns
# (e.g. "cannot import name") must come before the broader ones
# (e.g. "ModuleNotFoundError").
_ClassifierFn = Callable[[Dict[str, Any]], bool]

_CLASSIFIER_REGISTRY: Tuple[Tuple[RepairClassification, _ClassifierFn], ...] = (
    ("third_party_import_break",
     lambda c: bool(_CANNOT_IMPORT_NAME_RE.search(_failure_text(c)))),
    ("unittest_pytest_mix",
     lambda c: bool(_UNITTEST_HELPER_RE.search(_failure_text(c)))),
    ("missing_module_pythonpath",
     lambda c: bool(_MODULE_NOT_FOUND_RE.search(_failure_text(c)))),
    ("missing_fixture",
     lambda c: bool(_FIXTURE_NOT_FOUND_RE.search(_failure_text(c)))),
)


@traced("repair.classify")
def classify_verify_report(content: Dict[str, Any]) -> RepairClassification:
    """Map a VERIFY_REPORT content dict to a classification token.

    Only fires on reports where ``ran`` is true and ``outcome`` is one
    of the genuinely-failed tokens. Skipped / pytest_missing reports are
    not auto-repairable -- they're environment problems that
    BOOTSTRAP_ENV (or the user) owns. Walks the
    :data:`_CLASSIFIER_REGISTRY` in declared order and returns the first
    matching token; falls back to ``unknown`` so the executor emits an
    empty plan and the router escalates to ASK_USER.

    ``no_tests_collected`` (pytest exit 5) is a special case: pytest
    found the selected test file(s) but collected zero test functions --
    the canonical symptom of ``def test_*`` nested inside a fixture or
    another function rather than defined at module top level. It has no
    mechanical locator, so it maps to ``empty_test_suite`` which the
    REPAIR executor routes to a re-scaffold. The router only lets a
    ``no_tests_collected`` report reach here when test files were
    actually selected, so this never fires for genuinely test-free
    projects.
    """
    outcome = str(content.get("outcome") or "").strip()
    if outcome == "no_tests_collected":
        return "empty_test_suite"
    if outcome not in ("assertions_failed", "collection_error"):
        return "unknown"
    for token, predicate in _CLASSIFIER_REGISTRY:
        if predicate(content):
            return token
    return "unknown"


def third_party_import_breaks(
        content: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return ``((symbol, package), ...)`` pairs from "cannot import name".

    Used by the dependency-aware proposer to drive the PyPI lookup.
    The package name is the **top-level** distribution (e.g. ``werkzeug``
    for ``werkzeug.urls``) so it can be matched against installed
    packages in the BUILD_REPORT. Order-preserving and de-duplicated.
    """
    blob = _failure_text(content)
    out: List[Tuple[str, str]] = []
    for m in _CANNOT_IMPORT_NAME_RE.finditer(blob):
        symbol = m.group(1)
        pkg = m.group(2).split(".", 1)[0]
        pair = (symbol, pkg)
        if pair not in out:
            out.append(pair)
    return tuple(out)


def missing_module_names(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the dotted module names pytest reported as missing.

    Used by the locator to decide whether the missing modules
    correspond to project files (auto-fixable) or third-party packages
    (BOOTSTRAP_ENV's problem). Order-preserving and de-duplicated.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _MODULE_NOT_FOUND_RE.finditer(blob):
        name = m.group(1)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def missing_fixture_names(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the fixture names pytest reported as not found.

    The locator uses these to scan the project tree for a matching
    ``@pytest.fixture`` definition; if every name resolves to an
    on-disk fixture, the proposer hoists the bodies into a conftest.
    Order-preserving and de-duplicated.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _FIXTURE_NOT_FOUND_RE.finditer(blob):
        name = m.group(1)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def failure_signature(content: Dict[str, Any]) -> str:
    """Return a short hash that identifies "the same failure" across runs.

    The hash combines outcome + returncode + the first matching error
    line (truncated) so two VERIFY artifacts that fail on the same
    underlying error produce the same signature even if pytest's
    wall-clock-dependent output (durations, timestamps) differs.
    """
    outcome = str(content.get("outcome") or "")
    rc = str(content.get("returncode") or "")
    line = _first_error_line(_failure_text(content))[:120]
    raw = f"{outcome}|{rc}|{line}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:16]


def failure_text(content: Dict[str, Any]) -> str:
    """Public accessor for the concatenated VERIFY failure text.

    A thin wrapper over :func:`_failure_text` so callers outside this
    module (e.g. the bounded LLM-repair path in the REPAIR executor) can
    feed the exact same failure blob to a provider without importing a
    private name or re-deriving the concatenation order.
    """
    return _failure_text(content)


# --------------------- helpers ---------------------

def _failure_text(content: Dict[str, Any]) -> str:
    """Concatenate every textual failure source in a VERIFY_REPORT.

    Walks the structured ``failures`` list first (populated from
    ``--junitxml`` by VERIFY) so the classifier sees the
    captured traceback / exception type even when pytest's short stdout
    elided it; then falls back to ``stdout`` + ``stderr`` for runs that
    pre-date the junit pipeline or whose junit file failed to parse.
    Order matters: pytest writes the captured traceback to stdout under
    its default mode, with stderr carrying setup warnings.
    """
    parts: List[str] = []
    failures = content.get("failures")
    if isinstance(failures, list):
        for fail in failures:
            if not isinstance(fail, dict):
                continue
            # Synthesize a canonical ``<Type>: <message>`` line so the
            # existing regexes (which were built against pytest's stdout
            # rendering) match against junit attributes too.
            etype = fail.get("type") or ""
            msg = fail.get("message") or ""
            if isinstance(etype, str) and isinstance(msg, str) and etype:
                parts.append(f"{etype}: {msg}".rstrip())
            elif isinstance(msg, str) and msg:
                parts.append(msg)
            tb = fail.get("traceback")
            if isinstance(tb, str) and tb:
                parts.append(tb)
    for key in ("stdout", "stderr"):
        v = content.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    return "\n".join(parts)


def _first_error_line(blob: str) -> str:
    """Return the first line that looks like an exception or pytest header.

    Pytest exception lines start with ``E   `` (collection) or contain
    ``Error:`` / ``Exception:`` (assertion-style failures). Falls back
    to the first non-empty line so unknown failures still hash to
    something stable rather than the empty string.
    """
    if not blob:
        return ""
    for raw in blob.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("E ") or line.startswith("E\t"):
            return line
        if "Error:" in line or "Exception:" in line:
            return line
    for raw in blob.splitlines():
        line = raw.strip()
        if line:
            return line
    return ""
