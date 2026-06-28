


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
from typing import Any, Dict, List, Tuple


REPAIR_CLASSIFICATIONS: Tuple[str, ...] = (
    "unittest_pytest_mix",
    "missing_module_pythonpath",
    "missing_fixture",
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


def classify_verify_report(content: Dict[str, Any]) -> RepairClassification:
    """Map a VERIFY_REPORT content dict to a classification token.

    Only fires on reports where ``ran`` is true and ``outcome`` is one
    of the genuinely-failed tokens. Skipped / no_tests / pytest_missing
    reports are not auto-repairable in v1 -- they're environment
    problems that BOOTSTRAP_ENV (or the user) owns.
    """
    outcome = str(content.get("outcome") or "").strip()
    if outcome not in ("assertions_failed", "collection_error"):
        return "unknown"
    blob = _failure_text(content)
    if not blob:
        return "unknown"
    if _UNITTEST_HELPER_RE.search(blob):
        return "unittest_pytest_mix"
    if _MODULE_NOT_FOUND_RE.search(blob):
        return "missing_module_pythonpath"
    if _FIXTURE_NOT_FOUND_RE.search(blob):
        return "missing_fixture"
    return "unknown"


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


# --------------------- helpers ---------------------

def _failure_text(content: Dict[str, Any]) -> str:
    """Concatenate stdout + stderr from a VERIFY_REPORT for scanning.

    Order matters: pytest writes the captured traceback to stdout under
    its default mode, with stderr carrying setup warnings. The combined
    blob is what a human reads to triage a failure, so the classifier
    sees the same input.
    """
    parts = []
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
