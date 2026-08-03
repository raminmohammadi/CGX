


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
    "circular_import",
    "third_party_import_break",
    "first_party_symbol_mismatch",
    "relative_import_error",
    "unittest_pytest_mix",
    "missing_dependency",
    "missing_module_pythonpath",
    "missing_fixture",
    "empty_test_suite",
    "undefined_name",
    "collection_error",
    "assertion_drift",
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

# A library named the exact pip package it needs: optional-extra guards
# like starlette's ``testclient`` raise ``RuntimeError("The
# starlette.testclient module requires the httpx package to be
# installed...")`` when the transitive extra is absent. No first-party
# file imports the package directly, so BOOTSTRAP_ENV's file-scan
# preflight never installs it -- and regenerating source can never fix
# it. Matched *before* ``missing_module_pythonpath``: the same failure
# text usually carries the guard's internal ModuleNotFoundError, which
# would otherwise misroute the repair to a source regenerate.
_REQUIRES_PACKAGE_RE = re.compile(
    r"requires the\s+([A-Za-z0-9][A-Za-z0-9._-]*)\s+package to be installed"
)

# Pytest emits ``fixture '<name>' not found`` (with a leading ``E`` in
# the captured traceback) whenever a test function declares an argument
# that no @pytest.fixture / conftest in scope provides. The fix is to
# locate the fixture definition elsewhere in the tree and hoist it to a
# conftest.py the tests can see.
_FIXTURE_NOT_FOUND_RE = re.compile(
    r"fixture\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+not found"
)

# Names pytest reports as a "missing fixture" that no fixture can ever
# supply. ``self``/``cls`` mean a test method was collected outside a
# collected class (a helper class pytest picked up, or a ``Test`` class
# with an ``__init__``); ``request`` is a pytest builtin, so its absence
# means the plugin machinery is broken, not the tree. Treating any of
# them as a fixture sends the loop hunting a definition that cannot
# exist -- observed live, where a whole-tree regenerate was ordered to
# create a fixture named ``self``.
_NON_FIXTURE_NAMES = frozenset({"self", "cls", "request"})

# Traceback frame shapes that name a source file + line. Pytest renders
# its own frames as ``path/to/file.py:123: in func`` (and a trailing
# ``path/to/file.py:123: SomeError`` summary), while a captured Python
# traceback uses ``File "path/to/file.py", line 123``. Both let us
# localize the repair to the files the failure actually flowed through
# instead of only the files APPLY happened to write. Character classes
# are spelled out (no ``\d``/``\w``) for cross-engine portability.
_TB_PYTEST_FRAME_RE = re.compile(
    r"([A-Za-z0-9_./\\-]+\.py):[0-9]+")
_TB_FILE_LINE_RE = re.compile(
    r'File "([A-Za-z0-9_./\\-]+\.py)", line [0-9]+')

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

# A first-party import cycle. Python names the module it could not finish
# initialising in one of two shapes -- ``ImportError: cannot import name
# 'x' from partially initialized module 'm'`` or ``AttributeError:
# partially initialized module 'm' has no attribute 'x'`` -- usually
# suffixed with ``(most likely due to a circular import)``. Matched before
# every other pattern: no single-file patch can decide which import to
# break or where the shared symbols belong, so REPAIR re-authors the
# offending module(s) via a regenerate constraint (like
# ``relative_import_error``) instead of the bounded LLM patch, which a
# live run showed produces a no-op diff and burns the repair budget.
_PARTIALLY_INITIALIZED_MODULE_RE = re.compile(
    r"partially initialized module\s+"
    r"'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)'"
)
_CIRCULAR_IMPORT_RE = re.compile(
    r"partially initialized module|"
    r"most likely due to a circular import"
)

# A relative import that resolves above the package root -- ``from ..x
# import y`` in a module that is not deep enough, or a first-party module
# run without its package context. Python renders this as
# ``ImportError: attempted relative import beyond top-level package`` (or
# ``with no known parent package``). There is no mechanical patch: the
# scaffold authored an import that cannot resolve, so REPAIR re-authors the
# offending module(s) with the failure folded in as a regenerate constraint.
_RELATIVE_IMPORT_RE = re.compile(
    r"attempted relative import (?:beyond top-level package|"
    r"with no known parent package)"
)

# A bundler (Vite/Rollup) that cannot find the file it was told to start
# from: ``[UNRESOLVED_ENTRY] Cannot resolve entry module index.html`` /
# ``Could not resolve entry module "src/main.jsx"``. Unlike every other
# build error this one is not a defect *in* a generated file -- the file
# is absent -- so the regenerate loop must add it, not re-author around
# it.
_UNRESOLVED_ENTRY_RE = re.compile(
    r"(?:annot|ould not) resolve entry module\s+[\"']?"
    r"([A-Za-z0-9_.@/\\-]+?)[\"']?[.\s]*$",
    re.MULTILINE,
)

# A bundler (Vite/Rollup) that resolved its entry but could not resolve a
# relative import *inside* a generated file: ``Could not resolve
# "./index.css" from "src/main.jsx"``. Unlike the entry-module miss above,
# the offending file exists -- it just references a sibling that does not --
# so the fix is a targeted re-author of the importer(s), not a manifest
# extension. Group 1 is the unresolved specifier, group 2 the importer file.
_UNRESOLVED_IMPORT_RE = re.compile(
    r"(?:annot|ould not) resolve\s+[\"']([^\"']+)[\"']\s+from\s+"
    r"[\"']([^\"']+)[\"']",
)

# A name a generated module uses but never binds -- ``class
# Operation(str, enum.Enum)`` with no ``import enum``, a constant
# referenced in an f-string that was never assigned. The file parses,
# every import it *does* declare resolves, and it still dies the moment
# anything touches it: pytest aborts the whole run at collection
# (``NameError: name 'enum' is not defined``), often via a conftest
# chain, so no individual test is ever reported. There is no mechanical
# patch -- the missing binding could be an import, an assignment, or a
# definition the author forgot -- so this routes to a regenerate with
# the unbound names folded in as a constraint.
_UNDEFINED_NAME_RE = re.compile(
    r"NameError:\s+name\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+is not defined"
)


# --------------------- registry ---------------------

# Each entry is ``(token, predicate)`` where ``predicate(content)``
# returns ``True`` if the failure text matches the classifier. Order
# matters: the first matching entry wins, so the more-specific patterns
# (e.g. "cannot import name") must come before the broader ones
# (e.g. "ModuleNotFoundError").
_ClassifierFn = Callable[[Dict[str, Any]], bool]

_CLASSIFIER_REGISTRY: Tuple[Tuple[RepairClassification, _ClassifierFn], ...] = (
    ("circular_import",
     lambda c: bool(_CIRCULAR_IMPORT_RE.search(_failure_text(c)))),
    ("third_party_import_break",
     lambda c: bool(_CANNOT_IMPORT_NAME_RE.search(_failure_text(c)))),
    ("relative_import_error",
     lambda c: bool(_RELATIVE_IMPORT_RE.search(_failure_text(c)))),
    ("unittest_pytest_mix",
     lambda c: bool(_UNITTEST_HELPER_RE.search(_failure_text(c)))),
    ("missing_dependency",
     lambda c: bool(_REQUIRES_PACKAGE_RE.search(_failure_text(c)))),
    ("missing_module_pythonpath",
     lambda c: bool(_MODULE_NOT_FOUND_RE.search(_failure_text(c)))),
    ("missing_fixture",
     lambda c: bool(missing_fixture_names(c))),
    # Last: an undefined name is an authoring defect with no mechanical
    # locator, so every classification that *does* have one is preferred
    # when a run surfaces both.
    ("undefined_name",
     lambda c: bool(_UNDEFINED_NAME_RE.search(_failure_text(c)))),
)


@traced("repair.classify")
def classify_verify_report(content: Dict[str, Any]) -> RepairClassification:
    """Map a VERIFY_REPORT content dict to a classification token.

    Only fires on reports where ``ran`` is true and ``outcome`` is one
    of the genuinely-failed tokens. Skipped / pytest_missing reports are
    not auto-repairable -- they're environment problems that
    BOOTSTRAP_ENV (or the user) owns. Walks the
    :data:`_CLASSIFIER_REGISTRY` in declared order and returns the first
    matching token. A genuinely-failed report that no classifier matched
    falls back to ``assertion_drift`` (a plain assertion / status / message
    mismatch the targeted repair path owns) or ``collection_error`` (pytest
    could not import the suite at all -- the executor escalates). Skipped /
    passed / ``pytest_missing`` reports are not auto-repairable and map to
    ``unknown``.

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
    # No mechanical classifier matched. An *unrecognized* collection error
    # is one pytest could not even import the suite for -- a broken
    # conftest, a pytest CLI/setup error, or an import break outside the
    # generated first-party modules -- and a blind re-scaffold structurally
    # cannot fix it, so it surfaces as its own first-class token and the
    # executor escalates instead of burning the regenerate budget looping.
    if outcome == "collection_error":
        return "collection_error"
    # A plain assertion failure with no mechanical locator is ordinary
    # test<->implementation contract drift: the suite imported and ran and
    # an ``assert`` (or a status-code / message-string mismatch) tripped.
    # The tests encode the intended contract, so this routes to the bounded
    # LLM-repair / *targeted* regenerate path that aligns the implementation
    # file(s) the failure flowed through to the asserted contract -- not a
    # whole-tree regenerate that re-rolls both sides of the seam and
    # reproduces the same divergence.
    return "assertion_drift"


def import_name_breaks(
        content: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return ``((symbol, module), ...)`` pairs from "cannot import name".

    Unlike :func:`third_party_import_breaks` the module is the **full**
    dotted path (e.g. ``werkzeug.urls`` or ``backend.auth``), not the
    top-level distribution. The full path lets a caller with disk access
    tell a first-party symbol mismatch (the module is a project file that
    imported cleanly but never defines the symbol -- a regenerate) apart
    from a genuine third-party API break (the module is an installed
    package whose peer version drifted -- a pin). Order-preserving and
    de-duplicated.
    """
    blob = _failure_text(content)
    out: List[Tuple[str, str]] = []
    for m in _CANNOT_IMPORT_NAME_RE.finditer(blob):
        pair = (m.group(1), m.group(2))
        if pair not in out:
            out.append(pair)
    return tuple(out)


def third_party_import_breaks(
        content: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Return ``((symbol, package), ...)`` pairs from "cannot import name".

    Used by the dependency-aware proposer to drive the PyPI lookup.
    The package name is the **top-level** distribution (e.g. ``werkzeug``
    for ``werkzeug.urls``) so it can be matched against installed
    packages in the BUILD_REPORT. Derived from :func:`import_name_breaks`
    by collapsing each module to its top-level name. Order-preserving and
    de-duplicated.
    """
    out: List[Tuple[str, str]] = []
    for symbol, module in import_name_breaks(content):
        pair = (symbol, module.split(".", 1)[0])
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


def required_package_names(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the pip packages the failure explicitly asked to install.

    Extracted from the ``requires the <pkg> package to be installed``
    RuntimeError shape -- the library names the exact distribution, so
    the install-deps route can pass it straight to pip. Order-preserving
    and de-duplicated.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _REQUIRES_PACKAGE_RE.finditer(blob):
        name = m.group(1)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def circular_import_modules(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the dotted module names Python reported partially initialized.

    These are the members of the import cycle the failure flowed through;
    the REPAIR executor folds them into the regenerate constraint so the
    re-authored scaffold knows exactly which modules must stop importing
    each other. Order-preserving and de-duplicated.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _PARTIALLY_INITIALIZED_MODULE_RE.finditer(blob):
        name = m.group(1)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def undefined_names(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the names Python reported as not defined.

    Fed into the ``undefined_name`` regenerate constraint so the
    re-authored module knows exactly which bindings it must supply
    (an import, an assignment, or a definition). Order-preserving and
    de-duplicated.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _UNDEFINED_NAME_RE.finditer(blob):
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

    :data:`_NON_FIXTURE_NAMES` are dropped: pytest words the "collected
    a method outside a collected class" failure as a missing fixture
    named ``self``, and no repair can conjure that fixture. Filtering
    here also demotes the classification (the registry predicate reads
    this function), so such a report falls through to the classifiers
    that can actually act on it.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for m in _FIXTURE_NOT_FOUND_RE.finditer(blob):
        name = m.group(1)
        if name and name not in out and name not in _NON_FIXTURE_NAMES:
            out.append(name)
    return tuple(out)


def unresolved_entry_paths(text: str) -> Tuple[str, ...]:
    """Return the entry modules a bundler reported it could not resolve.

    Takes the raw build stderr (the SMOKE build-smoke tail) rather than a
    report dict because that is the only place the error appears. The
    paths are normalised to forward slashes and stripped of a leading
    ``./``; order-preserving and de-duplicated. Callers treat these as
    files that must be *added* to the manifest -- the bundler is looking
    for something the scaffold never generated.
    """
    out: List[str] = []
    for m in _UNRESOLVED_ENTRY_RE.finditer(str(text or "")):
        path = m.group(1).replace("\\", "/").strip().lstrip("./")
        if path and path not in out:
            out.append(path)
    return tuple(out)


def unresolved_import_sources(text: str) -> Tuple[str, ...]:
    """Return the importer files a bundler could not resolve an import in.

    Takes the raw build stderr (the SMOKE build-smoke tail) and matches
    ``Could not resolve "<spec>" from "<file>"`` -- the failure a generated
    source ships when it imports a sibling that was never generated. Returns
    the *importer* paths (the ``from`` side), normalised to forward slashes
    and stripped of a leading ``./``; order-preserving and de-duplicated.
    Callers treat these as files to *re-author* (a targeted regenerate),
    distinct from :func:`unresolved_entry_paths` whose paths must be added.
    An absolute path is returned verbatim for the caller to relativise
    against the project root.
    """
    out: List[str] = []
    for m in _UNRESOLVED_IMPORT_RE.finditer(str(text or "")):
        path = m.group(2).replace("\\", "/").strip()
        if not path.startswith("/"):
            path = path.lstrip("./")
        if path and path not in out:
            out.append(path)
    return tuple(out)


def traceback_source_files(content: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the ``.py`` paths named in the failure traceback(s).

    Scans the concatenated failure text for both pytest (``file.py:12:
    in func``) and standard (``File "file.py", line 12``) frame shapes so
    the REPAIR executor can localize the fix to the files the failure
    actually flowed through -- not just the files APPLY wrote. The paths
    are returned verbatim (still runner-relative or absolute), order-
    preserving and de-duplicated; the caller resolves them against the
    project root and drops anything not on disk.
    """
    blob = _failure_text(content)
    out: List[str] = []
    for regex in (_TB_FILE_LINE_RE, _TB_PYTEST_FRAME_RE):
        for m in regex.finditer(blob):
            path = m.group(1).replace("\\", "/").strip()
            if path and path not in out:
                out.append(path)
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


# Classification token for a RUNTIME_REPORT whose app failed to boot
# (import-time error, ``create_app`` wiring, or a boot that hangs). Unlike
# the VERIFY tokens in :data:`REPAIR_CLASSIFICATIONS` this never has a
# mechanical locator -- the fix is to re-author the failing entry module
# and what it imports -- so REPAIR routes it straight to a regenerate with
# the captured boot error folded in as a constraint.
RUNTIME_REPAIR_CLASSIFICATION = "runtime_failure"

# RUNTIME_REPORT outcomes that mean the app did not boot cleanly. A
# ``passed`` / ``skipped`` report never reaches classification (the router
# completes the session on those), so this covers the hard boot failures.
_RUNTIME_FAILED_OUTCOMES = frozenset({"failed", "timeout", "error"})


def classify_runtime_report(content: Dict[str, Any]) -> RepairClassification:
    """Map a RUNTIME_REPORT content dict to a classification token.

    Returns :data:`RUNTIME_REPAIR_CLASSIFICATION` for any hard boot
    outcome (``failed`` / ``timeout`` / ``error``) and ``unknown``
    otherwise, mirroring the conservative contract of
    :func:`classify_verify_report`.
    """
    outcome = str(content.get("outcome") or "").strip()
    if outcome in _RUNTIME_FAILED_OUTCOMES:
        return RUNTIME_REPAIR_CLASSIFICATION
    return "unknown"


def runtime_failure_text(content: Dict[str, Any]) -> str:
    """Concatenate the captured boot error of every failing entry probe.

    Each failing ``probes`` entry contributes a ``<file> (<kind>):`` header
    followed by its ``stderr_tail`` so the regenerate constraint carries
    the concrete traceback the model must fix, not just the entry name.
    Order-stable; returns an empty string when nothing failed.
    """
    parts: List[str] = []
    for probe in content.get("probes") or []:
        if not isinstance(probe, dict) or probe.get("ok"):
            continue
        rel = str(probe.get("file") or "").strip()
        kind = str(probe.get("kind") or "").strip()
        tail = str(probe.get("stderr_tail") or "").strip()
        header = f"{rel} ({kind}):" if kind else f"{rel}:"
        parts.append(f"{header}\n{tail}".rstrip())
    return "\n\n".join(parts)


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
