


"""Deterministic failure classifier + repair plan generator.

This package is consulted by the :class:`~cgx.session.tasks.repair`
executor (and indirectly by the router) to decide whether a failed
VERIFY can be auto-fixed without an LLM call. Three pieces live here:

* :mod:`classify` -- map a VERIFY_REPORT into a typed
  :class:`RepairClassification` token, plus a stable
  ``failure_signature`` used by the progress detector.
* :mod:`locate` -- AST/regex helpers that find the offending spans
  for the classifications that have a deterministic locator.
* :mod:`propose` -- per-classification diff generators. Each one
  returns a list of ``{"file", "patch"}`` unified diffs that the
  shared APPLY executor can write to disk.
* :mod:`context` -- :class:`FailureContext`, the single normalized
  input the DIAGNOSE reasoning rung consumes across all four gates.

The split is intentional: classification is the only step that has to
be conservative across the entire failure surface; location and
proposal can be incomplete (return ``None`` / empty list) without
breaking the loop -- the executor will simply emit an empty plan and
the router will hand off to ASK_USER instead of looping forever.
"""

from cgx.session.repair.classify import (
    REPAIR_CLASSIFICATIONS,
    RepairClassification,
    circular_import_modules,
    classify_verify_report,
    failure_signature,
    import_name_breaks,
    missing_fixture_names,
    missing_module_names,
    required_package_names,
    third_party_import_breaks,
    undefined_names,
)
from cgx.session.repair.context import FailureContext

__all__ = [
    "REPAIR_CLASSIFICATIONS",
    "RepairClassification",
    "FailureContext",
    "circular_import_modules",
    "classify_verify_report",
    "failure_signature",
    "import_name_breaks",
    "missing_fixture_names",
    "missing_module_names",
    "required_package_names",
    "third_party_import_breaks",
    "undefined_names",
]
