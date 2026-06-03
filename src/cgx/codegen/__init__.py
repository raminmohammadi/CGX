

"""Code generation safety net for CGX.

This subpackage validates LLM-generated changes before they touch a real repo:

- ``diff_apply``: parse ``diff path=...`` fenced blocks (or raw unified diffs),
  resolve targets to files on disk, and produce the post-patch file contents
  *in memory* without modifying anything.
- ``validate``: run language-aware syntax checks (Python AST, JSON, YAML when
  available) on the proposed post-patch contents and surface diagnostics.
- ``test_runner``: locate tests impacted by a set of changed files and run
  them in an isolated sandbox copy of the project.

The public entry point is :func:`cgx.codegen.validate_and_test` which a
calling LLM loop can use to verify a proposed plan before exposing it to the
user or before retrying with a follow-up prompt.
"""

from cgx.codegen.diff_apply import (
    PatchTarget,
    PatchResult,
    parse_fenced_diffs,
    apply_diffs_in_memory,
)
from cgx.codegen.validate import (
    SyntaxDiagnostic,
    validate_python_source,
    validate_patch_results,
)
from cgx.codegen.test_runner import (
    TestRunOutcome,
    discover_all_tests,
    ensure_project_venv,
    find_impacted_tests,
    run_impacted_tests,
    run_pytest_paths,
    run_tests_on_disk,
)
from cgx.codegen.pipeline import (
    CodegenReport,
    validate_and_test,
)

__all__ = [
    "PatchTarget",
    "PatchResult",
    "parse_fenced_diffs",
    "apply_diffs_in_memory",
    "SyntaxDiagnostic",
    "validate_python_source",
    "validate_patch_results",
    "TestRunOutcome",
    "discover_all_tests",
    "ensure_project_venv",
    "find_impacted_tests",
    "run_impacted_tests",
    "run_pytest_paths",
    "run_tests_on_disk",
    "CodegenReport",
    "validate_and_test",
]
