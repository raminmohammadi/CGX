# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""End-to-end validation of an LLM-generated code plan.

This module composes the diff parser, the syntax validator, and the impacted
test runner into a single function returning a structured report. UIs and
agent loops should treat the report's ``summary`` field as authoritative for
deciding whether to surface the plan to the user, retry with revised
instructions, or fail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cgx.codegen.diff_apply import (
    PatchResult,
    PatchTarget,
    apply_diffs_in_memory,
    parse_fenced_diffs,
)
from cgx.codegen.test_runner import TestRunOutcome, run_impacted_tests
from cgx.codegen.validate import SyntaxDiagnostic, validate_patch_results

logger = logging.getLogger(__name__)


@dataclass
class CodegenReport:
    """Full report from :func:`validate_and_test`."""
    targets: List[PatchTarget] = field(default_factory=list)
    patches: List[PatchResult] = field(default_factory=list)
    diagnostics: List[SyntaxDiagnostic] = field(default_factory=list)
    tests: Optional[TestRunOutcome] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets": [{"path": t.path} for t in self.targets],
            "patches": [
                {
                    "path": p.path,
                    "ok": p.ok,
                    "is_new_file": p.is_new_file,
                    "error": p.error,
                    "rejected_hunks": p.rejected_hunks,
                    "new_content_preview": (p.new_content or "")[:500] if p.new_content else None,
                }
                for p in self.patches
            ],
            "diagnostics": [
                {
                    "path": d.path, "ok": d.ok, "language": d.language,
                    "error": d.error, "line": d.line, "column": d.column,
                }
                for d in self.diagnostics
            ],
            "tests": (
                {
                    "ran": self.tests.ran,
                    "returncode": self.tests.returncode,
                    "tests_selected": self.tests.tests_selected,
                    "stdout_tail": (self.tests.stdout or "")[-2000:],
                    "stderr_tail": (self.tests.stderr or "")[-2000:],
                    "skipped_reason": self.tests.skipped_reason,
                }
                if self.tests is not None else None
            ),
            "summary": self.summary,
        }


def validate_and_test(
    project_root: str,
    plan_text: str,
    *,
    run_tests: bool = True,
    timeout_seconds: float = 120.0,
    allow_new_files: bool = True,
) -> CodegenReport:
    """Parse, apply (in memory), validate, and optionally test a code plan.

    Parameters
    ----------
    project_root
        Path to the project the plan targets.
    plan_text
        Raw plan / diff text emitted by the LLM. May include free-form
        markdown around fenced ``diff path=...`` blocks.
    run_tests
        When True, copies the project to a sandbox, materializes the patches,
        and runs impacted tests with pytest.
    """
    logger.info("codegen.pipeline: validate_and_test root=%s plan_len=%d run_tests=%s",
                project_root, len(plan_text or ""), run_tests)
    targets = parse_fenced_diffs(plan_text or "")
    patches = apply_diffs_in_memory(project_root, targets, allow_new_files=allow_new_files)
    diagnostics = validate_patch_results(patches)

    tests: Optional[TestRunOutcome] = None
    if run_tests and any(p.ok for p in patches):
        tests = run_impacted_tests(project_root, patches, timeout_seconds=timeout_seconds)

    summary: Dict[str, Any] = {
        "n_targets": len(targets),
        "n_patches_ok": sum(1 for p in patches if p.ok),
        "n_patches_failed": sum(1 for p in patches if not p.ok),
        "n_syntax_ok": sum(1 for d in diagnostics if d.ok),
        "n_syntax_failed": sum(1 for d in diagnostics if not d.ok),
        "tests_ran": bool(tests and tests.ran),
        "tests_passed": bool(tests and tests.ran and tests.returncode == 0),
        "empty_plan": len(targets) == 0,
    }
    summary["overall_ok"] = (
        not summary["empty_plan"]
        and summary["n_patches_failed"] == 0
        and summary["n_syntax_failed"] == 0
        and (not run_tests or summary["tests_passed"] or (tests and tests.skipped_reason))
    )
    logger.info("codegen.pipeline: report targets=%d patches_ok=%d patches_failed=%d "
                "syntax_ok=%d syntax_failed=%d tests_ran=%s tests_passed=%s overall_ok=%s",
                summary["n_targets"], summary["n_patches_ok"], summary["n_patches_failed"],
                summary["n_syntax_ok"], summary["n_syntax_failed"],
                summary["tests_ran"], summary["tests_passed"], summary["overall_ok"])
    return CodegenReport(
        targets=targets, patches=patches, diagnostics=diagnostics,
        tests=tests, summary=summary,
    )


def build_retry_feedback(report: CodegenReport) -> str:
    """Format a concise textual feedback string for an LLM retry pass."""
    lines: List[str] = ["The previous plan had issues:"]
    missing_hunks = False
    empty_plan = bool(report.summary.get("empty_plan"))
    if empty_plan:
        lines.append(
            "- no diff blocks were parsed from your previous reply. "
            "The plan must contain at least one concrete fenced "
            "```diff path=<relative/path>``` block; a markdown outline "
            "is not enough."
        )
    for p in report.patches:
        if not p.ok:
            lines.append(f"- patch FAILED for {p.path}: {p.error or 'unknown'}")
            if p.error and "No @@ hunks" in p.error:
                missing_hunks = True
            for rh in p.rejected_hunks[:2]:
                lines.append("  rejected hunk:")
                lines.append("  " + rh.replace("\n", "\n  ")[:600])
    for d in report.diagnostics:
        if not d.ok:
            loc = f" (line {d.line})" if d.line else ""
            lines.append(f"- {d.language} syntax error in {d.path}{loc}: {d.error}")
    if report.tests and report.tests.ran and report.tests.returncode != 0:
        tail = (report.tests.stdout or "")[-1500:]
        lines.append("- impacted tests FAILED. pytest tail:")
        lines.append(tail)
    lines.append(
        "Revise the diffs to fix these issues. Keep using fenced "
        "```diff path=<relative/path>``` blocks and respect existing indentation."
    )
    if missing_hunks or empty_plan:
        lines.append(
            "Every diff MUST contain a '@@' hunk header. Format for an EDIT:\n"
            "```diff path=pkg/mod.py\n"
            "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,3 +1,4 @@\n"
            " def add(a, b):\n     return a + b\n+def mul(a, b):\n+    return a * b\n"
            "```\n"
            "Format for a NEW file:\n"
            "```diff path=pkg/extra.py\n"
            "--- /dev/null\n+++ b/pkg/extra.py\n@@ -0,0 +1,2 @@\n"
            "+def hello():\n+    return 'hi'\n"
            "```"
        )
    return "\n".join(lines)
