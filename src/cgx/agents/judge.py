"""Judge: validate a task's produced artifact against acceptance criteria.

The Judge is deliberately conservative — it answers ``pass`` or ``fail``
with a short rationale and a confidence in [0.0, 1.0]. When an LLM is
available it grounds its verdict in the artifact + criteria; otherwise
it falls back to heuristic checks tied to the artifact's shape (answer
markdown, diff text, search hits).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cgx.agents.planner import _extract_json  # reuse balanced-brace parser
from cgx.agents.types import Task, TaskKind


SYSTEM_PROMPT = (
    "You are a strict reviewer. Given a TASK, its CRITERIA, and its "
    "OUTPUT, decide whether the output meets every criterion. Reply with "
    "strict JSON only:\n"
    "{\"verdict\": \"pass|fail\", \"confidence\": 0.0-1.0, "
    "\"rationale\": \"one sentence\"}\n"
    "Be conservative: any unmet criterion ⇒ fail."
)


@dataclass
class Verdict:
    verdict: str         # "pass" | "fail"
    confidence: float    # [0.0, 1.0]
    rationale: str
    checked_criteria: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "checked_criteria": int(self.checked_criteria),
        }

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


class Judge:
    """Evaluate task outputs against the task's acceptance criteria."""

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    def judge(self, task: Task) -> Verdict:
        if not task.criteria:
            return Verdict(verdict="pass", confidence=1.0,
                           rationale="No criteria specified.", checked_criteria=0)
        # 1) Cheap deterministic short-circuits.
        if task.output is None:
            return Verdict(verdict="fail", confidence=1.0,
                           rationale="Task produced no output.",
                           checked_criteria=len(task.criteria))
        if task.error:
            return Verdict(verdict="fail", confidence=1.0,
                           rationale=f"Task errored: {task.error[:120]}",
                           checked_criteria=len(task.criteria))
        # 2) Heuristic structural checks for known task shapes.
        struct = self._structural_check(task)
        if struct is not None and not struct.passed:
            return struct
        # SEARCH/APPLY/VERIFY outcomes are decided by ground truth — hits
        # exist or don't, files were written or weren't, pytest's
        # returncode is what it is. An LLM-judge handed these artifacts
        # has nothing meaningful to add and routinely fabricates
        # rationales (e.g. claiming "duplicate diff entries" when there
        # are none) that override the structural verdict. Short-circuit
        # on a structural pass for these kinds.
        if (struct is not None and struct.passed
                and task.kind in (TaskKind.SEARCH, TaskKind.APPLY, TaskKind.VERIFY)):
            return struct
        # 3) LLM judgement (optional).
        if self.provider is not None:
            v = self._llm_judge(task)
            if v is not None:
                return v
        # 4) Fallback: trust the structural check if it passed; otherwise
        #    pass with low confidence (the Tracker treats this as soft-ok).
        if struct is not None:
            return struct
        return Verdict(verdict="pass", confidence=0.5,
                       rationale="No LLM available; output produced and no "
                                 "structural issue detected.",
                       checked_criteria=len(task.criteria))

    # ------------------------------------------------------------------
    # Heuristic checks per task kind.
    # ------------------------------------------------------------------
    def _structural_check(self, task: Task) -> Optional[Verdict]:
        out = task.output or {}
        ncrit = len(task.criteria)
        if task.kind == TaskKind.PLAN:
            # Look for a unified-diff signature in any of the common shapes.
            diffs = out.get("diffs") or out.get("diffs_md") or ""
            if isinstance(diffs, list):
                joined = "\n".join(
                    str(d.get("patch") or d.get("diff") or "")
                    for d in diffs if isinstance(d, dict)
                ) or "\n".join(str(x) for x in diffs)
            else:
                joined = str(diffs)
            has_diff = bool(diffs) and (
                "diff path=" in joined or "--- " in joined or "+++ " in joined
                or "@@" in joined
            )
            if not has_diff:
                # Hard-fail only when both plan_md and diffs are absent — the
                # task produced nothing useful. When plan_md has content but no
                # diffs (e.g. a local LLM that followed the plan format but not
                # the diff format), fall through to the LLM judge so it can
                # assess whether the plan meets the criteria.
                plan_md = str(out.get("plan_md") or "").strip()
                if not plan_md:
                    return Verdict(verdict="fail", confidence=0.9,
                                   rationale="Plan output contains no recognisable diff block.",
                                   checked_criteria=ncrit)
                return None  # plan_md present, no diffs — let LLM judge decide
            # When the engine ran self-tests, trust the report verbatim.
            report = out.get("codegen_report") or {}
            if isinstance(report, dict):
                summary = report.get("summary") or {}
                if isinstance(summary, dict) and summary.get("overall_ok") is False:
                    return Verdict(verdict="fail", confidence=0.95,
                                   rationale="Codegen report flagged failures (patch/syntax/tests).",
                                   checked_criteria=ncrit)
                if report.get("ok") is False:
                    return Verdict(verdict="fail", confidence=0.95,
                                   rationale="Self-test report flagged failures.",
                                   checked_criteria=ncrit)
            return None  # let LLM decide further
        if task.kind == TaskKind.SCAFFOLD:
            diffs = out.get("diffs") or []
            plan_md = str(out.get("plan_md") or "").strip()
            if not diffs and not plan_md:
                return Verdict(verdict="fail", confidence=0.9,
                               rationale="Scaffold produced no files and no plan.",
                               checked_criteria=ncrit)
            if not diffs:
                # Has plan_md but no files — let LLM judge assess.
                return None
            # Check that the generated files match the requested technology.
            goal_text = (str(task.inputs.get("goal") or "") + " " + task.description).lower()
            file_paths = [str(d.get("file") or d.get("path") or "") for d in diffs]
            if re.search(r"\breact\b", goal_text):
                has_jsx = any(f.endswith((".jsx", ".tsx", ".js", ".ts")) for f in file_paths)
                has_python_only = all(
                    f.endswith((".py",)) for f in file_paths
                    if not f.endswith((".md", ".txt", ".cfg", ".ini", ".toml", ".yml", ".yaml"))
                )
                if not has_jsx or has_python_only:
                    return Verdict(
                        verdict="fail", confidence=0.9,
                        rationale=(
                            "Goal requests a React project but scaffold generated Python/non-JS files "
                            f"({[f for f in file_paths if f.endswith('.py')][:3]}). "
                            "Regenerate using React component files (App.jsx, index.js, package.json)."
                        ),
                        checked_criteria=ncrit,
                    )
            return None  # files generated, let LLM judge verify quality
        if task.kind == TaskKind.ASK:
            answer = str(out.get("answer_md") or out.get("answer") or "")
            if not answer.strip():
                return Verdict(verdict="fail", confidence=0.9,
                               rationale="Answer is empty.",
                               checked_criteria=ncrit)
            return None
        if task.kind == TaskKind.SEARCH:
            hits = out.get("hits") or []
            if not hits:
                return Verdict(verdict="fail", confidence=0.9,
                               rationale="Search returned zero hits.",
                               checked_criteria=ncrit)
            # Search is an intermediate retrieval step; the LLM-judge cannot
            # meaningfully assess qualitative criteria against a raw hits
            # payload (it tends to fail descriptive criteria like
            # "comprehend the purpose"). Trust the structural signal and let
            # downstream tasks be judged on their own artifacts.
            return Verdict(verdict="pass", confidence=0.8,
                           rationale=f"Search returned {len(hits)} hit(s).",
                           checked_criteria=ncrit)
        if task.kind == TaskKind.APPLY:
            applied = out.get("applied_files") or []
            failed = out.get("failed_files") or []
            if failed:
                return Verdict(verdict="fail", confidence=0.95,
                               rationale=f"{len(failed)} patch(es) or syntax check(s) failed.",
                               checked_criteria=ncrit)
            if not applied:
                return Verdict(verdict="fail", confidence=0.9,
                               rationale="No files were written to disk.",
                               checked_criteria=ncrit)
            return Verdict(verdict="pass", confidence=0.85,
                           rationale=f"Wrote {len(applied)} file(s); smoke check ok.",
                           checked_criteria=ncrit)
        if task.kind == TaskKind.VERIFY:
            ran = bool(out.get("ran"))
            if not ran:
                reason = str(out.get("skipped_reason") or "tests did not run")
                # Treat a "no impacted tests" outcome as a soft pass — the
                # change is just untested rather than wrong.
                if "no impacted" in reason.lower() or "no tests" in reason.lower():
                    return Verdict(verdict="pass", confidence=0.6,
                                   rationale=f"Skipped: {reason}",
                                   checked_criteria=ncrit)
                return Verdict(verdict="fail", confidence=0.85,
                               rationale=f"Verify did not run: {reason}",
                               checked_criteria=ncrit)
            if bool(out.get("tests_passed")):
                return Verdict(verdict="pass", confidence=0.95,
                               rationale="Impacted tests passed.",
                               checked_criteria=ncrit)
            rc = out.get("returncode")
            return Verdict(verdict="fail", confidence=0.95,
                           rationale=f"Impacted tests failed (rc={rc}).",
                           checked_criteria=ncrit)
        return None

    # ------------------------------------------------------------------
    # LLM judgement.
    # ------------------------------------------------------------------
    def _llm_judge(self, task: Task) -> Optional[Verdict]:
        artifact = self._render_artifact(task)
        user_msg = (
            f"TASK: {task.description}\n\n"
            f"CRITERIA:\n- " + "\n- ".join(task.criteria) + "\n\n"
            f"OUTPUT (truncated):\n{artifact[:4000]}\n"
        )
        try:
            resp = self.provider.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                force_json=True,
            )
        except Exception:
            return None
        if not isinstance(resp, dict) or resp.get("error"):
            return None
        data = _extract_json(str(resp.get("content") or ""))
        if not isinstance(data, dict):
            return None
        verdict = str(data.get("verdict") or "").lower().strip()
        if verdict not in {"pass", "fail"}:
            return None
        try:
            conf = float(data.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        rationale = str(data.get("rationale") or "")[:240]
        return Verdict(verdict=verdict, confidence=max(0.0, min(1.0, conf)),
                       rationale=rationale, checked_criteria=len(task.criteria))

    @staticmethod
    def _render_artifact(task: Task) -> str:
        out = task.output or {}
        if task.kind == TaskKind.PLAN:
            return str(out.get("plan_md") or out.get("answer_md") or "") + "\n\n" + \
                   str(out.get("diffs") or out.get("diffs_md") or "")
        if task.kind == TaskKind.ASK:
            return str(out.get("answer_md") or out.get("answer") or "")
        if task.kind == TaskKind.SEARCH:
            hits = out.get("hits") or []
            return json.dumps(hits[:10], default=str)[:4000]
        return json.dumps(out, default=str)[:4000]
