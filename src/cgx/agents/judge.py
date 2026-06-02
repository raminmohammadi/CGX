# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Judge: validate a task's produced artifact against acceptance criteria.

The Judge is deliberately conservative — it answers ``pass`` or ``fail``
with a short rationale and a confidence in [0.0, 1.0]. When an LLM is
available it grounds its verdict in the artifact + criteria; otherwise
it falls back to heuristic checks tied to the artifact's shape (answer
markdown, diff text, search hits).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cgx.agents.planner import _extract_json  # reuse balanced-brace parser
from cgx.agents.types import Task, TaskKind

logger = logging.getLogger(__name__)

# The ``skills`` package lives at the repo root; pyproject.toml + the
# test conftest both wire it into sys.path. The import is defensive so
# Judge still works in stripped-down environments without the package.
try:  # pragma: no cover - exercised indirectly through judge runs
    import skills as _skills
except Exception:  # pragma: no cover - degrade gracefully
    _skills = None  # type: ignore[assignment]


def _skill_objects_for_task(task: Task) -> List[Any]:
    """Resolve a Task's active skills.

    Prefers ``task.inputs['skills']`` (planner-attached, deterministic).
    Falls back to detecting from the task's ``goal``/``description`` when
    the planner didn't attach anything (older plans / direct invocation).
    """
    if _skills is None:
        return []
    inp = (task.inputs or {})
    names = inp.get("skills") or []
    if isinstance(names, list) and names:
        try:
            return _skills.skills_by_names([str(n) for n in names])
        except Exception:
            pass
    goal_text = str(inp.get("goal") or "").strip() or (task.description or "")
    if not goal_text:
        return []
    try:
        return _skills.detect_skills(goal_text)
    except Exception:
        return []


# Skill name → file extension/path hints used by the manifest goal-vs-layer
# check. The check passes when at least one manifest file matches one of the
# patterns for every active skill that has hints.
_SKILL_FILE_HINTS: Dict[str, tuple] = {
    "python": (".py",),
    "fastapi": (".py",),
    "flask": (".py",),
    "django": (".py",),
    "react": (".jsx", ".tsx"),
    "nextjs": (".jsx", ".tsx"),
    "vue": (".vue",),
    "svelte": (".svelte",),
    "express": (".js", ".ts", ".mjs"),
}

# Bare goal keywords that imply a backend layer even if no specific backend
# skill was detected. Matched against the lower-cased goal text.
_BACKEND_GOAL_KEYWORDS: tuple = (
    "backend", "server", "api endpoint", "rest api", "graphql",
)

# Generic "backend exists" fallback when the goal mentions a backend but
# names no language. Kept broad because any of these would be a valid
# server-side implementation.
_BACKEND_FILE_EXTS: tuple = (
    ".py", ".js", ".ts", ".mjs", ".go", ".rs", ".rb", ".java", ".cs",
)

# Language word in the goal → mandatory backend file extensions. Matched
# as whole words. When a language word fires, the broad fallback above is
# NOT used — the language constrains which extensions count as backend.
_LANGUAGE_BACKEND_EXTS: Dict[str, tuple] = {
    "python": (".py",),
    "node": (".js", ".ts", ".mjs"),
    "node.js": (".js", ".ts", ".mjs"),
    "nodejs": (".js", ".ts", ".mjs"),
    "express": (".js", ".ts", ".mjs"),
    "typescript": (".ts", ".mjs"),
    "go": (".go",),
    "golang": (".go",),
    "rust": (".rs",),
    "ruby": (".rb",),
    "java": (".java",),
}

# Canonical packaging/manifest files required when a skill is active. The
# check passes when at least one path in the manifest matches any of the
# listed canonical files (substring match on basename).
_SKILL_REQUIRED_FILES: Dict[str, tuple] = {
    "react": ("package.json",),
    "vue": ("package.json",),
    "nextjs": ("package.json",),
    "express": ("package.json",),
    "fastapi": ("pyproject.toml", "requirements.txt"),
    "flask": ("pyproject.toml", "requirements.txt"),
    "django": ("pyproject.toml", "requirements.txt", "manage.py"),
    "python_cli": ("pyproject.toml", "requirements.txt"),
}

# Skill-independent packaging requirements. When the goal mentions a
# language/framework by literal name and no skill resolved (typos,
# unsupported aliases), we still expect the canonical packaging file
# for that stack. Keys are matched as whole words against the goal.
_LANGUAGE_REQUIRED_FILES: Dict[str, tuple] = {
    "python": ("pyproject.toml", "requirements.txt", "setup.py"),
    "react": ("package.json",),
    "vue": ("package.json",),
    "node": ("package.json",),
    "nodejs": ("package.json",),
    "node.js": ("package.json",),
    "express": ("package.json",),
    "nextjs": ("package.json",),
    "next.js": ("package.json",),
    "typescript": ("package.json", "tsconfig.json"),
}


def _goal_language_backend_exts(goal_low: str) -> tuple:
    """Pick the most specific backend extensions implied by the goal text."""
    for lang, exts in _LANGUAGE_BACKEND_EXTS.items():
        if re.search(r"\b" + re.escape(lang) + r"\b", goal_low):
            return exts
    return ()


def _manifest_required_missing(
    layers: List[Dict[str, Any]],
    skill_names: List[str],
    goal: str,
) -> Optional[str]:
    """Return a rationale listing every required layer/skill missing from
    the manifest, or ``None`` when every requirement is satisfied.

    All violations are collected and joined into one rationale so the
    retry planner sees the full picture in a single round-trip rather
    than chasing one constraint at a time.

    Honors per-skill extension hints in :data:`_SKILL_FILE_HINTS`, the
    canonical packaging files in :data:`_SKILL_REQUIRED_FILES`, and the
    language-aware backend check via :data:`_LANGUAGE_BACKEND_EXTS`.
    """
    paths: List[str] = []
    for lay in layers or []:
        if not isinstance(lay, dict):
            continue
        for f in (lay.get("files") or []):
            if isinstance(f, dict):
                p = str(f.get("path") or "").lower().strip()
                if p:
                    paths.append(p)
    if not paths:
        return None  # base check already failed on this

    issues: List[str] = []

    for sk in skill_names or []:
        sk_low = str(sk).lower()
        hints = _SKILL_FILE_HINTS.get(sk_low)
        if hints and not any(p.endswith(hints) for p in paths):
            issues.append(f"Goal declares '{sk}' but manifest has no file "
                          f"matching {hints}.")
        req = _SKILL_REQUIRED_FILES.get(sk_low)
        if req and not any(p.rsplit("/", 1)[-1] in req for p in paths):
            issues.append(f"Goal declares '{sk}' but manifest has no "
                          f"packaging file ({', '.join(req)}).")

    goal_low = (goal or "").lower()
    lang_exts = _goal_language_backend_exts(goal_low)
    if lang_exts:
        if not any(p.endswith(lang_exts) for p in paths):
            issues.append(f"Goal mentions a language requiring {lang_exts} "
                          "but manifest has no matching source file.")
    elif any(kw in goal_low for kw in _BACKEND_GOAL_KEYWORDS):
        if not any(p.endswith(_BACKEND_FILE_EXTS) for p in paths):
            issues.append("Goal mentions a backend/server/API but manifest "
                          "has no server-side source file.")

    # Skill-independent packaging check: if the goal literally names a
    # language/framework (e.g. "python", "react"), the manifest must
    # contain its canonical packaging file even when skill resolution
    # returned nothing (typos, unsupported aliases).
    basenames = {p.rsplit("/", 1)[-1] for p in paths}
    for kw, req in _LANGUAGE_REQUIRED_FILES.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", goal_low):
            if not (basenames & set(req)):
                issues.append(f"Goal mentions '{kw}' but manifest has no "
                              f"packaging file ({', '.join(req)}).")

    # Python + backend keyword → require a runnable backend entry module.
    # Packaging files alone (pyproject.toml, setup.py) aren't enough: the
    # 3B model often ships a Python "backend" with no app.py / main.py
    # / server.py, leaving the README's `python app.py` instruction broken.
    if (re.search(r"\bpython\b", goal_low)
            and any(kw in goal_low for kw in _BACKEND_GOAL_KEYWORDS)):
        entry_names = ("app.py", "main.py", "server.py", "wsgi.py", "asgi.py",
                       "manage.py", "run.py")
        if not (basenames & set(entry_names)):
            issues.append("Goal asks for a Python backend but manifest has "
                          f"no entry module ({', '.join(entry_names)}).")

    if not issues:
        return None
    if len(issues) == 1:
        return issues[0]
    return "Multiple manifest issues: " + " | ".join(issues)


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
        kind_val = task.kind.value if task.kind else "?"
        logger.info("Judge: judging task id=%s kind=%s n_criteria=%d",
                    task.id, kind_val, len(task.criteria or []))
        if not task.criteria:
            logger.info("Judge: no criteria → pass id=%s", task.id)
            return Verdict(verdict="pass", confidence=1.0,
                           rationale="No criteria specified.", checked_criteria=0)
        # 1) Cheap deterministic short-circuits.
        if task.output is None:
            logger.info("Judge: no output → fail id=%s", task.id)
            return Verdict(verdict="fail", confidence=1.0,
                           rationale="Task produced no output.",
                           checked_criteria=len(task.criteria))
        if task.error:
            logger.info("Judge: task errored → fail id=%s error=%r",
                        task.id, task.error[:120])
            return Verdict(verdict="fail", confidence=1.0,
                           rationale=f"Task errored: {task.error[:120]}",
                           checked_criteria=len(task.criteria))
        # 2) Heuristic structural checks for known task shapes.
        struct = self._structural_check(task)
        if struct is not None and not struct.passed:
            logger.info("Judge: structural FAIL id=%s kind=%s rationale=%r",
                        task.id, kind_val, struct.rationale[:160])
            return struct
        # SEARCH/APPLY/VERIFY/SCAFFOLD outcomes are decided by ground
        # truth — hits exist or don't, files were written or weren't,
        # pytest's returncode is what it is, and a scaffold either
        # produced source files matching the requested stack or it
        # didn't. An LLM-judge handed these artifacts has nothing
        # meaningful to add and routinely fabricates rationales (e.g.
        # claiming "doesn't include input fields" or "doesn't support
        # +,-,*,/" while staring at a Calculator.js that demonstrably
        # does) that override the structural verdict. Local 3-7B judge
        # models are especially prone to this; short-circuit on a
        # structural pass for these kinds.
        if (struct is not None and struct.passed
                and task.kind in (TaskKind.SEARCH, TaskKind.APPLY,
                                  TaskKind.VERIFY, TaskKind.SCAFFOLD,
                                  TaskKind.SCAFFOLD_MANIFEST, TaskKind.SCAFFOLD_FILE)):
            logger.info("Judge: structural PASS short-circuit id=%s kind=%s "
                        "(skipping LLM grader)", task.id, kind_val)
            return struct
        # 3) LLM judgement (optional).
        if self.provider is not None:
            v = self._llm_judge(task)
            if v is not None:
                return v
        # 4) Fallback: trust the structural check if it passed; otherwise
        #    pass with low confidence (the Tracker treats this as soft-ok).
        if struct is not None:
            logger.info("Judge: structural PASS (no LLM) id=%s kind=%s",
                        task.id, kind_val)
            return struct
        logger.info("Judge: no structural check, no LLM → soft-pass id=%s kind=%s",
                    task.id, kind_val)
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
            # Run skill-level plan validators when any skill is active.
            # Skills abstain by default for plans (no fresh-layout
            # assumption); a non-None fail here signals a hard
            # anti-pattern (e.g. class component in a hooks codebase).
            active = _skill_objects_for_task(task)
            if _skills is not None and active and isinstance(diffs, list):
                goal_text = (str((task.inputs or {}).get("goal") or "")
                             or task.description)
                sv = _skills.validate_plan(active, diffs, goal=goal_text)
                if sv is not None and not sv.passed:
                    label = sv.skill or "skill"
                    return Verdict(
                        verdict="fail",
                        confidence=max(0.0, min(1.0, float(sv.confidence))),
                        rationale=f"[{label}] {sv.rationale}",
                        checked_criteria=ncrit,
                    )
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
            # Run each active skill's structural validator. The first
            # failure short-circuits to a Judge FAIL with the skill's
            # rationale; ``None`` (no opinion) is treated as a pass.
            active = _skill_objects_for_task(task)
            goal_text = (str((task.inputs or {}).get("goal") or "")
                         or task.description)
            if _skills is not None and active:
                sv = _skills.validate_scaffold(active, diffs, goal=goal_text)
                if sv is not None and not sv.passed:
                    label = sv.skill or "skill"
                    return Verdict(
                        verdict="fail",
                        confidence=max(0.0, min(1.0, float(sv.confidence))),
                        rationale=f"[{label}] {sv.rationale}",
                        checked_criteria=ncrit,
                    )
            # Collect non-fatal warnings (e.g. "no test files generated")
            # so the rationale tells operators what to improve without
            # failing the task.
            warning_notes: List[str] = []
            if _skills is not None and active:
                try:
                    for w in _skills.collect_scaffold_warnings(
                        active, diffs, goal=goal_text
                    ) or []:
                        label = w.skill or "skill"
                        warning_notes.append(f"[{label} warning] {w.rationale}")
                except Exception:  # pragma: no cover - defensive
                    pass
            # Ground truth: diffs were produced and every active skill
            # either passed or abstained. Return a positive verdict so
            # the SCAFFOLD short-circuit in :meth:`judge` skips the LLM
            # grader — small local models routinely fabricate
            # criteria-based fails ("doesn't include input fields")
            # against scaffolds that demonstrably satisfy them.
            base = f"Scaffold generated {len(diffs)} file(s) matching the requested stack."
            rationale = base if not warning_notes else base + " " + " ".join(warning_notes)
            return Verdict(
                verdict="pass", confidence=0.75,
                rationale=rationale,
                checked_criteria=ncrit,
            )
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
                # Distinguish "apply produced nothing despite having diffs"
                # (a real bug) from "apply had nothing to do because every
                # upstream SCAFFOLD_FILE soft-failed" (a benign no-op in a
                # scaffold-retry plan whose only failed file came back
                # empty again). The capability surfaces the latter via an
                # explicit ``error`` field; treating it as a soft pass
                # lets VERIFY proceed against what is already on disk so
                # the unrecoverable-verify demotion path can finish the
                # run cleanly instead of halting on a misleading APPLY
                # failure.
                err = str(out.get("error") or "").lower()
                if "no diffs" in err:
                    return Verdict(verdict="pass", confidence=0.7,
                                   rationale="Apply had no new diffs to write; "
                                             "previously-applied files remain on disk.",
                                   checked_criteria=ncrit)
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
        if task.kind == TaskKind.SCAFFOLD_MANIFEST:
            layers = out.get("layers")
            if not isinstance(layers, list) or not layers:
                return Verdict(verdict="fail", confidence=0.95,
                               rationale="Manifest returned no layers.",
                               checked_criteria=ncrit)
            n_files = sum(len(lay.get("files") or []) for lay in layers
                          if isinstance(lay, dict))
            if n_files == 0:
                return Verdict(verdict="fail", confidence=0.9,
                               rationale="Manifest layers contain no files.",
                               checked_criteria=ncrit)
            # Goal-vs-layer guard: required skills/backend keywords must be
            # represented by at least one file in the manifest.
            inp = task.inputs or {}
            skill_names = inp.get("skills") or []
            if isinstance(skill_names, list):
                skill_names = [str(n) for n in skill_names]
            else:
                skill_names = []
            goal_text = str(inp.get("goal") or task.description or "")
            missing = _manifest_required_missing(layers, skill_names, goal_text)
            if missing:
                return Verdict(verdict="fail", confidence=0.9,
                               rationale=missing, checked_criteria=ncrit)
            return Verdict(verdict="pass", confidence=0.9,
                           rationale=f"Manifest planned {n_files} file(s) across {len(layers)} layer(s).",
                           checked_criteria=ncrit)
        if task.kind == TaskKind.SCAFFOLD_FILE:
            fp = str(out.get("file") or "").strip()
            patch = str(out.get("patch") or "").strip()
            if not fp or not patch:
                return Verdict(verdict="fail", confidence=0.95,
                               rationale="File generation produced no content.",
                               checked_criteria=ncrit)
            if out.get("syntax_ok") is False:
                err = str(out.get("syntax_error") or "syntax error")[:120]
                return Verdict(verdict="fail", confidence=1.0,
                               rationale=f"Syntax validation failed for {fp}: {err}",
                               checked_criteria=ncrit)
            return Verdict(verdict="pass", confidence=0.85,
                           rationale=f"Generated {fp} with valid syntax.",
                           checked_criteria=ncrit)
        return None

    # ------------------------------------------------------------------
    # LLM judgement.
    # ------------------------------------------------------------------
    def _llm_judge(self, task: Task) -> Optional[Verdict]:
        kind_val = task.kind.value if task.kind else "?"
        artifact = self._render_artifact(task)
        # Include the user's original goal when the planner injected it on
        # the task — a per-task description like "Generate React UI
        # components" lacks the technology-stack context the judge needs to
        # decide whether functional criteria are met.
        goal = str((task.inputs or {}).get("goal") or "").strip()
        goal_block = f"USER GOAL: {goal[:400]}\n\n" if goal else ""
        user_msg = (
            f"{goal_block}"
            f"TASK: {task.description}\n\n"
            f"CRITERIA:\n- " + "\n- ".join(task.criteria) + "\n\n"
            f"OUTPUT:\n{artifact}\n"
        )
        logger.info("Judge: invoking LLM grader id=%s kind=%s artifact_len=%d",
                    task.id, kind_val, len(artifact))
        try:
            resp = self.provider.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1000,
                force_json=True,
            )
        except Exception as e:
            logger.warning("Judge: LLM grader raised %s: %s",
                           type(e).__name__, e)
            return None
        if not isinstance(resp, dict) or resp.get("error"):
            err = resp.get("error") if isinstance(resp, dict) else type(resp).__name__
            logger.warning("Judge: LLM grader returned error id=%s err=%r",
                           task.id, err)
            return None
        data = _extract_json(str(resp.get("content") or ""))
        if not isinstance(data, dict):
            logger.warning("Judge: LLM grader response not valid JSON id=%s "
                           "head=%r", task.id, str(resp.get("content") or "")[:160])
            return None
        verdict = str(data.get("verdict") or "").lower().strip()
        if verdict not in {"pass", "fail"}:
            logger.warning("Judge: LLM grader returned bad verdict id=%s verdict=%r",
                           task.id, verdict)
            return None
        try:
            conf = float(data.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        rationale = str(data.get("rationale") or "")[:240]
        conf_clamped = max(0.0, min(1.0, conf))
        logger.info("Judge: LLM verdict id=%s kind=%s verdict=%s confidence=%.2f",
                    task.id, kind_val, verdict, conf_clamped)
        return Verdict(verdict=verdict, confidence=conf_clamped,
                       rationale=rationale, checked_criteria=len(task.criteria))

    @staticmethod
    def _render_artifact(task: Task) -> str:
        out = task.output or {}
        if task.kind == TaskKind.PLAN:
            return str(out.get("plan_md") or out.get("answer_md") or "") + "\n\n" + \
                   str(out.get("diffs") or out.get("diffs_md") or "")
        if task.kind == TaskKind.SCAFFOLD:
            # Surface the plan summary, the list of generated files, and a
            # bounded preview of each file's contents so the LLM judge can
            # assess functional criteria (e.g. "includes input fields",
            # "supports +,-,*,/") against the actual code rather than a
            # JSON-truncated keyset. Source files are previewed before
            # metadata files (README/package.json/requirements.txt) so the
            # logic-bearing code always fits inside the budget.
            parts: List[str] = []
            plan_md = str(out.get("plan_md") or "").strip()
            if plan_md:
                parts.append("## Plan\n" + plan_md[:1200])
            diffs = out.get("diffs") or []
            files: List[Dict[str, str]] = []
            for d in diffs:
                if not isinstance(d, dict):
                    continue
                fp = str(d.get("file") or d.get("path") or "")
                patch = str(d.get("patch") or d.get("diff") or "")
                if fp:
                    files.append({"path": fp, "patch": patch})
            if files:
                parts.append("## Generated files (" + str(len(files)) + ")")
                parts.append("\n".join("- " + f["path"] for f in files))
                source_exts = (".jsx", ".tsx", ".js", ".ts", ".py",
                               ".vue", ".svelte", ".go", ".rs",
                               ".java", ".kt", ".rb", ".php",
                               ".html", ".css", ".scss")
                source_files = [f for f in files
                                if f["path"].lower().endswith(source_exts)]
                meta_files = [f for f in files if f not in source_files]
                ordered = source_files + meta_files
                # Preview up to 6 files, each capped so the prompt stays small.
                # Sources go first; raise the per-file cap so a ~1.4 KB
                # component (e.g. Calculator.js) fits in full.
                per_file = max(800, min(1600, 6400 // max(1, min(6, len(ordered)))))
                for f in ordered[:6]:
                    parts.append(f"\n### {f['path']}\n```\n"
                                 f"{f['patch'][:per_file]}\n```")
            return "\n\n".join(parts)[:7500]
        if task.kind == TaskKind.SCAFFOLD_MANIFEST:
            plan_md = str(out.get("plan_md") or "").strip()
            layers = out.get("layers") or []
            lines: List[str] = []
            if plan_md:
                lines.append(plan_md[:800])
            for lay in layers[:8]:
                if not isinstance(lay, dict):
                    continue
                lines.append(f"Layer: {lay.get('name')}")
                for f in (lay.get("files") or [])[:10]:
                    if isinstance(f, dict):
                        lines.append(f"  - {f.get('path')}: {f.get('description')}")
            return "\n".join(lines)[:3000]
        if task.kind == TaskKind.SCAFFOLD_FILE:
            fp = str(out.get("file") or "")
            content = str(out.get("content") or out.get("patch") or "")
            return f"File: {fp}\n\n```\n{content[:3000]}\n```"
        if task.kind == TaskKind.ASK:
            return str(out.get("answer_md") or out.get("answer") or "")
        if task.kind == TaskKind.SEARCH:
            hits = out.get("hits") or []
            return json.dumps(hits[:10], default=str)[:4000]
        return json.dumps(out, default=str)[:4000]
