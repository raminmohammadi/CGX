"""Planner: decompose a user goal into a sequence of agent tasks.

The Planner prefers the LLM for plan generation but always returns a
useful plan even when no LLM is available — the deterministic fallback
inspects the goal text via the existing intent classifier and emits a
single-task plan that matches the legacy single-shot behaviour.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import logging

from cgx.agents.types import Plan, Task, TaskKind
from cgx.answer.intent import detect_intent

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a senior software engineer decomposing a user request into the "
    "smallest sequence of atomic agent tasks. Reply with strict JSON only.\n\n"
    "Schema:\n"
    "{\n"
    '  "tasks": [\n'
    '    {"name": "short title (max 8 words)",\n'
    '     "description": "single imperative sentence the capability will execute",\n'
    '     "kind": "ask|plan|scaffold|search|summarize|verify",\n'
    '     "criteria": ["..."]}\n'
    "  ]\n"
    "}\n\n"
    "Task kinds (pick the cheapest that satisfies the goal):\n"
    "- 'search'    — retrieve relevant code/files from the index. Use this "
    "first whenever the goal references a file, symbol, or behaviour to "
    "inspect.\n"
    "- 'ask'       — answer a natural-language question grounded in the "
    "indexed code. Use for 'what / how / why / where / explain / describe' "
    "goals.\n"
    "- 'summarize' — condense the outputs of previous tasks into a brief "
    "human summary. Cheap; use as the final step for read-only goals.\n"
    "- 'verify'    — run the project's pytest suite against the working "
    "tree and report pass/fail. Use when the goal asks to run tests / "
    "check whether tests pass / verify the suite, with no code change.\n"
    "- 'plan'      — produce a code-change diff. ONLY use when the goal "
    "explicitly asks to add / implement / modify / refactor / fix / change "
    "/ update / remove / rename / replace / ensure / complete / fill code "
    "in an EXISTING codebase. "
    "NEVER use 'plan' for explanation, summarisation, or read-only "
    "inspection — it is the most expensive kind and must not run unless "
    "code is being written.\n"
    "- 'scaffold'  — generate files for a brand-new project (no existing "
    "codebase). 'apply' and 'verify' are always appended automatically.\n"
    "  Simple projects (1–3 files total) → ONE scaffold task.\n"
    "  Complex projects (UI + backend + tests + config) → 2–4 scaffold "
    "tasks, each covering ONE distinct layer so every call stays focused "
    "and produces complete, syntactically correct code. Assign one layer "
    "per task:\n"
    "    • core logic  e.g. 'Generate the calculator engine: arithmetic "
    "operations, input validation, and state management'\n"
    "    • UI layer    e.g. 'Generate the HTML/CSS/JS UI with themed "
    "color-scheme controls and result display'\n"
    "    • config/pkg  e.g. 'Generate requirements.txt, README.md, and "
    ".gitignore for the project'\n"
    "    • test suite  e.g. 'Generate pytest unit tests covering every "
    "calculator operation and edge case'\n"
    "  NEVER mix scaffold with plan/ask/search for new-project goals.\n\n"
    "Rules:\n"
    "- Task descriptions MUST preserve every technology/framework name from "
    "the goal (e.g. if the goal says 'React UI', write 'Generate React UI "
    "components', not just 'Generate UI'). Failing to name the technology "
    "will cause the wrong stack to be generated.\n"
    "- 2 to 6 tasks. Prefer the fewest tasks that fully cover the goal; "
    "never group unrelated concerns into one task.\n"
    "- New-project goal → 1 scaffold (simple) or 2–4 scaffold tasks "
    "(one per layer for complex projects); 'apply' and 'verify' are "
    "appended automatically. Think: what layers must this project have? "
    "Give each its own scaffold task so failures are isolated and fixable.\n"
    "- Read-only / explanatory goal → typical plan is [search, ask] or "
    "[search, summarize]; do NOT include a 'plan' task.\n"
    "- Run-tests-only goal → typical plan is [verify] or [search, verify]; "
    "do NOT include 'plan' or 'apply'.\n"
    "- Code-change goal (existing codebase) → typical plan is [search, plan] "
    "or just [plan]; an 'apply' and 'verify' will be appended automatically.\n"
    "- Each criterion is one short plain-English sentence the result must "
    "satisfy.\n"
)


# Verbs that signal the user wants code modified rather than explained.
# Kept module-level so both the LLM-path post-validator and the deterministic
# fallback share a single source of truth.
_CHANGE_VERB_RE = re.compile(
    r"\b(add|implement|modify|refactor|fix|change|update|remove|delete|"
    r"create|introduce|inject|replace|rename|extract|migrate|patch|write|"
    r"ensure|complete|fill|populate|finish)\b",
    re.IGNORECASE,
)

# Phrases that signal the user wants an entirely new project generated from scratch.
# Checked BEFORE the change-verb regex so "create a new project" doesn't route
# to a PLAN task against a non-existent codebase.
_SCAFFOLD_RE = re.compile(
    r"(?:"
    # scaffold verb + up to 5 arbitrary tokens + project noun
    # handles: "create a new FastAPI project", "build a REST API", etc.
    r"\b(create|build|generate|scaffold|bootstrap|initialize|init|start|make)\b"
    r"(?:\s+\S+){0,5}"
    r"\s*\b(?:app|application|project|service|api|website|web\s*app|cli|tool|"
    r"library|package|module|repo|repository|backend|frontend|fullstack|bot|chatbot)\b"
    r"|"
    # explicit "from scratch"
    r"\bfrom\s+scratch\b"
    r"|"
    # "new project/app/api/..."
    r"\bnew\s+(?:app|application|project|service|api|website|cli|tool|"
    r"library|package|module|repo|repository|backend|frontend|bot|chatbot)\b"
    r")",
    re.IGNORECASE,
)

# Phrases that signal the user wants tests executed (not changed).
# Matched independently of the change verbs: a verify-only goal does NOT
# trigger plan/apply, only a standalone verify task.
_VERIFY_ONLY_RE = re.compile(
    r"\b("
    r"run\s+(the\s+|all\s+)?tests?|"
    r"run\s+pytest|"
    r"execute\s+(the\s+)?tests?|"
    r"(do|does|will)\s+(the\s+|all\s+)?tests?\s+pass|"
    r"(are|is)\s+(the\s+|all\s+)?tests?\s+passing|"
    r"verify\s+(the\s+|all\s+)?tests?|"
    r"check\s+(if|that|whether)\s+(the\s+|all\s+)?tests?\s+pass|"
    r"tests?\s+will\s+pass"
    r")\b",
    re.IGNORECASE,
)


def _goal_is_scaffold(goal: str) -> bool:
    """True when *goal* asks to generate an entirely new project from scratch."""
    if not goal:
        return False
    return _SCAFFOLD_RE.search(goal) is not None


def _goal_is_change(goal: str) -> bool:
    """True when *goal* asks for a code modification.

    Combines :func:`cgx.answer.intent.detect_intent` (authoritative when it
    returns ``"change_plan"``) with a verb-based heuristic so we still
    recognise change requests when the classifier is uncertain.
    """
    if not goal:
        return False
    try:
        if detect_intent(goal) == "change_plan":
            return True
    except Exception:
        pass
    return _CHANGE_VERB_RE.search(goal) is not None


def _goal_is_verify_only(goal: str) -> bool:
    """True when *goal* asks to run tests but not modify code.

    Used by :meth:`Planner._enforce_kind_policy` to route goals like
    "do the tests pass?" or "run pytest" to a standalone ``verify`` task
    instead of a useless ``ask`` answer.
    """
    if not goal:
        return False
    if _goal_is_change(goal):
        return False
    return _VERIFY_ONLY_RE.search(goal) is not None


def _coerce_kind(raw: str) -> TaskKind:
    if not raw:
        return TaskKind.ASK
    raw = raw.strip().lower()
    try:
        return TaskKind(raw)
    except ValueError:
        return TaskKind.ASK


def _derive_name(name: str, description: str, *, limit: int = 72) -> str:
    """Pick a short, human-friendly title for a task.

    Prefers an explicit ``name`` (truncated for sanity), otherwise distils
    the first sentence/line of ``description`` so the UI always has a
    title to render even when older planner replies omit the field.
    """
    name = (name or "").strip()
    if name:
        return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"
    base = (description or "").strip()
    if not base:
        return ""
    head = re.split(r"(?<=[.!?])\s|\n", base, maxsplit=1)[0].strip()
    if len(head) <= limit:
        return head
    return head[: limit - 1].rstrip() + "…"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from an LLM response.

    Mirrors the engine's balanced-brace strategy so we don't pull in a new
    dependency just for plan parsing.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


class Planner:
    """Produce a :class:`Plan` for a user goal.

    Parameters
    ----------
    provider
        Optional LLM provider used to draft the plan. When ``None``, the
        planner falls back to a deterministic single-task plan derived
        from the intent classifier.
    max_tasks
        Hard cap on tasks returned (defends against runaway models).
    """

    def __init__(
        self,
        provider: Any = None,
        max_tasks: int = 8,
        *,
        retriever: Optional[Any] = None,
    ) -> None:
        self.provider = provider
        self.max_tasks = max(1, int(max_tasks))
        # ``retriever(goal) -> dict`` (matches ``run_query_auto``'s shape).
        # When supplied, the planner uses ``top_files`` to ground the LLM
        # prompt with real candidate paths from the index.
        self.retriever = retriever

    def plan(self, goal: str) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Planner.plan: goal must be non-empty")
        logger.info("Planner.plan: starting goal=%r provider=%s retriever=%s",
                    goal[:80], type(self.provider).__name__ if self.provider else "None",
                    "yes" if self.retriever else "no")
        candidates = self._retrieve_candidates(goal)
        tasks: List[Task]
        if self.provider is not None:
            logger.info("Planner: calling LLM for plan decomposition")
            tasks = self._llm_plan(goal, candidates=candidates)
            logger.info("Planner: LLM returned %d tasks", len(tasks))
        else:
            tasks = []
        if not tasks:
            logger.info("Planner: using deterministic fallback plan")
            tasks = self._fallback_plan(goal)
        tasks = self._enforce_kind_policy(goal, tasks)
        plan = Plan(goal=goal.strip(), tasks=tasks[: self.max_tasks])
        # Wire sequential dependencies so the UI can render the execution DAG.
        for i in range(1, len(plan.tasks)):
            plan.tasks[i].dependencies = [plan.tasks[i - 1].id]
        logger.info("Planner: plan ready id=%s tasks=%d kinds=%s",
                    plan.id, len(plan.tasks), [t.kind.value for t in plan.tasks])
        return plan

    def _retrieve_candidates(self, goal: str) -> List[str]:
        """Call the optional retriever and return up to 8 candidate file paths."""
        if self.retriever is None:
            return []
        try:
            result = self.retriever(goal)
        except Exception as e:
            logger.warning("Planner: retriever failed: %s", e)
            return []
        if not isinstance(result, dict):
            return []
        top_files = result.get("top_files") or []
        out: List[str] = []
        for f in top_files:
            if isinstance(f, dict):
                fp = str(f.get("file") or f.get("path") or "").strip()
                if fp:
                    out.append(fp)
            elif isinstance(f, str) and f.strip():
                out.append(f.strip())
            if len(out) >= 8:
                break
        return out

    def _enforce_kind_policy(self, goal: str, tasks: List[Task]) -> List[Task]:
        """Coerce LLM-emitted tasks into the operational kind contract.

        Three policies are enforced:

        - **Read-only goals** must not include ``plan`` / ``apply`` /
          ``verify`` tasks. Smaller planner models often emit one for
          purely explanatory goals, which then hangs or fails the Judge.
          Coerce those to the cheaper ``ask`` kind.
        - **Verify-only goals** ("run the tests", "do tests pass?")
          must conclude with a standalone ``verify`` task and must NOT
          trigger ``plan`` / ``apply``. Any stray ``plan`` / ``apply`` /
          ``ask`` is dropped or replaced.
        - **Code-change goals** must conclude with ``apply`` then
          ``verify`` after the final ``plan`` task so the produced diffs
          are actually written to disk and tested before the run ends.
        """
        if _goal_is_scaffold(goal):
            # Preserve the LLM's layer-by-layer scaffold decomposition.
            # Each scaffold task covers one subsystem (core logic, UI, tests,
            # config) so failures are isolated and every call stays focused.
            # Drop any APPLY/VERIFY/PLAN the LLM may have included — we always
            # append a fresh pair so prior_outputs ordering is well-defined.
            scaffold_tasks = [t for t in tasks if t.kind == TaskKind.SCAFFOLD]
            if not scaffold_tasks:
                # LLM produced nothing useful — fall back to single full-goal task.
                scaffold_tasks = [Task(
                    description=goal.strip(),
                    kind=TaskKind.SCAFFOLD,
                    name=_derive_name("", goal.strip()),
                    criteria=["Generates at least one source file.",
                              "All generated files have valid, working code."],
                )]
            # Cap at 4 scaffold tasks so APPLY+VERIFY stay within max_tasks.
            scaffold_tasks = scaffold_tasks[:4]
            # Always inject the original goal so the scaffold LLM receives the
            # full technology-stack context even when the planner LLM produced
            # a generic task description (e.g. "Generate project" instead of
            # "Generate React UI").
            for t in scaffold_tasks:
                t.inputs.setdefault("goal", goal.strip())
            scaffold_tasks.append(Task(
                description="Write all generated project files to the output directory.",
                kind=TaskKind.APPLY,
                name="Write project files",
                criteria=["All files written to disk without errors.",
                          "No files have syntax errors."],
            ))
            scaffold_tasks.append(Task(
                description="Verify the generated project runs or its tests pass.",
                kind=TaskKind.VERIFY,
                name="Verify project",
                criteria=["Tests pass if test files were generated."],
            ))
            return scaffold_tasks
        if _goal_is_verify_only(goal):
            # Keep upstream search/summarize tasks if any; drop everything
            # else and append a single standalone verify at the tail.
            kept: List[Task] = [
                t for t in tasks
                if t.kind in (TaskKind.SEARCH, TaskKind.SUMMARIZE)
            ]
            kept.append(Task(
                description="Run the project's pytest suite and report pass/fail.",
                kind=TaskKind.VERIFY,
                name="Verify tests pass",
                criteria=["Pytest is executed against the project root.",
                          "Test suite returns a zero exit code."],
            ))
            return kept
        if not _goal_is_change(goal):
            out: List[Task] = []
            for t in tasks:
                if t.kind not in (TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY):
                    out.append(t)
                    continue
                criteria = list(t.criteria) if t.criteria else [
                    "Answer cites at least one source from the indexed codebase."
                ]
                out.append(Task(
                    description=t.description,
                    kind=TaskKind.ASK,
                    name=t.name or _derive_name("", t.description),
                    inputs=dict(t.inputs),
                    criteria=criteria,
                ))
            return out
        # Change goal: drop stray apply/verify (we always append fresh
        # ones at the tail so prior_outputs ordering is well-defined),
        # then ensure the chain ends with [..., plan, apply, verify].
        # Also drop any stray scaffold tasks the LLM may have emitted for a
        # change goal (they would attempt to create a new project instead of
        # modifying the existing one).
        filtered = [t for t in tasks
                    if t.kind not in (TaskKind.APPLY, TaskKind.VERIFY, TaskKind.SCAFFOLD)]
        if not any(t.kind == TaskKind.PLAN for t in filtered):
            filtered.append(Task(
                description=goal.strip(),
                kind=TaskKind.PLAN,
                name=_derive_name("", goal.strip()),
                criteria=["Produces a unified diff against real files.",
                          "Cites at least one source from the index."],
            ))
        filtered.append(Task(
            description="Apply the proposed diffs to disk after a smoke test.",
            kind=TaskKind.APPLY,
            name="Apply diffs to disk",
            criteria=["Every diff applies without rejected hunks.",
                      "Modified files parse as valid syntax."],
        ))
        filtered.append(Task(
            description="Run impacted tests against the modified tree.",
            kind=TaskKind.VERIFY,
            name="Verify with tests",
            criteria=["Impacted tests located and executed.",
                      "Test suite returns a zero exit code."],
        ))
        return filtered

    def _llm_plan(self, goal: str, *,
                  candidates: Optional[List[str]] = None) -> List[Task]:
        user_parts = [f"Goal:\n{goal.strip()}"]
        if candidates:
            user_parts.append(
                "Candidate files surfaced by the index (prefer these when "
                "the goal targets specific code):\n- " + "\n- ".join(candidates)
            )
        try:
            resp = self.provider.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(user_parts) + "\n"},
                ],
                temperature=0.1,
                max_tokens=600,
                force_json=True,
            )
        except Exception:
            return []
        if not isinstance(resp, dict) or resp.get("error"):
            return []
        data = _extract_json(str(resp.get("content") or ""))
        if not isinstance(data, dict):
            return []
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            return []
        out: List[Task] = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            desc = str(t.get("description") or "").strip()
            if not desc:
                continue
            name = str(t.get("name") or t.get("title") or "").strip()
            criteria = [str(c) for c in (t.get("criteria") or []) if str(c).strip()]
            out.append(Task(
                description=desc,
                kind=_coerce_kind(str(t.get("kind") or "ask")),
                name=_derive_name(name, desc),
                criteria=criteria,
            ))
        return out

    def _fallback_plan(self, goal: str) -> List[Task]:
        """Deterministic plan when no LLM is available.

        Read-only goals collapse to a single ``ask`` task; verify-only
        goals collapse to a single ``verify`` task; code-change goals
        seed a single ``plan`` task — :meth:`_enforce_kind_policy` then
        appends the ``apply`` and ``verify`` follow-ups so the chain
        always terminates with a real-disk write + test gate.
        """
        goal_clean = goal.strip()
        if _goal_is_scaffold(goal):
            return [Task(
                description=goal_clean,
                kind=TaskKind.SCAFFOLD,
                name=_derive_name("", goal_clean),
                criteria=["Generates at least one source file.",
                          "All generated files have valid, working code."],
            )]
        if _goal_is_verify_only(goal):
            return [Task(
                description=goal_clean,
                kind=TaskKind.VERIFY,
                name=_derive_name("", goal_clean),
                criteria=["Pytest is executed against the project root.",
                          "Test suite returns a zero exit code."],
            )]
        if _goal_is_change(goal):
            return [Task(
                description=goal_clean,
                kind=TaskKind.PLAN,
                name=_derive_name("", goal_clean),
                criteria=["Diff is syntactically valid.",
                          "Diff targets the files implicated by the goal."],
            )]
        return [Task(
            description=goal_clean,
            kind=TaskKind.ASK,
            name=_derive_name("", goal_clean),
            criteria=["Answer cites at least one source from the indexed codebase."],
        )]
