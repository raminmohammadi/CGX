

"""Planner: decompose a user goal into a sequence of agent tasks.

The Planner prefers the LLM for plan generation but always returns a
useful plan even when no LLM is available -- the deterministic fallback
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

# The ``skills`` package lives at the repo root alongside ``src/``. It is
# wired into sys.path by ``tests/conftest.py`` for tests and by
# ``pyproject.toml``'s package discovery for installed runs. The import
# is defensive so the planner still works in stripped-down environments
# (single-file unit tests, docs builds) where the package isn't present.
try:  # pragma: no cover - exercised indirectly through every planner run
    import skills as _skills
except Exception:  # pragma: no cover - fall back to no skills
    _skills = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _detected_skill_names(goal: str) -> List[str]:
    """Return the names of skills that fire on *goal* (highest score first).

    Returns an empty list when the ``skills`` package is unavailable or
    when no skill's ``detect`` score crosses the registry threshold.
    """
    if _skills is None or not goal:
        return []
    try:
        return _skills.skill_names(_skills.detect_skills(goal))
    except Exception:
        return []


def _goal_has_supported_skill(goal: str) -> bool:
    """True when at least one non-style skill fires on *goal*.

    Style-only skills (Tailwind) cannot stand alone -- they require a
    frontend host -- so they're not strong enough on their own to flip
    the routing decision to SCAFFOLD.
    """
    if _skills is None or not goal:
        return False
    try:
        detected = _skills.detect_skills(goal)
    except Exception:
        return False
    return any(getattr(s, "role", "") != "style" for s in detected)


SYSTEM_PROMPT = (
    "You are a senior software engineer decomposing a user request into the "
    "smallest sequence of atomic agent tasks. Reply with strict JSON only.\n\n"
    "Schema:\n"
    "{\n"
    '  "rationale": "2-4 plain sentences explaining why this decomposition '
    'fits the goal: which technology layers you identified, why each task '
    'covers exactly one of them, and what risks the plan defends against.",\n'
    '  "tasks": [\n'
    '    {"name": "short title (max 8 words)",\n'
    '     "description": "single imperative sentence the capability will execute",\n'
    '     "kind": "ask|plan|scaffold|search|summarize|verify",\n'
    '     "criteria": ["..."]}\n'
    "  ]\n"
    "}\n\n"
    "Task kinds (pick the cheapest that satisfies the goal):\n"
    "- 'search'    -- retrieve relevant code/files from the index. Use this "
    "first whenever the goal references a file, symbol, or behaviour to "
    "inspect.\n"
    "- 'ask'       -- answer a natural-language question grounded in the "
    "indexed code. Use for 'what / how / why / where / explain / describe' "
    "goals.\n"
    "- 'summarize' -- condense the outputs of previous tasks into a brief "
    "human summary. Cheap; use as the final step for read-only goals.\n"
    "- 'verify'    -- run the project's pytest suite against the working "
    "tree and report pass/fail. Use when the goal asks to run tests / "
    "check whether tests pass / verify the suite, with no code change.\n"
    "- 'plan'      -- produce a code-change diff. ONLY use when the goal "
    "explicitly asks to add / implement / modify / refactor / fix / change "
    "/ update / remove / rename / replace / ensure / complete / fill code "
    "in an EXISTING codebase. "
    "NEVER use 'plan' for explanation, summarisation, or read-only "
    "inspection -- it is the most expensive kind and must not run unless "
    "code is being written.\n"
    "- 'scaffold'  -- generate files for a brand-new project (no existing "
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

# Verbs that imply *starting a brand-new project*. Kept separate from
# ``_CHANGE_VERB_RE`` (which also includes verbs like "fix"/"refactor"/"add"
# that only make sense against an existing codebase) so the scaffold
# detector can decide intent independently of the change detector.
_SCAFFOLD_VERB_RE = re.compile(
    r"\b(create|build|generate|scaffold|bootstrap|initialize|init|start|"
    r"make|design|develop|implement|write)\b",
    re.IGNORECASE,
)

# Frameworks / languages / UI libraries that, when paired with a scaffold
# verb, strongly indicate a brand-new project request even if no generic
# project noun (app/project/cli/...) is mentioned. Covers the common case
# "create a <thing> using React" / "build a <thing> with FastAPI".
_TECH_RE = re.compile(
    r"\b(react|vue|angular|svelte|next\.?js|nuxt|remix|"
    r"fastapi|flask|django|express|node\.?js|nest\.?js|"
    r"tkinter|pyqt|pyside|qt|electron|streamlit|gradio|"
    r"react\s+native|flutter|swiftui|tauri|"
    r"rails|laravel|spring|asp\.?net|"
    r"python|javascript|typescript|rust|golang|kotlin|java|"
    r"html|css|tailwind|bootstrap)\b",
    re.IGNORECASE,
)

# Phrases that explicitly point at an existing codebase. When any of these
# fire we keep the goal on the change-goal path even if a scaffold verb +
# tech name appear together (e.g. "add a React component to our existing app").
_EXISTING_CODE_HINT_RE = re.compile(
    r"\b("
    r"existing|current|legacy|"
    r"(?:in|to|inside|within)\s+(?:the|our|this|my|an?\s+existing)\s+"
    r"(?:app|application|project|codebase|code\s*base|repo|repository|"
    r"module|file|service|api|backend|frontend|component)|"
    r"already\s+(?:have|exists|implemented)|"
    r"refactor|modif(?:y|ies|ied)|fix\s+(?:the\s+)?bug"
    r")\b",
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
    r"library|package|module|repo|repository|backend|frontend|fullstack|bot|chatbot|"
    # Concrete project archetypes that come up in real user prompts but
    # weren't in the original noun list ("create a calculator using React").
    r"calculator|dashboard|todo|blog|game|chat|editor|tracker|portfolio|landing\s*page|"
    r"form|page|site|gui|interface|ui)\b"
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

# Phrases that signal an *exploratory* read-only goal -- the user is
# asking for suggestions / ideas / forward-looking directions rather
# than a factual lookup of what the code currently does. Answers to
# these goals are inherently partly speculative, so the Judge should
# not demand source citations for every suggestion.
_EXPLORATORY_RE = re.compile(
    r"\b("
    # forward-looking verbs
    r"improve|enhance|optimi[sz]e|"
    # solicitations for ideas
    r"suggest|recommend|propose|brainstorm|"
    r"ideas?\s+(?:for|on|about|to)|"
    # open-ended interrogatives
    r"how\s+(?:can|could|might|should|would)\s+(?:i|we|you|one|the)|"
    r"what\s+(?:are\s+)?(?:some\s+|other\s+)*"
    r"(?:ways?|approaches?|options?|methods?|strategies?|techniques?|"
    r"directions?|improvements?|alternatives?|possibilities)|"
    r"alternative\s+(?:approach|method|way|technique|strategy)|"
    r"explore|investigate|consider"
    r")\b",
    re.IGNORECASE,
)


def _goal_mentions_existing_code(goal: str) -> bool:
    """True when *goal* explicitly references an existing codebase / file.

    Used to keep "add a React component to our existing app" on the
    change-goal path even though it pairs a scaffold-friendly verb with a
    tech name.
    """
    if not goal:
        return False
    return _EXISTING_CODE_HINT_RE.search(goal) is not None


def _goal_is_scaffold(goal: str) -> bool:
    """True when *goal* asks to generate an entirely new project from scratch.

    Four signals are accepted:
      1. The original verb + project-noun regex (`create a new FastAPI app`).
      2. A scaffold-friendly verb together with a framework / language name
         from ``_TECH_RE`` and no "existing-codebase" hint
         (`create a calculator using React`). ``_TECH_RE`` is intentionally
         broader than the skills registry so unsupported techs (Angular,
         Svelte, Tkinter, Electron, Rails, …) still route to SCAFFOLD
         even when no dedicated skill exists.
      3. A scaffold-friendly verb together with at least one supported,
         non-style skill firing via :func:`skills.detect_skills`. This
         path is more precise than ``_TECH_RE`` (no false matches against
         words that appear unrelated to a real technology mention).
      4. Explicit `from scratch` / `new <project-noun>` phrasing.
    """
    if not goal:
        return False
    if _SCAFFOLD_RE.search(goal) is not None:
        return True
    has_scaffold_verb = _SCAFFOLD_VERB_RE.search(goal) is not None
    no_existing_hint = not _goal_mentions_existing_code(goal)
    if has_scaffold_verb and no_existing_hint:
        # Pattern (2): scaffold verb + tech-regex mention.
        if _TECH_RE.search(goal) is not None:
            return True
        # Pattern (3): scaffold verb + skill-detected technology.
        if _goal_has_supported_skill(goal):
            return True
    return False


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


def _goal_is_exploratory(goal: str) -> bool:
    """True when *goal* asks for suggestions/ideas rather than a factual lookup.

    Read-only goals like "improve indexing accuracy" or "what are some ways
    to speed this up?" produce inherently speculative answers; the Judge
    must not demand source citations for every forward-looking suggestion.
    Code-change and verify-only goals are excluded so they keep their own
    routing.
    """
    if not goal:
        return False
    if _goal_is_change(goal) or _goal_is_verify_only(goal):
        return False
    return _EXPLORATORY_RE.search(goal) is not None


# Engine mode name used for exploratory ASK tasks. Defined here so the
# planner, judge, and tests share a single source of truth.
CLARIFY_PATHS_MODE = "clarify_paths"


def _default_ask_criteria(goal: str, *, continuation: bool = False) -> List[str]:
    """Default acceptance criteria for a read-only ASK task.

    Exploratory goals (see :func:`_goal_is_exploratory`) get a
    clarify-shaped criterion: the answer must enumerate multiple
    concrete directions or ask the user to narrow the goal, rather
    than asserting unverifiable speculation. All other read-only
    goals keep the citation requirement so a factual answer is
    grounded in the indexed codebase.

    When ``continuation`` is True, the goal is treated as a narrowed
    follow-up (the user has already replied to a previous clarify
    ASK) so the clarify-shaped criterion is suppressed and the answer
    is held to the standard citation requirement.
    """
    if not continuation and _goal_is_exploratory(goal):
        return [
            "Answer enumerates multiple concrete directions or asks a "
            "clarifying question to narrow the open-ended goal.",
        ]
    return ["Answer cites at least one source from the indexed codebase."]


def _ask_inputs_for_goal(
    goal: str, base: Optional[Dict[str, Any]] = None,
    *, continuation: bool = False,
) -> Dict[str, Any]:
    """Compose the ``inputs`` dict for a read-only ASK task.

    Exploratory goals route through the engine's ``clarify_paths`` mode
    so the answer enumerates options + asks a follow-up question rather
    than producing a single grounded answer. Non-exploratory goals keep
    whatever inputs the caller already had (typically empty).
    """
    inputs: Dict[str, Any] = dict(base or {})
    if not continuation and _goal_is_exploratory(goal):
        inputs.setdefault("mode_override", CLARIFY_PATHS_MODE)
        inputs.setdefault("goal", goal.strip())
    return inputs


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


def _fallback_rationale(goal: str, tasks: List[Task]) -> str:
    """Compose a short rationale string when the LLM didn't supply one.

    The text is purely descriptive \u2014 it names the routing path the
    planner took (scaffold / verify-only / change / read-only) and the
    detected skills if any \u2014 so the UI's "Plan Rationale" card has
    something to show even on the deterministic fallback path.
    """
    if not tasks:
        return ""
    kinds = [t.kind.value for t in tasks]
    skills = _detected_skill_names(goal)
    skill_phrase = ""
    if skills:
        skill_phrase = (f" Detected skills: {', '.join(skills)}.")
    if _goal_is_scaffold(goal) or TaskKind.SCAFFOLD in {t.kind for t in tasks} \
            or TaskKind.SCAFFOLD_MANIFEST in {t.kind for t in tasks}:
        return ("Routing as a new-project scaffold: a manifest call plans the "
                "file structure, then one focused call per file generates the "
                "content, followed by apply + verify."
                + skill_phrase)
    if _goal_is_verify_only(goal):
        return ("Routing as a verify-only goal: running the existing test "
                "suite without producing or applying any diffs.")
    if _goal_is_change(goal):
        return ("Routing as a code-change against the existing codebase: "
                "produce a diff, apply it, then run the impacted tests."
                + skill_phrase)
    return ("Routing as a read-only question: retrieve relevant code and "
            "answer without modifying the codebase.")


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

    def plan(self, goal: str, *, continuation: bool = False,
             prior_goal: Optional[str] = None) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Planner.plan: goal must be non-empty")
        # ``prior_goal`` is only meaningful when ``continuation`` is True:
        # the user is replying to a previous clarify ASK and ``goal`` is the
        # narrowed pick; ``prior_goal`` is the original open-ended request
        # they're still trying to make progress on. Drop it otherwise so the
        # LLM prompt isn't muddied with stale context.
        carried_prior = (prior_goal or "").strip() if continuation else ""
        logger.info("Planner.plan: starting goal=%r provider=%s retriever=%s "
                    "continuation=%s prior_goal=%r",
                    goal[:80], type(self.provider).__name__ if self.provider else "None",
                    "yes" if self.retriever else "no", bool(continuation),
                    carried_prior[:60])
        candidates = self._retrieve_candidates(goal)
        tasks: List[Task]
        rationale_box: List[str] = []
        if self.provider is not None:
            logger.info("Planner: calling LLM for plan decomposition")
            tasks = self._llm_plan(goal, candidates=candidates,
                                   rationale_box=rationale_box,
                                   prior_goal=carried_prior or None)
            logger.info("Planner: LLM returned %d tasks", len(tasks))
        else:
            tasks = []
        if not tasks:
            logger.info("Planner: using deterministic fallback plan")
            tasks = self._fallback_plan(goal, continuation=continuation)
        tasks = self._enforce_kind_policy(goal, tasks, continuation=continuation)
        if not continuation:
            tasks = self._enforce_exploratory_clarification(goal, tasks)
        rationale = (rationale_box[0] if rationale_box
                     else _fallback_rationale(goal, tasks))
        plan = Plan(goal=goal.strip(),
                    tasks=tasks[: self.max_tasks],
                    rationale=rationale)
        # Wire sequential dependencies so the UI can render the execution DAG.
        for i in range(1, len(plan.tasks)):
            plan.tasks[i].dependencies = [plan.tasks[i - 1].id]
        logger.info("Planner: plan ready id=%s tasks=%d kinds=%s",
                    plan.id, len(plan.tasks), [t.kind.value for t in plan.tasks])
        return plan

    def plan_fix(
        self,
        goal: str,
        *,
        broken_files: Optional[List[str]] = None,
        already_good_files: Optional[List[str]] = None,
        prior_owned_files: Optional[Dict[str, str]] = None,
    ) -> Plan:
        """Produce a retry plan that never re-scaffolds.

        Bypasses the goal-classification heuristics in
        :meth:`_enforce_kind_policy`. Those heuristics inspect the goal
        text and will route any string containing the embedded
        ``Original goal: create a calculator using React ...`` phrasing
        back through SCAFFOLD_MANIFEST + APPLY + VERIFY, which re-emits
        every scaffold task and overwrites files that already passed.

        The returned plan is always ``[PLAN, APPLY, VERIFY]``:

        * ``PLAN`` carries the diagnostic fix-goal as its description and,
          when supplied, ``inputs["target_files"]`` / ``inputs["do_not_change"]``
          so the engine can scope its diff generation.
        * ``APPLY`` writes the resulting diffs to disk (only the broken
          files are diff-ed, so nothing else is touched).
        * ``VERIFY`` reruns impacted tests.

        ``prior_owned_files`` (the manifest from the previous attempt) is
        copied onto the new plan so the Tracker can recognise which files
        are already on disk and skip any stray scaffold work.
        """
        if not goal or not goal.strip():
            raise ValueError("Planner.plan_fix: goal must be non-empty")
        broken_files = list(broken_files or [])
        already_good_files = list(already_good_files or [])
        detected = _detected_skill_names(goal)
        plan_task = Task(
            description=goal.strip(),
            kind=TaskKind.PLAN,
            name=_derive_name("", "Fix failing files"),
            criteria=["Diff targets only the broken files.",
                      "Diff is syntactically valid."],
        )
        if broken_files:
            plan_task.inputs["target_files"] = broken_files
        if already_good_files:
            plan_task.inputs["do_not_change"] = already_good_files
        if detected:
            plan_task.inputs.setdefault("skills", list(detected))
        apply_task = Task(
            description="Apply the proposed diffs to disk after a smoke test.",
            kind=TaskKind.APPLY,
            name="Apply diffs to disk",
            criteria=["Every diff applies without rejected hunks.",
                      "Modified files parse as valid syntax."],
        )
        verify_task = Task(
            description="Run impacted tests against the modified tree.",
            kind=TaskKind.VERIFY,
            name="Verify with tests",
            criteria=["Impacted tests located and executed.",
                      "Test suite returns a zero exit code."],
        )
        plan = Plan(
            goal=goal.strip(),
            tasks=[plan_task, apply_task, verify_task],
            rationale=("Fix-plan: regenerate only the broken file(s) and "
                       "re-verify; files already on disk are preserved."),
            owned_files=dict(prior_owned_files or {}),
        )
        for i in range(1, len(plan.tasks)):
            plan.tasks[i].dependencies = [plan.tasks[i - 1].id]
        logger.info(
            "Planner.plan_fix: id=%s broken=%d kept=%d carried_owned=%d",
            plan.id, len(broken_files), len(already_good_files),
            len(plan.owned_files),
        )
        return plan

    def _retrieve_candidates(self, goal: str) -> List[str]:
        """Call the optional retriever and return up to 8 candidate file paths.

        When the goal's scope resolves to ``"src"`` (production code), test
        paths are dropped from ``top_files`` even though they may rank high
        in the hybrid retriever's output. ``run_query_auto`` only soft-
        penalises ``hits``; ``top_files`` aggregates per-file ranks before
        the penalty is applied, so test files can still surface here and
        mislead the LLM into building VERIFY tasks around unrelated test
        modules.
        """
        if self.retriever is None:
            return []
        try:
            result = self.retriever(goal)
        except Exception as e:
            # Some failure modes (FAISS dim-mismatch assertions, KeyErrors with
            # no message) produce a blank ``str(e)`` which hides the real cause
            # in the logs. Surface type+repr so future debugging is possible.
            logger.warning(
                "Planner: retriever failed: %s: %r", type(e).__name__, e,
            )
            return []
        if not isinstance(result, dict):
            return []
        top_files = result.get("top_files") or []
        from cgx.answer.scope import detect_scope, _is_test_path
        scope = detect_scope(goal)
        out: List[str] = []
        dropped_tests = 0
        for f in top_files:
            if isinstance(f, dict):
                fp = str(f.get("file") or f.get("path") or "").strip()
            elif isinstance(f, str):
                fp = f.strip()
            else:
                continue
            if not fp:
                continue
            if scope == "src" and _is_test_path(fp):
                dropped_tests += 1
                continue
            out.append(fp)
            if len(out) >= 8:
                break
        if dropped_tests:
            logger.info(
                "Planner: dropped %d test-path candidate(s) for scope=src",
                dropped_tests,
            )
        return out

    def _enforce_kind_policy(
        self, goal: str, tasks: List[Task],
        *, continuation: bool = False,
    ) -> List[Task]:
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
        # When the LLM smartly emitted scaffold tasks for a goal whose
        # phrasing didn't trip the regex (e.g. "create a calculator using
        # React + python"), trust that decomposition as long as the goal
        # doesn't explicitly reference an existing codebase.
        llm_emitted_scaffold = any(t.kind == TaskKind.SCAFFOLD for t in tasks)
        scaffold_via_llm = (
            llm_emitted_scaffold
            and not _goal_mentions_existing_code(goal)
        )
        if scaffold_via_llm and not _goal_is_scaffold(goal):
            logger.info(
                "Planner: routing as SCAFFOLD because LLM emitted scaffold "
                "task(s) and goal has no existing-codebase hint")
        if _goal_is_scaffold(goal) or scaffold_via_llm:
            detected = _detected_skill_names(goal)
            logger.info(
                "Planner: kind-policy SCAFFOLD path → SCAFFOLD_MANIFEST "
                "(regex=%s llm=%s skills=%s)",
                _goal_is_scaffold(goal), scaffold_via_llm, detected)
            # Emit a single SCAFFOLD_MANIFEST task. At runtime the manifest
            # capability calls plan_scaffold_manifest() (cheap LLM call, no
            # file contents) and injects one SCAFFOLD_FILE task per file into
            # the Tracker's plan list before APPLY runs. This gives the UI
            # full per-file visibility and validates each file before moving on.
            manifest_task = Task(
                description=goal.strip(),
                kind=TaskKind.SCAFFOLD_MANIFEST,
                name="Plan project file structure",
                criteria=["Returns a non-empty file manifest with at least one layer.",
                          "All planned files have relative POSIX paths."],
            )
            manifest_task.inputs["goal"] = goal.strip()
            if detected:
                manifest_task.inputs["skills"] = list(detected)
            return [
                manifest_task,
                Task(
                    description="Write all generated project files to the output directory.",
                    kind=TaskKind.APPLY,
                    name="Write project files",
                    criteria=["All files written to disk without errors.",
                              "No files have syntax errors."],
                ),
                Task(
                    description="Verify the generated project runs or its tests pass.",
                    kind=TaskKind.VERIFY,
                    name="Verify project",
                    criteria=["Tests pass if test files were generated."],
                ),
            ]
        if _goal_is_verify_only(goal):
            logger.info("Planner: kind-policy VERIFY-ONLY path")
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
            rewrote = 0
            for t in tasks:
                if t.kind not in (TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY):
                    out.append(t)
                    continue
                rewrote += 1
                criteria = (list(t.criteria) if t.criteria
                            else _default_ask_criteria(
                                goal, continuation=continuation))
                out.append(Task(
                    description=t.description,
                    kind=TaskKind.ASK,
                    name=t.name or _derive_name("", t.description),
                    inputs=_ask_inputs_for_goal(
                        goal, t.inputs, continuation=continuation),
                    criteria=criteria,
                ))
            if rewrote:
                logger.info(
                    "Planner: kind-policy READ-ONLY path (downgraded %d "
                    "plan/apply/verify task(s) → ask)", rewrote)
            else:
                logger.info(
                    "Planner: kind-policy READ-ONLY path (no rewriting needed)")
            return out
        detected = _detected_skill_names(goal)
        logger.info(
            "Planner: kind-policy CHANGE-GOAL path (plan→apply→verify) skills=%s",
            detected)
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
        # Attach detected skill names to every PLAN task so the engine
        # can compose plan-time prompt fragments and the Judge can run
        # plan-time skill validators. Use per-task detection (over the
        # task description) when available so a layered change-goal
        # plan attaches only the relevant skill to each PLAN task, and
        # fall back to the global goal-level detection otherwise.
        for t in filtered:
            if t.kind == TaskKind.PLAN:
                per_task = _detected_skill_names(t.description or "")
                attach = per_task or list(detected)
                if attach:
                    t.inputs.setdefault("skills", attach)
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

    def _enforce_exploratory_clarification(
        self, goal: str, tasks: List[Task],
    ) -> List[Task]:
        """Reshape an exploratory plan so it terminates in one clarify ASK.

        ``_fallback_plan`` already terminates exploratory goals in an ASK
        carrying ``mode_override="clarify_paths"``. LLM-emitted plans for
        the same goals are inconsistent and routinely break the contract
        in three ways at once:

        * A mid-plan ``SUMMARIZE`` describes the existing code rather
          than recommending directions; the LLM-judge correctly fails
          it, which halts the plan before the terminal ASK ever runs.
        * Multiple ASK tasks chained at the tail duplicate work and
          waste tokens; only the last one needs to enumerate options.
        * A terminal ASK without ``mode_override`` falls back to the
          engine's default intent prompt instead of ``clarify_paths``.

        Policy applied here (all SEARCH tasks are kept verbatim because
        they're cheap retrieval that grounds the final ASK via
        ``_prior_outputs``):

        1. Drop every ``SUMMARIZE`` task.
        2. Drop every ASK that isn't the terminal task; the trailing
           clarify ASK consumes prior search outputs and supersedes
           any earlier descriptive ASK.
        3. Ensure the result ends in exactly one ASK with
           ``mode_override="clarify_paths"`` -- stamping an existing
           terminal ASK in place when present, otherwise appending a
           fresh one. The appended ASK's description is inherited from
           the dropped terminal (if any) so the UI keeps a meaningful
           title, and falls back to the goal text.

        Non-exploratory goals are left untouched.
        """
        if not _goal_is_exploratory(goal) or not tasks:
            return tasks
        goal_clean = goal.strip()
        original_kinds = [t.kind.value for t in tasks]
        # Step 1+2: keep non-ASK, non-SUMMARIZE tasks (mostly SEARCH /
        # PLAN-already-downgraded-to-ASK won't appear here because
        # `_enforce_kind_policy` ran first and turned plan/apply/verify
        # into ASK). We track the terminal task separately so we can
        # inherit its description if it carried something distinctive.
        kept: List[Task] = []
        for t in tasks[:-1]:
            if t.kind == TaskKind.SUMMARIZE:
                continue
            if t.kind == TaskKind.ASK:
                continue
            kept.append(t)
        terminal = tasks[-1]
        # Step 3: derive the terminal clarify ASK. Stamp in place when
        # the LLM already chose ASK; otherwise compose a fresh one and
        # carry the dropped task's description forward when useful.
        if terminal.kind == TaskKind.ASK:
            if (terminal.inputs or {}).get("mode_override") != CLARIFY_PATHS_MODE:
                terminal.inputs = _ask_inputs_for_goal(goal, terminal.inputs)
                if not terminal.criteria:
                    terminal.criteria = _default_ask_criteria(goal)
            kept.append(terminal)
        else:
            kept.append(Task(
                description=(terminal.description or goal_clean),
                kind=TaskKind.ASK,
                name=_derive_name("", goal_clean),
                inputs=_ask_inputs_for_goal(goal_clean),
                criteria=_default_ask_criteria(goal_clean),
            ))
        new_kinds = [t.kind.value for t in kept]
        if new_kinds != original_kinds:
            logger.info(
                "Planner: exploratory contract: %s → %s",
                original_kinds, new_kinds)
        return kept

    def _llm_plan(self, goal: str, *,
                  candidates: Optional[List[str]] = None,
                  rationale_box: Optional[List[str]] = None,
                  prior_goal: Optional[str] = None) -> List[Task]:
        """Call the LLM and return the parsed tasks.

        When ``rationale_box`` is supplied, the model-provided
        ``rationale`` (if any) is appended to it so :meth:`plan` can
        attach it to the :class:`Plan` without changing the function's
        primary return shape.

        When ``prior_goal`` is supplied, the user's broader objective is
        prepended to the prompt so the LLM keeps the narrowed follow-up
        anchored in the original request rather than drifting into a
        generic deep-dive on whatever the user clicked.
        """
        user_parts: List[str] = []
        prior = (prior_goal or "").strip()
        if prior:
            user_parts.append(
                "Original user objective (the broader goal the user is "
                "still trying to achieve; keep the plan aligned with "
                f"this):\n{prior}"
            )
            user_parts.append(
                "Narrowed follow-up the user picked to focus on next "
                f"(treat as the immediate goal):\n{goal.strip()}"
            )
        else:
            user_parts.append(f"Goal:\n{goal.strip()}")
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
                max_tokens=1000,
                force_json=True,
            )
        except Exception as e:
            logger.warning("Planner: LLM chat raised %s: %s", type(e).__name__, e)
            return []
        if not isinstance(resp, dict):
            logger.warning("Planner: LLM returned non-dict response: %r", type(resp).__name__)
            return []
        if resp.get("error"):
            logger.warning("Planner: LLM returned error -- %s", resp.get("error"))
            return []
        data = _extract_json(str(resp.get("content") or ""))
        if not isinstance(data, dict):
            logger.warning("Planner: LLM response was not valid JSON (content head=%r)",
                           str(resp.get("content") or "")[:160])
            return []
        if rationale_box is not None:
            rat = str(data.get("rationale") or "").strip()
            if rat:
                rationale_box.append(rat)
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

    def _fallback_plan(
        self, goal: str, *, continuation: bool = False,
    ) -> List[Task]:
        """Deterministic plan when no LLM is available.

        Routing summary:
        * Scaffold goals → ``[scaffold_manifest]``; the file-content
          tasks are injected at runtime by the manifest capability.
        * Verify-only goals → ``[verify]``.
        * Code-change goals → ``[plan]``; :meth:`_enforce_kind_policy`
          appends ``apply``+``verify`` so the chain always terminates
          with a real-disk write + test gate.
        * Exploratory read-only goals (see :func:`_goal_is_exploratory`)
          → ``[search, ask]`` so the UI shows a visible investigation
          step before the clarification, and the ASK can ground its
          enumerated options in the search's hits. Suppressed when
          ``continuation`` is True (the user is replying to a prior
          clarify ASK and expects a focused answer).
        * Plain read-only goals → ``[ask]``.
        """
        goal_clean = goal.strip()
        if _goal_is_scaffold(goal):
            return [Task(
                description=goal_clean,
                kind=TaskKind.SCAFFOLD_MANIFEST,
                name="Plan project file structure",
                criteria=["Returns a non-empty file manifest with at least one layer.",
                          "All planned files have relative POSIX paths."],
                inputs={"goal": goal_clean},
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
        if not continuation and _goal_is_exploratory(goal_clean):
            return [
                Task(
                    description=goal_clean,
                    kind=TaskKind.SEARCH,
                    name="Locate relevant code",
                    inputs={"goal": goal_clean},
                    criteria=["Search returns at least one indexed hit."],
                ),
                Task(
                    description=goal_clean,
                    kind=TaskKind.ASK,
                    name=_derive_name("", goal_clean),
                    inputs=_ask_inputs_for_goal(goal_clean),
                    criteria=_default_ask_criteria(goal_clean),
                ),
            ]
        return [Task(
            description=goal_clean,
            kind=TaskKind.ASK,
            name=_derive_name("", goal_clean),
            inputs=_ask_inputs_for_goal(goal_clean, continuation=continuation),
            criteria=_default_ask_criteria(goal_clean, continuation=continuation),
        )]
