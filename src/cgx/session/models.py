

"""Domain dataclasses for the session-shaped agent backbone.

Plain :mod:`dataclasses` (no Pydantic) to keep the package
dependency-light and JSON-serialise cleanly via ``to_dict()`` --
matching the convention already used by :mod:`cgx.sessions`. Pydantic
stays at the webui wire boundary.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def _now() -> float:
    return time.time()


def _gen_id(prefix: str = "") -> str:
    h = uuid.uuid4().hex[:16]
    return f"{prefix}{h}" if prefix else h


# --------------------- enums ---------------------

class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class TaskNodeStatus(str, enum.Enum):
    """Lifecycle states for a :class:`TaskNode`.

    ``PENDING`` is the initial state; ``BLOCKED`` is set when blockers
    are present; ``READY`` is set by the router once blockers clear and
    an executor can pick the task up.
    """
    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    ABANDONED = "abandoned"


class TaskKind(str, enum.Enum):
    """Typed kinds of work the router can spawn.

    Each kind has its own input/output schema and executor (added in
    later phases). EXPLORE/INVESTIGATE/RECOMMEND form the read-only
    continuation loop; PLAN_CHANGE/APPLY/VERIFY drive code changes;
    ASK_USER is the structured-pause primitive; SEARCH/SUMMARIZE are
    utility kinds the router may interleave.

    Greenfield kinds (CLARIFY_REQUIREMENTS, DECOMPOSE, SCAFFOLD,
    BOOTSTRAP_ENV) drive the new-project path that spawns when
    ``Session.mode == 'greenfield'``.

    REPAIR is the auto-correction kind: spawned after a VERIFY that
    failed with a classifiable cause (e.g. unittest/pytest helper
    mismatch). It emits a typed plan; the existing APPLY executor
    consumes it like any other diff source.

    RUNTIME_VERIFY is the post-VERIFY runtime gate (greenfield only):
    once the unit suite the model wrote passes, it boots the scaffolded
    app / import-and-call smokes the entry modules so "the tests pass"
    becomes "the app actually runs".
    """
    EXPLORE = "explore"
    INVESTIGATE = "investigate"
    RECOMMEND = "recommend"
    PLAN_CHANGE = "plan_change"
    APPLY = "apply"
    VERIFY = "verify"
    RUNTIME_VERIFY = "runtime_verify"
    ASK_USER = "ask_user"
    SEARCH = "search"
    SUMMARIZE = "summarize"
    CLARIFY_REQUIREMENTS = "clarify_requirements"
    DECOMPOSE = "decompose"
    SCAFFOLD = "scaffold"
    BOOTSTRAP_ENV = "bootstrap_env"
    API_CHECK = "api_check"
    SMOKE = "smoke"
    REPAIR = "repair"
    AST_REGENERATE = "ast_regenerate"


class FactKind(str, enum.Enum):
    FILE = "file"
    SYMBOL = "symbol"
    PARAMETER = "parameter"
    ANCHOR = "anchor"
    LLM_CALL = "llm_call"


class ArtifactKind(str, enum.Enum):
    DIRECTIONS_LIST = "directions_list"
    FINDINGS_BUNDLE = "findings_bundle"
    RECOMMENDATION_LIST = "recommendation_list"
    CODE_CHANGE_PLAN = "code_change_plan"
    APPLIED_CHANGES = "applied_changes"
    VERIFY_REPORT = "verify_report"
    RUNTIME_REPORT = "runtime_report"
    SESSION_DIGEST = "session_digest"
    REQUIREMENTS_SHEET = "requirements_sheet"
    WORK_PLAN = "work_plan"
    SCAFFOLD_PATCHES = "scaffold_patches"
    BUILD_REPORT = "build_report"
    API_CHECK_REPORT = "api_check_report"
    SMOKE_REPORT = "smoke_report"
    REPAIR_PLAN = "repair_plan"


class DecisionKind(str, enum.Enum):
    CHOOSE_PATH = "choose_path"
    CHOOSE_RECOMMENDATION = "choose_recommendation"
    APPROVE = "approve"
    FREEFORM = "freeform"
    CLARIFY_ANSWERS = "clarify_answers"
    APPROVE_PLAN = "approve_plan"


# --------------------- core dataclasses ---------------------

@dataclass
class Fact:
    """A single piece of session knowledge.

    Append-only. Updates set ``stale=True`` rather than mutating
    ``content``; refresh is the responsibility of consuming tasks
    (mtime check / re-retrieval).
    """
    fact_id: str
    session_id: str
    kind: FactKind
    content: Dict[str, Any]
    surfaced_in_task_id: Optional[str] = None
    stale: bool = False
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    @classmethod
    def new(cls, session_id: str, kind: FactKind, content: Dict[str, Any],
            surfaced_in_task_id: Optional[str] = None) -> "Fact":
        return cls(
            fact_id=_gen_id("fact_"),
            session_id=session_id,
            kind=kind,
            content=dict(content),
            surfaced_in_task_id=surfaced_in_task_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d



@dataclass
class Decision:
    """Structured record of a user choice resolving an ``ASK_USER`` task.

    Downstream tasks reference decisions by ``decision_id`` rather than
    re-parsing user text. ``chosen`` is a typed dict whose shape
    depends on :class:`DecisionKind` (e.g. ``{"anchor_chunk_id": ...,
    "title": ...}`` for ``CHOOSE_PATH``).
    """
    decision_id: str
    session_id: str
    resolved_task_id: str
    kind: DecisionKind
    question: str
    chosen: Dict[str, Any]
    rationale: Optional[str] = None
    made_at: float = field(default_factory=_now)

    @classmethod
    def new(cls, session_id: str, resolved_task_id: str, kind: DecisionKind,
            question: str, chosen: Dict[str, Any],
            rationale: Optional[str] = None) -> "Decision":
        return cls(
            decision_id=_gen_id("dec_"),
            session_id=session_id,
            resolved_task_id=resolved_task_id,
            kind=kind,
            question=question,
            chosen=dict(chosen),
            rationale=rationale,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class Artifact:
    """A typed output produced by a finished task (e.g. a recommendation
    list, code-change plan, verify report). Survives across turns and
    is the system of record for "what the agent has produced so far".
    """
    artifact_id: str
    session_id: str
    produced_by_task_id: str
    kind: ArtifactKind
    content: Dict[str, Any]
    created_at: float = field(default_factory=_now)

    @classmethod
    def new(cls, session_id: str, produced_by_task_id: str,
            kind: ArtifactKind, content: Dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=_gen_id("art_"),
            session_id=session_id,
            produced_by_task_id=produced_by_task_id,
            kind=kind,
            content=dict(content),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class TaskNode:
    """A node in the per-session task tree.

    Children are spawned dynamically by the router as parents complete;
    blockers list sibling/ancestor task ids that must finish before
    ``status`` transitions to ``READY``.
    """
    task_id: str
    session_id: str
    kind: TaskKind
    name: str
    description: str = ""
    parent_task_id: Optional[str] = None
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    consumed_decision_ids: List[str] = field(default_factory=list)
    produced_artifact_id: Optional[str] = None
    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @classmethod
    def new(cls, session_id: str, kind: TaskKind, name: str, *,
            description: str = "", parent_task_id: Optional[str] = None,
            inputs: Optional[Dict[str, Any]] = None,
            blockers: Optional[List[str]] = None) -> "TaskNode":
        status = (TaskNodeStatus.BLOCKED if blockers
                  else TaskNodeStatus.READY)
        return cls(
            task_id=_gen_id("task_"),
            session_id=session_id,
            kind=kind,
            name=name,
            description=description,
            parent_task_id=parent_task_id,
            inputs=dict(inputs or {}),
            blockers=list(blockers or []),
            status=status,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        return d


class SessionMode(str, enum.Enum):
    """How the router drives the session.

    * ``EXPLORE`` -- the default, FAISS-backed read/write loop over an
      existing indexed codebase (``EXPLORE -> INVESTIGATE -> RECOMMEND
      -> PLAN_CHANGE -> APPLY -> VERIFY``).
    * ``GREENFIELD`` -- new-project scaffolding loop
      (``CLARIFY_REQUIREMENTS -> DECOMPOSE -> SCAFFOLD -> APPLY ->
      VERIFY``) that does not require a prebuilt index.
    """
    EXPLORE = "explore"
    GREENFIELD = "greenfield"


@dataclass
class Session:
    """The long-lived unit of persistent agent work.

    A session owns one task tree, one knowledge base, and one decision
    log. New objectives spawn new sessions; continuations stay within
    the same session.
    """
    session_id: str
    title: str
    original_objective: str
    status: SessionStatus = SessionStatus.ACTIVE
    mode: SessionMode = SessionMode.EXPLORE
    current_focus: Optional[str] = None
    root_task_id: Optional[str] = None
    project_root: Optional[str] = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    # Optional per-session resource budget (E1). ``None`` means unlimited
    # for that axis. The runner escalates when either cap is reached --
    # an interactive session pauses on a fresh ASK_USER, a ``headless``
    # session ends terminally FAILED -- rather than looping until the
    # drain's coarse ``max_steps`` safety valve trips.
    max_task_runs: Optional[int] = None
    max_wall_seconds: Optional[float] = None
    headless: bool = False
    # Live counters the runner maintains as it dispatches work tasks
    # (everything except the ASK_USER pause primitive).
    task_runs: int = 0
    first_task_started_at: Optional[float] = None
    # Explicit skill names chosen at session creation (e.g. via an Agent
    # Profile), used by the plan/scaffold executors in place of
    # auto-detecting from goal text. Empty means "auto-detect" -- see
    # ``cgx.answer.engine._resolve_skills``.
    skills: List[str] = field(default_factory=list)

    @classmethod
    def new(cls, original_objective: str, *, title: Optional[str] = None,
            project_root: Optional[str] = None,
            mode: SessionMode = SessionMode.EXPLORE,
            max_task_runs: Optional[int] = None,
            max_wall_seconds: Optional[float] = None,
            headless: bool = False,
            skills: Optional[List[str]] = None) -> "Session":
        t = (title or original_objective).strip()
        if len(t) > 80:
            t = t[:77] + "..."
        return cls(
            session_id=_gen_id("ses_"),
            title=t or "Untitled",
            original_objective=original_objective,
            project_root=project_root,
            mode=mode,
            max_task_runs=max_task_runs,
            max_wall_seconds=max_wall_seconds,
            headless=headless,
            skills=list(skills or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["mode"] = self.mode.value
        return d


# --------------------- aggregate views ---------------------

@dataclass
class KnowledgeBase:
    """In-memory aggregate of a session's :class:`Fact`s.

    The persistent backing store is SQLite; this dataclass is the
    convenient view passed to executors / the router. It is rebuilt
    from disk on session resume by ``SessionStore.load_kb``.
    """
    session_id: str
    facts: Dict[str, Fact] = field(default_factory=dict)

    def add(self, fact: Fact) -> None:
        self.facts[fact.fact_id] = fact

    def of_kind(self, kind: FactKind) -> List[Fact]:
        return [f for f in self.facts.values() if f.kind is kind]

    def find_anchor(self, chunk_id: str) -> Optional[Fact]:
        for f in self.facts.values():
            if f.kind is FactKind.ANCHOR and f.content.get("chunk_id") == chunk_id:
                return f
        return None

    def mark_stale(self, fact_ids: Iterable[str]) -> None:
        for fid in fact_ids:
            f = self.facts.get(fid)
            if f is not None:
                f.stale = True
                f.updated_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
        }


@dataclass
class DecisionLog:
    """In-memory aggregate of a session's :class:`Decision`s.

    Rebuilt from disk on session resume by
    ``SessionStore.load_decisions``.
    """
    session_id: str
    decisions: Dict[str, Decision] = field(default_factory=dict)

    def add(self, decision: Decision) -> None:
        self.decisions[decision.decision_id] = decision

    def for_task(self, task_id: str) -> Optional[Decision]:
        for d in self.decisions.values():
            if d.resolved_task_id == task_id:
                return d
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
        }
