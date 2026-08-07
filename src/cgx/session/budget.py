

"""Typed loop budgets for the session router's bounded retry loops.

Every recovery loop the router runs (repair, regenerate, re-plan,
decompose-retry) is bounded by an attempt counter threaded through
``TaskNode.inputs``. Historically each router edge copied its counters
by hand as loose dict keys, so adding an edge (or forgetting one key on
an existing edge) silently reset a budget and re-opened an unbounded
loop. :class:`LoopBudget` centralizes those counters into one immutable
object: routers read it with :meth:`LoopBudget.from_inputs`, spend it
with the ``spend_*`` helpers, and serialize it back onto successor
tasks with :meth:`LoopBudget.repair_chain_inputs`. The wire format is
unchanged -- the same flat input keys as before -- so persisted
in-flight sessions resume cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

# Absolute ceiling on repair rounds for a single greenfield write loop.
# The progress-aware gate (``_repair_progress_stalled`` in the router)
# ends a loop as soon as the failing-test count stops strictly dropping,
# so most loops terminate well before this cap; the cap only bounds a
# loop whose failure signature keeps mutating with no usable count
# trend. Raised from the original 2-shot limit so a genuinely-progressing
# hard task can iterate further without ever running unbounded.
REPAIR_BUDGET = 4

# Maximum number of targeted regenerate attempts per SCAFFOLD ancestor
# chain. Each attempt re-generates only the files that dropped, seeding
# the survivors from the prior checkpoint, so a retry is fast and does
# not disturb good work. A local/flaky model routinely drops a single
# file to a read timeout or an empty patch, so a shallow budget escalated
# straight to a disruptive re-plan (and re-approval); a few in-place
# retries clear the common case before the manifest is ever blamed.
REGENERATE_BUDGET = 3

# Maximum number of *semantic-repair* regenerations per SCAFFOLD ancestor
# chain -- deliberately separate from :data:`REGENERATE_BUDGET`. That
# budget bounds *syntax churn* (SCAFFOLD/APPLY dropping files that do not
# parse) *before* the tree is applied; this one bounds a whole-tree
# rewrite of a tree that already generated and applied cleanly but
# references a symbol a downstream gate (API_CHECK / VERIFY /
# RUNTIME_VERIFY) proved wrong. Conflating the two on one counter let a
# scaffold that spent its whole syntax budget converging to a clean tree
# arrive at the *first* semantic repair with nothing left. The counter is
# scaffold-carried because ``repair_attempt`` does not survive a scaffold
# regeneration (``propose_regenerate`` copies the SCAFFOLD's inputs, not
# the REPAIR's).
REPAIR_REGENERATE_BUDGET = 2

# Maximum number of *re-plan* escalations per session. When a SCAFFOLD or
# APPLY spends its per-manifest regenerate budget the manifest itself is
# the suspect, so the router escalates once to a fresh DECOMPOSE with the
# accumulated failure folded into its goal. When this budget is also
# spent the router proceeds with the surviving files rather than failing
# terminally -- only a scaffold that produced nothing usable is a dead end.
REPLAN_BUDGET = 1

# Maximum number of constraint-folded DECOMPOSE retries after a failure
# the executor marked retryable (a plan-quality problem: an empty or
# unbuildable manifest). One retry with the concrete failure folded into
# the goal is enough to move a deterministic (temperature 0) planner off
# the broken output; a second identical failure means the model cannot
# satisfy the constraint and the session fails terminally.
DECOMPOSE_RETRY_BUDGET = 1

# Outer per-session circuit breaker for autonomous greenfield builds. The
# per-subtree budgets above bound each recovery loop individually, but
# nested re-plans (each seeding a fresh SCAFFOLD subtree with its own
# fresh counters) can still multiply total work beyond what any single
# counter sees. These caps are the session-wide backstop the runner
# enforces before dispatching each work task: a build that has not
# converged after this many compute-task runs -- or this much wall-clock
# -- halts and escalates (interactive: an ASK_USER pause; headless:
# terminal FAILED) instead of grinding on through nested budgets that look
# infinite. ASK_USER waits are budget-exempt, so a build that legitimately
# needs clarification is never charged for the pause. Explore mode stays
# unbounded by default -- its loops are user-gated, not autonomous.
GREENFIELD_MAX_TASK_RUNS = 60
GREENFIELD_MAX_WALL_SECONDS = 3600.0

# Default safety-valve ceiling on the number of tasks a single drain pass
# dispatches. Explore/greenfield task graphs are small and bounded well
# under it, so the flat cap only ever fires on a router bug. A SWARM
# session instead spawns one DEVELOPER task per planned file followed by a
# VERIFY, so a large plan legitimately needs more; :func:`drain_step_ceiling`
# scales the ceiling to the plan's file count for SWARM sessions only.
DRAIN_STEP_CEILING = 64


def drain_step_ceiling(store: Any, session_id: str, *,
                       default: int = DRAIN_STEP_CEILING) -> int:
    """Plan-aware ceiling for a session drain loop.

    Returns ``default`` for explore/greenfield sessions, whose task
    graphs stay well under it. A SWARM session dispatches one DEVELOPER
    task per planned file then a VERIFY, so a plan with more than ~20
    files would be truncated mid-chain by the flat default; scale the
    ceiling to the plan's file count -- with headroom for one regenerate
    retry per file plus planning/verify/terminal overhead -- so a large
    build runs to quiescence while a runaway is still bounded. The file
    count is read from the WORK_PLAN artifact, falling back to the number
    of DEVELOPER tasks already spawned before the plan lands. Best-effort:
    any lookup failure falls back to ``default`` so the drain never blocks.
    """
    try:
        from cgx.session.models import ArtifactKind, SessionMode, TaskKind
        session = store.get_session(session_id)
        if session is None or session.mode is not SessionMode.SWARM:
            return default
        file_count = 0
        for art in store.list_artifacts(session_id):
            if art.kind is ArtifactKind.WORK_PLAN and art.content:
                paths = art.content.get("paths") or []
                file_count = max(file_count, len(paths))
        dev = sum(1 for t in store.list_tasks(session_id)
                  if t.kind is TaskKind.SWARM_DEVELOPER)
        file_count = max(file_count, dev)
        return max(default, file_count * 3 + 16)
    except Exception:  # pragma: no cover - defensive; never block a drain
        return default


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort ``int`` coercion; returns ``None`` for missing/garbage."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LoopBudget:
    """Immutable snapshot of every bounded-loop counter on a task node.

    Four fields form the *repair chain* threaded along the
    APPLY -> BOOTSTRAP_ENV -> API_CHECK -> SMOKE -> VERIFY ->
    RUNTIME_VERIFY -> REPAIR edges:

    * ``repair_attempt`` -- rounds spent in the shared repair loop,
      capped by :data:`REPAIR_BUDGET`.
    * ``prior_failure_signatures`` -- the flap backstop: a repeated
      signature means the loop is churning without progress.
    * ``prior_failing_counts`` / ``prior_passing_counts`` -- the
      coverage-aware progress ledger read by the VERIFY gate.

    The remaining counters are carried by the node families that own
    them: ``regenerate_attempt`` / ``repair_regenerate_attempt`` /
    ``replan_attempt`` on SCAFFOLD nodes, ``decompose_retry`` (plus
    ``replan_attempt``) on DECOMPOSE nodes.

    ``repair_ledger_fact_id`` is not a counter but rides the same chain:
    it points at the ``FactKind.REPAIR_LEDGER`` working-memory fact so
    ``DIAGNOSE`` can read what was already tried and never re-propose a
    proven dead end. The router only threads the id -- it never reads the
    fact's contents, so it stays pure (see docs/diagnose-design.md §7).
    """

    repair_attempt: int = 0
    prior_failure_signatures: Tuple[str, ...] = ()
    prior_failing_counts: Tuple[int, ...] = ()
    prior_passing_counts: Tuple[int, ...] = ()
    regenerate_attempt: int = 0
    repair_regenerate_attempt: int = 0
    replan_attempt: int = 0
    decompose_retry: int = 0
    repair_ledger_fact_id: Optional[str] = None

    # --------------- construction / serialization ---------------

    @classmethod
    def from_inputs(cls, inputs: Optional[Mapping[str, Any]]) -> "LoopBudget":
        """Read the budget off a ``TaskNode.inputs`` mapping.

        Missing or garbage values coerce to the field defaults, so the
        router never crashes on a hand-crafted or legacy-persisted node.
        Count lists silently drop non-int entries (mirroring the
        router's old per-site ``_coerce_count`` filtering).
        """
        src = inputs or {}
        return cls(
            repair_attempt=_coerce_int(src.get("repair_attempt")) or 0,
            prior_failure_signatures=tuple(
                str(s) for s in (src.get("prior_failure_signatures") or [])),
            prior_failing_counts=tuple(
                c for c in (_coerce_int(x)
                            for x in (src.get("prior_failing_counts") or []))
                if c is not None),
            prior_passing_counts=tuple(
                c for c in (_coerce_int(x)
                            for x in (src.get("prior_passing_counts") or []))
                if c is not None),
            regenerate_attempt=_coerce_int(
                src.get("regenerate_attempt")) or 0,
            repair_regenerate_attempt=_coerce_int(
                src.get("repair_regenerate_attempt")) or 0,
            replan_attempt=_coerce_int(src.get("replan_attempt")) or 0,
            decompose_retry=_coerce_int(src.get("decompose_retry")) or 0,
            repair_ledger_fact_id=(
                str(src["repair_ledger_fact_id"]).strip() or None
                if src.get("repair_ledger_fact_id") else None),
        )

    def repair_chain_inputs(self) -> Dict[str, Any]:
        """Serialize the repair-chain fields as successor-task inputs.

        Every edge on the repair chain threads exactly this dict, so a
        counter (or the ledger id) can no longer be dropped by an edge that
        forgot one key. The keys match the historical flat wire format;
        ``repair_ledger_fact_id`` is emitted only when set so a chain that
        never opened a ledger keeps the identical wire shape as before.
        """
        chain: Dict[str, Any] = {
            "repair_attempt": self.repair_attempt,
            "prior_failure_signatures": list(self.prior_failure_signatures),
            "prior_failing_counts": list(self.prior_failing_counts),
            "prior_passing_counts": list(self.prior_passing_counts),
        }
        if self.repair_ledger_fact_id:
            chain["repair_ledger_fact_id"] = self.repair_ledger_fact_id
        return chain

    def with_repair_ledger(self, fact_id: Optional[str]) -> "LoopBudget":
        """Return a copy pointing at the repair chain's ledger fact.

        Threaded on the edge out of DIAGNOSE so the ledger the executor
        just appended survives the hop to the next repair round.
        """
        fid = str(fact_id).strip() if fact_id else None
        return replace(self, repair_ledger_fact_id=fid or None)

    # --------------- exhaustion predicates ---------------

    @property
    def repair_exhausted(self) -> bool:
        """True once :data:`REPAIR_BUDGET` repair rounds are spent."""
        return self.repair_attempt >= REPAIR_BUDGET

    @property
    def regenerate_exhausted(self) -> bool:
        """True once the syntax-churn regenerate budget is spent."""
        return self.regenerate_attempt >= REGENERATE_BUDGET

    @property
    def repair_regenerate_exhausted(self) -> bool:
        """True once the semantic-repair regenerate budget is spent."""
        return self.repair_regenerate_attempt >= REPAIR_REGENERATE_BUDGET

    @property
    def replan_exhausted(self) -> bool:
        """True once the re-plan escalation budget is spent."""
        return self.replan_attempt >= REPLAN_BUDGET

    @property
    def decompose_retry_exhausted(self) -> bool:
        """True once the retryable-DECOMPOSE retry budget is spent."""
        return self.decompose_retry >= DECOMPOSE_RETRY_BUDGET

    def seen(self, signature: str) -> bool:
        """True when ``signature`` already appears in the flap ledger."""
        return signature in self.prior_failure_signatures

    def signature_repeats(self, signature: str) -> int:
        """How many times ``signature`` already appears in the flap ledger.

        A gate that only asks *whether* a signature repeated has one
        response left -- stop. Callers that can escalate their strategy
        instead (API_CHECK: install_deps, then a regenerate that removes
        the offending import) read the count to pick the next rung.
        """
        return self.prior_failure_signatures.count(str(signature))

    # --------------- spending (each returns a new copy) ---------------

    def spend_repair(self, signature: str,
                     failing_count: Optional[int] = None,
                     passing_count: Optional[int] = None) -> "LoopBudget":
        """Charge one repair round: bump the attempt, record the ledger.

        ``signature`` is appended to the flap ledger; the counts are
        appended to the progress ledger only when supplied (a
        non-assertion outcome has no meaningful count).
        """
        return replace(
            self,
            repair_attempt=self.repair_attempt + 1,
            prior_failure_signatures=(
                self.prior_failure_signatures + (str(signature),)),
            prior_failing_counts=(
                self.prior_failing_counts + (failing_count,)
                if failing_count is not None else self.prior_failing_counts),
            prior_passing_counts=(
                self.prior_passing_counts + (passing_count,)
                if passing_count is not None else self.prior_passing_counts),
        )

    def with_repair_attempt(self, attempt: int) -> "LoopBudget":
        """Return a copy with ``repair_attempt`` pinned to ``attempt``.

        Used on the REPAIR -> APPLY / REPAIR -> BOOTSTRAP_ENV edges,
        where the authoritative attempt number is echoed back by the
        REPAIR executor's outputs rather than incremented by the router.
        """
        return replace(self, repair_attempt=int(attempt))

    def with_signature(self, signature: str) -> "LoopBudget":
        """Return a copy whose flap ledger contains ``signature``.

        Appends only when absent, so re-recording the signature REPAIR
        already charged does not double-count it.
        """
        if signature in self.prior_failure_signatures:
            return self
        return replace(
            self,
            prior_failure_signatures=(
                self.prior_failure_signatures + (str(signature),)))

    def spend_regenerate(self) -> "LoopBudget":
        """Charge one syntax-churn regenerate attempt."""
        return replace(self, regenerate_attempt=self.regenerate_attempt + 1)

    def spend_repair_regenerate(self) -> "LoopBudget":
        """Charge one semantic-repair regenerate attempt."""
        return replace(
            self,
            repair_regenerate_attempt=self.repair_regenerate_attempt + 1)

    def spend_replan(self) -> "LoopBudget":
        """Charge one re-plan escalation."""
        return replace(self, replan_attempt=self.replan_attempt + 1)

    def spend_decompose_retry(self) -> "LoopBudget":
        """Charge one retryable-DECOMPOSE retry."""
        return replace(self, decompose_retry=self.decompose_retry + 1)
