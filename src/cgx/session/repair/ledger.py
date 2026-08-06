"""``RepairLedger`` -- durable working memory for DIAGNOSE (Workstream B4).

Today's recovery ladder is *stateless*: every repair round re-derives its
fix from the current failure alone, so a fix that did not help is proposed
again on the next round until a budget forces a whole-tree regenerate. The
ledger is the fix. A :class:`~cgx.session.models.FactKind.REPAIR_LEDGER`
fact threads along one repair chain (its id rides
``LoopBudget.repair_chain_inputs``) and records, per round, the action
``DIAGNOSE`` proposed, the files/packages it targeted, the failure
signature it was addressing, and -- once the next round is reached and the
outcome is known -- whether that action left the *identical* failure
standing.

This module is pure (no I/O): the executor loads the fact content into a
:class:`RepairLedger`, reasons over it, and emits an appended copy. That
keeps the memory trivially unit-testable and the router LLM-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

# Attempt outcomes. ``pending`` is the just-proposed action whose effect is
# not yet known; the next DIAGNOSE round resolves it by comparing the live
# failure signature against the one the attempt was addressing.
OUTCOME_PENDING = "pending"
OUTCOME_STILL_FAILING = "still_failing"
OUTCOME_CHANGED = "changed"

# Only a fix that left the *identical* signature standing is a proven dead
# end DIAGNOSE must never re-propose; a changed signature is progress on a
# now-different failure, so the same action may legitimately reappear.
_BLOCKING_OUTCOMES = frozenset({OUTCOME_STILL_FAILING})

# Bound how many attempts are rendered into a model prompt so a long repair
# chain cannot blow a small model's context window.
LEDGER_RENDER_LIMIT = 8


def _norm_targets(value: Any) -> Tuple[str, ...]:
    """Normalize a targets field to a sorted, de-duped tuple of strings.

    Sorting makes ``has_attempted`` order-insensitive so ``[a, b]`` and
    ``[b, a]`` count as the same attempt.
    """
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return ()
    out = []
    for it in items:
        s = str(it).strip()
        if s and s not in out:
            out.append(s)
    return tuple(sorted(out))


@dataclass(frozen=True)
class RepairAttempt:
    """One proposed-and-observed repair action on the chain."""

    action: str
    targets: Tuple[str, ...] = ()
    outcome: str = OUTCOME_PENDING
    signature: str = ""
    rationale: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RepairAttempt":
        return cls(
            action=str(d.get("action") or "").strip(),
            targets=_norm_targets(d.get("targets")),
            outcome=str(d.get("outcome") or OUTCOME_PENDING).strip()
            or OUTCOME_PENDING,
            signature=str(d.get("signature") or "").strip(),
            rationale=str(d.get("rationale") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "targets": list(self.targets),
            "outcome": self.outcome,
            "signature": self.signature,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RepairLedger:
    """Append-only list of :class:`RepairAttempt` on one repair chain."""

    attempts: Tuple[RepairAttempt, ...] = ()

    @classmethod
    def from_content(cls, content: Dict[str, Any]) -> "RepairLedger":
        items = (content or {}).get("attempts") or []
        return cls(tuple(
            RepairAttempt.from_dict(i) for i in items if isinstance(i, dict)))

    def to_content(self) -> Dict[str, Any]:
        return {"attempts": [a.to_dict() for a in self.attempts]}

    def has_attempted(self, action: str, targets: Any) -> bool:
        """True when ``(action, targets)`` already proved a dead end.

        Only attempts whose recorded outcome is in
        :data:`_BLOCKING_OUTCOMES` count -- a still-``pending`` proposal or
        one that moved the failure does not block a re-proposal.
        """
        want = _norm_targets(targets)
        return any(
            a.action == action and a.targets == want
            and a.outcome in _BLOCKING_OUTCOMES
            for a in self.attempts)

    def finalize_pending(self, current_signature: str) -> "RepairLedger":
        """Resolve the trailing ``pending`` attempt now the outcome is known.

        Called at the top of a DIAGNOSE round: if the last proposed action
        left the *same* failure signature standing it is marked
        ``still_failing`` (a proven dead end); otherwise ``changed``.
        """
        if not self.attempts or self.attempts[-1].outcome != OUTCOME_PENDING:
            return self
        last = self.attempts[-1]
        outcome = (OUTCOME_STILL_FAILING
                   if str(current_signature) == last.signature
                   else OUTCOME_CHANGED)
        return RepairLedger(self.attempts[:-1] + (replace(last, outcome=outcome),))

    def append(self, action: str, targets: Any, signature: str,
               rationale: str = "",
               outcome: str = OUTCOME_PENDING) -> "RepairLedger":
        """Return a copy with one freshly-proposed attempt appended."""
        attempt = RepairAttempt(
            action=str(action).strip(), targets=_norm_targets(targets),
            outcome=outcome, signature=str(signature or "").strip(),
            rationale=str(rationale or "").strip())
        return RepairLedger(self.attempts + (attempt,))

    def render(self) -> str:
        """A compact, bounded summary for the DIAGNOSE ReAct prompt."""
        if not self.attempts:
            return "(no prior repair attempts on this chain)"
        lines = []
        for i, a in enumerate(self.attempts[-LEDGER_RENDER_LIMIT:], 1):
            tgt = ", ".join(a.targets) or "-"
            lines.append(
                f"{i}. action={a.action} targets=[{tgt}] outcome={a.outcome}")
        return "\n".join(lines)
