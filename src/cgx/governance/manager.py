

"""The :class:`QuotaManager`: budget config + usage meter + enforcement.

Resolves the *owner* to attribute spend to (from the trace context, since auth
is out of scope), checks a day's accumulated usage against the configured
ceilings (soft-warn then hard-stop), and records every governed call to the
meter while mirroring per-owner spend to the metrics registry.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from cgx import metrics as _metrics
from cgx.governance.budget import BudgetConfig, BudgetExceeded
from cgx.governance.meter import UsageMeter

_COST_HELP = "Per-owner LLM cost in USD (governed calls)."
_TOK_HELP = "Per-owner LLM tokens by direction (governed calls)."
_EVT_HELP = "Budget governance events by owner/state (warn|exceeded)."
_DAY_COST_HELP = "Per-owner LLM cost in USD accrued in the current day window."


def resolve_owner() -> str:
    """Owner id for the active call: trace ``owner`` -> ``session_id`` -> default."""
    try:
        from cgx.trace import trace_context
        ctx = trace_context.get() or {}
    except Exception:  # pragma: no cover - trace is optional
        ctx = {}
    return str(ctx.get("owner") or ctx.get("session_id") or "default")


class QuotaManager:
    """Owns a :class:`BudgetConfig` + :class:`UsageMeter` and enforces both."""

    def __init__(self, config: Optional[BudgetConfig] = None, *,
                 meter: Optional[UsageMeter] = None,
                 project_root: Optional[str | Path] = None,
                 db_path: Optional[str | Path] = None) -> None:
        self._config = config or BudgetConfig.from_env()
        self._meter = meter or UsageMeter(db_path, project_root=project_root)

    @property
    def config(self) -> BudgetConfig:
        return self._config

    @property
    def meter(self) -> UsageMeter:
        return self._meter

    def close(self) -> None:
        self._meter.close()

    # ----------------------- enforcement -----------------------
    def check(self, owner: Optional[str] = None, *,
              enforce: bool = True) -> Dict[str, Any]:
        """Evaluate ``owner``'s current-day usage against its ceilings.

        Returns a decision dict ``{owner, state, cost_*, tokens_*}`` where
        ``state`` is ``ok``/``warn``/``exceeded``. When ``enforce`` and the
        config is enabled, an already-exceeded ceiling raises
        :class:`BudgetExceeded` (a hard-stop before the next call runs).
        """
        owner = owner or resolve_owner()
        totals = self._meter.totals(owner)
        cost_limit, token_limit = self._config.limits_for(owner)
        state, resource, used, limit = self._evaluate(
            totals["cost_usd"], cost_limit,
            totals["tokens_total"], token_limit)
        decision = {"owner": owner, "state": state,
                    "cost_used": totals["cost_usd"], "cost_limit": cost_limit,
                    "tokens_used": totals["tokens_total"],
                    "tokens_limit": token_limit}
        if state in ("warn", "exceeded"):
            _metrics.inc("cgx_budget_events_total", help=_EVT_HELP,
                         owner=owner, state=state)
        if state == "exceeded" and enforce and self._config.enabled:
            raise BudgetExceeded(owner, resource, used, limit)
        return decision

    def _evaluate(self, cost_used: float, cost_limit: float,
                  tokens_used: int, token_limit: float):
        state, resource, used, limit = "ok", "", 0.0, 0.0
        for name, u, lim in (("cost", cost_used, cost_limit),
                             ("tokens", float(tokens_used), token_limit)):
            if lim <= 0:
                continue
            if u >= lim:
                return "exceeded", name, u, lim
            if u >= lim * self._config.soft_ratio:
                state, resource, used, limit = "warn", name, u, lim
        return state, resource, used, limit

    # ----------------------- recording -----------------------
    def record_usage(self, owner: Optional[str] = None, *, tokens_in: int,
                     tokens_out: int, cost_usd: float,
                     model: Optional[str] = None,
                     provider: Optional[str] = None) -> Dict[str, Any]:
        """Persist one governed call's usage and mirror it to metrics."""
        owner = owner or resolve_owner()
        self._meter.record(owner, tokens_in=tokens_in, tokens_out=tokens_out,
                           cost_usd=cost_usd, model=model, provider=provider)
        if cost_usd:
            _metrics.inc("cgx_owner_cost_usd_total", cost_usd, help=_COST_HELP,
                         owner=owner, model=model or "unknown",
                         provider=provider or "unknown")
        _metrics.inc("cgx_owner_tokens_total", int(tokens_in), help=_TOK_HELP,
                     owner=owner, direction="in")
        _metrics.inc("cgx_owner_tokens_total", int(tokens_out),
                     owner=owner, direction="out")
        totals = self._meter.totals(owner)
        _metrics.set_gauge("cgx_owner_cost_usd_day", totals["cost_usd"],
                           help=_DAY_COST_HELP, owner=owner)
        return totals

    def status(self, owner: Optional[str] = None) -> Dict[str, Any]:
        """Totals + limits + current state for the usage-meter API."""
        owner = owner or resolve_owner()
        decision = self.check(owner, enforce=False)
        decision.update(self._meter.totals(owner))
        return decision


# --- process-wide default manager -----------------------------------------
_DEFAULT: Optional[QuotaManager] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_quota_manager() -> QuotaManager:
    """Return the lazily-constructed process-wide :class:`QuotaManager`."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = QuotaManager()
    return _DEFAULT
