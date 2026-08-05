

"""Budget configuration + the hard-stop signal for cost/quota governance.

A :class:`BudgetConfig` carries a global daily ceiling (cost + tokens) plus an
optional per-owner override map, all env-driven so operators tune limits
without code changes (mirrors :class:`cgx.monitor.checks.MonitorThresholds`).
A limit of ``0`` means *unlimited* -- so the default config meters usage but
never blocks, and enforcement only kicks in once a real ceiling is set.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised on a hard-stop: ``owner`` is over its ``resource`` ceiling."""

    def __init__(self, owner: str, resource: str, used: float,
                 limit: float) -> None:
        self.owner = owner
        self.resource = resource
        self.used = used
        self.limit = limit
        super().__init__(
            f"budget exceeded for owner={owner!r}: {resource} "
            f"used {used:.4g} >= limit {limit:.4g}")


def _envf(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class BudgetConfig:
    """Cost/token ceilings, overridable via ``CGX_BUDGET_*`` env vars.

    ``per_owner`` maps an owner id to ``{"cost": x, "tokens": y}`` overrides
    (loaded from the ``CGX_BUDGET_OWNERS`` JSON env var); missing keys fall
    back to the global ceilings.
    """

    enabled: bool = True
    daily_cost_usd: float = 0.0   # 0 == unlimited
    daily_tokens: float = 0.0     # 0 == unlimited
    soft_ratio: float = 0.8       # warn once utilisation crosses this fraction
    per_owner: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        return cls(
            enabled=_envb("CGX_BUDGET_ENABLED", True),
            daily_cost_usd=_envf("CGX_BUDGET_DAILY_COST_USD", 0.0),
            daily_tokens=_envf("CGX_BUDGET_DAILY_TOKENS", 0.0),
            soft_ratio=_envf("CGX_BUDGET_SOFT_RATIO", 0.8),
            per_owner=cls._parse_owners(os.getenv("CGX_BUDGET_OWNERS")),
        )

    @staticmethod
    def _parse_owners(raw: Optional[str]) -> Dict[str, Dict[str, float]]:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("CGX_BUDGET_OWNERS is not valid JSON; ignoring")
            return {}
        out: Dict[str, Dict[str, float]] = {}
        if isinstance(data, dict):
            for owner, limits in data.items():
                if not isinstance(limits, dict):
                    continue
                entry: Dict[str, float] = {}
                for key in ("cost", "tokens"):
                    if key in limits:
                        try:
                            entry[key] = float(limits[key])
                        except (TypeError, ValueError):
                            continue
                out[str(owner)] = entry
        return out

    def limits_for(self, owner: str) -> Tuple[float, float]:
        """Return ``(cost_limit, token_limit)`` for ``owner`` (0 == unlimited)."""
        override = self.per_owner.get(owner, {})
        cost = override.get("cost", self.daily_cost_usd)
        tokens = override.get("tokens", self.daily_tokens)
        return float(cost), float(tokens)
