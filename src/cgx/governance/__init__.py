"""Cost & quota governance for CGX (Subsystem I).

Turns the truthful token/cost accounting from :mod:`cgx.usage` into an
enforceable per-owner budget: a :class:`GovernedProvider` wraps the LLM
provider at the request choke-point, checks each call against the owner's
day ceiling (soft-warn then hard-stop) and meters actual spend. The
:class:`UsageMeter` persists per-owner/day usage (SQLite, WAL,
``$CGX_CONFIG_DIR``-aware) and the :class:`QuotaManager` ties config +
meter + metrics together for both enforcement and the usage-meter API.
"""

from cgx.governance.budget import BudgetConfig, BudgetExceeded
from cgx.governance.manager import (
    QuotaManager,
    get_default_quota_manager,
    resolve_owner,
)
from cgx.governance.meter import UsageMeter, default_usage_db_path, today
from cgx.governance.provider import GovernedProvider, govern

__all__ = [
    "BudgetConfig",
    "BudgetExceeded",
    "UsageMeter",
    "default_usage_db_path",
    "today",
    "QuotaManager",
    "get_default_quota_manager",
    "resolve_owner",
    "GovernedProvider",
    "govern",
]
