"""Retention + right-to-erasure driver for the observation stores.

Ties the per-store ``purge`` / ``delete_run`` / ``delete_owner`` primitives
(Subsystem M) together behind three operator actions:

* :func:`purge_expired` -- TTL sweep: drop rows older than
  ``policy.retention_days`` across activity, feedback, alerts and usage.
* :func:`erase_run` -- delete everything tied to one ``run_id`` (activity,
  feedback, alerts).
* :func:`erase_owner` -- delete everything tied to one owner (activity, usage).

Each function is best-effort per store -- a failing/missing store is logged
and skipped -- and returns ``{store_name: rows_deleted}`` so the caller can
report exactly what was removed. Stores may be injected for tests; otherwise
the process-wide default singletons are used.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from cgx.govdata.policy import GovernanceConfig

logger = logging.getLogger(__name__)


def _default_stores() -> Dict[str, Any]:
    """Resolve the process-wide default stores, skipping any that fail."""
    out: Dict[str, Any] = {}
    resolvers: Dict[str, Callable[[], Any]] = {
        "activity": lambda: __import__(
            "cgx.activity", fromlist=["get_default_run_store"]
        ).get_default_run_store(),
        "feedback": lambda: __import__(
            "cgx.feedback", fromlist=["get_default_store"]
        ).get_default_store(),
        "alerts": lambda: __import__(
            "cgx.monitor", fromlist=["get_default_monitor"]
        ).get_default_monitor().store,
        "usage": lambda: __import__(
            "cgx.governance", fromlist=["get_default_quota_manager"]
        ).get_default_quota_manager().meter,
    }
    for name, resolve in resolvers.items():
        try:
            out[name] = resolve()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("govdata: store %s unavailable: %s", name, e)
    return out


def _call(store: Any, method: str, **kwargs: Any) -> Optional[int]:
    fn = getattr(store, method, None)
    if not callable(fn):
        return None
    try:
        return int(fn(**kwargs))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("govdata: %s.%s failed: %s", type(store).__name__, method, e)
        return None


def purge_expired(policy: Optional[GovernanceConfig] = None, *,
                  stores: Optional[Dict[str, Any]] = None,
                  now: Optional[float] = None) -> Dict[str, int]:
    """Delete rows older than ``policy.retention_days`` from every store.

    A ``retention_days`` of 0 disables the sweep (returns ``{}``) so retention
    stays strictly opt-in.
    """
    policy = policy or GovernanceConfig.from_env()
    if policy.retention_days <= 0:
        return {}
    cutoff = (now if now is not None else time.time()) - policy.retention_days * 86400
    stores = stores if stores is not None else _default_stores()
    result: Dict[str, int] = {}
    for name, store in stores.items():
        deleted = _call(store, "purge", before=cutoff)
        if deleted is not None:
            result[name] = deleted
    return result


def erase_run(run_id: str, *,
              stores: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Right-to-erasure by ``run_id`` across activity / feedback / alerts."""
    stores = stores if stores is not None else _default_stores()
    result: Dict[str, int] = {}
    for name in ("activity", "feedback", "alerts"):
        store = stores.get(name)
        if store is None:
            continue
        deleted = _call(store, "delete_run", run_id=run_id)
        if deleted is not None:
            result[name] = deleted
    return result


def erase_owner(owner: str, *,
                stores: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Right-to-erasure by ``owner`` across activity / usage."""
    stores = stores if stores is not None else _default_stores()
    result: Dict[str, int] = {}
    for name in ("activity", "usage"):
        store = stores.get(name)
        if store is None:
            continue
        deleted = _call(store, "delete_owner", owner=owner)
        if deleted is not None:
            result[name] = deleted
    return result
