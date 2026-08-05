"""The :class:`Monitor` façade wiring checks -> alert store + metrics.

A single ``Monitor`` owns an :class:`~cgx.monitor.alerts.AlertStore` and a
:class:`~cgx.monitor.checks.MonitorThresholds`. The ``observe_*`` methods run
the matching pure check, persist every resulting alert, and mirror it to the
in-process metrics registry so both the admin page (via the store) and the
Prometheus scrape (via ``/api/metrics``) see the same findings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from cgx import metrics as _metrics
from cgx.monitor.alerts import Alert, AlertStore
from cgx.monitor.checks import (
    MonitorThresholds,
    check_cost_anomaly,
    check_groundedness,
    check_repair_health,
    check_retrieval_drift,
)

_ALERTS_HELP = "AIOps monitor alerts by code/severity."
_VALUE_HELP = "Last numeric value that triggered a monitor alert, by code."


class Monitor:
    """Runs AIOps checks, persisting and exporting any alerts they raise."""

    def __init__(self, store: Optional[AlertStore] = None, *,
                 thresholds: Optional[MonitorThresholds] = None,
                 project_root: Optional[str | Path] = None,
                 db_path: Optional[str | Path] = None) -> None:
        if store is not None:
            self._store = store
        else:
            self._store = AlertStore(db_path, project_root=project_root)
        self._thresholds = thresholds or MonitorThresholds.from_env()

    @property
    def store(self) -> AlertStore:
        return self._store

    @property
    def thresholds(self) -> MonitorThresholds:
        return self._thresholds

    def close(self) -> None:
        self._store.close()

    # ----------------------- emission -----------------------

    def _emit(self, alerts: List[Alert]) -> List[Alert]:
        for a in alerts:
            self._store.record(a)
            _metrics.inc("cgx_monitor_alerts_total", help=_ALERTS_HELP,
                         code=a.code, severity=a.severity)
            if a.value is not None:
                _metrics.set_gauge("cgx_monitor_alert_value", a.value,
                                   help=_VALUE_HELP, code=a.code)
        return alerts

    # ----------------------- observers -----------------------

    def observe_answer(self, answer: Dict[str, Any], *,
                       run_id: Optional[str] = None) -> List[Alert]:
        """Run groundedness checks on an answer payload and record findings."""
        return self._emit(check_groundedness(
            answer, self._thresholds, run_id=run_id))

    def observe_codegen(self, report: Dict[str, Any], *,
                        run_id: Optional[str] = None) -> List[Alert]:
        """Run repair-loop health checks on a ``codegen_report`` dict."""
        return self._emit(check_repair_health(
            report, self._thresholds, run_id=run_id))

    def observe_retrieval(self, current: Dict[str, Any],
                          baseline: Dict[str, Any], *,
                          run_id: Optional[str] = None) -> List[Alert]:
        """Compare a retrieval snapshot against a baseline for drift."""
        return self._emit(check_retrieval_drift(
            current, baseline, self._thresholds, run_id=run_id))

    def observe_cost(self, window: Dict[str, Any], baseline: Dict[str, Any],
                     *, run_id: Optional[str] = None) -> List[Alert]:
        """Check a cost/error window against a baseline for anomalies."""
        return self._emit(check_cost_anomaly(
            window, baseline, self._thresholds, run_id=run_id))

    # ----------------------- reads -----------------------

    def recent(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Proxy to :meth:`AlertStore.recent` for the admin page/API."""
        return self._store.recent(**kwargs)


# --- process-wide default monitor -----------------------------------------
# Live request paths (webui handlers) observe through this singleton so
# monitoring is a one-liner at the call site and shares a single alert DB
# (``~/.cgx/monitor.db``), mirroring the fallback used by other CGX stores.
_DEFAULT: Optional[Monitor] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_monitor() -> Monitor:
    """Return the lazily-constructed process-wide :class:`Monitor`."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = Monitor()
    return _DEFAULT
