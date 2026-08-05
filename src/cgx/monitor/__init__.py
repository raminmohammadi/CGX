"""AIOps monitoring for CGX.

Drift / quality / cost monitors that turn already-computed pipeline signals
(answer payloads, codegen reports, retrieval snapshots, cost windows) into
persisted, metric-exported :class:`Alert` records. The public surface is the
:class:`Monitor` façade plus the pure ``check_*`` functions and the
:class:`AlertStore` for read access from the web UI.
"""

from cgx.monitor.alerts import Alert, AlertStore, default_alert_db_path
from cgx.monitor.checks import (
    MonitorThresholds,
    check_cost_anomaly,
    check_groundedness,
    check_repair_health,
    check_retrieval_drift,
)
from cgx.monitor.monitor import Monitor, get_default_monitor

__all__ = [
    "Alert",
    "AlertStore",
    "default_alert_db_path",
    "MonitorThresholds",
    "Monitor",
    "get_default_monitor",
    "check_groundedness",
    "check_repair_health",
    "check_retrieval_drift",
    "check_cost_anomaly",
]
