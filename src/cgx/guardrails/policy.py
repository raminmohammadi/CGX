

"""Guardrail policy primitives: config, findings, kill-switch, emission.

Central types shared by the input (:mod:`cgx.guardrails.injection`) and output
(:mod:`cgx.guardrails.output`) scanners. A :class:`Finding` is one guardrail
hit; :class:`GuardrailConfig` is env-driven (mirrors
:class:`cgx.monitor.checks.MonitorThresholds` / :class:`cgx.governance.BudgetConfig`);
:func:`assert_llm_enabled` implements the operator kill-switch; and
:func:`record_findings` mirrors findings to metrics + the AIOps alert store so
they surface on the admin page alongside monitor alerts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEVERITIES = ("info", "warning", "critical")


class GuardrailTripped(RuntimeError):
    """Raised when a hard guardrail blocks an operation (e.g. the kill-switch)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class Finding:
    """One guardrail hit. ``detail`` carries a short, redaction-safe excerpt."""

    code: str
    severity: str
    message: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "detail": self.detail}


def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class GuardrailConfig:
    """Toggles for the guardrail subsystem, overridable via ``CGX_GUARDRAIL_*``.

    ``kill_switch`` (``CGX_LLM_DISABLED``) is the operator panic button: when
    set, every provider resolution raises :class:`GuardrailTripped` so no LLM
    call runs. ``block_secret_output`` escalates a secret-shaped literal in
    generated code from advisory to a hard block.
    """

    enabled: bool = True
    kill_switch: bool = False
    scan_input: bool = True
    scan_output: bool = True
    block_secret_output: bool = False

    @classmethod
    def from_env(cls) -> "GuardrailConfig":
        return cls(
            enabled=_envb("CGX_GUARDRAIL_ENABLED", True),
            kill_switch=_envb("CGX_LLM_DISABLED", False),
            scan_input=_envb("CGX_GUARDRAIL_SCAN_INPUT", True),
            scan_output=_envb("CGX_GUARDRAIL_SCAN_OUTPUT", True),
            block_secret_output=_envb("CGX_GUARDRAIL_BLOCK_SECRETS", False),
        )


def assert_llm_enabled(config: Optional[GuardrailConfig] = None) -> None:
    """Raise :class:`GuardrailTripped` when the operator kill-switch is set."""
    cfg = config or GuardrailConfig.from_env()
    if cfg.enabled and cfg.kill_switch:
        raise GuardrailTripped(
            "llm_kill_switch",
            "LLM calls are disabled by the operator kill-switch "
            "(unset CGX_LLM_DISABLED to re-enable).")


_EVT_HELP = "Guardrail findings by code/severity/kind (input|output)."


def record_findings(findings: List[Finding], *, run_id: Optional[str] = None,
                    kind: str = "input") -> None:
    """Mirror findings to metrics + the AIOps alert store (best-effort).

    Guardrails must never break a request, so every sink is wrapped: a metrics
    counter per ``(code, severity, kind)`` and -- for ``warning``/``critical``
    findings -- a persisted ``guardrail_*`` :class:`~cgx.monitor.alerts.Alert`
    so the admin page lists them next to drift/quality/cost incidents.
    """
    if not findings:
        return
    try:
        from cgx import metrics as _metrics
        for f in findings:
            _metrics.inc("cgx_guardrail_events_total", help=_EVT_HELP,
                         code=f.code, severity=f.severity, kind=kind)
    except Exception:  # pragma: no cover - metrics must never break a request
        pass
    try:
        from cgx.monitor import Alert, get_default_monitor
        store = get_default_monitor().store
        for f in findings:
            if f.severity in ("warning", "critical"):
                store.record(Alert(
                    code=f"guardrail_{f.code}", severity=f.severity,
                    message=f.message, run_id=run_id,
                    labels={"kind": kind, "detail": f.detail[:200]}))
    except Exception as e:  # pragma: no cover - alerting is non-critical
        logger.warning("record_findings: alert store failed: %s", e)


__all__ = [
    "SEVERITIES",
    "GuardrailTripped",
    "Finding",
    "GuardrailConfig",
    "assert_llm_enabled",
    "record_findings",
]
