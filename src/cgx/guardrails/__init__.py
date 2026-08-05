"""Guardrails & safety for CGX (Subsystem K).

Three defensive layers around the LLM call path:

* **Input** (:mod:`cgx.guardrails.injection`) -- prompt-injection heuristics on
  the user's question/task and on *retrieved* repo chunks (indirect injection).
* **Output** (:mod:`cgx.guardrails.output`) -- secret-shaped literals in
  generated code and diff targets that escape ``project_root``.
* **Kill-switch** (:func:`assert_llm_enabled`) -- an operator panic button
  (``CGX_LLM_DISABLED``) enforced at the provider choke-point.

Findings are advisory by default: they are mirrored to metrics + the AIOps
alert store (so they show on the admin page) and surfaced in response meta,
without mutating the prompt or silently dropping a request.
"""

from cgx.guardrails.injection import scan_context, scan_text
from cgx.guardrails.output import check_diffs, scan_secret_literals
from cgx.guardrails.policy import (
    SEVERITIES,
    Finding,
    GuardrailConfig,
    GuardrailTripped,
    assert_llm_enabled,
    record_findings,
)

__all__ = [
    "Finding",
    "GuardrailConfig",
    "GuardrailTripped",
    "SEVERITIES",
    "assert_llm_enabled",
    "record_findings",
    "scan_text",
    "scan_context",
    "check_diffs",
    "scan_secret_literals",
]
