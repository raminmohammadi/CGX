"""AIOps quality/drift/cost checks.

Each ``check_*`` function is pure: it takes already-computed signals (an
answer payload, a codegen report, retrieval stats, a cost window) plus a
:class:`MonitorThresholds` and returns a list of :class:`~cgx.monitor.alerts.Alert`.
Persistence and metrics live in :class:`cgx.monitor.monitor.Monitor`; keeping the
checks side-effect free makes them trivial to unit test and reuse offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cgx.monitor.alerts import Alert

# Matches ``[[chunk_id]]`` citation tokens in answer Markdown.
_INLINE_CITATION_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


def _envf(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


@dataclass(frozen=True)
class MonitorThresholds:
    """Tunable bounds for the monitors, overridable via ``CGX_MON_*`` env vars."""

    min_confidence: float = 0.35
    min_citation_coverage: float = 1.0
    max_repair_attempts: int = 2
    cost_spike_ratio: float = 3.0
    max_error_rate: float = 0.25
    drift_score_drop: float = 0.30

    @classmethod
    def from_env(cls) -> "MonitorThresholds":
        return cls(
            min_confidence=_envf("CGX_MON_MIN_CONFIDENCE", 0.35),
            min_citation_coverage=_envf("CGX_MON_MIN_CITATION_COVERAGE", 1.0),
            max_repair_attempts=int(_envf("CGX_MON_MAX_REPAIR_ATTEMPTS", 2)),
            cost_spike_ratio=_envf("CGX_MON_COST_SPIKE_RATIO", 3.0),
            max_error_rate=_envf("CGX_MON_MAX_ERROR_RATE", 0.25),
            drift_score_drop=_envf("CGX_MON_DRIFT_SCORE_DROP", 0.30),
        )


def _source_ids(answer: Dict[str, Any]) -> set[str]:
    debug = answer.get("debug") or {}
    sources = debug.get("sources") or []
    return {str(s.get("chunk_id")) for s in sources if s.get("chunk_id")}


def check_groundedness(answer: Dict[str, Any], thresholds: MonitorThresholds,
                       *, run_id: Optional[str] = None) -> List[Alert]:
    """Validate citation coverage/validity and confidence of an answer payload.

    Emits ``citation_invalid`` when any cited (or inline) chunk_id is absent
    from the retrieved SOURCES, ``ungrounded_answer`` when a non-abstaining
    answer carries no citations, and ``low_confidence`` below the bound. A
    genuine abstention (no citations + low confidence) yields a single info
    alert rather than a false ungrounded/low-confidence pair.
    """
    out: List[Alert] = []
    source_ids = _source_ids(answer)
    citations = answer.get("citations") or []
    cited = [str(c.get("chunk_id")) for c in citations
             if isinstance(c, dict) and c.get("chunk_id")]
    inline = _INLINE_CITATION_RE.findall(answer.get("answer_md") or "")
    referenced = cited + inline
    confidence = answer.get("confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else None

    if referenced and source_ids:
        valid = [c for c in referenced if c in source_ids]
        coverage = len(valid) / len(referenced)
        if coverage < thresholds.min_citation_coverage:
            out.append(Alert(
                code="citation_invalid",
                severity="critical" if coverage < 0.5 else "warning",
                message=(f"{len(referenced) - len(valid)}/{len(referenced)} "
                         "citations reference chunks absent from SOURCES"),
                value=round(coverage, 4),
                threshold=thresholds.min_citation_coverage,
                run_id=run_id,
                labels={"referenced": len(referenced), "valid": len(valid)},
            ))

    abstained = not referenced and (conf is None or conf <= 0.25)
    if abstained:
        out.append(Alert(
            code="answer_abstained", severity="info",
            message="answer produced no citations and low confidence",
            value=conf, run_id=run_id,
        ))
        return out

    if not referenced:
        out.append(Alert(
            code="ungrounded_answer", severity="warning",
            message="answer carries no citations despite available SOURCES",
            value=0.0, threshold=1.0, run_id=run_id,
            labels={"sources": len(source_ids)},
        ))

    if conf is not None and conf < thresholds.min_confidence:
        out.append(Alert(
            code="low_confidence", severity="warning",
            message=f"answer confidence {conf:.2f} below bound",
            value=round(conf, 4), threshold=thresholds.min_confidence,
            run_id=run_id,
        ))
    return out


def check_repair_health(report: Dict[str, Any], thresholds: MonitorThresholds,
                        *, run_id: Optional[str] = None) -> List[Alert]:
    """Assess the self-test/repair loop from a ``codegen_report`` dict.

    ``repair_failed`` fires when the loop exhausted its retries without
    reaching ``overall_ok``; ``repair_loop_churn`` warns when the number of
    retry attempts met/exceeded the budget (a sign of a flapping fix even if
    it eventually passed); ``empty_plan`` flags a plan with no diffs.
    """
    out: List[Alert] = []
    if report.get("error"):
        out.append(Alert(
            code="repair_error", severity="warning",
            message=f"codegen self-test raised: {report['error']}",
            run_id=run_id,
        ))
        return out
    summary = report.get("summary") or {}
    attempts = int(report.get("attempts", 0) or 0)
    overall_ok = bool(summary.get("overall_ok"))
    if summary.get("empty_plan"):
        out.append(Alert(
            code="empty_plan", severity="warning",
            message="codegen produced no applicable diffs", run_id=run_id,
        ))
    if not overall_ok:
        out.append(Alert(
            code="repair_failed", severity="critical",
            message=(f"repair loop did not converge after {attempts} retry(ies)"),
            value=float(attempts), threshold=float(thresholds.max_repair_attempts),
            run_id=run_id,
            labels={k: summary.get(k) for k in
                    ("n_patches_failed", "n_syntax_failed", "tests_passed")},
        ))
    elif attempts >= thresholds.max_repair_attempts:
        out.append(Alert(
            code="repair_loop_churn", severity="warning",
            message=(f"repair converged but used {attempts} retry(ies)"),
            value=float(attempts), threshold=float(thresholds.max_repair_attempts),
            run_id=run_id,
        ))
    return out


def check_retrieval_drift(current: Dict[str, Any], baseline: Dict[str, Any],
                          thresholds: MonitorThresholds,
                          *, run_id: Optional[str] = None) -> List[Alert]:
    """Compare a current retrieval snapshot against a recorded baseline.

    Both dicts carry ``embed_model`` and ``mean_top_score``. An embedding
    model change invalidates score comparability, so it is surfaced on its
    own; otherwise a relative drop in mean top-hit score beyond
    ``drift_score_drop`` raises a warning.
    """
    out: List[Alert] = []
    cur_model = current.get("embed_model")
    base_model = baseline.get("embed_model")
    if cur_model and base_model and cur_model != base_model:
        out.append(Alert(
            code="embedding_model_drift", severity="warning",
            message=f"embed model changed {base_model!r} -> {cur_model!r}",
            run_id=run_id,
            labels={"baseline": base_model, "current": cur_model},
        ))
        return out
    base_score = baseline.get("mean_top_score")
    cur_score = current.get("mean_top_score")
    if isinstance(base_score, (int, float)) and base_score > 0 \
            and isinstance(cur_score, (int, float)):
        drop = (base_score - cur_score) / base_score
        if drop >= thresholds.drift_score_drop:
            out.append(Alert(
                code="retrieval_score_drift", severity="warning",
                message=(f"mean top score fell {drop*100:.0f}% "
                         f"({base_score:.3f} -> {cur_score:.3f})"),
                value=round(drop, 4), threshold=thresholds.drift_score_drop,
                run_id=run_id,
            ))
    return out


def check_cost_anomaly(window: Dict[str, Any], baseline: Dict[str, Any],
                       thresholds: MonitorThresholds,
                       *, run_id: Optional[str] = None) -> List[Alert]:
    """Flag provider cost spikes and elevated error rates over a window.

    ``window``/``baseline`` carry ``cost_usd``; ``window`` may also carry
    ``calls`` and ``errors`` for an error-rate check. A cost exceeding
    ``cost_spike_ratio`` x the baseline, or an error rate above
    ``max_error_rate``, raises an alert.
    """
    out: List[Alert] = []
    cur_cost = float(window.get("cost_usd", 0.0) or 0.0)
    base_cost = float(baseline.get("cost_usd", 0.0) or 0.0)
    if base_cost > 0 and cur_cost > base_cost * thresholds.cost_spike_ratio:
        out.append(Alert(
            code="cost_spike", severity="warning",
            message=(f"window cost ${cur_cost:.4f} exceeds "
                     f"{thresholds.cost_spike_ratio:g}x baseline ${base_cost:.4f}"),
            value=round(cur_cost / base_cost, 3),
            threshold=thresholds.cost_spike_ratio, run_id=run_id,
        ))
    calls = int(window.get("calls", 0) or 0)
    errors = int(window.get("errors", 0) or 0)
    if calls > 0:
        rate = errors / calls
        if rate > thresholds.max_error_rate:
            out.append(Alert(
                code="provider_error_rate", severity="critical",
                message=f"provider error rate {rate*100:.0f}% over {calls} calls",
                value=round(rate, 4), threshold=thresholds.max_error_rate,
                run_id=run_id, labels={"calls": calls, "errors": errors},
            ))
    return out
