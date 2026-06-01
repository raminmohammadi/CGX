"""Static renderers for agent execution state.

Two surfaces are produced:

* :func:`status_rows` — table rows ``[#, kind, description, status, judge, ms]``
  consumed by the React task table on the Agent page.
* :func:`render_graph_html` — a self-contained, dependency-free HTML chip
  flow that visualises the plan as a horizontal arrow chain.

Both are pure functions over a :class:`~cgx.agents.types.Plan`; the
streaming handler calls them on each event so the UI updates live.
"""

from __future__ import annotations

from html import escape
from typing import Any, List

from cgx.agents.types import Plan, Task, TaskStatus


_STATUS_GLYPH = {
    TaskStatus.PENDING: "•",
    TaskStatus.RUNNING: "▶",
    TaskStatus.DONE: "✔",
    TaskStatus.FAILED: "✖",
    TaskStatus.SKIPPED: "⊝",
}

_STATUS_COLOR = {
    TaskStatus.PENDING: ("#f1f3f5", "#495057"),
    TaskStatus.RUNNING: ("#fff3bf", "#5c3c00"),
    TaskStatus.DONE: ("#d3f9d8", "#1b4332"),
    TaskStatus.FAILED: ("#ffe3e3", "#842029"),
    TaskStatus.SKIPPED: ("#e9ecef", "#495057"),
}


def _duration_ms(task: Task) -> str:
    if task.started_at is None or task.ended_at is None:
        return ""
    ms = max(0, int((task.ended_at - task.started_at) * 1000))
    return str(ms)


def _judge_cell(task: Task) -> str:
    j = task.judge or {}
    if not j:
        return ""
    v = str(j.get("verdict") or "")
    try:
        conf = float(j.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return f"{v} ({conf:.2f})" if v else ""


def status_rows(plan: Plan) -> List[List[Any]]:
    """Return ``[[#, kind, description, status, judge, ms], …]`` rows."""
    rows: List[List[Any]] = []
    for i, t in enumerate(plan.tasks, 1):
        rows.append([
            i,
            t.kind.value,
            t.description,
            f"{_STATUS_GLYPH[t.status]} {t.status.value}",
            _judge_cell(t),
            _duration_ms(t),
        ])
    return rows


def render_graph_html(plan: Plan) -> str:
    """Render the plan as an inline HTML chip flow.

    The output is intentionally self-contained: no external CSS, no JS,
    and uses inline styles so it survives the React renderer's HTML
    sanitisation when injected via ``dangerouslySetInnerHTML``.
    """
    if not plan or not plan.tasks:
        return "<div style='color:#868e96'>No plan yet.</div>"
    chips = []
    for i, t in enumerate(plan.tasks, 1):
        bg, fg = _STATUS_COLOR[t.status]
        glyph = _STATUS_GLYPH[t.status]
        desc = escape(t.description[:80])
        kind = escape(t.kind.value)
        chip = (
            f"<div style='display:inline-block;vertical-align:top;"
            f"max-width:220px;margin:4px 0;padding:8px 12px;border-radius:10px;"
            f"background:{bg};color:{fg};border:1px solid rgba(0,0,0,0.08);"
            f"font-family:system-ui,-apple-system,Segoe UI,sans-serif;"
            f"font-size:12px;line-height:1.3'>"
            f"<div style='font-weight:600;letter-spacing:.02em'>"
            f"{glyph}&nbsp;#{i} · {kind}"
            f"</div>"
            f"<div style='margin-top:4px;color:inherit;opacity:0.9'>"
            f"{desc}"
            f"</div>"
            f"</div>"
        )
        chips.append(chip)
        if i < len(plan.tasks):
            chips.append(
                "<span style='display:inline-block;margin:0 6px;color:#868e96;"
                "font-size:18px;vertical-align:top;line-height:48px'>→</span>"
            )
    legend = (
        "<div style='margin-top:8px;color:#868e96;font-size:11px'>"
        "Legend: ▶ running · ✔ done · ✖ failed · ⊝ skipped · • pending"
        "</div>"
    )
    return (
        "<div style='padding:6px 2px;overflow-x:auto;white-space:nowrap'>"
        + "".join(chips)
        + "</div>"
        + legend
    )
