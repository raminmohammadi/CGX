"""Tests for the agent execution graph visualizer."""

from __future__ import annotations

from cgx.agents.types import Plan, Task, TaskKind, TaskStatus
from cgx.agents.viz import render_graph_html, status_rows


def _plan() -> Plan:
    return Plan(goal="g", tasks=[
        Task(description="find module", kind=TaskKind.SEARCH,
             status=TaskStatus.DONE,
             started_at=100.0, ended_at=100.250,
             judge={"verdict": "pass", "confidence": 0.9,
                    "rationale": "x", "checked_criteria": 1}),
        Task(description="add csv export", kind=TaskKind.PLAN,
             status=TaskStatus.RUNNING),
        Task(description="explain change", kind=TaskKind.ASK,
             status=TaskStatus.PENDING),
    ])


def test_status_rows_shape_and_content():
    rows = status_rows(_plan())
    assert len(rows) == 3
    assert [r[0] for r in rows] == [1, 2, 3]
    assert [r[1] for r in rows] == ["search", "plan", "ask"]
    # status cell contains glyph + status name.
    assert "done" in rows[0][3] and "✔" in rows[0][3]
    assert "running" in rows[1][3] and "▶" in rows[1][3]
    assert "pending" in rows[2][3]
    # judge cell only populated when the judge has run.
    assert rows[0][4].startswith("pass") and "0.90" in rows[0][4]
    assert rows[1][4] == ""
    # duration in ms only for completed tasks.
    assert rows[0][5] == "250"
    assert rows[1][5] == ""


def test_render_graph_html_contains_each_task_and_legend():
    html = render_graph_html(_plan())
    assert "find module" in html
    assert "add csv export" in html
    assert "explain change" in html
    # arrow between chips (only 2 arrows for 3 tasks).
    assert html.count("→") == 2
    assert "Legend" in html


def test_render_graph_html_empty_plan_is_safe():
    empty = Plan(goal="g", tasks=[])
    html = render_graph_html(empty)
    assert "No plan yet" in html


def test_status_rows_html_escapes_dangerous_input():
    p = Plan(goal="g", tasks=[Task(
        description="<script>alert(1)</script>",
        kind=TaskKind.ASK, status=TaskStatus.PENDING,
    )])
    html = render_graph_html(p)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
