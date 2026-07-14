"""Tests for the interactive terminal dashboard (cgx.cli.tui).

Exercises the pure rendering helpers and the command-dispatch state
machine. No network or engine work happens here: heavy operations
(index/agent/serve) are surfaced as ``action`` tokens by ``dispatch`` and
executed by the loop, which these tests deliberately do not drive.
"""

from __future__ import annotations

import os
import threading

from cgx.cli.tui import ansi, ops, render
from cgx.cli.tui.app import Dashboard, DashboardState
from cgx.cli.tui.runner import Printer, run_stream


def _dash(**kw) -> Dashboard:
    state = DashboardState(project_root=os.getcwd(), model="test-model", **kw)
    d = Dashboard(state=state, out=lambda *_a, **_k: None)
    d.color = False
    return d


# --- rendering -------------------------------------------------------

def test_banner_has_six_rows_and_no_color_when_disabled():
    banner = render.render_banner(enabled=False)
    assert len(banner.splitlines()) == 6
    assert "\x1b[" not in banner


def test_banner_is_colored_when_enabled():
    assert "\x1b[38;5;" in render.render_banner(enabled=True)


def test_visible_len_ignores_ansi():
    colored = ansi.fg("hello", 220, enabled=True)
    assert render.visible_len(colored) == 5


def test_status_bar_shows_model_and_context():
    bar = render.render_status_bar(
        cwd="/tmp/x", index_state="index ready", model="qwen",
        context_pct=100, width=80, enabled=False)
    assert "qwen" in bar and "100% context left" in bar
    assert "index ready" in bar


def test_status_bar_context_pct_clamped():
    bar = render.render_status_bar(
        cwd="/tmp/x", index_state="s", model="m", context_pct=250,
        width=80, enabled=False)
    assert "100% context left" in bar


def test_abbreviate_path_uses_tilde():
    home = os.path.expanduser("~")
    assert render.abbreviate_path(os.path.join(home, "proj")) == "~/proj"


def test_help_lists_core_commands():
    text = render.render_help()
    for cmd in ("/index", "/model", "/provider", "/quit", "/help"):
        assert cmd in text


def test_input_box_is_three_lines():
    box = render.render_input_box("type here", width=60, enabled=False)
    assert len(box.splitlines()) == 3


# --- dispatch --------------------------------------------------------

def test_plain_text_routes_to_agent():
    r = _dash().dispatch("add a CSV export function")
    assert r.action == "agent"
    assert r.arg == "add a CSV export function"


def test_empty_line_is_noop():
    r = _dash().dispatch("   ")
    assert r.action == "text" and r.output == ""


def test_help_command_returns_reference():
    r = _dash().dispatch("/help")
    assert r.action == "text" and "Commands:" in r.output


def test_model_command_updates_state():
    d = _dash()
    r = d.dispatch("/model gpt-4o")
    assert d.state.model == "gpt-4o"
    assert "gpt-4o" in r.output


def test_model_command_without_arg_shows_usage():
    r = _dash().dispatch("/model")
    assert "usage" in r.output.lower()


def test_provider_kind_switch():
    d = _dash()
    r = d.dispatch("/provider openai")
    assert d.state.provider_kind == "openai"
    assert d.state.profile_name is None
    assert "openai" in r.output


def test_unknown_provider_reports_error():
    r = _dash().dispatch("/provider nope-not-real")
    assert "no profile or kind" in r.output


def test_quit_and_clear_and_status_actions():
    d = _dash()
    assert d.dispatch("/quit").action == "quit"
    assert d.dispatch("/exit").action == "quit"
    assert d.dispatch("/clear").action == "clear"
    assert d.dispatch("/status").action == "status"
    assert d.dispatch("/serve").action == "serve"


def test_index_action_carries_path_arg():
    r = _dash().dispatch("/index /some/path")
    assert r.action == "index" and r.arg == "/some/path"


def test_project_switch_rejects_missing_dir():
    r = _dash().dispatch("/project /definitely/not/here/xyz")
    assert "not a directory" in r.output


def test_project_switch_accepts_existing_dir(tmp_path):
    d = _dash()
    r = d.dispatch(f"/project {tmp_path}")
    assert d.state.project_root == str(tmp_path)
    assert "project ->" in r.output


def test_unknown_command_hints_help():
    r = _dash().dispatch("/frobnicate")
    assert "unknown command" in r.output and "/help" in r.output


def test_ask_dispatch_carries_question():
    r = _dash().dispatch("/ask how does indexing work?")
    assert r.action == "ask" and r.arg == "how does indexing work?"


def test_ask_without_arg_shows_usage():
    r = _dash().dispatch("/ask")
    assert r.action == "text" and "usage" in r.output.lower()


def test_help_lists_ask_command():
    assert "/ask" in render.render_help()


# --- map_event (pure render mapping) ---------------------------------

def test_map_event_answer_delta_is_inline():
    instr = ops.map_event("answer_delta", {"delta": "hello"}, enabled=False)
    assert instr.op == "inline" and instr.text == "hello"


def test_map_event_intent_sets_status():
    instr = ops.map_event("intent", {"mode": "explain"}, enabled=False)
    assert instr.op == "status" and "explain" in instr.text


def test_map_event_error_is_line():
    instr = ops.map_event("error", {"message": "boom"}, enabled=False)
    assert instr.op == "line" and "boom" in instr.text


def test_map_event_agent_plan_counts_tasks():
    payload = {"plan": {"tasks": [{"kind": "ask"}, {"kind": "plan"}]}}
    instr = ops.map_event("plan", payload, enabled=False)
    assert instr.op == "line" and "2 task(s)" in instr.text
    assert "ask" in instr.text and "plan" in instr.text


def test_map_event_index_result_shows_counts():
    payload = {"summary": {"counts": {"intent": 5}}}
    instr = ops.map_event("result", payload, enabled=False)
    assert instr.op == "line" and "index ready" in instr.text


def test_map_event_summary_extracts_answer_md():
    payload = {"plan": {"tasks": [
        {"kind": "search", "output": {}},
        {"kind": "summarize", "output": {"answer_md": "the answer"}},
    ]}}
    instr = ops.map_event("summary", payload, enabled=False)
    assert instr.op == "line" and "the answer" in instr.text


def test_map_event_unknown_is_nothing():
    assert ops.map_event("totally-new-event", {}, enabled=False).op == "nothing"


# --- threaded runner: streaming + cancellation -----------------------

def _collect_printer():
    """A non-TTY Printer that records every full-line write."""
    lines: list = []
    printer = Printer(write=lambda s: lines.append(s), is_tty=False,
                      enabled=False)
    return printer, lines


def test_run_stream_drains_events_in_order():
    printer, lines = _collect_printer()
    seen: list = []

    def make_events():
        yield "a", {"n": 1}
        yield "b", {"n": 2}

    status = run_stream(make_events, on_event=seen.append, printer=printer)
    assert status == "ok"
    assert seen == [("a", {"n": 1}), ("b", {"n": 2})]


def test_run_stream_reraises_generator_error():
    printer, _ = _collect_printer()

    def make_events():
        yield "x", {}
        raise ValueError("kaboom")

    try:
        run_stream(make_events, on_event=lambda _e: None, printer=printer)
    except ValueError as exc:
        assert "kaboom" in str(exc)
    else:
        raise AssertionError("expected ValueError to propagate")


def test_run_stream_cancels_and_sets_event():
    printer, lines = _collect_printer()
    cancel = threading.Event()

    def make_events():
        # Simulate the user hitting Ctrl-C while the first event is handled.
        yield "first", {}
        yield "second", {}

    def on_event(_ev):
        raise KeyboardInterrupt

    status = run_stream(make_events, on_event=on_event, printer=printer,
                        cancel_event=cancel)
    assert status == "cancelled"
    assert cancel.is_set()


def test_printer_line_writes_through_injected_writer():
    printer, lines = _collect_printer()
    printer.line("hello world")
    assert "hello world\n" in lines
