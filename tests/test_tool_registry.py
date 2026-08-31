"""Unit tests for the unified swarm tool registry + tolerant parser."""

from cgx.session.tasks.tool_registry import (
    REGISTRY, RiskLevel, ToolContext, ToolRegistry, ToolSpec, parse_tool_calls)


def _echo(args, ctx):
    return f"echo:{args.get('x', '')}"


def _reg():
    r = ToolRegistry()
    r.register(ToolSpec(name="echo", description="echo x", handler=_echo,
                        risk=RiskLevel.LOW, arg_hint='{"x": "..."}'))
    return r


def test_parse_single_call():
    calls = parse_tool_calls('<call_tool name="echo">{"x": "hi"}</call_tool>')
    assert len(calls) == 1
    assert calls[0].name == "echo"
    assert calls[0].args == {"x": "hi"}


def test_parse_multiple_calls():
    text = ('<call_tool name="a">{"x": 1}</call_tool> then '
            '<call_tool name="b">{"y": 2}</call_tool>')
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]


def test_parse_tolerates_single_quotes_and_whitespace():
    calls = parse_tool_calls("<call_tool  name = 'echo' >{\"x\": 1}</call_tool>")
    assert len(calls) == 1 and calls[0].name == "echo"


def test_parse_non_json_body_kept_as_input():
    calls = parse_tool_calls('<call_tool name="echo">just text</call_tool>')
    assert calls[0].args == {"input": "just text"}


def test_dispatch_runs_handler():
    r = _reg()
    (call,) = parse_tool_calls('<call_tool name="echo">{"x": "yo"}</call_tool>')
    assert r.dispatch(call, ToolContext(root=".")) == "echo:yo"


def test_dispatch_unknown_tool():
    r = _reg()
    (call,) = parse_tool_calls('<call_tool name="nope">{}</call_tool>')
    assert "Unknown tool" in r.dispatch(call, ToolContext(root="."))


def test_describe_for_prompt_lists_tools():
    r = _reg()
    text = r.describe_for_prompt(["echo"])
    assert "echo" in text and "<call_tool" in text


def test_native_tools_registered():
    # register_native_tools runs at import of swarm_tools.
    import cgx.session.tasks.swarm_tools  # noqa: F401
    for name in ("run_python_probe", "file_skeleton", "list_symbols",
                 "search_web", "query_codebase"):
        assert REGISTRY.get(name) is not None


class _DenyGate:
    def request(self, name, args, risk):
        return type("D", (), {"approved": False, "reason": "test-deny"})()


def test_dispatch_respects_approval_gate():
    r = _reg()
    (call,) = parse_tool_calls('<call_tool name="echo">{"x": 1}</call_tool>')
    out = r.dispatch(call, ToolContext(root=".", approval_gate=_DenyGate()))
    assert "not approved" in out and "test-deny" in out


# --- diagnose tools now dispatch through the shared registry mechanism --------

class _FC:
    installed_packages = ["fastapi", "pytest"]


def test_diagnose_run_tool_inspect_packages(tmp_path):
    from pathlib import Path
    from cgx.session.tasks.diagnose import _run_tool
    out = _run_tool("inspect_packages", {}, _FC(), Path(tmp_path))
    assert "fastapi" in out and "pytest" in out


def test_diagnose_run_tool_read_file_refuses_traversal(tmp_path):
    from pathlib import Path
    from cgx.session.tasks.diagnose import _run_tool
    out = _run_tool("read_file", {"path": "../../etc/passwd"}, _FC(),
                    Path(tmp_path))
    assert "refused" in out or "no such file" in out


def test_diagnose_run_tool_unknown(tmp_path):
    from pathlib import Path
    from cgx.session.tasks.diagnose import _run_tool
    out = _run_tool("bogus", {}, _FC(), Path(tmp_path))
    assert "Unknown tool" in out
