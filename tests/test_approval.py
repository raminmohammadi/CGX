"""Tests for the human-in-the-loop approval gate."""

import threading
import time

import pytest

from cgx.session.approval import (
    ApprovalDecision, ApprovalGate, ApprovalMode, mode_from_env)
from cgx.session.tasks.tool_registry import RiskLevel


def test_off_mode_auto_approves():
    gate = ApprovalGate(mode=ApprovalMode.OFF)
    d = gate.request("run_python_probe", {}, RiskLevel.HIGH)
    assert d.approved


def test_risky_mode_allows_low_blocks_high():
    gate = ApprovalGate(mode=ApprovalMode.RISKY,
                        responder=lambda r: ApprovalDecision(False, "no"))
    assert gate.request("query_codebase", {}, RiskLevel.LOW).approved
    assert not gate.request("run_python_probe", {}, RiskLevel.HIGH).approved


def test_all_mode_gates_even_low():
    seen = []
    gate = ApprovalGate(mode=ApprovalMode.ALL,
                        responder=lambda r: seen.append(r) or ApprovalDecision(True))
    assert gate.request("query_codebase", {}, RiskLevel.LOW).approved
    assert len(seen) == 1


def test_responder_approves():
    gate = ApprovalGate(mode=ApprovalMode.RISKY,
                        responder=lambda r: ApprovalDecision(True, "ok"))
    d = gate.request("edit_file", {"path": "x"}, RiskLevel.HIGH)
    assert d.approved and d.reason == "ok"


def test_ttl_auto_reject():
    gate = ApprovalGate(mode=ApprovalMode.RISKY, ttl_seconds=0.2)
    start = time.time()
    d = gate.request("run_python_probe", {}, RiskLevel.HIGH)
    assert not d.approved
    assert "auto-rejected" in d.reason
    assert time.time() - start >= 0.2


def test_out_of_band_resolve_unblocks():
    gate = ApprovalGate(mode=ApprovalMode.RISKY, ttl_seconds=5)
    result = {}

    def worker():
        result["d"] = gate.request("edit_file", {}, RiskLevel.HIGH)

    t = threading.Thread(target=worker)
    t.start()
    # Wait for the request to register, then approve it.
    for _ in range(50):
        if gate.pending():
            break
        time.sleep(0.01)
    pend = gate.pending()
    assert len(pend) == 1
    assert gate.resolve(pend[0]["request_id"], ApprovalDecision(True, "yes"))
    t.join(timeout=2)
    assert result["d"].approved


def test_mode_from_env(monkeypatch):
    monkeypatch.setenv("CGX_APPROVAL_MODE", "all")
    assert mode_from_env() is ApprovalMode.ALL
    monkeypatch.setenv("CGX_APPROVAL_MODE", "garbage")
    assert mode_from_env() is ApprovalMode.RISKY


def test_gate_integrates_with_registry_dispatch():
    from cgx.session.approval import use_gate
    from cgx.session.tasks.tool_registry import (
        REGISTRY, RiskLevel as RL, ToolContext, ToolRegistry, ToolSpec,
        parse_tool_calls)
    r = ToolRegistry()
    r.register(ToolSpec(name="danger", description="d",
                        handler=lambda a, c: "ran", risk=RL.HIGH))
    (call,) = parse_tool_calls('<call_tool name="danger">{}</call_tool>')
    gate = ApprovalGate(mode=ApprovalMode.RISKY,
                        responder=lambda req: ApprovalDecision(False, "denied"))
    with use_gate(gate):
        out = r.dispatch(call, ToolContext(root="."))
    assert "not approved" in out
