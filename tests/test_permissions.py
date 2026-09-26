"""Tests for the shared sensitivity scorer + programmable permission policy."""

from __future__ import annotations

import pytest

from cgx.answer.sensitivity import Sensitivity, score_path, score_paths
from cgx.guardrails.command import (
    HARD_DENY,
    evaluate_action,
    gate_action,
    policy_mode,
)
from cgx.session.tasks.tool_registry import (
    RiskLevel,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)

# --- sensitivity scorer --------------------------------------------------

@pytest.mark.parametrize("path,level", [
    ("config/.env", Sensitivity.RESTRICTED),
    ("app/.env.production", Sensitivity.RESTRICTED),
    ("/home/u/.ssh/id_rsa", Sensitivity.RESTRICTED),
    ("certs/server.pem", Sensitivity.RESTRICTED),
    ("infra/prod.tfstate", Sensitivity.RESTRICTED),
    ("terraform.tfvars", Sensitivity.RESTRICTED),
    ("docs/guide.md", Sensitivity.PUBLIC),
    ("node_modules/lib/index.js", Sensitivity.PUBLIC),
    (".venv/lib/site.py", Sensitivity.PUBLIC),
    ("src/app/main.py", Sensitivity.STANDARD),
    ("", Sensitivity.STANDARD),
])
def test_score_path(path, level):
    assert score_path(path) is level


def test_score_paths_takes_most_restrictive():
    assert score_paths(["docs/a.md", "src/b.py"]) is Sensitivity.STANDARD
    assert score_paths(["docs/a.md", ".env"]) is Sensitivity.RESTRICTED
    assert score_paths(["docs/a.md"]) is Sensitivity.PUBLIC
    assert score_paths([]) is Sensitivity.STANDARD


# --- evaluate_action -----------------------------------------------------

def test_evaluate_ssrf_is_hard_deny():
    pol = evaluate_action("fetch_url", {"url": "http://127.0.0.1:8080/admin"})
    assert pol.verdict == "deny" and pol.is_hard_deny


def test_evaluate_sensitive_path_deny():
    pol = evaluate_action("run_python_probe",
                          {"code": "open('/home/u/.ssh/id_rsa').read()"})
    assert pol.verdict == "deny" and pol.confidence >= HARD_DENY


def test_evaluate_restricted_file_deny():
    pol = evaluate_action("write_file", {"path": "infra/prod.tfstate",
                                         "content": "x"})
    assert pol.verdict == "deny" and pol.is_hard_deny


def test_evaluate_secret_literal_deny():
    pol = evaluate_action("write_file",
                          {"content": "KEY = 'sk-abcd1234efgh5678ijkl'"})
    assert pol.verdict == "deny"


def test_evaluate_network_egress_soft_deny_unless_deploy():
    code = {"code": "import requests\nrequests.post('https://api.example.com', json=d)"}
    soft = evaluate_action("run_python_probe", code)
    assert soft.verdict == "deny" and not soft.is_hard_deny  # soft (0.7)
    ok = evaluate_action("run_python_probe", code, task_scope="deploy")
    assert ok.verdict != "deny"  # allowed under a deploy-scoped task


def test_evaluate_readonly_allow_and_default_ask():
    assert evaluate_action("read_file", {"path": "src/a.py"},
                           risk=RiskLevel.LOW).verdict == "allow"
    assert evaluate_action("write_file", {"path": "src/a.py", "content": "x=1"},
                           risk=RiskLevel.HIGH).verdict == "ask"


# --- gate_action / policy_mode ------------------------------------------

def test_policy_mode_default_advisory(monkeypatch):
    monkeypatch.delenv("CGX_POLICY_MODE", raising=False)
    assert policy_mode() == "advisory"
    monkeypatch.setenv("CGX_POLICY_MODE", "ENFORCE")
    assert policy_mode() == "enforce"
    monkeypatch.setenv("CGX_POLICY_MODE", "junk")
    assert policy_mode() == "advisory"


def test_gate_action_advisory_enforces_hard_only(monkeypatch):
    monkeypatch.setenv("CGX_POLICY_MODE", "advisory")
    # hard deny (sensitive path) is enforced
    assert gate_action("run_python_probe",
                       {"code": "open('~/.ssh/id_rsa')"}) is not None
    # soft deny (egress) is logged, not enforced, in advisory mode
    assert gate_action("run_python_probe",
                       {"code": "import socket; socket.socket()"}) is None


def test_gate_action_enforce_blocks_soft(monkeypatch):
    monkeypatch.setenv("CGX_POLICY_MODE", "enforce")
    assert gate_action("run_python_probe",
                       {"code": "import socket; socket.socket()"}) is not None


def test_gate_action_off_never_blocks(monkeypatch):
    monkeypatch.setenv("CGX_POLICY_MODE", "off")
    assert gate_action("run_python_probe",
                       {"code": "open('~/.ssh/id_rsa')"}) is None


# --- dispatch integration ------------------------------------------------

def _reg():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="probe", description="run code",
                          handler=lambda a, c: "RAN", risk=RiskLevel.HIGH))
    reg.register(ToolSpec(name="read_x", description="read",
                          handler=lambda a, c: "BODY", risk=RiskLevel.LOW))
    return reg


def test_dispatch_denies_sensitive_action_by_default(monkeypatch):
    monkeypatch.delenv("CGX_POLICY_MODE", raising=False)  # advisory default
    reg = _reg()
    out = reg.dispatch(
        ToolCall("probe", {"code": "open('/home/u/.ssh/id_rsa')"}, ""),
        ToolContext(root="."))
    assert "denied by policy" in out


def test_dispatch_allows_benign_action(monkeypatch):
    monkeypatch.delenv("CGX_POLICY_MODE", raising=False)
    reg = _reg()
    # No gate installed -> a benign call runs its handler unchanged.
    out = reg.dispatch(ToolCall("read_x", {"path": "src/a.py"}, ""),
                       ToolContext(root="."))
    assert out == "BODY"
    out2 = reg.dispatch(ToolCall("probe", {"code": "print(1 + 1)"}, ""),
                        ToolContext(root="."))
    assert out2 == "RAN"
