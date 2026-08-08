"""Tests for cost & quota governance (budget + meter + manager + provider)."""

from __future__ import annotations

import asyncio
import json

import pytest

from cgx.governance import (
    BudgetConfig,
    BudgetExceeded,
    GovernedProvider,
    QuotaManager,
    UsageMeter,
    govern,
)


class FakeProvider:
    """Minimal LLMProvider stand-in for governance tests."""

    model = "test-model"
    parallel_scaffold_capable = True

    def chat(self, messages, temperature=0.2, max_tokens=None,
             force_json=True, **kwargs):
        return {"content": "hello world response"}

    def chat_stream(self, messages, temperature=0.2, max_tokens=None,
                    force_json=False, **kwargs):
        for w in ("a", "b", "c"):
            yield w


def _mgr(config=None):
    return QuotaManager(config or BudgetConfig(), meter=UsageMeter(":memory:"))


# --------------------------------------------------------------------------
# BudgetConfig: env parsing + per-owner overrides
# --------------------------------------------------------------------------
def test_budget_config_from_env(monkeypatch):
    monkeypatch.setenv("CGX_BUDGET_DAILY_COST_USD", "5")
    monkeypatch.setenv("CGX_BUDGET_DAILY_TOKENS", "1000")
    monkeypatch.setenv("CGX_BUDGET_OWNERS",
                       '{"alice": {"cost": 1.5, "tokens": 200}}')
    cfg = BudgetConfig.from_env()
    assert cfg.enabled is True
    assert cfg.daily_cost_usd == 5.0 and cfg.daily_tokens == 1000.0
    assert cfg.limits_for("alice") == (1.5, 200.0)
    assert cfg.limits_for("bob") == (5.0, 1000.0)  # falls back to global


# --------------------------------------------------------------------------
# UsageMeter: record + aggregate
# --------------------------------------------------------------------------
def test_meter_totals_owners_summary():
    m = UsageMeter(":memory:")
    m.record("a", tokens_in=1, tokens_out=2, cost_usd=0.001)
    m.record("b", tokens_in=3, tokens_out=4, cost_usd=0.002)
    m.record("a", tokens_in=5, tokens_out=6, cost_usd=0.003)
    assert m.owners() == ["a", "b"]
    ta = m.totals("a")
    assert ta["calls"] == 2 and ta["tokens_total"] == 14
    assert abs(ta["cost_usd"] - 0.004) < 1e-9
    assert len(m.summary()) == 2
    m.close()


# --------------------------------------------------------------------------
# QuotaManager: soft-warn then hard-stop
# --------------------------------------------------------------------------
def test_quota_manager_states_and_enforcement():
    mgr = _mgr(BudgetConfig(daily_tokens=100, soft_ratio=0.8))
    assert mgr.check("o", enforce=False)["state"] == "ok"
    mgr.meter.record("o", tokens_in=85, tokens_out=0, cost_usd=0.0)
    assert mgr.check("o", enforce=False)["state"] == "warn"
    mgr.meter.record("o", tokens_in=20, tokens_out=0, cost_usd=0.0)
    assert mgr.check("o", enforce=False)["state"] == "exceeded"
    with pytest.raises(BudgetExceeded):
        mgr.check("o")  # enforce=True by default


def test_quota_manager_record_usage():
    mgr = _mgr()
    totals = mgr.record_usage("o", tokens_in=10, tokens_out=5,
                              cost_usd=0.02, model="gpt-4o")
    assert totals["tokens_total"] == 15 and totals["calls"] == 1


# --------------------------------------------------------------------------
# GovernedProvider: metering + passthrough + hard-stop
# --------------------------------------------------------------------------
def test_governed_provider_meters_chat_and_stream():
    mgr = _mgr()
    gp = GovernedProvider(FakeProvider(), manager=mgr)
    assert gp.model == "test-model"
    assert gp.parallel_scaffold_capable is True  # __getattr__ passthrough

    resp = gp.chat([{"role": "user", "content": "a question here"}], force_json=False)
    assert resp["content"] == "hello world response"
    assert "".join(gp.chat_stream([{"role": "user", "content": "hi"}], force_json=False)) == "abc"

    totals = mgr.meter.totals("default")
    assert totals["calls"] == 2 and totals["tokens_total"] > 0


def test_governed_provider_hard_stop():
    mgr = _mgr(BudgetConfig(daily_tokens=5))
    mgr.meter.record("default", tokens_in=10, tokens_out=0, cost_usd=0.0)
    gp = GovernedProvider(FakeProvider(), manager=mgr)
    with pytest.raises(BudgetExceeded):
        gp.chat([{"role": "user", "content": "x"}], force_json=False)


def test_govern_gating_and_idempotency():
    fp = FakeProvider()
    disabled = _mgr(BudgetConfig(enabled=False))
    assert govern(fp, manager=disabled) is fp  # disabled -> unchanged
    assert govern(None, manager=disabled) is None

    enabled = _mgr(BudgetConfig(enabled=True))
    wrapped = govern(fp, manager=enabled)
    assert isinstance(wrapped, GovernedProvider)
    assert govern(wrapped, manager=enabled) is wrapped  # already governed


# --------------------------------------------------------------------------
# Route: /api/usage (+ /api/usage/summary)
# --------------------------------------------------------------------------
async def _asgi(app, method, path):
    scope = {"type": "http", "http_version": "1.1", "method": method,
             "path": path.split("?")[0], "raw_path": path.encode(),
             "query_string": path.split("?", 1)[1].encode() if "?" in path else b"",
             "headers": [], "client": ("test", 1), "server": ("test", 80),
             "scheme": "http", "root_path": ""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    out = {"status": None, "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return out


def test_usage_route(monkeypatch):
    import cgx.governance.manager as mgr_mod
    from cgx.webui.server import create_app

    mgr = _mgr(BudgetConfig(daily_tokens=100))
    mgr.meter.record("alice", tokens_in=10, tokens_out=5, cost_usd=0.01)
    monkeypatch.setattr(mgr_mod, "_DEFAULT", mgr)
    app = create_app()

    r = asyncio.run(_asgi(app, "GET", "/api/usage?owner=alice"))
    body = json.loads(r["body"])
    assert body["tokens_total"] == 15 and body["state"] == "ok"

    s = asyncio.run(_asgi(app, "GET", "/api/usage/summary"))
    assert json.loads(s["body"])["count"] == 1
