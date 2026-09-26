"""Tests for the llama-server decision provider + jev logprob routing."""

from __future__ import annotations

import math

import pytest

import cgx.answer.jev as jev
import cgx.answer.providers as pv
from cgx.answer.schemas import SWARM_RELEVANCE_SCHEMA

_BOOL_SCHEMA = {"type": "object", "properties": {"relevant": {"type": "boolean"}}}


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def _payload(content='{"relevant": true}', logprob=-0.1):
    return {"choices": [{
        "message": {"content": content},
        "logprobs": {"content": [
            {"token": "true", "logprob": logprob,
             "top_logprobs": [{"token": "true", "logprob": logprob}]}]},
    }]}


def test_openai_compat_requests_and_returns_logprobs(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return _Resp(_payload())

    monkeypatch.setattr(pv.requests, "post", fake_post)
    prov = pv.OpenAICompatProvider(model="m", base_url="http://x",
                                   allow_no_auth=True, supports_logprobs=True)
    out = prov.chat(messages=[{"role": "user", "content": "q"}],
                    force_json=True, json_schema=_BOOL_SCHEMA,
                    logprobs=True, top_logprobs=5)
    assert captured["body"].get("logprobs") is True
    assert captured["body"].get("top_logprobs") == 5
    assert out.get("logprobs")            # surfaced to the caller
    assert out["content"] == '{"relevant": true}'


def test_openai_compat_omits_logprobs_when_unsupported(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return _Resp(_payload())

    monkeypatch.setattr(pv.requests, "post", fake_post)
    prov = pv.OpenAICompatProvider(model="m", base_url="http://x",
                                   allow_no_auth=True, supports_logprobs=False)
    prov.chat(messages=[{"role": "user", "content": "q"}], force_json=True,
              logprobs=True)
    assert "logprobs" not in captured["body"]  # not requested


def test_decision_provider_from_env(monkeypatch):
    monkeypatch.setenv("CGX_JEV_BASE_URL", "http://127.0.0.1:8080")
    dp = jev.decision_provider()
    assert dp is not None and dp.supports_logprobs is True
    monkeypatch.delenv("CGX_JEV_BASE_URL", raising=False)
    assert jev.decision_provider() is None


def test_decide_routes_to_decision_provider_for_logprobs(monkeypatch):
    used = {}

    class DP:
        supports_logprobs = True

        def chat(self, **kwargs):
            used["hit"] = True
            return {"content": '{"relevant": true}',
                    "logprobs": {"content": [
                        {"token": "true", "logprob": math.log(0.8)}]}}

    monkeypatch.setattr(jev, "decision_provider", lambda: DP())

    class Plain:  # no logprobs; would self-report otherwise
        supports_logprobs = False

        def chat(self, **kwargs):
            return {"content": '{"relevant": false}'}

    d = jev.decide(Plain(), "q", SWARM_RELEVANCE_SCHEMA,
                   decision_key="relevant", prefer_logprobs=True)
    assert used.get("hit")                       # routed to the logprob endpoint
    assert d.value is True
    assert d.probability == pytest.approx(0.8, abs=1e-6)


def test_decide_stays_on_provider_without_prefer(monkeypatch):
    # Even with a decision provider available, no prefer_logprobs -> stay put.
    monkeypatch.setattr(jev, "decision_provider",
                        lambda: (_ for _ in ()).throw(AssertionError("built")))

    class Plain:
        supports_logprobs = False

        def chat(self, **kwargs):
            return {"content": '{"relevant": false}'}

    d = jev.decide(Plain(), "q", SWARM_RELEVANCE_SCHEMA, decision_key="relevant")
    assert d.value is False
