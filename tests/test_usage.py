"""Unit tests for truthful token + cost accounting (cgx.usage)."""

from __future__ import annotations

import json

from cgx import usage


def test_extract_usage_gemini_provider_counts():
    resp = {"content": "hi", "provider": "gemini",
            "raw": {"usageMetadata": {"promptTokenCount": 100,
                                      "candidatesTokenCount": 40}}}
    u = usage.extract_usage(resp, prompt_text="x" * 400, response_text="y" * 40)
    assert u["tokens_in"] == 100
    assert u["tokens_out"] == 40
    assert u["tokens_total"] == 140
    assert u["token_source"] == "provider"


def test_extract_usage_openai_and_ollama():
    oai = {"raw": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}}
    assert usage.extract_usage(oai)["tokens_in"] == 7
    olm = {"raw": {"prompt_eval_count": 11, "eval_count": 5}}
    got = usage.extract_usage(olm)
    assert (got["tokens_in"], got["tokens_out"]) == (11, 5)
    assert got["token_source"] == "provider"


def test_extract_usage_falls_back_to_estimate():
    resp = {"content": "abcd", "raw": {}}
    u = usage.extract_usage(resp, prompt_text="a" * 8, response_text="b" * 4)
    assert u["token_source"] == "estimated"
    assert u["tokens_in"] == 2  # ceil(8/4)
    assert u["tokens_out"] == 1


def test_estimate_cost_default_table_prefix_match():
    c = usage.estimate_cost("gpt-4o-2024-08-06", 1_000_000, 1_000_000)
    # gpt-4o default: (2.50 in, 10.0 out) per 1M.
    assert c["cost_usd"] == 12.5
    assert c["cost_source"] == "default"


def test_estimate_cost_unknown_model_is_zero():
    c = usage.estimate_cost("some-local-model", 1000, 1000)
    assert c["cost_usd"] == 0.0
    assert c["cost_source"] == "unknown"


def test_estimate_cost_env_override(monkeypatch):
    monkeypatch.setenv(
        "CGX_MODEL_PRICING",
        json.dumps({"some-local-model": {"in": 1.0, "out": 2.0}}))
    c = usage.estimate_cost("some-local-model", 1_000_000, 500_000)
    assert c["cost_usd"] == 2.0  # 1.0 + 0.5*2.0
    assert c["cost_source"] == "config"
