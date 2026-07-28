"""Schema-constrained decoding: pure translators + per-provider request shape.

Covers the JSON-Schema -> native-request translation helpers and asserts that
each provider (a) emits the constrained request when handed a ``json_schema``
and (b) degrades gracefully to plain JSON mode when the backend rejects it.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from cgx.answer import providers
from cgx.answer.schemas import (
    JUDGE_SCHEMA,
    PLAN_SCHEMA,
    to_gemini_schema,
    to_openai_response_format,
)


# ---------------------------------------------------------------------------
# Pure translators
# ---------------------------------------------------------------------------
def test_openai_response_format_wraps_schema_non_strict():
    rf = to_openai_response_format(PLAN_SCHEMA, name="plan")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "plan"
    assert rf["json_schema"]["strict"] is False
    assert rf["json_schema"]["schema"] is PLAN_SCHEMA


def test_gemini_schema_uppercases_types_and_strips_unknown_keys():
    src = {
        "type": "object",
        "additionalProperties": False,   # unsupported -> dropped
        "title": "ignore me",            # unsupported -> dropped
        "properties": {
            "tasks": {
                "type": "array",
                "items": {"type": "string", "enum": ["a", "b"]},
            },
        },
        "required": ["tasks"],
    }
    out = to_gemini_schema(src)
    assert out["type"] == "OBJECT"
    assert "additionalProperties" not in out and "title" not in out
    assert out["properties"]["tasks"]["type"] == "ARRAY"
    assert out["properties"]["tasks"]["items"]["type"] == "STRING"
    assert out["properties"]["tasks"]["items"]["enum"] == ["a", "b"]
    assert out["required"] == ["tasks"]


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status: int, payload: Dict[str, Any]):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Recorder:
    """Records each posted body (deep-copied) and returns scripted statuses."""

    def __init__(self, statuses: List[int], payload: Dict[str, Any]):
        self._statuses = list(statuses)
        self._payload = payload
        self.bodies: List[Dict[str, Any]] = []

    def __call__(self, url, json=None, timeout=None, headers=None, stream=False, **kw):  # noqa: A002
        self.bodies.append(copy.deepcopy(json))
        status = self._statuses.pop(0) if self._statuses else 200
        return _FakeResp(status, self._payload)


_OLLAMA_OK = {"message": {"content": "{}"}}
_GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
_OAI_OK = {"choices": [{"message": {"content": "{}"}}]}


# ---------------------------------------------------------------------------
# Ollama: format=<schema>, fallback to "json"
# ---------------------------------------------------------------------------
def test_ollama_sends_schema_format_and_falls_back(monkeypatch):
    ok = _Recorder([200], _OLLAMA_OK)
    monkeypatch.setattr(providers.requests, "post", ok)
    providers.OllamaProvider(model="llama3").chat(
        [{"role": "user", "content": "hi"}], json_schema=PLAN_SCHEMA,
    )
    assert ok.bodies[0]["format"] == PLAN_SCHEMA

    bad = _Recorder([400, 200], _OLLAMA_OK)
    monkeypatch.setattr(providers.requests, "post", bad)
    providers.OllamaProvider(model="llama3").chat(
        [{"role": "user", "content": "hi"}], json_schema=PLAN_SCHEMA,
    )
    assert len(bad.bodies) == 2
    assert bad.bodies[0]["format"] == PLAN_SCHEMA
    assert bad.bodies[1]["format"] == "json"


# ---------------------------------------------------------------------------
# Ollama: keep_alive keeps the model resident between requests
# ---------------------------------------------------------------------------
def test_ollama_sends_keep_alive_by_default(monkeypatch):
    rec = _Recorder([200], _OLLAMA_OK)
    monkeypatch.setattr(providers.requests, "post", rec)
    providers.OllamaProvider(model="llama3").chat(
        [{"role": "user", "content": "hi"}], force_json=False,
    )
    assert rec.bodies[0]["keep_alive"] == providers.DEFAULT_OLLAMA_KEEP_ALIVE


def test_ollama_keep_alive_omitted_when_none(monkeypatch):
    rec = _Recorder([200], _OLLAMA_OK)
    monkeypatch.setattr(providers.requests, "post", rec)
    providers.OllamaProvider(model="llama3", keep_alive=None).chat(
        [{"role": "user", "content": "hi"}], force_json=False,
    )
    assert "keep_alive" not in rec.bodies[0]


def test_build_provider_flows_keep_alive_to_ollama():
    from cgx.webui.helpers import build_provider
    prov = build_provider(kind="ollama", model="llama3",
                          base_url="http://localhost:11434")
    assert isinstance(prov, providers.OllamaProvider)
    assert prov.keep_alive == providers.DEFAULT_OLLAMA_KEEP_ALIVE


# ---------------------------------------------------------------------------
# Gemini: responseSchema present, dropped on 400
# ---------------------------------------------------------------------------
def test_gemini_sends_response_schema_and_falls_back(monkeypatch):
    bad = _Recorder([400, 200], _GEMINI_OK)
    monkeypatch.setattr(providers.requests, "post", bad)
    providers.GeminiProvider(model="gemini-2.5-flash", api_key="x").chat(
        [{"role": "user", "content": "hi"}], json_schema=JUDGE_SCHEMA,
    )
    assert len(bad.bodies) == 2
    assert "responseSchema" in bad.bodies[0]["generationConfig"]
    assert bad.bodies[0]["generationConfig"]["responseSchema"]["type"] == "OBJECT"
    assert "responseSchema" not in bad.bodies[1]["generationConfig"]
    assert bad.bodies[1]["generationConfig"]["responseMimeType"] == "application/json"


# ---------------------------------------------------------------------------
# OpenAI-compat: json_schema -> json_object -> plain ladder
# ---------------------------------------------------------------------------
def test_openai_compat_walks_response_format_ladder(monkeypatch):
    rec = _Recorder([400, 400, 200], _OAI_OK)
    monkeypatch.setattr(providers.requests, "post", rec)
    providers.OpenAICompatProvider(model="m", base_url="http://x").chat(
        [{"role": "user", "content": "hi"}], json_schema=PLAN_SCHEMA,
    )
    assert len(rec.bodies) == 3
    assert rec.bodies[0]["response_format"]["type"] == "json_schema"
    assert rec.bodies[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in rec.bodies[2]
