"""Schema-constrained decoding: pure translators, per-provider request shape,
boundary validation, and the bounded re-ask at each LLM executor boundary.

Covers the JSON-Schema -> native-request translation helpers, asserts that
each provider (a) emits the constrained request when handed a ``json_schema``
and (b) degrades gracefully to plain JSON mode when the backend rejects it,
and exercises the three greenfield boundaries (manifest / clarify / repair):
schema threaded into the call, one re-ask with actionable violations on a
shape miss, and a hard stop after the second miss.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from cgx.answer import providers
from cgx.answer.schemas import (
    CLARIFY_QUESTIONS_SCHEMA,
    MANIFEST_SCHEMA,
    REPAIR_FILES_SCHEMA,
    to_gemini_schema,
    to_openai_response_format,
    validate_json_schema,
)


# ---------------------------------------------------------------------------
# Pure translators
# ---------------------------------------------------------------------------
def test_openai_response_format_wraps_schema_non_strict():
    rf = to_openai_response_format(MANIFEST_SCHEMA, name="manifest")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "manifest"
    assert rf["json_schema"]["strict"] is False
    assert rf["json_schema"]["schema"] is MANIFEST_SCHEMA


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
        [{"role": "user", "content": "hi"}], json_schema=MANIFEST_SCHEMA,
    )
    assert ok.bodies[0]["format"] == MANIFEST_SCHEMA

    bad = _Recorder([400, 200], _OLLAMA_OK)
    monkeypatch.setattr(providers.requests, "post", bad)
    providers.OllamaProvider(model="llama3").chat(
        [{"role": "user", "content": "hi"}], json_schema=MANIFEST_SCHEMA,
    )
    assert len(bad.bodies) == 2
    assert bad.bodies[0]["format"] == MANIFEST_SCHEMA
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
        [{"role": "user", "content": "hi"}], json_schema=CLARIFY_QUESTIONS_SCHEMA,
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
        [{"role": "user", "content": "hi"}], json_schema=REPAIR_FILES_SCHEMA,
    )
    assert len(rec.bodies) == 3
    assert rec.bodies[0]["response_format"]["type"] == "json_schema"
    assert rec.bodies[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in rec.bodies[2]


# ---------------------------------------------------------------------------
# Boundary validator (pure)
# ---------------------------------------------------------------------------
_VALID_MANIFEST = {
    "plan_md": "overview",
    "contracts": {},
    "layers": [{"name": "core", "files": [
        {"path": "src/app.py", "description": "entry",
         "depends_on": ["src/db.py"]},
    ]}],
}


def test_validate_json_schema_accepts_conforming_payloads():
    assert validate_json_schema(_VALID_MANIFEST, MANIFEST_SCHEMA) == []
    assert validate_json_schema(
        {"questions": [{"prompt": "Which framework?"}]},
        CLARIFY_QUESTIONS_SCHEMA) == []
    # An empty files array is REPAIR's explicit "no fix" signal.
    assert validate_json_schema({"files": []}, REPAIR_FILES_SCHEMA) == []


def test_validate_json_schema_reports_pathed_violations():
    errs = validate_json_schema({"layers": {}}, MANIFEST_SCHEMA)
    assert errs == ["$.layers: expected an array, got dict"]

    errs = validate_json_schema(
        {"layers": [{"name": "core"}]}, MANIFEST_SCHEMA)
    assert errs == ["$.layers[0].files: required key is missing"]

    errs = validate_json_schema(
        {"files": [{"path": "a.py"}, "junk"]}, REPAIR_FILES_SCHEMA)
    assert errs == [
        "$.files[0].content: required key is missing",
        "$.files[1]: expected an object, got str",
    ]


def test_validate_json_schema_checks_min_items_scalars_and_enum():
    assert validate_json_schema({"layers": []}, MANIFEST_SCHEMA) == [
        "$.layers: expected at least 1 item(s), got 0"]
    assert validate_json_schema(
        {"questions": [{"prompt": 7}]}, CLARIFY_QUESTIONS_SCHEMA,
    ) == ["$.questions[0].prompt: expected a string, got int"]
    enum_schema = {"type": "string", "enum": ["pass", "fail"]}
    assert validate_json_schema("pass", enum_schema) == []
    assert validate_json_schema("maybe", enum_schema) == [
        "$: expected one of ['pass', 'fail']"]
    num_schema = {"type": "number"}
    assert validate_json_schema(True, num_schema) == [
        "$: expected a number, got bool"]


# ---------------------------------------------------------------------------
# Executor boundaries: schema threaded + one bounded re-ask on violation
# ---------------------------------------------------------------------------
class _ScriptedChat:
    """Returns scripted replies in order; records every chat kwargs."""

    def __init__(self, replies: List[str]):
        self._replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages=None, **kwargs):
        kwargs["messages"] = messages
        self.calls.append(kwargs)
        return {"content": self._replies.pop(0) if self._replies else "{}"}


def test_plan_scaffold_manifest_reasks_once_on_schema_violation():
    from cgx.answer.engine import plan_scaffold_manifest

    provider = _ScriptedChat([
        json.dumps({"layers": {}}),        # wrong shape -> re-ask
        json.dumps(_VALID_MANIFEST),       # conforming
    ])
    out = plan_scaffold_manifest("build a calculator", provider,
                                 goal="build a calculator")
    assert len(provider.calls) == 2
    assert all(c["json_schema"] is MANIFEST_SCHEMA for c in provider.calls)
    reask = provider.calls[1]["messages"]
    assert reask[-2]["role"] == "assistant"
    assert "Violations" in reask[-1]["content"]
    assert "$.layers: expected an array" in reask[-1]["content"]
    paths = [f["path"] for layer in out["layers"] for f in layer["files"]]
    assert "src/app.py" in paths


def test_generate_repair_files_reasks_once_then_gives_up():
    from cgx.answer.engine import generate_repair_files

    fixed = "def add(a, b):\n    return a + b\n"
    good = json.dumps({"files": [{"path": "src/calc.py", "content": fixed}]})
    files = [{"path": "src/calc.py",
              "content": "def add(a, b):\n    return a - b\n"}]

    provider = _ScriptedChat([json.dumps({"files": {"path": "x"}}), good])
    out = generate_repair_files(provider, goal="g", failure_text="E",
                                files=files)
    assert out == {"src/calc.py": fixed}
    assert len(provider.calls) == 2
    assert all(c["json_schema"] is REPAIR_FILES_SCHEMA
               for c in provider.calls)
    assert "Violations" in provider.calls[1]["messages"][-1]["content"]

    # Second miss: no third call, empty mapping (caller regenerates).
    provider = _ScriptedChat(["not json", "still not json"])
    out = generate_repair_files(provider, goal="g", failure_text="E",
                                files=files)
    assert out == {}
    assert len(provider.calls) == 2


def test_clarify_questions_reask_recovers_then_falls_back():
    from cgx.session.tasks.base import ExecutorDeps
    from cgx.session.tasks.clarify_requirements import _ask_llm_for_questions

    good = json.dumps({"questions": [
        {"id": "q1", "prompt": "Which framework?"},
        {"id": "q2", "prompt": "Which database?"},
        {"id": "q3", "prompt": "Need auth?"},
    ]})

    # Shape miss -> one re-ask carrying the violation -> recovered.
    provider = _ScriptedChat([json.dumps({"question": "one?"}), good])
    qs = _ask_llm_for_questions("build an app", ExecutorDeps(provider=provider))
    assert [q["id"] for q in qs] == ["q1", "q2", "q3"]
    assert len(provider.calls) == 2
    assert all(c["json_schema"] is CLARIFY_QUESTIONS_SCHEMA
               for c in provider.calls)
    reask = provider.calls[1]["messages"][-1]["content"]
    assert "Violations" in reask
    assert "$.questions: required key is missing" in reask

    # Second miss returns [] so the executor's fallback bank takes over.
    provider = _ScriptedChat(["nope", "still nope"])
    qs = _ask_llm_for_questions("build an app", ExecutorDeps(provider=provider))
    assert qs == []
    assert len(provider.calls) == 2


def test_to_gemini_schema_open_object():
    src = {"type": "object"}
    out = to_gemini_schema(src)
    assert out["type"] == "OBJECT"
    assert "properties" in out
    assert "_extra" in out["properties"]
