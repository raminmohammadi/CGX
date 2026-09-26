"""Tests for the JEV typed-decision layer (cgx.answer.jev) and the Phase-0
schema/bug-fix wiring that depends on it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cgx.answer.jev import TypedDecision, decide
from cgx.answer.schemas import (
    DIAGNOSIS_SCHEMA,
    SWARM_RELEVANCE_SCHEMA,
    validate_json_schema,
)


class FakeProvider:
    """A provider whose chat() replays a queue of preset ``content`` strings."""

    def __init__(self, replies, *, supports_logprobs=False, raise_exc=False):
        self._replies = list(replies)
        self.calls = 0
        self.supports_logprobs = supports_logprobs
        self._raise = raise_exc
        self.last_kwargs = None

    def chat(self, messages, temperature=0.0, max_tokens=None,
             force_json=False, json_schema=None, **kwargs):
        self.calls += 1
        self.last_kwargs = dict(
            temperature=temperature, max_tokens=max_tokens,
            force_json=force_json, json_schema=json_schema, **kwargs)
        if self._raise:
            raise RuntimeError("provider boom")
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        return {"content": reply}


def test_decide_valid_with_decision_key():
    p = FakeProvider([json.dumps({"relevant": True, "reason": "same topic"})])
    d = decide(p, "Is it relevant?", SWARM_RELEVANCE_SCHEMA,
               decision_key="relevant", system="judge")
    assert isinstance(d, TypedDecision)
    assert d.ok and d.value is True
    assert d.raw["reason"] == "same topic"
    assert d.violations == []
    assert p.calls == 1
    # force_json + schema were actually requested of the provider.
    assert p.last_kwargs["force_json"] is True
    assert p.last_kwargs["json_schema"] is SWARM_RELEVANCE_SCHEMA


def test_decide_whole_object_when_no_decision_key():
    # A DIAGNOSE-style tool-call turn: value is the whole parsed object.
    p = FakeProvider([json.dumps({"tool": "read_file", "path": "a.py"})])
    d = decide(p, [{"role": "user", "content": "go"}], DIAGNOSIS_SCHEMA)
    assert d.ok and d.value == {"tool": "read_file", "path": "a.py"}
    assert d.violations == []


def test_decide_reasks_once_on_violation_then_succeeds():
    # First reply violates the schema (relevant is a string); second conforms.
    p = FakeProvider([
        json.dumps({"relevant": "maybe"}),
        json.dumps({"relevant": False, "reason": "unrelated"}),
    ])
    d = decide(p, "q", SWARM_RELEVANCE_SCHEMA, decision_key="relevant")
    assert p.calls == 2, "should issue exactly one corrective re-ask"
    assert d.ok and d.value is False and d.violations == []


def test_decide_degrades_when_unparseable():
    p = FakeProvider(["not json at all", "still not json"])
    d = decide(p, "q", SWARM_RELEVANCE_SCHEMA, decision_key="relevant")
    assert not d.ok and d.raw == {} and d.value is None


def test_decide_never_raises_on_provider_crash():
    p = FakeProvider(["ignored"], raise_exc=True)
    d = decide(p, "q", SWARM_RELEVANCE_SCHEMA, decision_key="relevant")
    assert not d.ok and d.value is None


def test_decide_probability_clamped_from_field():
    schema = {"type": "object",
              "properties": {"confidence": {"type": "number"}},
              "required": ["confidence"]}
    p = FakeProvider([json.dumps({"confidence": 0.73})])
    d = decide(p, "q", schema)
    assert d.probability == pytest.approx(0.73)

    p2 = FakeProvider([json.dumps({"confidence": 4.2})])
    d2 = decide(p2, "q", schema)
    assert d2.probability == 1.0  # clamped into [0, 1]


def test_decide_prefers_logprob_over_self_report():
    import math
    content = json.dumps({"relevant": True})
    reply = {"content": content,
             "logprobs": {"content": [
                 {"token": "true", "logprob": math.log(0.9)}]}}

    class LPProvider(FakeProvider):
        def chat(self, *a, **k):
            self.calls += 1
            return reply

    p = LPProvider([], supports_logprobs=True)
    d = decide(p, "q", SWARM_RELEVANCE_SCHEMA, decision_key="relevant",
               prefer_logprobs=True)
    assert d.value is True
    assert d.probability == pytest.approx(0.9, abs=1e-6)


# --- schema guards -------------------------------------------------------

def test_min_max_validation():
    schema = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    assert validate_json_schema(0.5, schema) == []
    assert validate_json_schema(-0.1, schema)  # below minimum
    assert validate_json_schema(1.5, schema)   # above maximum


def test_diagnosis_schema_rejects_out_of_enum():
    # The exact bug this closes: a valid-JSON but out-of-enum action.
    bad = validate_json_schema({"minimal_action": "nuke_everything"},
                               DIAGNOSIS_SCHEMA)
    assert any("minimal_action" in e for e in bad)
    bad_tool = validate_json_schema({"tool": "rm_rf"}, DIAGNOSIS_SCHEMA)
    assert any("tool" in e for e in bad_tool)
    # But a legitimate verdict and a legitimate tool call both pass.
    assert validate_json_schema({"minimal_action": "patch_files"},
                                DIAGNOSIS_SCHEMA) == []
    assert validate_json_schema({"tool": "grep_files", "pattern": "x"},
                                DIAGNOSIS_SCHEMA) == []


def test_diagnosis_schema_enums_track_source_of_truth():
    from cgx.session.tasks.diagnose import MINIMAL_ACTIONS
    props = DIAGNOSIS_SCHEMA["properties"]
    assert set(props["minimal_action"]["enum"]) == set(MINIMAL_ACTIONS)
    assert set(props["tool"]["enum"]) == {
        "read_file", "grep_files", "inspect_packages"}


# --- query_codebase empty-snippet fix -----------------------------------

def test_query_codebase_resolves_file_and_body(tmp_path, monkeypatch):
    import cgx.session.tasks.swarm_tools as st

    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(
        {"id": "pkg/mod.py::function::do_it", "text": "def do_it():\n    return 1"}
    ) + "\n", encoding="utf-8")

    def fake_run_query_auto(**kwargs):
        return {"hits": [{"chunk_id": "pkg/mod.py::function::do_it",
                          "score": 1.0, "rank": 1, "provenance": {}}]}

    monkeypatch.setattr(st, "run_query_auto", fake_run_query_auto)
    deps = SimpleNamespace(index_dir=str(tmp_path), records_path=str(records),
                           provider=None, embed_model=None)
    out = st.query_codebase("find do_it", deps)
    assert "pkg/mod.py" in out and "do_it" in out
    assert "unknown" not in out and "(body unavailable)" not in out
    assert "def do_it" in out
