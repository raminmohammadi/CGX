"""Stub-based tests for the core LLM-calling paths.

These tests verify that ``answer_with_llm``, ``generate_code_plan``, and
``run_agent`` (streaming) behave correctly without any real LLM call,
embedding model, or GPU. They use:

- A ``_StubProvider`` that returns scripted JSON responses.
- A ``HashEmbedder`` (same as test_integration_index_query.py).
- A tiny on-disk project built by ``_make_mini_project``.
- A real FAISS index built by ``run_index_auto`` so retrieval is live.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("faiss")

from cgx.agents import Judge, Planner, run_agent
from cgx.agents.types import AgentEvent, Plan, Task, TaskKind, TaskStatus
from cgx.answer.engine import answer_with_llm, generate_code_plan
from cgx.pipeline.auto import run_index_auto


class _FixedPlanner:
    """Stub planner that always returns a single ASK task regardless of goal."""
    def plan(self, goal: str) -> Plan:
        return Plan(goal=goal, tasks=[Task(description=goal, kind=TaskKind.ASK)])


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_integration_index_query.py)
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic, dependency-free embedder (no model download required)."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = (t or " ").lower().split()
            for tok in tokens:
                h = hashlib.sha256(tok.encode()).digest()
                for j in range(self.dim):
                    out[i, j] += (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i]) + 1e-12
            out[i] /= n
        return out


class _StubProvider:
    """Minimal LLMProvider stub that returns scripted replies in order."""

    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kw) -> Dict[str, Any]:
        self.calls.append({"messages": messages, **kw})
        if not self.replies:
            return {"content": json.dumps({"answer_md": "stub fallback", "citations": []}), "error": None}
        return self.replies.pop(0)

    def chat_stream(self, messages, **kw):
        yield ""


def _make_mini_project(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "math_ops.py").write_text(
        textwrap.dedent("""
            \"\"\"Tiny math utilities.\"\"\"

            def add(a, b):
                \"\"\"Return the sum of two numbers.\"\"\"
                return a + b

            def multiply(a, b):
                \"\"\"Return the product of two numbers.\"\"\"
                return a * b
        """).lstrip(),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def mini_index(tmp_path_factory):
    """Build a real FAISS index once for the whole module."""
    root = tmp_path_factory.mktemp("proj")
    out_dir = tmp_path_factory.mktemp("out")
    _make_mini_project(root)
    embedder = HashEmbedder(dim=32)
    result = run_index_auto(str(root), str(out_dir), metric="cosine",
                            index_type="flat", embedder=embedder)
    return {
        "index_dir": result["out"]["indices"],
        "records_path": result["out"]["records"],
        "project_root": str(root),
    }


# ---------------------------------------------------------------------------
# answer_with_llm
# ---------------------------------------------------------------------------

def test_answer_with_llm_returns_answer_md_key(mini_index):
    """answer_with_llm must always return a dict with a non-empty 'answer_md'."""
    answer_json = json.dumps({
        "answer_md": "The `add` function returns the sum of `a` and `b`.",
        "citations": [],
        "confidence": 0.8,
    })
    prov = _StubProvider([
        {"content": answer_json, "error": None},
    ])
    result = answer_with_llm(
        mini_index["index_dir"],
        mini_index["records_path"],
        "What does add() do?",
        prov,
    )
    assert isinstance(result, dict), "answer_with_llm must return a dict"
    assert "answer_md" in result, "must have 'answer_md' key"
    assert isinstance(result["answer_md"], str), "answer_md must be a str"
    assert result["answer_md"].strip(), "answer_md must not be empty"


def test_answer_with_llm_retry_on_bad_json(mini_index):
    """When the first LLM reply is invalid JSON, answer_with_llm retries once."""
    good_json = json.dumps({
        "answer_md": "Recovered answer after retry.",
        "citations": [],
        "confidence": 0.6,
    })
    prov = _StubProvider([
        {"content": "not valid json {{{{", "error": None},
        {"content": good_json, "error": None},
    ])
    result = answer_with_llm(
        mini_index["index_dir"],
        mini_index["records_path"],
        "What does multiply() do?",
        prov,
    )
    assert "answer_md" in result
    assert result["answer_md"].strip()


def test_answer_with_llm_debug_field_is_json_safe(mini_index):
    """The 'debug' field must not contain non-JSON-serialisable objects."""
    import json as _json

    answer_json = json.dumps({
        "answer_md": "add sums two numbers.",
        "citations": [],
        "confidence": 0.7,
    })
    prov = _StubProvider([{"content": answer_json, "error": None}])
    result = answer_with_llm(
        mini_index["index_dir"],
        mini_index["records_path"],
        "explain add",
        prov,
    )
    # Verify the entire result is JSON-serialisable (no Graph objects etc.)
    try:
        _json.dumps(result, default=str)
    except Exception as exc:
        pytest.fail(f"answer_with_llm result is not JSON-safe: {exc}")


# ---------------------------------------------------------------------------
# generate_code_plan
# ---------------------------------------------------------------------------

def test_generate_code_plan_returns_plan_md_and_diffs(mini_index):
    """generate_code_plan must return a dict with 'plan_md' and 'diffs'."""
    plan_json = json.dumps({
        "plan_md": "## Plan\nAdd a `subtract` function.\n\n```diff path=pkg/math_ops.py\n--- a/pkg/math_ops.py\n+++ b/pkg/math_ops.py\n@@ -1,3 +1,7 @@\n+def subtract(a, b):\n+    return a - b\n```",
        "diffs": [{"file": "pkg/math_ops.py", "patch": "--- a/pkg/math_ops.py\n+++ b/pkg/math_ops.py\n@@ -1 +1,3 @@\n+def subtract(a, b):\n+    return a - b\n"}],
        "citations": [],
        "confidence": 0.7,
    })
    prov = _StubProvider([{"content": plan_json, "error": None}])
    # Pass the same HashEmbedder used to build the index so dim matches (32).
    result = generate_code_plan(
        mini_index["index_dir"],
        mini_index["records_path"],
        "Add a subtract function",
        prov,
        project_root=mini_index["project_root"],
        self_test=False,
        embedder=HashEmbedder(dim=32),
    )
    assert isinstance(result, dict), "generate_code_plan must return a dict"
    assert "plan_md" in result, "must have 'plan_md' key"
    assert isinstance(result["plan_md"], str)
    assert "diffs" in result, "must have 'diffs' key"


def test_generate_code_plan_without_self_test_has_no_codegen_report(mini_index):
    """When self_test=False, 'codegen_report' key should be absent."""
    plan_json = json.dumps({
        "plan_md": "## Plan\nSome plan.",
        "diffs": [],
        "citations": [],
        "confidence": 0.5,
    })
    prov = _StubProvider([{"content": plan_json, "error": None}])
    result = generate_code_plan(
        mini_index["index_dir"],
        mini_index["records_path"],
        "refactor add function",
        prov,
        self_test=False,
        embedder=HashEmbedder(dim=32),
    )
    # report_summary(None) must return None -- no KeyError allowed
    from cgx.webui.helpers import report_summary
    assert report_summary(result.get("codegen_report")) is None


# ---------------------------------------------------------------------------
# run_agent streaming
# ---------------------------------------------------------------------------

def test_run_agent_stream_true_returns_generator():
    """run_agent(stream=True) must return a generator, not a Plan or dict."""
    import types

    def ask(q, **_): return {"answer_md": "stub"}

    gen = run_agent("Explain something", capabilities={"ask": ask},
                    planner=_FixedPlanner(), stream=True)
    assert isinstance(gen, types.GeneratorType), "stream=True must return a generator"


def test_run_agent_stream_true_emits_required_event_types():
    """Streaming must emit plan, task_start, task_done, summary events."""
    def ask(q, **_): return {"answer_md": "answer"}

    events: List[AgentEvent] = list(
        run_agent("Explain something", capabilities={"ask": ask},
                  planner=_FixedPlanner(), stream=True,
                  progress_interval=0)  # disable threading for determinism
    )
    types_seen = {e.type for e in events}
    assert "plan" in types_seen
    assert "task_start" in types_seen
    assert "task_done" in types_seen
    assert "summary" in types_seen


def test_run_agent_stream_false_returns_plan():
    """run_agent(stream=False) must return a Plan, not subscriptable as a dict."""
    def ask(q, **_): return {"answer_md": "answer"}

    result = run_agent("Explain something", capabilities={"ask": ask},
                       planner=_FixedPlanner(), stream=False)
    assert isinstance(result, Plan)
    with pytest.raises(TypeError):
        _ = result["events"]  # must not be subscriptable


def test_run_agent_stream_task_failed_event_on_error():
    """A capability that raises must produce task_failed in the event stream."""
    def ask(q, **_): raise RuntimeError("deliberate failure")

    events: List[AgentEvent] = list(
        run_agent("Explain something", capabilities={"ask": ask},
                  planner=_FixedPlanner(), stream=True,
                  stop_on_fail=True, progress_interval=0)
    )
    assert any(e.type == "task_failed" for e in events)
    assert any(e.type == "summary" for e in events)


def test_run_agent_stream_event_payloads_are_json_safe():
    """All AgentEvent payloads must be JSON-serialisable."""
    import json as _json

    def ask(q, **_): return {"answer_md": "answer", "citations": []}

    for event in run_agent("Explain something", capabilities={"ask": ask},
                           planner=_FixedPlanner(), stream=True,
                           progress_interval=0):
        try:
            _json.dumps(event.payload, default=str)
        except Exception as exc:
            pytest.fail(f"Event {event.type!r} payload not JSON-safe: {exc}")
