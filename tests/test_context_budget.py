"""Tests for the deterministic num_ctx guard (cgx.session.context_budget)."""

from __future__ import annotations

from cgx.session.context_budget import (
    cap_text,
    estimate_tokens,
    fit_react_messages,
    messages_tokens,
)


def _sys(c="SYSTEM PROMPT"):
    return {"role": "system", "content": c}


def _u(c):
    return {"role": "user", "content": c}


def _a(c):
    return {"role": "assistant", "content": c}


def test_estimate_and_messages_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # ceil(5/4)
    assert messages_tokens([_u("abcd"), _a("abcd")]) == 2


def test_cap_text_noop_when_short():
    assert cap_text("hello", 100) == "hello"


def test_cap_text_keeps_head_and_tail():
    text = "H" * 100 + "M" * 100 + "T" * 100
    out = cap_text(text, 120)
    assert len(out) <= 120
    assert out.startswith("H")
    assert out.rstrip().endswith("T")   # tail preserved (middle-out)
    assert "[trimmed" in out


def test_fit_noop_when_within_budget():
    msgs = [_sys(), _u("objective"), _a("short"), _u("obs")]
    out = fit_react_messages(msgs, effective_ctx=100_000, reserve_output=1000)
    assert [m["content"] for m in out] == [m["content"] for m in msgs]


def test_fit_does_not_mutate_input():
    msgs = [_sys(), _u("obj"), _u("X" * 10_000)]
    before = msgs[2]["content"]
    fit_react_messages(msgs, effective_ctx=2000, reserve_output=200,
                       per_msg_cap_chars=400)
    assert msgs[2]["content"] == before  # original untouched


def test_fit_caps_oversized_middle_body_but_not_head():
    big = "X" * 20_000
    msgs = [_sys("S" * 5_000), _u("OBJECTIVE" * 10), _u(big), _a("recent")]
    out = fit_react_messages(msgs, effective_ctx=100_000, reserve_output=0,
                             per_msg_cap_chars=500)
    # head (system + objective) preserved byte-identical
    assert out[0]["content"] == msgs[0]["content"]
    assert out[1]["content"] == msgs[1]["content"]
    # the oversized middle body was capped
    assert len(out[2]["content"]) <= 500


def test_fit_drops_oldest_middle_and_preserves_head_and_tail():
    # Many oversized middle turns; tiny window forces dropping the oldest.
    head = [_sys("S"), _u("OBJECTIVE")]
    middle = [_u(f"obs-{i}-" + "Z" * 4_000) for i in range(6)]
    tail = [_a("assistant-latest"), _u("observation-latest")]
    msgs = head + middle + tail
    out = fit_react_messages(msgs, effective_ctx=2_000, reserve_output=200,
                             keep_recent=2, per_msg_cap_chars=None)
    contents = [m["content"] for m in out]
    # stable head byte-identical
    assert contents[0] == "S" and contents[1] == "OBJECTIVE"
    # most recent turns always kept
    assert "assistant-latest" in contents
    assert "observation-latest" in contents
    # an elision marker replaced the dropped oldest middle turns
    assert any("elided" in c for c in contents)
    # and it actually shrank
    assert messages_tokens(out) < messages_tokens(msgs)


def test_fit_empty():
    assert fit_react_messages([], effective_ctx=1000) == []
