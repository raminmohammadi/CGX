"""Tests for cgx.answer.model_caps."""

import json
from typing import Any, Dict, List

from cgx.answer.model_caps import (
    DEFAULT_CONTEXT_TOKENS,
    TIERS,
    get_capability_tier,
    get_context_map_budget,
    get_model_context_window,
    get_prompt_strategy,
    get_summary_budget,
    provider_model_name,
)


# ---------------------------------------------------------------------------
# get_model_context_window
# ---------------------------------------------------------------------------
def test_window_exact_gemini():
    assert get_model_context_window("gemini-2.5-flash") == 1_000_000
    assert get_model_context_window("gemini-1.5-pro") == 2_000_000


def test_window_case_insensitive_and_strips_whitespace():
    assert get_model_context_window("  Gemini-2.5-Flash  ") == 1_000_000


def test_window_strips_ollama_tag_suffix():
    # Ollama tags like ":3b", ":7b-instruct" must collapse to the base.
    assert get_model_context_window("qwen2.5-coder:3b") == 32_768
    assert get_model_context_window("llama3.1:8b-instruct-q4_0") == 128_000
    assert get_model_context_window("deepseek-r1:14b") == 128_000


def test_window_strips_param_size_suffix():
    # "-3b" / "-70b" / "-8x7b" appended after the base name.
    assert get_model_context_window("qwen2.5-coder-7b") == 32_768
    assert get_model_context_window("mixtral-8x7b") == 32_768
    assert get_model_context_window("llama3.1-70b") == 128_000


def test_window_family_substring_match():
    # Unknown specific tag, but the family is recognisable.
    assert get_model_context_window("gemini-2.5-flash-preview-09-2025") \
        == 1_000_000
    assert get_model_context_window("gpt-4o-2024-08-06") == 128_000


def test_window_unknown_falls_back_to_default():
    assert get_model_context_window("totally-made-up-model") \
        == DEFAULT_CONTEXT_TOKENS
    assert get_model_context_window("") == DEFAULT_CONTEXT_TOKENS
    assert get_model_context_window(None) == DEFAULT_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# provider_model_name
# ---------------------------------------------------------------------------
class _Prov:
    def __init__(self, model: str = ""):
        self.model = model


def test_provider_model_name_reads_model_attr():
    assert provider_model_name(_Prov("gemini-2.5-flash")) == "gemini-2.5-flash"


def test_provider_model_name_none_when_provider_or_model_missing():
    assert provider_model_name(None) is None
    assert provider_model_name(_Prov("")) is None
    assert provider_model_name(object()) is None  # no .model attribute


# ---------------------------------------------------------------------------
# get_summary_budget tiers
# ---------------------------------------------------------------------------
def test_budget_tier_tiny_local():
    # Llama 3 base, Gemma -- 8K window.
    b = get_summary_budget(_Prov("llama3"))
    assert b == {"max_chars": 400, "max_files": 12, "output_tokens": 2_000}


def test_budget_tier_mid_local():
    # Qwen / Mistral 32K class.
    b = get_summary_budget(_Prov("qwen2.5-coder:3b"))
    assert b == {"max_chars": 800, "max_files": 30, "output_tokens": 4_000}


def test_budget_tier_large_cloud():
    # GPT-4o / Llama 3.1 -- 128K.
    b = get_summary_budget(_Prov("gpt-4o"))
    assert b == {"max_chars": 1_500, "max_files": 60, "output_tokens": 6_000}


def test_budget_tier_huge_cloud():
    # Gemini 2.5 -- 1M, Claude -- 200K. Both should land in the top tier.
    b_gem = get_summary_budget(_Prov("gemini-2.5-flash"))
    b_claude = get_summary_budget(_Prov("claude-3-5-sonnet"))
    assert b_gem == {"max_chars": 3_000, "max_files": 120,
                     "output_tokens": 8_000}
    assert b_claude == {"max_chars": 3_000, "max_files": 120,
                        "output_tokens": 8_000}


def test_budget_unknown_provider_uses_smallest_tier():
    # Unknown -> DEFAULT_CONTEXT_TOKENS (8K) -> tiny tier.
    b = get_summary_budget(_Prov("nope"))
    assert b["max_chars"] == 400
    assert b["output_tokens"] == 2_000


# ---------------------------------------------------------------------------
# Integration: generate_single_scaffold_file honours the provider budget
# ---------------------------------------------------------------------------
class _RecordingProvider:
    """Returns a canned JSON reply and records each chat() call."""

    def __init__(self, model: str, content: str):
        self.model = model
        self._content = content
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kw):  # noqa: ANN001 -- duck type
        self.calls.append({"messages": messages, **kw})
        return {"content": self._content}


def test_generate_single_scaffold_file_uses_provider_budget():
    from cgx.answer.engine import generate_single_scaffold_file

    canned = json.dumps({"content": "print('ok')\n"})
    # 50 prior files; tiny-tier (Llama 3) should cap at 12 in the prompt.
    existing = [
        {"path": f"f{i}.py", "content": f"def fn_{i}(): pass\n"}
        for i in range(50)
    ]

    prov = _RecordingProvider("llama3", canned)
    generate_single_scaffold_file(
        "new.py", "make a new file", prov,
        existing_files_with_content=existing, goal="g",
    )
    user_msg = prov.calls[0]["messages"][1]["content"]
    # max_files=12 → exactly 12 "### " context-block headers.
    assert user_msg.count("### f") == 12, user_msg.count("### f")
    # output_tokens=2000 for the tiny tier.
    assert prov.calls[0]["max_tokens"] == 2_000

    prov2 = _RecordingProvider("gemini-2.5-flash", canned)
    generate_single_scaffold_file(
        "new.py", "make a new file", prov2,
        existing_files_with_content=existing, goal="g",
    )
    # Top tier: all 50 prior files fit (cap 120).
    user_msg2 = prov2.calls[0]["messages"][1]["content"]
    assert user_msg2.count("### f") == 50
    assert prov2.calls[0]["max_tokens"] == 8_000


# ---------------------------------------------------------------------------
# get_capability_tier
# ---------------------------------------------------------------------------
def test_capability_tier_bands_from_context_window():
    # One representative model per band; tiers derive solely from the
    # context window (8K -> small, 32K -> medium, 128K -> large, 200K+ ->
    # xlarge).
    assert get_capability_tier(_Prov("llama3")) == "small"
    assert get_capability_tier(_Prov("qwen2.5-coder")) == "medium"
    assert get_capability_tier(_Prov("gpt-4o")) == "large"
    assert get_capability_tier(_Prov("gemini-2.5-flash")) == "xlarge"
    assert get_capability_tier(_Prov("claude-3-5-sonnet")) == "xlarge"


def test_capability_tier_accepts_provider_or_raw_string():
    # A raw model-id string and a provider carrying the same id agree.
    assert get_capability_tier("gpt-4o") == get_capability_tier(_Prov("gpt-4o"))
    # Unknown / missing model falls back to the smallest tier.
    assert get_capability_tier(_Prov("nope")) == "small"
    assert get_capability_tier(None) == "small"
    assert all(t in TIERS for t in {"small", "medium", "large", "xlarge"})


# ---------------------------------------------------------------------------
# Budgets derive from the tier and return safe copies
# ---------------------------------------------------------------------------
def test_budgets_are_tier_indexed_and_return_copies():
    # Same tier -> identical budget for two different models in that band.
    assert get_summary_budget(_Prov("gpt-4o")) == \
        get_summary_budget(_Prov("o1-mini"))
    assert get_context_map_budget(_Prov("gpt-4o")) == \
        get_context_map_budget(_Prov("o1-mini"))
    # Mutating the returned dict must not corrupt the shared table.
    b = get_summary_budget(_Prov("llama3"))
    b["max_files"] = 9999
    assert get_summary_budget(_Prov("llama3"))["max_files"] == 12


# ---------------------------------------------------------------------------
# get_prompt_strategy tiers
# ---------------------------------------------------------------------------
def test_prompt_strategy_hardens_weak_models_and_frees_strong_ones():
    small = get_prompt_strategy(_Prov("llama3"))
    assert small["tier"] == "small"
    assert small["reinforce_json"] is True
    assert small["plan_max_tokens"] == 1_000

    large = get_prompt_strategy(_Prov("gemini-2.5-flash"))
    assert large["tier"] == "xlarge"
    assert large["reinforce_json"] is False
    assert large["plan_max_tokens"] == 2_000
