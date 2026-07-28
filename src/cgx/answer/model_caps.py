

"""Model capability registry.

Different LLMs have wildly different context windows (Gemini 2.5 Flash
~1M tokens, Qwen2.5-Coder 3B ~32K, Llama 3 ~8K). Callers that build long
prompts -- most notably the project scaffolder that embeds "already
generated" files as context for each new sibling file -- need to know
how much room they actually have so they neither overflow small local
models nor waste capacity on large cloud ones.

This module exposes a small, deliberately conservative registry plus a
set of accessors:

* :func:`get_model_context_window` -- token count for a given model id.
* :func:`get_capability_tier` -- coarse ``small`` / ``medium`` /
  ``large`` / ``xlarge`` classification derived *solely* from the
  context window. This is the single source of truth every budget /
  prompt-strategy decision keys off, so a new model only needs a window
  entry and never a per-model branch at the call sites.
* :func:`get_summary_budget` / :func:`get_context_map_budget` --
  tier-indexed prompt/response budgets (looked up in the tables below).
* :func:`get_prompt_strategy` -- tier-driven prompt knobs (planning
  ``max_tokens``, whether to reinforce the strict-JSON contract) so
  weak local models get the hardening they need while strong models
  take the lean path.

The registry is intentionally a flat ``dict`` so new models can be
added without changing the public API. Matching is case-insensitive
and tolerant of Ollama ``model:tag`` suffixes and ``-3b``/``-7b``
parameter-size suffixes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Conservative fallback for unknown models. 8K is the smallest window
# we still encounter in practice (Llama 3 base, Gemma).
DEFAULT_CONTEXT_TOKENS = 8_192

# Approximate context windows in tokens. Numbers come from each
# vendor's model card and are rounded down where the upstream spec is
# fuzzy, so callers always have headroom for the response.
_MODEL_CONTEXT_TOKENS: Dict[str, int] = {
    # Google Gemini -------------------------------------------------------
    "gemini-2.5-pro":          1_000_000,
    "gemini-2.5-flash":        1_000_000,
    "gemini-2.5-flash-lite":   1_000_000,
    "gemini-2.0-flash":        1_000_000,
    "gemini-2.0-flash-lite":   1_000_000,
    "gemini-1.5-pro":          2_000_000,
    "gemini-1.5-flash":        1_000_000,
    "gemini-1.5-flash-8b":     1_000_000,
    # OpenAI --------------------------------------------------------------
    "gpt-4o":                    128_000,
    "gpt-4o-mini":               128_000,
    "gpt-4-turbo":               128_000,
    "gpt-4":                       8_192,
    "gpt-3.5-turbo":              16_384,
    "o1":                        200_000,
    "o1-mini":                   128_000,
    "o3":                        200_000,
    "o3-mini":                   200_000,
    # Anthropic Claude (via OpenAI-compat gateways) -----------------------
    "claude-3-5-sonnet":         200_000,
    "claude-3-5-haiku":          200_000,
    "claude-3-opus":             200_000,
    "claude-3-sonnet":           200_000,
    "claude-3-haiku":            200_000,
    "claude-opus-4":             200_000,
    "claude-sonnet-4":           200_000,
    # Ollama / local ------------------------------------------------------
    "qwen2.5-coder":              32_768,
    "qwen2.5":                    32_768,
    "qwen3":                      32_768,
    "qwen3-coder":                32_768,
    "llama3.1":                  128_000,
    "llama3.2":                  128_000,
    "llama3.3":                  128_000,
    "llama3":                      8_192,
    "deepseek-coder-v2":         128_000,
    "deepseek-coder":             16_384,
    "deepseek-v3":               128_000,
    "deepseek-r1":               128_000,
    "codellama":                  16_384,
    "mistral":                    32_768,
    "mistral-nemo":              128_000,
    "mixtral":                    32_768,
    "phi3":                      128_000,
    "phi4":                       16_384,
    "gemma2":                      8_192,
    "gemma3":                    128_000,
    "gemma4":                    128_000,
    "gemma":                       8_192,
    "starcoder2":                 16_384,
}


def get_model_context_window(model: Optional[str]) -> int:
    """Return the approximate context window (tokens) for ``model``.

    Matching order:
      1. exact match (case-insensitive)
      2. drop Ollama ``:tag`` (``qwen2.5-coder:3b`` → ``qwen2.5-coder``)
      3. drop trailing parameter-size suffix (``-3b``, ``-70b``, ``-8x7b``)
      4. family substring match
      5. :data:`DEFAULT_CONTEXT_TOKENS`
    """
    if not model:
        return DEFAULT_CONTEXT_TOKENS
    m = model.strip().lower()
    if m in _MODEL_CONTEXT_TOKENS:
        return _MODEL_CONTEXT_TOKENS[m]
    base = m.split(":", 1)[0]
    if base in _MODEL_CONTEXT_TOKENS:
        return _MODEL_CONTEXT_TOKENS[base]
    base2 = re.sub(r"-\d+(?:x\d+)?\.?\d*b$", "", base)
    if base2 in _MODEL_CONTEXT_TOKENS:
        return _MODEL_CONTEXT_TOKENS[base2]
    for key, ctx in _MODEL_CONTEXT_TOKENS.items():
        if key in base2 or base2 in key:
            return ctx
    return DEFAULT_CONTEXT_TOKENS


def provider_model_name(provider: Any) -> Optional[str]:
    """Best-effort extraction of the model id from any provider instance."""
    if provider is None:
        return None
    name = getattr(provider, "model", None)
    return str(name) if name else None


# Model families that expose a native reasoning / "thinking" phase. Matching
# is substring-based against the lowercased model id so Ollama tags
# (``deepseek-r1:8b``) and vendor variants (``o4-mini``) all resolve without
# an exact-name entry. Ordinary instruct models (gemma, plain llama, phi3,
# mistral) are absent on purpose -- for those the ASK sketch phase is pure
# latency with no reasoning payoff, so it is skipped.
_THINKING_MODEL_KEYS = (
    # Local / Ollama reasoning models
    "deepseek-r1",
    "qwq",
    "qwen3",
    "gpt-oss",
    "magistral",
    "cogito",
    "smallthinker",
    "phi4-reasoning",
    "phi-4-reasoning",
    "granite3.2",
    "granite3.3",
    # Cloud reasoning models
    "o1",
    "o3",
    "o4-mini",
    "gemini-2.5",
)


def model_supports_thinking(model: Optional[str]) -> bool:
    """Return True when ``model`` exposes a native reasoning/"thinking" phase.

    The ASK stream uses this to decide whether the (slow) thought sketch is
    worth running: only reasoning-capable models produce a meaningful
    thinking pass, so non-reasoning models (e.g. gemma, plain llama) skip it
    and answer directly. Matching mirrors :func:`get_model_context_window`:
    case-insensitive and tolerant of Ollama ``model:tag`` suffixes.
    """
    if not model:
        return False
    m = model.strip().lower()
    return any(key in m for key in _THINKING_MODEL_KEYS)


# Capability tiers, ordered small -> xlarge. A single ladder keyed on the
# context window so every downstream budget / prompt-strategy decision
# derives from one classification instead of re-deriving ad-hoc
# thresholds at each call site. The four bands are retained (rather than
# a literal three) because the top two cloud bands (128K vs 200K+) still
# warrant distinct budgets.
CapabilityTier = str  # one of: "small", "medium", "large", "xlarge"

TIERS = ("small", "medium", "large", "xlarge")

# (upper-exclusive context-window bound, tier); the catch-all top tier is
# returned when no bound matches.
_TIER_LADDER = (
    (16_000, "small"),
    (64_000, "medium"),
    (200_000, "large"),
)
_TOP_TIER: CapabilityTier = "xlarge"


def get_capability_tier(provider_or_model: Any) -> CapabilityTier:
    """Classify a provider (or raw model-id string) into a capability tier.

    The tier is derived only from the context window via
    :func:`get_model_context_window`, so adding a model to the registry
    is enough to tier it -- no per-model branching anywhere else.
    """
    model = (
        provider_or_model
        if isinstance(provider_or_model, str)
        else provider_model_name(provider_or_model)
    )
    ctx = get_model_context_window(model)
    for bound, tier in _TIER_LADDER:
        if ctx < bound:
            return tier
    return _TOP_TIER


# Tier -> budget tables. Values are unchanged from the previous inline
# thresholds; centralising them here is what lets every accessor share
# one tier classification.
_SUMMARY_BUDGETS: Dict[CapabilityTier, Dict[str, int]] = {
    "small":  {"max_chars": 400,   "max_files": 12,  "output_tokens": 2_000},
    "medium": {"max_chars": 800,   "max_files": 30,  "output_tokens": 4_000},
    "large":  {"max_chars": 1_500, "max_files": 60,  "output_tokens": 6_000},
    "xlarge": {"max_chars": 3_000, "max_files": 120, "output_tokens": 8_000},
}

_CONTEXT_MAP_BUDGETS: Dict[CapabilityTier, Dict[str, int]] = {
    "small": {
        "primary_chars": 900,  "neighbor_chars": 220,
        "primary_max": 8,      "neighbor_max": 12,
        "total_chars": 6_000,
    },
    "medium": {
        "primary_chars": 1_400, "neighbor_chars": 320,
        "primary_max": 12,      "neighbor_max": 24,
        "total_chars": 18_000,
    },
    "large": {
        "primary_chars": 2_200, "neighbor_chars": 420,
        "primary_max": 20,      "neighbor_max": 40,
        "total_chars": 48_000,
    },
    "xlarge": {
        "primary_chars": 3_500, "neighbor_chars": 520,
        "primary_max": 32,      "neighbor_max": 60,
        "total_chars": 120_000,
    },
}

# Tier -> prompt-strategy knobs. ``plan_max_tokens`` caps the planner /
# decision LLM completion; small tiers keep the historical 2K ceiling
# while stronger models are allowed longer plans. ``reinforce_json``
# adds an explicit strict-JSON reminder for weak local models that
# otherwise drift out of JSON mode.
_PROMPT_STRATEGIES: Dict[CapabilityTier, Dict[str, Any]] = {
    "small":  {"plan_max_tokens": 1_000, "reinforce_json": True},
    "medium": {"plan_max_tokens": 1_200, "reinforce_json": True},
    "large":  {"plan_max_tokens": 1_600, "reinforce_json": False},
    "xlarge": {"plan_max_tokens": 2_000, "reinforce_json": False},
}


def get_summary_budget(provider: Any) -> Dict[str, int]:
    """Return per-call prompt/response budgets scaled to the provider's model.

    Keys returned:
      * ``max_chars``     -- per-file summary char cap for prior-file context
      * ``max_files``     -- max number of prior files to include verbatim
      * ``output_tokens`` -- suggested ``max_tokens`` for the completion

    The tiers are coarse on purpose: any cloud-class model gets a generous
    budget, mid-size local models get a comfortable one, and tiny 8K-window
    models get a tight one so we never overflow.
    """
    return dict(_SUMMARY_BUDGETS[get_capability_tier(provider)])


def get_context_map_budget(provider: Any) -> Dict[str, int]:
    """Return a tiered SLM context budget scaled to the provider's model.

    Used by :func:`cgx.answer.context_map.build_tiered_context` to size the
    primary (full-window) and neighbor (stub) tiers without hard-coded
    magic numbers in the call sites.

    Keys returned:
      * ``primary_chars``  -- per-chunk char cap for primary (full-window) sources
      * ``neighbor_chars`` -- per-chunk char cap for neighbor stub sources
      * ``primary_max``    -- max number of primary chunks
      * ``neighbor_max``   -- max number of neighbor stubs
      * ``total_chars``    -- hard ceiling on the concatenated body text across tiers
    """
    return dict(_CONTEXT_MAP_BUDGETS[get_capability_tier(provider)])


def get_prompt_strategy(provider: Any) -> Dict[str, Any]:
    """Return tier-driven prompt-strategy knobs for the active provider.

    Keys returned:
      * ``tier``            -- the resolved :data:`CapabilityTier`
      * ``plan_max_tokens`` -- ``max_tokens`` for planning/decision calls
      * ``reinforce_json``  -- whether to append an explicit strict-JSON
        reminder to the prompt (on for weak local models)

    Callers select behaviour from the returned tier/flags instead of
    special-casing individual model ids.
    """
    tier = get_capability_tier(provider)
    strategy = dict(_PROMPT_STRATEGIES[tier])
    strategy["tier"] = tier
    return strategy
