

"""Model capability registry.

Different LLMs have wildly different context windows (Gemini 2.5 Flash
~1M tokens, Qwen2.5-Coder 3B ~32K, Llama 3 ~8K). Callers that build long
prompts — most notably the project scaffolder that embeds "already
generated" files as context for each new sibling file — need to know
how much room they actually have so they neither overflow small local
models nor waste capacity on large cloud ones.

This module exposes a small, deliberately conservative registry plus
two accessors:

* :func:`get_model_context_window` — token count for a given model id.
* :func:`get_summary_budget` — provider-aware ``max_chars`` /
  ``max_files`` / ``output_tokens`` triple to be used when building the
  "ALREADY GENERATED FILES" prompt block in
  :func:`cgx.answer.engine.generate_single_scaffold_file`.

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


def get_summary_budget(provider: Any) -> Dict[str, int]:
    """Return per-call prompt/response budgets scaled to the provider's model.

    Keys returned:
      * ``max_chars``     — per-file summary char cap for prior-file context
      * ``max_files``     — max number of prior files to include verbatim
      * ``output_tokens`` — suggested ``max_tokens`` for the completion

    The tiers are coarse on purpose: any cloud-class model gets a generous
    budget, mid-size local models get a comfortable one, and tiny 8K-window
    models get a tight one so we never overflow.
    """
    ctx = get_model_context_window(provider_model_name(provider))
    if ctx < 16_000:
        return {"max_chars": 400,  "max_files": 12,  "output_tokens": 2_000}
    if ctx < 64_000:
        return {"max_chars": 800,  "max_files": 30,  "output_tokens": 4_000}
    if ctx < 200_000:
        return {"max_chars": 1_500, "max_files": 60, "output_tokens": 6_000}
    return {"max_chars": 3_000, "max_files": 120, "output_tokens": 8_000}


def get_context_map_budget(provider: Any) -> Dict[str, int]:
    """Return a tiered SLM context budget scaled to the provider's model.

    Used by :func:`cgx.answer.context_map.build_tiered_context` to size the
    primary (full-window) and neighbor (stub) tiers without hard-coded
    magic numbers in the call sites.

    Keys returned:
      * ``primary_chars``  — per-chunk char cap for primary (full-window) sources
      * ``neighbor_chars`` — per-chunk char cap for neighbor stub sources
      * ``primary_max``    — max number of primary chunks
      * ``neighbor_max``   — max number of neighbor stubs
      * ``total_chars``    — hard ceiling on the concatenated body text across tiers
    """
    ctx = get_model_context_window(provider_model_name(provider))
    if ctx < 16_000:
        return {
            "primary_chars": 900,  "neighbor_chars": 220,
            "primary_max": 8,      "neighbor_max": 12,
            "total_chars": 6_000,
        }
    if ctx < 64_000:
        return {
            "primary_chars": 1_400, "neighbor_chars": 320,
            "primary_max": 12,      "neighbor_max": 24,
            "total_chars": 18_000,
        }
    if ctx < 200_000:
        return {
            "primary_chars": 2_200, "neighbor_chars": 420,
            "primary_max": 20,      "neighbor_max": 40,
            "total_chars": 48_000,
        }
    return {
        "primary_chars": 3_500, "neighbor_chars": 520,
        "primary_max": 32,      "neighbor_max": 60,
        "total_chars": 120_000,
    }
