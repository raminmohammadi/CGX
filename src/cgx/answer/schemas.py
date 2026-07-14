"""JSON schemas for structured LLM outputs + provider translation helpers.

CGX already forces JSON mode (`force_json`) and defensively re-extracts the
first balanced-brace object from whatever the model returns. On weak local
models that is not always enough: the reply can be valid JSON of the *wrong*
shape (missing ``tasks``, an object instead of an array, prose smuggled into a
field). Schema-constrained decoding closes that gap by handing the provider a
machine-checkable contract so the sampler can only emit conforming tokens.

Each provider expresses constrained decoding differently, so this module keeps
the canonical JSON-Schema definitions in one place plus small, pure translation
helpers that map a schema onto each backend's native request shape:

* Ollama                -- ``format`` accepts the JSON schema object verbatim.
* OpenAI-compatible     -- ``response_format={"type":"json_schema", ...}``.
* Gemini                -- ``generationConfig.responseSchema`` (OpenAPI subset).

The helpers are deliberately side-effect free and dependency-free so they can
be unit-tested without a live provider, and every caller keeps its existing
``force_json`` + balanced-brace fallback so an unsupported backend degrades
gracefully instead of failing.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Canonical schemas (draft-07 subset shared by all providers)
# ---------------------------------------------------------------------------

# Planner decomposition. Mirrors planner.SYSTEM_PROMPT exactly: a single object
# with an optional prose ``rationale`` and a non-empty ``tasks`` array whose
# ``kind`` is drawn from the six planner-visible capabilities.
PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "ask", "plan", "scaffold",
                            "search", "summarize", "verify",
                        ],
                    },
                    "criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description", "kind"],
            },
        },
    },
    "required": ["tasks"],
}

# Judge verdict. A closed vocabulary verdict, a bounded confidence and a short
# rationale -- the shape Judge._llm_judge parses.
JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict"],
}


# ---------------------------------------------------------------------------
# Provider translation helpers (pure)
# ---------------------------------------------------------------------------
def to_openai_response_format(
    schema: Dict[str, Any], name: str = "cgx_response",
) -> Dict[str, Any]:
    """Wrap a JSON schema as an OpenAI ``response_format`` block.

    ``strict`` is left False: strict mode additionally requires
    ``additionalProperties:false`` and every property to be ``required`` on
    every object, which many OpenAI-compatible servers (llama.cpp, vLLM,
    LM Studio) do not implement. Non-strict json_schema is the widely
    supported middle ground; callers still fall back to ``json_object`` and
    then plain text on rejection.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": False},
    }


# JSON-Schema keys Gemini's OpenAPI-subset ``Schema`` understands. Anything
# else (``additionalProperties``, ``$schema``, ``title`` ...) triggers a 400,
# so we drop unknown keys rather than forward them.
_GEMINI_KEYS = {
    "type", "description", "enum", "items", "properties", "required", "nullable",
}


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON-Schema (subset) to Gemini's ``responseSchema`` shape.

    Recursively keeps only Gemini-supported keys and upper-cases ``type``
    tokens (Gemini expects ``OBJECT`` / ``ARRAY`` / ``STRING`` ...). Unknown
    keys are dropped so an over-specified schema still yields a valid request.
    """
    if not isinstance(schema, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, val in schema.items():
        if key not in _GEMINI_KEYS:
            continue
        if key == "type" and isinstance(val, str):
            out[key] = val.upper()
        elif key == "properties" and isinstance(val, dict):
            out[key] = {k: to_gemini_schema(v) for k, v in val.items()}
        elif key == "items" and isinstance(val, dict):
            out[key] = to_gemini_schema(val)
        else:
            out[key] = val
    return out
