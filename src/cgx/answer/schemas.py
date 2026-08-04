"""JSON schemas for structured LLM outputs + provider translation helpers.

CGX already forces JSON mode (`force_json`) and defensively re-extracts the
first balanced-brace object from whatever the model returns. On weak local
models that is not always enough: the reply can be valid JSON of the *wrong*
shape (missing ``layers``, an object instead of an array, prose smuggled into
a field). Schema-constrained decoding closes that gap by handing the provider
a machine-checkable contract so the sampler can only emit conforming tokens,
and :func:`validate_json_schema` re-checks the parsed reply at the executor
boundary so a backend that silently ignored the schema still gets caught with
an actionable violation list (folded into one bounded re-ask).

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

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Canonical schemas (draft-07 subset shared by all providers)
# ---------------------------------------------------------------------------

# DECOMPOSE manifest. Mirrors engine._MANIFEST_SYSTEM: a prose ``plan_md``,
# optional shared ``contracts`` (left an open object -- its four categories
# hold heterogeneous nested shapes and are normalised leniently downstream),
# and a non-empty ``layers`` array of ``{name, files:[{path, description,
# depends_on}]}``.
MANIFEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_md": {"type": "string"},
        "contracts": {"type": "object"},
        "layers": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "description": {"type": "string"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": ["files"],
            },
        },
    },
    "required": ["layers"],
}

# CLARIFY_REQUIREMENTS questions. Mirrors the executor's _SYSTEM_PROMPT:
# 3-6 short questions, each with a required ``prompt`` plus optional
# ``id`` / ``hint`` / ``suggested`` chips.
CLARIFY_QUESTIONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "hint": {"type": "string"},
                    "suggested": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    "required": ["questions"],
}

# CLARIFY_PATHS schema for answer engine clarify_paths mode.
CLARIFY_PATHS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "restatement": {"type": "string"},
        "options": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["title", "rationale", "chunk_id"],
            },
        },
        "follow_up_question": {"type": "string"},
    },
    "required": ["restatement", "options", "follow_up_question"],
}

# REPAIR whole-file rewrites. Mirrors engine._LOGIC_REPAIR_SYSTEM: the model
# returns complete corrected files as ``{files:[{path, content}]}`` (an empty
# array is its explicit "no fix" signal).
REPAIR_FILES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "required": ["files"],
}


# ---------------------------------------------------------------------------
# Boundary validation (pure)
# ---------------------------------------------------------------------------
def validate_json_schema(
    obj: Any, schema: Dict[str, Any], path: str = "$",
) -> List[str]:
    """Validate ``obj`` against the draft-07 subset this module emits.

    Constrained decoding is best-effort -- a backend may silently ignore the
    schema (older Ollama, ``json_object``-only servers), so executors re-check
    the parsed reply here and fold the returned violations into a bounded
    re-ask. Supports exactly the keys the schemas above use (``type``,
    ``properties``, ``required``, ``items``, ``minItems``, ``enum``); each
    violation is a human-readable ``"$.layers[0].files: ..."`` string the
    model can act on. Empty list means conforming.
    """
    errs: List[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            return [f"{path}: expected an object, got {type(obj).__name__}"]
        for key in schema.get("required") or []:
            if key not in obj:
                errs.append(f"{path}.{key}: required key is missing")
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj and isinstance(sub, dict):
                errs.extend(validate_json_schema(obj[key], sub, f"{path}.{key}"))
    elif t == "array":
        if not isinstance(obj, list):
            return [f"{path}: expected an array, got {type(obj).__name__}"]
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(obj) < min_items:
            errs.append(f"{path}: expected at least {min_items} item(s), "
                        f"got {len(obj)}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, val in enumerate(obj):
                errs.extend(validate_json_schema(val, items, f"{path}[{i}]"))
    elif t == "string":
        if not isinstance(obj, str):
            errs.append(f"{path}: expected a string, got {type(obj).__name__}")
    elif t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            errs.append(f"{path}: expected a number, got {type(obj).__name__}")
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            errs.append(f"{path}: expected an integer, got {type(obj).__name__}")
    elif t == "boolean":
        if not isinstance(obj, bool):
            errs.append(f"{path}: expected a boolean, got {type(obj).__name__}")
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and obj not in enum:
        errs.append(f"{path}: expected one of {enum!r}")
    return errs


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
    if out.get("type") == "OBJECT" and not out.get("properties"):
        # Gemini API rejects OBJECT schemas that lack properties.
        # Inject optional known schema properties (e.g., for contracts)
        # plus a nullable fallback so OpenAPI schema validation succeeds.
        out["properties"] = {
            "endpoints": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "path": {"type": "STRING"},
                        "method": {"type": "STRING"},
                        "description": {"type": "STRING"},
                    },
                },
            },
            "schemas": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"name": {"type": "STRING"}},
                },
            },
            "functions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "signature": {"type": "STRING"},
                    },
                },
            },
            "constants": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"name": {"type": "STRING"}},
                },
            },
            "_extra": {"type": "STRING", "nullable": True},
        }
    return out
