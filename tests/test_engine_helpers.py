"""Tests for engine.py pure-Python helpers (no LLM / embedder)."""

from cgx.answer.engine import (
    SYSTEM_PROMPTS,
    _extract_json_object,
    _get_system_prompt,
    _parse_plan_freeform,
    _window_text,
)


def test_window_text_centers_on_focus():
    text = "\n".join(f"line {i}" for i in range(60))
    out = _window_text(text, ["line 30"], max_chars=200, context_lines=3)
    assert "line 30" in out
    assert "line 0" not in out  # window should not start at the top
    assert "line 59" not in out


def test_window_text_falls_back_when_no_match():
    text = "abc\ndef\nghi"
    out = _window_text(text, ["nope"], max_chars=100)
    assert out == text


def test_extract_json_object_balanced():
    text = 'prose {\n"a": 1, "b": "}"\n} trailing'
    obj = _extract_json_object(text)
    assert obj == {"a": 1, "b": "}"}


def test_extract_json_object_empty_on_garbage():
    assert _extract_json_object("not json at all") == {}


def test_get_system_prompt_known_and_fallback():
    for mode in SYSTEM_PROMPTS:
        assert isinstance(_get_system_prompt(mode), str)
    # unknown mode falls back to the default SYSTEM string
    default = _get_system_prompt("definitely-not-a-mode")
    assert "senior codebase assistant" in default.lower()


def test_parse_plan_freeform_extracts_diffs():
    text = (
        "## Plan\n"
        "Add a hello function.\n\n"
        "## Diffs\n"
        "```diff path=src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@\n"
        "+def hello():\n"
        "+    return 1\n"
        "```\n"
        "Cite as [[src/foo.py::function::hello]]"
    )
    parsed = _parse_plan_freeform(text)
    assert parsed["diffs"] and parsed["diffs"][0]["file"] == "src/foo.py"
    assert "def hello" in parsed["diffs"][0]["patch"]
    assert parsed["citations"] and parsed["citations"][0]["chunk_id"].startswith("src/foo.py")
