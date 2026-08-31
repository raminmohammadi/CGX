"""Tests for the Ollama installed-model readiness helper (`_closest_installed`).

Regression guard for the tag-mismatch confusion: a requested tag that differs
only in case/naming from an installed one (e.g. a manually imported GGUF) must
be suggested, while exact matching stays the contract the ping enforces.
"""

from cgx.webui.routes.setup import _closest_installed


INSTALLED = {
    "Qwen2.5-Coder-14B-Instruct-GGUF:latest",
    "glm-ocr:latest",
    "qwen3-vl:8b",
}


def test_suggests_case_and_name_mismatch():
    # The exact reported scenario: profile has the registry-style tag, install
    # is a CamelCase GGUF. We should suggest the installed one.
    assert _closest_installed("qwen2.5-coder:14b-instruct", INSTALLED) == \
        "Qwen2.5-Coder-14B-Instruct-GGUF:latest"


def test_case_insensitive_exact_match_wins():
    assert _closest_installed("QWEN3-VL:8B", INSTALLED) == "qwen3-vl:8b"


def test_no_plausible_match_returns_none():
    assert _closest_installed("llama3.1:8b", INSTALLED) is None


def test_prefers_shortest_family_candidate():
    installed = {"qwen2.5-coder:3b", "qwen2.5-coder:14b-instruct"}
    # Both share the family root; the shortest (least specific) is suggested.
    assert _closest_installed("qwen2.5-coder:xyz", installed) == \
        "qwen2.5-coder:3b"


def test_empty_installed_returns_none():
    assert _closest_installed("anything:1b", set()) is None
