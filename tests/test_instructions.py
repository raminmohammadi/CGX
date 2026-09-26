"""Tests for the user-instruction preamble builder (CGX.md + skills)."""

from __future__ import annotations

import pytest

from cgx import context_files as cf
from cgx.answer.instructions import build_instruction_preamble
from skills import loader as skill_loader


@pytest.fixture(autouse=True)
def _clear(monkeypatch, tmp_path):
    cf.clear_cache()
    # Isolate the global markdown-skills dir so detection is deterministic.
    d = tmp_path / "cfg_skills"
    d.mkdir()
    monkeypatch.setattr(skill_loader, "CUSTOM_SKILLS_DIR", d)
    monkeypatch.setattr(skill_loader, "_md_cache", {})
    yield
    cf.clear_cache()


def test_empty_when_nothing_present(tmp_path):
    assert build_instruction_preamble(str(tmp_path), "any question") == ""
    assert build_instruction_preamble(None, "q") == ""


def test_includes_cgx_md(tmp_path):
    (tmp_path / "CGX.md").write_text("Always cite file paths.", encoding="utf-8")
    out = build_instruction_preamble(str(tmp_path), "how does auth work")
    assert "Always cite file paths." in out
    assert "PROJECT INSTRUCTIONS" in out


def test_includes_detected_repo_skill(tmp_path):
    (tmp_path / ".cgx" / "skills" / "auth").mkdir(parents=True)
    (tmp_path / ".cgx" / "skills" / "auth" / "SKILL.md").write_text(
        "---\nname: authskill\ntriggers: [auth]\nsurfaces: [chat]\n---\n"
        "Auth uses JWT in this repo.", encoding="utf-8")
    out = build_instruction_preamble(str(tmp_path), "how does auth work")
    assert "Auth uses JWT in this repo." in out
    assert "PROJECT SKILLS" in out


def test_skill_not_injected_when_no_trigger_match(tmp_path):
    (tmp_path / ".cgx" / "skills" / "auth").mkdir(parents=True)
    (tmp_path / ".cgx" / "skills" / "auth" / "SKILL.md").write_text(
        "---\nname: authskill\ntriggers: [auth]\n---\nAuth uses JWT.",
        encoding="utf-8")
    out = build_instruction_preamble(str(tmp_path), "what is the database schema")
    assert "Auth uses JWT" not in out


def test_pinned_skill_overrides_detection(tmp_path):
    skill_loader.save_markdown_skill(
        "pinme", "---\nname: pinme\n---\nPinned guidance body.")
    # No trigger match, but explicit pin forces it in.
    out = build_instruction_preamble(str(tmp_path), "unrelated", pinned_skills=["pinme"])
    assert "Pinned guidance body." in out


def test_respects_max_chars_budget(tmp_path):
    (tmp_path / "CGX.md").write_text("A" * 5000, encoding="utf-8")
    out = build_instruction_preamble(str(tmp_path), "q", max_chars=500)
    assert len(out) <= 600  # block body capped near max_chars (+ header)
    assert "truncated" in out


def test_always_on_skill_injected_without_trigger(tmp_path):
    (tmp_path / ".cgx" / "skills" / "houserules").mkdir(parents=True)
    (tmp_path / ".cgx" / "skills" / "houserules" / "SKILL.md").write_text(
        "---\nname: houserules\nalways_on: true\n---\nNever use print().",
        encoding="utf-8")
    out = build_instruction_preamble(str(tmp_path), "literally anything")
    assert "Never use print()." in out
