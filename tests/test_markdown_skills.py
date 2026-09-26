"""Tests for the markdown-skill format: frontmatter parsing, the
``MarkdownSkill`` adapter, the loader CRUD/discovery, and registry
integration (detection, description, composition)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import skills as registry
from skills import loader as skill_loader
from skills import markdown_skill as md
from skills.frontmatter import parse_frontmatter


# --------------------------------------------------------------------------
# frontmatter parser
# --------------------------------------------------------------------------

def test_parse_frontmatter_scalars_and_lists():
    text = textwrap.dedent("""\
        ---
        name: my-skill
        description: One line.
        triggers: [foo, "bar baz", qux]
        roles:
          - backend
          - data
        always_on: true
        priority: 5
        ---
        Body text here.
        Second line.
        """)
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "my-skill"
    assert meta["description"] == "One line."
    assert meta["triggers"] == ["foo", "bar baz", "qux"]
    assert meta["roles"] == ["backend", "data"]
    assert meta["always_on"] is True
    assert meta["priority"] == 5
    assert body == "Body text here.\nSecond line."


def test_parse_frontmatter_none_when_no_fence():
    meta, body = parse_frontmatter("# Just a heading\nsome text")
    assert meta == {}
    assert body == "# Just a heading\nsome text"


def test_parse_frontmatter_unterminated_fence_is_body():
    meta, body = parse_frontmatter("---\nname: x\nno closing fence")
    assert meta == {}
    assert "no closing fence" in body


# --------------------------------------------------------------------------
# MarkdownSkill detection + surface gating
# --------------------------------------------------------------------------

def _skill(content: str, name_hint: str = "demo") -> md.MarkdownSkill:
    return md.from_source(content, name_hint=name_hint)


def test_always_on_detects_everything_including_empty_goal():
    s = _skill("---\nalways_on: true\n---\nAlways.")
    assert s.detect("anything at all") == 1.0
    assert s.detect("") == 1.0


def test_trigger_keyword_word_boundary():
    s = _skill("---\ntriggers: [eob]\n---\nEOB rules.")
    assert s.detect("please validate the EOB document") == pytest.approx(0.9)
    # word-boundaried: 'eob' must not fire inside 'neobank'
    assert s.detect("this is a neobank feature") == 0.0


def test_trigger_regex_matches():
    # On disk the frontmatter reads:  trigger_regex: "\bRx\b"
    # The parser preserves backslashes literally (no YAML escape handling),
    # which is exactly what regex authoring needs.
    s = _skill('---\ntrigger_regex: "\\bRx\\b"\n---\nRx handling.')
    assert s.detect("handle the Rx receipt") == pytest.approx(0.9)
    assert s.detect("unrelated") == 0.0


def test_no_triggers_is_pin_only():
    s = _skill("---\ndescription: x\n---\nBody.")
    assert s.detect("body mentions nothing") == 0.0


def test_surface_gating_defaults_to_chat_only():
    s = _skill("---\ntriggers: [x]\n---\nHouse rules.")
    assert s.ask_system_prompt() == "House rules."
    assert s.scaffold_system_prompt() == ""
    assert s.plan_system_prompt() == ""


def test_surface_all_targets_every_surface():
    s = _skill("---\nsurfaces: [all]\ntriggers: [x]\n---\nRule.")
    assert s.ask_system_prompt() == "Rule."
    assert s.scaffold_system_prompt() == "Rule."
    assert s.plan_system_prompt() == "Rule."


def test_surface_plan_only():
    s = _skill("---\nsurfaces: [plan]\ntriggers: [x]\n---\nPlan rule.")
    assert s.ask_system_prompt() == ""
    assert s.plan_system_prompt() == "Plan rule."


# --------------------------------------------------------------------------
# check_source validation
# --------------------------------------------------------------------------

def test_check_source_valid():
    ok, kind, _detail, meta = md.check_source(
        "---\nname: good-skill\ndescription: d\ntriggers: [a]\n---\nBody.",
        name_hint="good-skill")
    assert ok is True
    assert kind == ""
    assert meta["name"] == "good-skill"
    assert "body" not in meta  # body stripped from returned meta


def test_check_source_empty_body():
    ok, kind, _d, _m = md.check_source("---\nname: x\n---\n   ", name_hint="x")
    assert not ok and kind == "empty_body"


def test_check_source_invalid_name():
    ok, kind, _d, _m = md.check_source("---\nname: bad/name\n---\nBody",
                                       name_hint="bad/name")
    assert not ok and kind == "invalid_name"


def test_check_source_missing_name():
    ok, kind, _d, _m = md.check_source("---\ndescription: d\n---\nBody",
                                       name_hint="")
    assert not ok and kind == "invalid_name"


def test_check_source_invalid_surface():
    ok, kind, _d, _m = md.check_source(
        "---\nname: x\nsurfaces: [bogus]\n---\nBody", name_hint="x")
    assert not ok and kind == "invalid_field"


def test_check_source_invalid_role():
    ok, kind, _d, _m = md.check_source(
        "---\nname: x\nroles: [wizard]\n---\nBody", name_hint="x")
    assert not ok and kind == "invalid_field"


def test_check_source_bad_regex():
    ok, kind, _d, _m = md.check_source(
        "---\nname: x\ntrigger_regex: \"([\"\n---\nBody", name_hint="x")
    assert not ok and kind == "invalid_field"


def test_check_source_too_large():
    big = "---\nname: x\n---\n" + ("a" * (md.MAX_SKILL_CHARS + 1))
    ok, kind, _d, _m = md.check_source(big, name_hint="x")
    assert not ok and kind == "too_large"


def test_check_source_name_collision():
    ok, kind, _d, _m = md.check_source(
        "---\nname: react\n---\nBody", name_hint="react",
        known_names={"react"})
    assert not ok and kind == "name_collision"


# --------------------------------------------------------------------------
# loader CRUD + discovery
# --------------------------------------------------------------------------

@pytest.fixture()
def md_skills_dir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr(skill_loader, "CUSTOM_SKILLS_DIR", d)
    monkeypatch.setattr(skill_loader, "_md_cache", {})
    return d


def test_save_read_delete_roundtrip(md_skills_dir):
    content = "---\nname: house\ntriggers: [style]\n---\nUse tabs."
    skill_loader.save_markdown_skill("house", content)
    assert (md_skills_dir / "house" / "SKILL.md").is_file()
    assert skill_loader.read_markdown_skill_source("house") == content
    assert "house" in skill_loader.list_markdown_skill_names()

    loaded = skill_loader.load_markdown_skills(force=True)
    assert [s.name for s in loaded] == ["house"]
    assert loaded[0].detect("what is our style") == pytest.approx(0.9)

    assert skill_loader.delete_markdown_skill("house") is True
    assert skill_loader.read_markdown_skill_source("house") is None
    assert skill_loader.delete_markdown_skill("house") is False


def test_flat_form_is_discovered(md_skills_dir):
    (md_skills_dir / "quick.md").write_text(
        "---\ntriggers: [quick]\n---\nQuick body.", encoding="utf-8")
    loaded = skill_loader.load_markdown_skills(force=True)
    names = [s.name for s in loaded]
    assert "quick" in names


def test_save_migrates_flat_to_dir_form(md_skills_dir):
    (md_skills_dir / "mig.md").write_text("---\nname: mig\n---\nOld.",
                                          encoding="utf-8")
    skill_loader.save_markdown_skill("mig", "---\nname: mig\n---\nNew.")
    assert not (md_skills_dir / "mig.md").exists()
    assert (md_skills_dir / "mig" / "SKILL.md").is_file()


def test_repo_markdown_skills_loaded_from_project_root(tmp_path):
    repo = tmp_path / "proj"
    (repo / ".cgx" / "skills" / "domain").mkdir(parents=True)
    (repo / ".cgx" / "skills" / "domain" / "SKILL.md").write_text(
        "---\nname: domain\ntriggers: [invoice]\n---\nInvoice rules.",
        encoding="utf-8")
    loaded = skill_loader.load_repo_markdown_skills(str(repo))
    assert [s.name for s in loaded] == ["domain"]
    assert loaded[0].scope == "repo"


def test_load_is_mtime_cached(md_skills_dir):
    skill_loader.save_markdown_skill("c", "---\nname: c\n---\nOne.")
    first = skill_loader.load_markdown_skills()
    again = skill_loader.load_markdown_skills()
    assert [s.name for s in first] == [s.name for s in again]


def test_bad_skill_file_is_skipped_not_raised(md_skills_dir):
    # A directory with no SKILL.md is simply ignored; a truly broken read
    # path must not raise.
    (md_skills_dir / "empty_dir").mkdir()
    loaded = skill_loader.load_markdown_skills(force=True)
    assert loaded == []


# --------------------------------------------------------------------------
# registry integration
# --------------------------------------------------------------------------

def test_registry_detects_and_composes_markdown_skill(md_skills_dir):
    skill_loader.save_markdown_skill(
        "biz", "---\nname: biz\ntriggers: [refund]\nsurfaces: [chat]\n---\n"
               "Refunds must cite a receipt.")
    detected = registry.detect_skills("how do I process a refund")
    names = [s.name for s in detected]
    assert "biz" in names

    active = registry.skills_by_names(["biz"])
    composed = registry.compose_ask_prompt(active)
    assert "Refunds must cite a receipt." in composed
    assert "### biz" in composed
    # markdown knowledge skills do NOT leak into the scaffold surface
    assert registry.compose_scaffold_prompt(active) == ""


def test_describe_skills_marks_markdown_format(md_skills_dir):
    skill_loader.save_markdown_skill("md1", "---\nname: md1\n---\nBody.")
    described = {d["name"]: d for d in registry.describe_skills()}
    assert described["md1"]["format"] == "markdown"
    assert described["md1"]["is_custom"] is True
    # a built-in stays classified as builtin
    assert described["react"]["format"] == "builtin"


def test_known_skill_names_includes_markdown(md_skills_dir):
    skill_loader.save_markdown_skill("uniq", "---\nname: uniq\n---\nBody.")
    assert "uniq" in registry.known_skill_names()


def test_describe_skills_includes_repo_scope(tmp_path, md_skills_dir):
    repo = tmp_path / "proj"
    (repo / ".cgx" / "skills" / "rp").mkdir(parents=True)
    (repo / ".cgx" / "skills" / "rp" / "SKILL.md").write_text(
        "---\nname: rp\nalways_on: true\n---\nRepo rule.", encoding="utf-8")
    described = {d["name"]: d for d in registry.describe_skills(str(repo))}
    assert described["rp"]["scope"] == "repo"
    assert described["rp"]["always_on"] is True
