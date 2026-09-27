"""Tests for markdown-skill declarative validation + detect_context (Phase 4).

Markdown skills were prompt-only: they could steer generation but never CATCH a
bad diff. These cover the new declarative frontmatter (require/forbid files +
patch regexes, gated by validate_surfaces) and condition-bound activation
(context_globs / context_exts -> detect_context)."""

from __future__ import annotations

from skills.markdown_skill import check_source, from_source


def _md(front: str, body: str = "Do the thing.") -> str:
    return f"---\n{front}\n---\n{body}"


def test_forbid_files_is_fatal_on_scaffold():
    s = from_source(_md("name: no-pkg\nsurfaces: [scaffold]\n"
                        "forbid_files: [package.json]"))
    v = s.validate_scaffold([{"path": "frontend/package.json", "content": "{}"}])
    assert v is not None and not v.passed and "package.json" in v.rationale
    # a clean diff passes
    assert s.validate_scaffold([{"path": "index.html", "content": "<html>"}]) is None


def test_require_files_on_scaffold():
    s = from_source(_md("name: needs-readme\nsurfaces: [scaffold]\n"
                        "require_files: [README.md]"))
    assert s.validate_scaffold([{"path": "app.py", "content": "x"}]) is not None
    assert s.validate_scaffold([{"path": "README.md", "content": "x"}]) is None


def test_patch_regexes():
    forb = from_source(_md('name: no-secret\nsurfaces: [scaffold]\n'
                           'forbid_patch_regex: ["password ="]'))
    assert forb.validate_scaffold(
        [{"path": "a.py", "content": "password = 'x'"}]) is not None
    assert forb.validate_scaffold(
        [{"path": "a.py", "content": "print(1)"}]) is None
    req = from_source(_md('name: needs-main\nsurfaces: [scaffold]\n'
                          'require_patch_regex: ["def main"]'))
    assert req.validate_scaffold([{"path": "a.py", "content": "x=1"}]) is not None
    assert req.validate_scaffold(
        [{"path": "a.py", "content": "def main():\n    pass"}]) is None


def test_validate_surfaces_gates_off_chat_only():
    # A chat-only skill must never validate a codegen diff.
    s = from_source(_md("name: chatty\nsurfaces: [chat]\nforbid_files: [x.py]"))
    assert s.validate_scaffold([{"path": "x.py", "content": "1"}]) is None
    assert s.validate_plan([{"path": "x.py"}]) is None


def test_validate_plan_forbids_only_never_requires():
    s = from_source(_md("name: no-pkg\nsurfaces: [plan]\n"
                        "forbid_files: [package.json]\nrequire_files: [README.md]"))
    # forbid_files fires at plan time (paths-only, safe direction)
    assert s.validate_plan([{"path": "package.json"}]) is not None
    # require_files must NOT fire on a small incremental edit
    assert s.validate_plan([{"path": "app.py"}]) is None


def test_detect_context_globs_and_exts():
    s = from_source(_md("name: css-style\ncontext_globs: [*.css]\n"
                        "context_exts: [scss]"))
    assert s.detect_context({"files": ["frontend/style.css"]}) > 0
    assert s.detect_context({"files_touched": ["a/b.scss"]}) > 0
    assert s.detect_context({"files": ["app.py"]}) == 0.0
    # a skill with no context config never activates by context
    plain = from_source(_md("name: plain\ntriggers: [foo]"))
    assert plain.detect_context({"files": ["a.css"]}) == 0.0


def test_check_source_rejects_bad_patch_regex():
    ok, kind, detail, _ = check_source(
        _md('name: bad\nforbid_patch_regex: ["("]'))
    assert not ok and kind == "invalid_field"


def test_from_source_parses_new_fields():
    s = from_source(_md("name: full\nsurfaces: [scaffold]\n"
                        "forbid_files: [secrets.py]\ncontext_exts: [.tf]"))
    assert s.forbid_files == ["secrets.py"]
    assert s.context_exts == (".tf",)
