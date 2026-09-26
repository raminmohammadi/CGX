"""Tests for JEV conditional instructions: per-directory GOTCHAS + tail append."""

from __future__ import annotations

from pathlib import Path

from cgx.answer.instructions import build_instruction_preamble
from cgx.context_files import clear_cache, load_directory_notes


def _repo(tmp_path: Path) -> Path:
    clear_cache()
    (tmp_path / "GOTCHAS.md").write_text("ROOT NOTE", encoding="utf-8")
    billing = tmp_path / "billing"
    billing.mkdir()
    (billing / "GOTCHAS.md").write_text("BILLING NOTE", encoding="utf-8")
    (tmp_path / "web").mkdir()  # no GOTCHAS here
    return tmp_path


def test_directory_notes_nearest_first(tmp_path):
    root = _repo(tmp_path)
    out = load_directory_notes(str(root), ["billing/invoice.py"])
    assert "BILLING NOTE" in out and "ROOT NOTE" in out
    # nearest directory's notes come before the root's
    assert out.index("billing/GOTCHAS.md") < out.index("./GOTCHAS.md")


def test_directory_notes_only_touched_dirs(tmp_path):
    root = _repo(tmp_path)
    out = load_directory_notes(str(root), ["web/app.py"])
    # web/ has no GOTCHAS; only the ancestor root note loads, not billing's
    assert "BILLING NOTE" not in out
    assert "ROOT NOTE" in out


def test_directory_notes_empty_when_no_files(tmp_path):
    root = _repo(tmp_path)
    assert load_directory_notes(str(root), []) == ""
    assert load_directory_notes(str(root), None) == ""
    assert load_directory_notes(None, ["billing/x.py"]) == ""


def test_directory_notes_ignores_paths_outside_repo(tmp_path):
    root = _repo(tmp_path)
    # A path that climbs out of the repo must not load anything spurious.
    out = load_directory_notes(str(root), ["../../etc/passwd"])
    assert out == "" or "ROOT NOTE" not in out


def test_preamble_appends_notes_at_tail(tmp_path):
    root = _repo(tmp_path)
    (root / "CGX.md").write_text("Always use tabs.", encoding="utf-8")
    clear_cache()
    with_notes = build_instruction_preamble(
        str(root), "how does billing work?", files_touched=["billing/invoice.py"])
    assert "Always use tabs." in with_notes         # CGX.md present
    assert "DIRECTORY NOTES" in with_notes           # notes present
    assert "BILLING NOTE" in with_notes
    # CGX.md prefix comes before the tail-appended directory notes
    assert with_notes.index("Always use tabs.") < with_notes.index("DIRECTORY NOTES")


def test_preamble_without_files_touched_has_no_notes(tmp_path):
    root = _repo(tmp_path)
    (root / "CGX.md").write_text("Always use tabs.", encoding="utf-8")
    clear_cache()
    out = build_instruction_preamble(str(root), "how does billing work?")
    assert "DIRECTORY NOTES" not in out


def test_skill_detect_context_default_zero():
    from skills.base import Skill

    class _S(Skill):
        name = "t"

        def detect(self, goal: str) -> float:
            return 0.0

    assert _S().detect_context({"files": ["a.css"]}) == 0.0
