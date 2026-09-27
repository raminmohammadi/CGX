"""Layer 2: the learning loop -- store, injection, and post-mortem extraction.

These lock in the self-improving behaviour the user asked for: a failure is
distilled into a durable, generalised lesson, auto-appended to the right scope
(project vs skill), fed back into future generation, and fully reversible.
"""

from __future__ import annotations

import json

from cgx.learning import lessons as L
from cgx.learning.postmortem import run_postmortem


# --------------------------------------------------------------------------- #
# Store: record / read / render / prune (project scope, hermetic in tmp)
# --------------------------------------------------------------------------- #
def test_project_lesson_lifecycle(tmp_path):
    root = str(tmp_path)
    lid = L.record_lesson(
        "project", "Bind SQLite to an absolute path, not a relative one.",
        root=root, root_cause="relative sqlite URI", session_id="s1",
        task_id="t1", now=1_000.0)
    assert lid and lid.startswith("lsn_")
    # JSONL ledger + rendered md both exist and carry the lesson.
    ledger = tmp_path / ".cgx" / "lessons.jsonl"
    md = tmp_path / ".cgx" / "lessons.md"
    assert ledger.exists() and md.exists()
    entries = L.read_lessons("project", root=root)
    assert len(entries) == 1 and entries[0]["id"] == lid
    assert entries[0]["session_id"] == "s1"          # auditable
    assert "absolute path" in L.project_lessons_text(root)
    assert "PROJECT MEMORY" in L.project_lessons_text(root)
    # Reversible: prune removes it and re-renders empty.
    assert L.prune_lesson(lid, scope="project", root=root) is True
    assert L.read_lessons("project", root=root) == []
    assert L.project_lessons_text(root) == ""


def test_duplicate_lesson_is_refreshed_not_appended(tmp_path):
    root = str(tmp_path)
    L.record_lesson("project", "Do X.", root=root, now=1.0)
    L.record_lesson("project", "  do   x.  ", root=root, now=2.0)  # same, normalised
    entries = L.read_lessons("project", root=root)
    assert len(entries) == 1
    assert entries[0]["created_at"] == 2.0            # refreshed timestamp


def test_store_is_bounded(tmp_path):
    root = str(tmp_path)
    for i in range(L._MAX_LESSONS + 10):
        L.record_lesson("project", f"lesson number {i}", root=root, now=float(i))
    entries = L.read_lessons("project", root=root)
    assert len(entries) == L._MAX_LESSONS
    # Newest win.
    assert entries[-1]["lesson"] == f"lesson number {L._MAX_LESSONS + 9}"


def test_project_lesson_needs_root():
    assert L.record_lesson("project", "no root given", root=None) is None


# --------------------------------------------------------------------------- #
# Skill scope + feedback injection into the scaffold prompt
# --------------------------------------------------------------------------- #
def test_skill_lesson_injected_into_scaffold_prompt(tmp_path, monkeypatch):
    # Isolate the skills dir so the test never writes into the real repo skills/.
    monkeypatch.setattr(L, "_SKILLS_DIR", str(tmp_path / "skills"))
    L.record_lesson(
        "skill",
        "A Flask blueprint route must not repeat the url_prefix it is "
        "registered under.",
        skill="flask", root_cause="double prefix", now=1.0)
    assert "url_prefix" in L.skill_lessons_text("flask")

    import skills as _sk
    from skills.flask import FlaskSkill
    prompt = _sk.compose_scaffold_prompt([FlaskSkill()])
    assert "LESSONS FROM PAST RUNS" in prompt
    assert "must not repeat the url_prefix" in prompt
    # A skill with no lessons contributes no lesson block.
    monkeypatch.setattr(L, "_SKILLS_DIR", str(tmp_path / "empty"))
    assert "LESSONS FROM PAST RUNS" not in _sk.compose_scaffold_prompt(
        [FlaskSkill()])


# --------------------------------------------------------------------------- #
# Post-mortem extraction (the model-in-the-loop "learn" step)
# --------------------------------------------------------------------------- #
class _FakeProvider:
    """Returns a fixed lessons JSON for jev.decide's provider.chat contract."""

    supports_logprobs = False

    def __init__(self, payload):
        self._payload = payload

    def chat(self, **kwargs):
        return {"content": json.dumps(self._payload)}


def test_postmortem_records_skill_and_project_lessons(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("CGX_SWARM_LEARNING", "on")
    provider = _FakeProvider({"lessons": [
        {"scope": "skill", "skill": "flask",
         "lesson": "Every ORM/raw-SQL table needs a matching db.Model.",
         "root_cause": "missing model", "generalizable": True},
        {"scope": "project",
         "lesson": "This app stores orders as a JSON column.",
         "generalizable": True},
        {"scope": "skill", "skill": "flask",
         "lesson": "one-off, ignore me", "generalizable": False},  # dropped
    ]})
    recorded = run_postmortem(
        provider, goal="coffee shop", skills=["flask"], root=str(tmp_path),
        incidents=[{"phase": "structural", "fixed": True,
                    "detail": "Order model referenced but undefined"}],
        session_id="s", task_id="t")
    scopes = sorted((r["scope"], r.get("skill", "")) for r in recorded)
    assert ("project", "") in scopes and ("skill", "flask") in scopes
    assert len(recorded) == 2                          # generalizable=False dropped
    assert "db.Model" in L.skill_lessons_text("flask")
    assert "JSON column" in L.project_lessons_text(str(tmp_path))


def test_postmortem_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CGX_SWARM_LEARNING", "off")
    provider = _FakeProvider({"lessons": [
        {"scope": "project", "lesson": "should not be recorded"}]})
    assert run_postmortem(provider, goal="g", skills=[], root=str(tmp_path),
                          incidents=[{"detail": "x"}]) == []


def test_postmortem_no_incidents_is_noop(tmp_path):
    provider = _FakeProvider({"lessons": [{"scope": "project", "lesson": "x"}]})
    assert run_postmortem(provider, goal="g", skills=[], root=str(tmp_path),
                          incidents=[]) == []


def test_postmortem_unknown_skill_demoted_to_project(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("CGX_SWARM_LEARNING", "on")
    # Model claims a 'skill' lesson but names no active/known skill -> project.
    provider = _FakeProvider({"lessons": [
        {"scope": "skill", "skill": "nonsense_stack",
         "lesson": "A rule that cannot be attributed to a real skill."}]})
    recorded = run_postmortem(
        provider, goal="g", skills=["flask", "react"], root=str(tmp_path),
        incidents=[{"detail": "x"}])
    assert recorded and recorded[0]["scope"] == "project"
