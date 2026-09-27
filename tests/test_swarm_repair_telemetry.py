"""Tests for truthful semantic-repair telemetry + re-gating the repaired body.

Before: a `semantic_repair` beat carried the *triggering* error (so the live
feed showed FAILED even when the repair then succeeded), and a repair was
written with ok=True WITHOUT re-checking the gate it was meant to fix. These
lock in: the beat reports its real outcome (ok + triggered_by), and a repair
that doesn't re-pass the gate is a truthful failure that VERIFY regenerates."""

from __future__ import annotations

import cgx.session.tasks.swarm_generate as sg


def _gen(tmp_path, **over):
    kw = dict(path="pkg/mod.py", description="d", depends_on=[], contracts={},
              goal="g", root=str(tmp_path), provider=object(),
              layer="pkg/mod.py", manifest_paths=["pkg/mod.py"],
              log_root=str(tmp_path))
    kw.update(over)
    return sg.generate_file(**kw)


def test_regate_repaired_python_syntax(tmp_path):
    assert sg._regate_repaired(
        "a.py", "def f():\n    return 1\n", {}, None, str(tmp_path)) is None
    err = sg._regate_repaired("a.py", "def f(:\n", {}, None, str(tmp_path))
    assert err and "syntax" in err.lower()
    # non-Python bodies aren't gated here (VERIFY's polyglot build is authority)
    assert sg._regate_repaired(
        "a.js", "const x = ;", {}, None, str(tmp_path)) is None


def test_semantic_repair_success_is_reported_truthfully(tmp_path, monkeypatch):
    beats = []
    monkeypatch.setattr(sg, "swarm_beat",
                        lambda *a, **k: beats.append((a[2], k)))
    # both full-file attempts fail a (non-contract) gate -> semantic_repair path
    monkeypatch.setattr(sg, "_full_file_attempt",
                        lambda *a, **k: ("def f(:", "syntax gate failed"))
    # the repair produces valid, gate-passing content
    monkeypatch.setattr(sg, "_semantic_repair_fallback",
                        lambda *a, **k: ("def f():\n    return 1\n", ""))
    out = _gen(tmp_path)
    assert out.ok and out.method == "semantic-repair"
    sr = [k for ph, k in beats if ph == "semantic_repair"]
    assert sr, "a semantic_repair beat must be emitted"
    assert sr[-1].get("ok") is True
    assert "error" not in sr[-1]                 # success carries no error...
    assert sr[-1].get("triggered_by")            # ...just the trigger context


def test_semantic_repair_that_fails_regate_is_a_failure(tmp_path, monkeypatch):
    beats = []
    monkeypatch.setattr(sg, "swarm_beat",
                        lambda *a, **k: beats.append((a[2], k)))
    monkeypatch.setattr(sg, "_full_file_attempt",
                        lambda *a, **k: ("def f(:", "syntax gate failed"))
    # the "repair" is still broken -> re-gate must reject it
    monkeypatch.setattr(sg, "_semantic_repair_fallback",
                        lambda *a, **k: ("def f(:\n", ""))
    out = _gen(tmp_path)
    assert not out.ok and out.method == "failed"   # not written as success
    sr = [k for ph, k in beats if ph == "semantic_repair"]
    assert sr and sr[-1].get("ok") is False and sr[-1].get("error")
