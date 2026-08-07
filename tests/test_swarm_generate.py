"""Phase 2: the per-file generation ladder for the swarm Developer.

The ladder tries a full-file generation first (grounded on the real on-disk
content of the file's dependencies), gated on ``syntax_ok`` and re-asked once;
on a second failure it degrades to the deterministic AST assembler, whose
required symbols come from the plan contracts. A file that fails both rungs is
reported failed rather than silently written empty.
"""

import cgx.answer.engine as engine
from cgx.session.tasks import swarm_generate as sg


class StubProvider:
    """Minimal provider for the AST-fallback rungs (``chat`` only)."""

    def __init__(self, header="import os", symbol="def run():\n    return 1"):
        self.header, self.symbol = header, symbol

    def chat(self, messages=None, force_json=False, **kw):
        prompt = messages[-1]["content"] if messages else ""
        body = self.header if "file header" in prompt else self.symbol
        return {"content": body}


def _stub_full_file(monkeypatch, content, syntax_ok):
    def fake(path, description, provider, **kw):
        return {"file": path, "content": content, "syntax_ok": syntax_ok,
                "syntax_error": "" if syntax_ok else "bad syntax"}
    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake)


def test_full_file_success_short_circuits(monkeypatch):
    _stub_full_file(monkeypatch, "print('hi')\n", True)
    out = sg.generate_file(
        path="src/app.py", description="entry", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok and out.method == "full-file"
    assert out.content == "print('hi')\n"


def test_falls_back_to_ast_when_full_file_fails(monkeypatch):
    _stub_full_file(monkeypatch, "", False)
    contracts = {"functions": [{"name": "run", "module": "src/app.py"}]}
    out = sg.generate_file(
        path="src/app.py", description="entry", depends_on=[],
        contracts=contracts, goal="demo", root=".", provider=StubProvider())
    assert out.ok and out.method == "ast-fallback"
    assert "def run" in out.content


def test_both_rungs_fail_reports_failed(monkeypatch):
    _stub_full_file(monkeypatch, "", False)
    # No contracts -> AST fallback has no required symbols and, with an empty
    # header, degrades to an empty module which the assembler rejects.
    out = sg.generate_file(
        path="src/app.py", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider(header="", symbol=""))
    assert not out.ok and out.method == "failed"
    assert out.error


def test_ast_fallback_refuses_non_python(monkeypatch):
    _stub_full_file(monkeypatch, "", False)
    out = sg.generate_file(
        path="src/app.js", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert not out.ok
    assert "only supports .py" in (out.error or "")


def test_dependency_context_reads_on_disk(tmp_path):
    dep = tmp_path / "src"
    dep.mkdir()
    (dep / "models.py").write_text("class User: pass\n", encoding="utf-8")
    ctx = sg._dep_context(["src/models.py", "src/missing.py"], str(tmp_path))
    assert ctx == [{"path": "src/models.py", "content": "class User: pass\n"}]


def test_full_file_retried_once_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake(path, description, provider, **kw):
        calls["n"] += 1
        ok = calls["n"] >= 2
        return {"file": path, "content": "x = 1\n" if ok else "",
                "syntax_ok": ok, "syntax_error": "" if ok else "bad"}
    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake)
    out = sg.generate_file(
        path="src/app.py", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok and out.method == "full-file" and calls["n"] == 2


def test_phantom_import_is_stripped_before_return(monkeypatch):
    # A syntactically valid body carrying an unused import must never ship:
    # even when the model repeats it on the re-ask, it is stripped on return.
    _stub_full_file(monkeypatch, "import os\n\nx = 1\n", True)
    out = sg.generate_file(
        path="src/app.py", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok and out.method == "full-file"
    assert "import os" not in out.content and "x = 1" in out.content


def test_phantom_import_triggers_reask(monkeypatch):
    calls = {"n": 0}

    def fake(path, description, provider, **kw):
        calls["n"] += 1
        # First body has a phantom import (gate failure -> re-ask); the second
        # is clean and short-circuits.
        content = "import os\nx = 1\n" if calls["n"] == 1 else "y = 2\n"
        return {"file": path, "content": content, "syntax_ok": True,
                "syntax_error": ""}
    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake)
    out = sg.generate_file(
        path="src/app.py", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok and calls["n"] == 2
    assert out.content == "y = 2\n"
