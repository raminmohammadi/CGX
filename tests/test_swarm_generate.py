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
    assert out.ok and out.method == "semantic-repair"
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


def test_non_python_file_gets_repair_path(monkeypatch):
    # Polyglot swarm: a non-Python file is no longer refused -- semantic repair
    # is language-aware and produces content (the old ".py only" guard is gone).
    _stub_full_file(monkeypatch, "", False)
    out = sg.generate_file(
        path="src/app.js", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok
    assert out.method == "semantic-repair"


def test_fence_lang_maps_extensions():
    assert sg._fence_lang("a.py") == "python"
    assert sg._fence_lang("src/App.jsx") == "jsx"
    assert sg._fence_lang("x.ts") == "typescript"
    assert sg._fence_lang("noext") == ""


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


def test_unused_import_is_stripped_not_reasked(monkeypatch):
    # An unused import is no longer a gate failure that forces a wasted re-ask
    # (that churned nearly every test file over its `import pytest`); the
    # accepted body is sanitized instead.
    calls = {"n": 0}

    def fake(path, description, provider, **kw):
        calls["n"] += 1
        return {"file": path, "content": "import os\nx = 1\n", "syntax_ok": True,
                "syntax_error": ""}
    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake)
    out = sg.generate_file(
        path="src/app.py", description="x", depends_on=[], contracts={},
        goal="demo", root=".", provider=StubProvider())
    assert out.ok and calls["n"] == 1          # accepted first try, no re-ask
    assert "import os" not in out.content       # unused import stripped
    assert "x = 1" in out.content


def test_requirements_generated_from_real_imports(tmp_path):
    # requirements.txt is source-derived: scan the manifest's .py files, drop
    # stdlib (os) and first-party (app) roots, map yaml -> PyYAML, and always
    # pin pytest. The AST ladder is never consulted for it.
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text(
        "import os\nimport yaml\nimport app\n", encoding="utf-8")
    out = sg.generate_file(
        path="requirements.txt", description="deps", depends_on=[],
        contracts={}, goal="demo", root=str(tmp_path),
        provider=StubProvider(), manifest_paths=["src/app.py"])
    assert out.ok and out.method == "template"
    lines = out.content.split()
    assert "PyYAML" in lines and "pytest" in lines
    assert "os" not in lines and "app" not in lines


def test_conftest_is_deterministic_template():
    out = sg.generate_file(
        path="conftest.py", description="pytest setup", depends_on=[],
        contracts={}, goal="demo", root=".", provider=StubProvider())
    assert out.ok and out.method == "template"
    assert out.content == sg._CONFTEST_TEMPLATE
    # A valid, importable module (no hallucinated symbols).
    import ast as _ast
    _ast.parse(out.content)


def test_readme_uses_freeform_model_then_falls_back(monkeypatch):
    # A responsive provider authors the README free-form (no AST ladder).
    class _Doc:
        def chat(self, messages=None, force_json=False, **kw):
            return {"content": "# Title\n\nGreat project."}
    out = sg.generate_file(
        path="README.md", description="overview", depends_on=[], contracts={},
        goal="demo", root=".", provider=_Doc(), manifest_paths=["src/app.py"])
    assert out.ok and out.method == "freeform"
    assert out.content.startswith("# Title")

    # A crashing provider degrades to the deterministic fallback README.
    class _Broken:
        def chat(self, *a, **k):
            raise RuntimeError("down")
    out2 = sg.generate_file(
        path="README.md", description="overview", depends_on=[], contracts={},
        goal="Build a widget", root=".", provider=_Broken(),
        manifest_paths=["src/app.py"])
    assert out2.ok and out2.method == "freeform"
    assert "# Build a widget" in out2.content
    assert "pip install -r requirements.txt" in out2.content


def test_non_source_routing_leaves_other_source_on_the_ladder(monkeypatch):
    # A .js file is real source, not scaffolding: it must stay on the code
    # ladder (full-file then AST fallback), not be diverted to freeform docs.
    assert sg._is_non_source("README.md")
    assert sg._is_non_source("requirements.txt")
    assert sg._is_non_source("conftest.py")
    assert not sg._is_non_source("src/app.js")
    assert not sg._is_non_source("src/app.py")


# --------------------- no-stub contract gate (engine) ---------------------

def test_contract_stub_symbols_flags_placeholder_bodies():
    # A module-level function, a dotted method, and a bare-name method, each
    # with a placeholder body (pass / ... / docstring-only / NotImplemented),
    # are all flagged; the one with a real body is not.
    src = (
        "def encode(u):\n    pass\n\n"
        "def real(u):\n    return u[::-1]\n\n"
        "class Store:\n"
        "    def put(self, k, v):\n        ...\n"
        "    def get(self, k):\n        raise NotImplementedError\n")
    contracts = {"functions": [
        {"name": "encode", "module": "src/app.py"},
        {"name": "real", "module": "src/app.py"},
        {"name": "Store.put", "module": "src/app.py"},
        {"name": "get", "module": "src/app.py"}]}
    stubs = engine._contract_stub_symbols(src, "src/app.py", contracts)
    assert set(stubs) == {"encode", "Store.put", "get"}


def test_contract_stub_symbols_ignores_other_modules_and_bad_syntax():
    src = "def encode(u):\n    pass\n"
    # A contract pinned to a different module never matches this file.
    assert engine._contract_stub_symbols(
        src, "src/app.py",
        {"functions": [{"name": "encode", "module": "src/other.py"}]}) == []
    # Unparseable content abstains rather than raising.
    assert engine._contract_stub_symbols(
        "def broken(:\n", "src/app.py",
        {"functions": [{"name": "broken", "module": "src/app.py"}]}) == []


def test_body_is_stub_true_and_false():
    import ast
    stub = ast.parse('def f():\n    """doc"""\n    pass\n').body[0]
    real = ast.parse("def g():\n    x = 1\n    return x\n").body[0]
    docstring_only = ast.parse('def h():\n    """just a doc"""\n').body[0]
    assert engine._body_is_stub(stub)
    assert engine._body_is_stub(docstring_only)
    assert not engine._body_is_stub(real)


# --------------------- C3: existing-repo safety + grounding ---------------------

def test_edit_file_backs_up_existing_and_keeps_pristine(tmp_path):
    from cgx.session.tasks.swarm_tools import edit_file
    (tmp_path / "a.py").write_text("original\n")
    edit_file("a.py", "new1\n", str(tmp_path))
    bak = tmp_path / ".cgx-backups" / "swarm" / "a.py"
    assert bak.read_text() == "original\n"
    assert (tmp_path / "a.py").read_text() == "new1\n"
    edit_file("a.py", "new2\n", str(tmp_path))  # second write keeps 1st backup
    assert bak.read_text() == "original\n"


def test_edit_file_no_backup_for_new_file(tmp_path):
    from cgx.session.tasks.swarm_tools import edit_file
    edit_file("b.py", "hi\n", str(tmp_path))
    assert not (tmp_path / ".cgx-backups").exists()


def test_dev_tools_query_codebase_index_awareness():
    from cgx.session.tasks.swarm_generate import _dev_tools
    assert "query_codebase" not in _dev_tools(None)

    class _DepsIdx:
        index_dir = "/tmp/x"
        records_path = "/tmp/y"
    assert "query_codebase" in _dev_tools(_DepsIdx())


def test_generate_file_modify_existing_grounds_on_current(tmp_path, monkeypatch):
    import cgx.answer.engine as engine
    import cgx.session.tasks.swarm_generate as sg
    (tmp_path / "svc.py").write_text("def keep():\n    return 1\n")
    captured = {}

    def fake_gen(path, description, provider, **kw):
        captured["description"] = description
        return {"content": "def keep():\n    return 1\n", "syntax_ok": True}

    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake_gen)

    class _P:
        def chat(self, *a, **k):
            return {"content": "", "syntax_ok": True}

    out = sg.generate_file(path="svc.py", description="add feature X",
                           depends_on=[], contracts={}, goal="g",
                           root=str(tmp_path), provider=_P(),
                           modify_existing=True)
    assert out.ok
    assert "ALREADY EXISTS" in captured["description"]
    assert "def keep()" in captured["description"]


# ---------- gate no longer thrashes on framework/unused imports (PIP fixes) ----------

def test_gate_allows_undeclared_third_party_import(tmp_path):
    """An undeclared but real third-party import (e.g. flask) is advisory, not a
    gate failure -- verify's dependency reconcile installs/pins it."""
    from cgx.session.tasks.swarm_generate import _gate_generated_content
    content = "import flask\napp = flask.Flask(__name__)\n"
    err = _gate_generated_content(
        "backend/app.py", content, contracts={"third_party_dependencies": []},
        manifest_paths=["backend/app.py"], root=str(tmp_path))
    assert err is None


def test_gate_still_flags_missing_first_party_symbol(tmp_path):
    """A first-party import of a symbol no planned file defines still gates."""
    from cgx.session.tasks.swarm_generate import _gate_generated_content
    # `backend.ghost` is not a planned path and not on disk -> a first-party
    # import that resolves against neither still gates (a genuine bug).
    content = "from backend.ghost import Thing\n"
    err = _gate_generated_content(
        "backend/app.py", content, contracts={},
        manifest_paths=["backend/app.py"], root=str(tmp_path))
    assert err is not None


def test_generate_file_accepts_unused_pytest_import(tmp_path, monkeypatch):
    """A test file importing pytest but not referencing it is accepted (the
    sanitizer strips the unused import) rather than rejected + regenerated."""
    import cgx.answer.engine as engine
    import cgx.session.tasks.swarm_generate as sg
    calls = {"n": 0}

    def fake_gen(path, description, provider, **kw):
        calls["n"] += 1
        return {"content": "import pytest\n\ndef test_ok():\n    assert True\n",
                "syntax_ok": True}

    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake_gen)

    class _P:
        def chat(self, *a, **k):
            return {"content": "", "syntax_ok": True}

    out = sg.generate_file(path="tests/test_ok.py", description="a test",
                           depends_on=[], contracts={}, goal="g",
                           root=str(tmp_path), provider=_P())
    assert out.ok and out.method == "full-file"
    assert calls["n"] == 1          # accepted on the first attempt, no regen churn
    assert "import pytest" not in out.content  # unused import stripped
