"""Cross-file call-resolution correctness in the knowledge graph.

Regression cover for the bare-name resolver that ignored the call receiver and
imports: a qualified ``obj.save()`` used to wire to every global ``save`` in the
repo, aliased externals produced a phantom ``module::<alias>`` node, and
imported project symbols never linked to their real function node.
"""

from __future__ import annotations

from cgx.graph.build_graph import build_knowledge_graph


def _file(fid, module_path):
    return {"id": fid, "file": fid, "type": "file", "name": fid,
            "module_path": module_path, "meta": {}}


def _func(fid, name, file, imports=None, class_name=None):
    meta = {}
    if imports:
        meta["imports_used"] = imports
    if class_name:
        meta["class_name"] = class_name
    return {"id": fid, "file": file, "type": "function", "name": name, "meta": meta}


def _calls_edge(G, u, v) -> bool:
    """True iff a ``calls`` edge u->v exists in the MultiDiGraph."""
    if not (G.has_node(u) and G.has_node(v)):
        return False
    data = G.get_edge_data(u, v) or {}
    return any((ed or {}).get("type") == "calls" for ed in data.values())


def _build():
    chunks = [
        _file("app/models.py", "app.models"),
        _file("app/repo.py", "app.repo"),
        _file("app/ui.py", "app.ui"),
        _file("app/main.py", "app.main"),
        _file("app/plot.py", "app.plot"),
        _func("app/models.py::function::load", "load", "app/models.py"),
        _func("app/repo.py::function::save", "save", "app/repo.py"),          # distractor global
        _func("app/ui.py::method::Widget.render", "render", "app/ui.py", class_name="Widget"),
        _func("app/ui.py::method::Widget.paint", "paint", "app/ui.py", class_name="Widget"),
        _func("app/main.py::function::caller", "caller", "app/main.py",
              imports={"load": "app.models.load"}),
        _func("app/plot.py::function::draw", "draw", "app/plot.py",
              imports={"np": "numpy"}),
    ]
    calls = [
        # Qualified call on an unknown receiver: must NOT bind to the global save.
        {"caller_id": "app/ui.py::method::Widget.render", "callee_name": "save",
         "callee_fullname": "obj.save"},
        # self.method: resolves within the class.
        {"caller_id": "app/ui.py::method::Widget.render", "callee_name": "paint",
         "callee_fullname": "self.paint"},
        # Bare call to an imported project symbol: links to the real function.
        {"caller_id": "app/main.py::function::caller", "callee_name": "load",
         "callee_fullname": "load"},
        # Aliased external call: resolves to the real module, not a phantom alias.
        {"caller_id": "app/plot.py::function::draw", "callee_name": "array",
         "callee_fullname": "np.array"},
    ]
    return build_knowledge_graph(chunks, calls)


def test_qualified_call_does_not_bind_to_bare_global():
    G = _build()
    assert not _calls_edge(G, "app/ui.py::method::Widget.render",
                           "app/repo.py::function::save"), \
        "obj.save() must not wire to an unrelated global save()"


def test_self_method_resolves_within_class():
    G = _build()
    assert _calls_edge(G, "app/ui.py::method::Widget.render",
                       "app/ui.py::method::Widget.paint")


def test_bare_imported_project_symbol_links_to_real_function():
    G = _build()
    assert _calls_edge(G, "app/main.py::function::caller",
                       "app/models.py::function::load")


def test_aliased_external_resolves_to_real_module_not_phantom_alias():
    G = _build()
    assert _calls_edge(G, "app/plot.py::function::draw", "module::numpy")
    assert not G.has_node("module::np"), "aliased import must not create module::np"
