"""Deterministic per-file import hints (cgx.session.tasks.swarm_skeleton)."""

from cgx.session.tasks.swarm_skeleton import render_import_hints


def test_emits_flat_import_for_src_dependency():
    deps = {"src/store.py": "class ChatManager:\n    pass\n\n"
                            "def save_message(m):\n    return m\n"}
    out = render_import_hints("src/api.py", deps)
    assert "from store import ChatManager, save_message" in out


def test_keeps_package_prefix_for_non_src_dependency():
    deps = {"backend/models.py": "class User:\n    pass\n"}
    out = render_import_hints("backend/api.py", deps)
    assert "from backend.models import User" in out


def test_excludes_private_and_reexported_symbols():
    deps = {"src/util.py": "import os\n\n_secret = 1\n\ndef helper():\n    return os\n"}
    out = render_import_hints("src/app.py", deps)
    # public defined symbol only -- not `os` (re-export) or `_secret` (private)
    assert "helper" in out and "os" not in out and "_secret" not in out


def test_abstains_on_unparseable_dependency():
    deps = {"src/broken.py": "def f(:\n"}   # syntax error
    assert render_import_hints("src/app.py", deps) == ""


def test_non_python_target_returns_empty():
    deps = {"src/store.py": "def f():\n    return 1\n"}
    assert render_import_hints("src/App.jsx", deps) == ""


def test_no_deps_returns_empty():
    assert render_import_hints("src/app.py", {}) == ""


def test_skips_non_python_dependency():
    deps = {"src/styles.css": "body{}"}
    assert render_import_hints("src/app.py", deps) == ""
