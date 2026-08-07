"""Phase 1: read-only grounding tools for the swarm Developer.

The tools expose the real public surface of a dependency (imports, signatures,
docstrings) so a consumer is authored against symbols that exist. They must be
path-safe (never read outside the project root) and degrade -- not crash -- on
missing or unparseable files.
"""

import pytest

from cgx.session.tasks import swarm_ground as g

MODELS = (
    '"""Data models."""\n'
    'from dataclasses import dataclass\n'
    'MAX = 10\n'
    'def make(x: int) -> "User":\n'
    '    """Build a user."""\n'
    '    ...\n'
    'class User:\n'
    '    """A user."""\n'
    '    pass\n'
)


@pytest.fixture()
def root(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "models.py").write_text(MODELS, encoding="utf-8")
    (src / "broken.py").write_text(
        "def oops(:\n   bad\n" + "x\n" * 40, encoding="utf-8")
    return str(tmp_path)


def test_file_skeleton_summarizes_public_surface(root):
    sk = g.file_skeleton("src/models.py", root)
    assert '"""Data models."""' in sk
    assert "from dataclasses import dataclass" in sk
    assert "def make(x: int)" in sk
    assert "class User" in sk


def test_file_skeleton_missing_is_empty(root):
    assert g.file_skeleton("src/ghost.py", root) == ""


def test_list_symbols(root):
    kinds = {s["name"]: s["kind"] for s in g.list_symbols("src/models.py", root)}
    assert kinds == {"make": "function", "User": "class"}


def test_get_signature(root):
    assert g.get_signature("src/models.py", "make", root).startswith("def make")
    assert g.get_signature("src/models.py", "nope", root) is None


def test_describe_file(root):
    d = g.describe_file("src/models.py", root)
    assert d["exists"] and d["parses"]
    assert d["docstring"] == "Data models."
    assert "from dataclasses import dataclass" in d["imports"]
    assert {s["name"] for s in d["symbols"]} == {"make", "User"}


def test_describe_missing_file(root):
    d = g.describe_file("src/ghost.py", root)
    assert d["exists"] is False
    assert d["symbols"] == [] and d["skeleton"] == ""


def test_ground_dependencies_sections_and_skips_missing(root):
    block = g.ground_dependencies(["src/models.py", "src/ghost.py"], root)
    assert "# From src/models.py" in block
    assert "ghost" not in block


def test_ground_dependencies_is_bounded(root):
    block = g.ground_dependencies(["src/models.py"], root, limit=20)
    assert "truncated" in block


def test_unsafe_paths_are_refused(root):
    assert g.file_skeleton("/etc/passwd", root) == ""
    assert g.file_skeleton("../../etc/passwd", root) == ""
    assert g.describe_file("../secrets.py", root)["exists"] is False


def test_unparseable_file_degrades_to_raw_head(root):
    raw = g.file_skeleton("src/broken.py", root)
    assert "def oops" in raw
    assert len(raw.splitlines()) <= 20
    assert g.list_symbols("src/broken.py", root) == []
