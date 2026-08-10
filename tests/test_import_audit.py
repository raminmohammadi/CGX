"""Phase 6: deterministic import invariants (:mod:`cgx.session.import_audit`).

Two correctness gates a weak model routinely trips: a *phantom* import (a
syntactically valid module carrying an unused, invented dependency, which makes
the whole tree fail at collection time) and a *misrooted* first-party import (a
``from X import ...`` that resolves against neither the project root nor
``root/src`` -- the layout pytest is given). Both are checked purely from text
and the planned manifest, so the tests stay hermetic.
"""

from cgx.session.import_audit import (
    resolve_first_party_imports, strip_unused_imports, unused_imports)


def test_unused_import_is_flagged():
    assert unused_imports("import os\n\nx = 1\n") == ["os"]


def test_used_import_is_kept():
    assert unused_imports("import os\nprint(os.getcwd())\n") == []


def test_future_and_all_are_never_flagged():
    assert unused_imports("from __future__ import annotations\nx = 1\n") == []
    assert unused_imports("import os\n__all__ = ['os']\n") == []


def test_init_files_are_skipped():
    # Package re-exports are intentional, so __init__.py abstains wholesale.
    assert unused_imports("from .a import b\n", path="pkg/__init__.py") == []


def test_strip_removes_whole_statement():
    text, removed = strip_unused_imports("import os\n\nx = 1\n")
    assert removed == ["os"]
    assert "import os" not in text and "x = 1" in text


def test_strip_keeps_used_alias_in_mixed_import():
    text, removed = strip_unused_imports("from m import a, b\nprint(b)\n")
    assert removed == ["a"]
    assert text == "from m import b\nprint(b)\n"


def test_strip_is_noop_when_all_used():
    src = "import os\nprint(os.getcwd())\n"
    assert strip_unused_imports(src) == (src, [])


def test_phantom_pydantic_settings_is_stripped():
    # The exact hallucination the E2E surfaced on a trivial is_prime module.
    src = ("from pydantic_settings import BaseSettings\n\n\n"
           "def is_prime(n):\n    return n > 1\n")
    text, removed = strip_unused_imports(src)
    assert removed == ["BaseSettings"]
    assert "pydantic_settings" not in text
    assert "def is_prime" in text


def test_first_party_src_import_resolves():
    # src/mathutils.py on PYTHONPATH means `from mathutils import` resolves.
    contents = {"tests/test_math.py": "from mathutils import is_prime\n"}
    assert resolve_first_party_imports(
        contents, ["src/mathutils.py", "tests/test_math.py"]) == []


def test_first_party_src_rooted_import_resolves():
    contents = {"tests/test_math.py": "from src.mathutils import is_prime\n"}
    assert resolve_first_party_imports(
        contents, ["src/mathutils.py", "tests/test_math.py"]) == []


def test_third_party_import_abstains():
    contents = {"src/a.py": "import requests\n"}
    assert resolve_first_party_imports(contents, ["src/a.py"]) == []


def test_misrooted_first_party_import_is_flagged():
    # `pkg` is first-party (a planned top-level dir) but `pkg.missing` names no
    # planned module, so it resolves nowhere pytest looks.
    contents = {"pkg/app.py": "from pkg.missing import thing\n"}
    warnings = resolve_first_party_imports(contents, ["pkg/app.py"])
    assert len(warnings) == 1
    assert warnings[0]["file"] == "pkg/app.py"
    assert warnings[0]["module"] == "pkg.missing"


def test_syntax_error_abstains():
    assert resolve_first_party_imports({"src/a.py": "def (:\n"},
                                       ["src/a.py"]) == []
