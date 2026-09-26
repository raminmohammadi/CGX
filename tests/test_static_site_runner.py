"""Tests for the on-disk static-site VERIFY runner + its link/asset checker."""

from __future__ import annotations

from cgx.codegen.static_site_check import check_static_site
from cgx.codegen.test_runners import StaticSiteRunner, detect_test_runners


def _write(root, rel, content=""):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_detects_plain_static_site(tmp_path):
    _write(tmp_path, "index.html", "<h1>hi</h1>")
    runners = detect_test_runners(str(tmp_path))
    names = {r.name for r in runners}
    assert "static_site" in names
    # pytest / npm must NOT claim a plain static site
    assert "pytest" not in names and "npm" not in names


def test_defers_when_package_json_present(tmp_path):
    _write(tmp_path, "index.html")
    _write(tmp_path, "package.json", "{}")
    assert StaticSiteRunner().detect(str(tmp_path)) is False


def test_defers_when_python_project(tmp_path):
    _write(tmp_path, "index.html")
    _write(tmp_path, "requirements.txt", "flask\n")
    assert StaticSiteRunner().detect(str(tmp_path)) is False


def test_clean_site_passes(tmp_path):
    _write(tmp_path, "index.html",
           '<link rel="stylesheet" href="css/style.css">'
           '<a href="about.html">About</a>'
           '<img src="assets/logo.png" alt="logo">'
           '<script src="js/script.js"></script>')
    _write(tmp_path, "about.html", '<link rel="stylesheet" href="css/style.css">')
    _write(tmp_path, "css/style.css", "body{background:url('../assets/bg.png')}")
    _write(tmp_path, "js/script.js", "console.log(1)")
    _write(tmp_path, "assets/logo.png", "x")
    _write(tmp_path, "assets/bg.png", "x")

    broken, checked = check_static_site(str(tmp_path))
    assert broken == []
    assert checked == 3  # 2 html + 1 css

    oc = StaticSiteRunner().run(str(tmp_path), [])
    assert oc.ran and oc.returncode == 0 and oc.ran_tests is True


def test_broken_link_fails(tmp_path):
    _write(tmp_path, "index.html",
           '<a href="missing.html">x</a>'
           '<link rel="stylesheet" href="css/style.css">')
    _write(tmp_path, "css/style.css", "body{}")
    broken, _checked = check_static_site(str(tmp_path))
    refs = [r for _s, r in broken]
    assert "missing.html" in refs

    oc = StaticSiteRunner().run(str(tmp_path), [])
    assert oc.returncode == 1 and "missing.html" in oc.stderr


def test_external_and_anchor_refs_ignored(tmp_path):
    _write(tmp_path, "index.html",
           '<a href="https://example.com">e</a>'
           '<a href="#top">a</a>'
           '<a href="mailto:x@y.z">m</a>'
           '<img src="data:image/png;base64,AAAA">')
    broken, _ = check_static_site(str(tmp_path))
    assert broken == []


def test_query_and_fragment_stripped(tmp_path):
    _write(tmp_path, "index.html",
           '<link rel="stylesheet" href="css/style.css?v=2">'
           '<a href="about.html#team">t</a>')
    _write(tmp_path, "css/style.css", "body{}")
    _write(tmp_path, "about.html", "")
    broken, _ = check_static_site(str(tmp_path))
    assert broken == []


def test_root_relative_ref_resolves(tmp_path):
    _write(tmp_path, "index.html", '<link rel="stylesheet" href="/css/style.css">')
    _write(tmp_path, "css/style.css", "body{}")
    broken, _ = check_static_site(str(tmp_path))
    assert broken == []


def test_traversal_escape_is_broken(tmp_path):
    _write(tmp_path, "index.html", '<img src="../../etc/passwd">')
    broken, _ = check_static_site(str(tmp_path))
    assert any(r == "../../etc/passwd" for _s, r in broken)
