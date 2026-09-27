"""Hardening tests for StaticSiteSkill's link scan over JavaScript.

The skill's own recommended multi-page layout builds the whole nav as a
string inside ``js/layout.js`` and injects it into every page. Before this
fix the link/asset validator only read ``.html``/``.css`` bodies, so a nav
pointing at a page that was never emitted "shipped green". These tests pin:

* JS-emitted references (href=/src= in template strings AND bare quoted
  ``*.html`` URLs) are scanned and resolved against emitted files;
* the recommended layout (nav in ``js/layout.js`` at ``js/`` referencing
  ROOT pages) does NOT false-positive -- JS refs resolve relative to the
  page that injects them, not the .js file's directory;
* external / CDN / ``#`` refs in JS stay excluded;
* a multi-page site with a page linked from no nav is flagged (orphan),
  as a non-fatal warning, and home (index.html) is never an orphan.
"""

from __future__ import annotations

import pytest

from skills.static_site import StaticSiteSkill


@pytest.fixture()
def skill() -> StaticSiteSkill:
    return StaticSiteSkill()


def _diff(path, patch=""):
    return {"file": path, "patch": patch}


# A page shell that pulls in the shared layout script (defer) and stylesheet.
# ``script`` names the JS file to <script src=> (must be emitted by the test);
# pass "" for no script tag.
def _page(script: str = "js/layout.js") -> str:
    tag = f'<script src="{script}" defer></script>' if script else ""
    return (
        '<link rel="stylesheet" href="css/style.css">'
        '<header id="site-header"></header>'
        + tag
    )


# --- JS nav: broken link now caught ---------------------------------------

def test_broken_nav_link_in_layout_js_is_flagged(skill):
    """A nav in js/layout.js pointing at a page that is never emitted must be
    caught -- this is exactly the "broken nav ships green" regression."""
    layout = (
        "document.getElementById('site-header').innerHTML = `"
        '<nav>'
        '<a href="index.html">Home</a>'
        '<a href="about.html">About</a>'
        '<a href="team.html">Team</a>'  # team.html is NOT emitted
        '</nav>`;'
    )
    diffs = [
        _diff("index.html", _page()),
        _diff("about.html", _page()),
        _diff("css/style.css", "body{}"),
        _diff("js/layout.js", layout),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert v.severity == "warning"  # broken links are advisory, not fatal
    assert "team.html" in v.rationale


def test_broken_string_url_in_js_is_flagged(skill):
    """A bare quoted local page URL (e.g. location.href = '...') is scanned."""
    diffs = [
        _diff("index.html", _page("js/nav.js")),
        _diff("css/style.css", "body{}"),
        _diff("js/nav.js", "window.location.href = 'thankyou.html';"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "thankyou.html" in v.rationale


# --- JS nav: recommended layout must NOT false-positive --------------------

def test_recommended_layout_nav_resolves_no_false_positive(skill):
    """js/layout.js links ROOT pages; those resolve relative to the page that
    injects the nav, so nothing is broken and no orphan warning fires."""
    layout = (
        "const header = `<nav>"
        '<a href="index.html">Home</a>'
        '<a href="about.html">About</a>'
        '<a href="contact.html">Contact</a>'
        "</nav>`;\n"
        "document.getElementById('site-header').innerHTML = header;"
    )
    diffs = [
        _diff("index.html", _page()),
        _diff("about.html", _page()),
        _diff("contact.html", _page()),
        _diff("css/style.css", "body{}"),
        _diff("js/layout.js", layout),
    ]
    assert skill.validate_scaffold(diffs) is None
    warns = skill.scaffold_warnings(diffs)
    assert not any("orphan" in w.rationale.lower() for w in warns)


def test_js_nav_resolves_page_in_subdir(skill):
    """A page under pages/ that injects ../js/layout.js: the nav's root refs
    still resolve because we try the site root as a candidate base."""
    layout = 'document.body.innerHTML = `<a href="index.html">Home</a>`;'
    diffs = [
        _diff("index.html", _page()),
        _diff("pages/services.html",
              '<link rel="stylesheet" href="../css/style.css">'
              '<script src="../js/layout.js" defer></script>'),
        _diff("css/style.css", "body{}"),
        _diff("js/layout.js", layout),
    ]
    # index.html is a real emitted page -> the JS ref resolves, not broken.
    assert skill.validate_scaffold(diffs) is None


# --- JS: external / CDN / anchor refs excluded -----------------------------

def test_js_external_and_anchor_refs_excluded(skill):
    layout = (
        "el.innerHTML = `"
        '<a href="https://cdn.example.com/app.html">ext</a>'
        '<a href="#top">anchor</a>'
        '<script src="https://cdn.tailwindcss.com"></script>`;\n'
        "const api = 'https://api.example.com/data.html';"
    )
    diffs = [
        _diff("index.html", _page()),
        _diff("css/style.css", "body{}"),
        _diff("js/layout.js", layout),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_min_js_is_not_scanned(skill):
    """A minified/vendor .min.js must not be mined for stray '.html' tokens."""
    diffs = [
        _diff("index.html", _page("js/vendor.min.js")),
        _diff("css/style.css", "body{}"),
        _diff("js/vendor.min.js", "var t='template.html';function n(){}"),
    ]
    # No warning/error from the vendored file's incidental string.
    v = skill.validate_scaffold(diffs)
    assert v is None


# --- orphan pages ----------------------------------------------------------

def test_orphan_page_warns_when_not_in_nav(skill):
    """Multi-page site whose nav (in layout.js) omits a page -> orphan warning."""
    layout = (
        "el.innerHTML = `<nav>"
        '<a href="index.html">Home</a>'
        '<a href="about.html">About</a>'
        "</nav>`;"
    )
    diffs = [
        _diff("index.html", _page()),
        _diff("about.html", _page()),
        _diff("secret.html", _page()),  # emitted but linked from nowhere
        _diff("css/style.css", "body{}"),
        _diff("js/layout.js", layout),
    ]
    warns = skill.scaffold_warnings(diffs)
    orphan = [w for w in warns if "orphan" in w.rationale.lower()]
    assert orphan and orphan[0].severity == "warning"
    assert not orphan[0].passed
    assert "secret.html" in orphan[0].rationale
    # A linked page is NOT reported.
    assert "about.html" not in orphan[0].rationale


def test_index_html_never_orphan(skill):
    """Home page is the entry point; never flag it as orphan even if no nav
    links back to it."""
    diffs = [
        _diff("index.html",
              '<link rel="stylesheet" href="css/style.css">'
              '<a href="about.html">About</a>'),
        _diff("about.html",
              '<link rel="stylesheet" href="css/style.css">'
              '<a href="about.html">About</a>'),
        _diff("css/style.css", "body{}"),
    ]
    warns = skill.scaffold_warnings(diffs)
    assert not any("index.html" in w.rationale for w in warns
                   if "orphan" in w.rationale.lower())


def test_single_page_site_has_no_orphan_check(skill):
    diffs = [_diff("index.html", _page()), _diff("css/style.css", "body{}")]
    warns = skill.scaffold_warnings(diffs)
    assert not any("orphan" in w.rationale.lower() for w in warns)


# --- regression: existing HTML/CSS behavior preserved ----------------------

def test_html_broken_link_still_flagged(skill):
    diffs = [
        _diff("index.html", '<a href="missing.html">x</a>'
                            '<link rel="stylesheet" href="css/style.css">'),
        _diff("css/style.css", "body{}"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "missing.html" in v.rationale


def test_clean_site_with_plain_js_passes(skill):
    diffs = [
        _diff("index.html",
              '<link rel="stylesheet" href="css/style.css">'
              '<a href="about.html">About</a>'
              '<script src="js/script.js"></script>'),
        _diff("about.html", '<link rel="stylesheet" href="css/style.css">'
                            '<a href="index.html">Home</a>'),
        _diff("css/style.css", "body { margin: 0; }"),
        _diff("js/script.js", "console.log('hi');"),
    ]
    assert skill.validate_scaffold(diffs) is None
    assert not any("orphan" in w.rationale.lower()
                   for w in skill.scaffold_warnings(diffs))
