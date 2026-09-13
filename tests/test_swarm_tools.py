"""Swarm web tools: search_web (title+URL+snippet) and fetch_url.

These let the Developer/Tech Lead find and READ real API docs so an integration
is implemented for real instead of stubbed. fetch_url must be safe (http/https
only, no SSRF into private/loopback hosts), size-capped, and never raise.
"""

import urllib.request

from cgx.session.tasks import swarm_tools as st


class _Resp:
    def __init__(self, body: bytes, content_type: str = "text/html"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self, n: int = -1) -> bytes:
        return self._body[:n] if n and n >= 0 else self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, resp_or_exc):
    """Patch the module-global urlopen (used by search_web)."""
    def fake(req, timeout=None):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc
    monkeypatch.setattr(urllib.request, "urlopen", fake)


class _FakeOpener:
    def __init__(self, resp_or_exc):
        self._r = resp_or_exc

    def open(self, req, timeout=None):
        if isinstance(self._r, Exception):
            raise self._r
        return self._r


def _patch_opener(monkeypatch, resp_or_exc):
    """Patch the safe opener seam (used by fetch_url)."""
    monkeypatch.setattr(st, "_safe_opener", lambda: _FakeOpener(resp_or_exc))


# ---------------------- fetch_url safety ----------------------

def test_fetch_url_rejects_non_http_schemes():
    assert "only http/https" in st.fetch_url("file:///etc/passwd")
    assert "only http/https" in st.fetch_url("ftp://host/x")


def test_fetch_url_refuses_loopback_and_private_hosts():
    for u in ("http://localhost/x", "http://127.0.0.1/x",
              "http://10.0.0.5/x", "http://169.254.1.1/x",
              "http://192.168.1.9/x"):
        assert "private/loopback" in st.fetch_url(u), u


# ---------------------- fetch_url behavior ----------------------

def test_fetch_url_reduces_html_to_text(monkeypatch):
    html = (b"<html><head><style>a{color:red}</style>"
            b"<script>evil()</script></head><body>"
            b"<h1>Gemini API</h1><p>Use genai.GenerativeModel(...)</p>"
            b"</body></html>")
    _patch_opener(monkeypatch, _Resp(html))
    out = st.fetch_url("https://ai.google.dev/docs")
    assert "Gemini API" in out and "genai.GenerativeModel" in out
    assert "evil()" not in out and "color:red" not in out  # script/style gone


def test_fetch_url_truncates_large_body(monkeypatch):
    _patch_opener(monkeypatch, _Resp(b"A" * 100_000, content_type="text/plain"))
    out = st.fetch_url("https://example.org/big", max_bytes=1000)
    assert out.endswith("[truncated]")
    assert len(out) < 1100


def test_fetch_url_reports_errors_without_raising(monkeypatch):
    _patch_opener(monkeypatch, TimeoutError("timed out"))
    assert st.fetch_url("https://example.org").startswith("Error fetching")


# ---------------------- search_web ----------------------

def test_ddg_real_url_decodes_redirect():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fai.google.dev%2Fdocs&rut=z"
    assert st._ddg_real_url(href) == "https://ai.google.dev/docs"


def test_search_web_returns_title_url_and_snippet(monkeypatch):
    html = (b'<a class="result__a" href="//duckduckgo.com/l/?uddg='
            b'https%3A%2F%2Fai.google.dev%2Fgemini-api%2Fdocs">Gemini API docs</a>'
            b'<a class="result__snippet">Call genai.GenerativeModel to chat.</a>')
    _patch_urlopen(monkeypatch, _Resp(html))
    out = st.search_web("gemini api python quickstart")
    assert "Gemini API docs" in out
    assert "https://ai.google.dev/gemini-api/docs" in out
    assert "genai.GenerativeModel" in out


# ---------------------- registration / advertisement ----------------------

def test_fetch_url_registered_and_advertised():
    from cgx.session.tasks.tool_registry import REGISTRY
    from cgx.session.tasks.swarm_generate import _dev_tools
    from cgx.session.tasks.swarm_tech_lead import _planner_tools
    assert "fetch_url" in REGISTRY.names()
    assert "fetch_url" in _dev_tools(None) and "search_web" in _dev_tools(None)
    assert "fetch_url" in _planner_tools()
