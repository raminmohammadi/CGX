import subprocess
import os
import json
import logging
from typing import Any, Dict

from cgx.pipeline.auto import run_query_auto
from cgx.session.tasks.base import ExecutorDeps

logger = logging.getLogger(__name__)

def query_codebase(query: str, deps: ExecutorDeps) -> str:
    """Wrapper around run_query_auto to search the indexed codebase."""
    if not deps.index_dir or not deps.records_path:
        return "Error: Index not available for querying."
    
    try:
        result = run_query_auto(
            index_dir=deps.index_dir,
            records_path=deps.records_path,
            query=query,
            model_name=deps.embed_model or "jinaai/jina-embeddings-v2-base-code",
            embedder=deps.provider, # Attempt to use the provider if it supports embeddings
            top_k_per_view=5
        )
        # Format the result to a readable string for the LLM
        hits = result.get("hits", [])
        if not hits:
            return "No relevant files found."
        
        output = []
        for hit in hits:
            path = hit.get("file", "unknown")
            text = hit.get("text", "")
            output.append(f"File: {path}\nContent snippet:\n{text}\n---")
        return "\n".join(output)
    except Exception as e:
        logger.exception("query_codebase failed")
        return f"Error querying codebase: {e}"

def _backup_existing(full_path: str, cwd: str, rel_path: str) -> None:
    """Mirror an existing file under ``.cgx-backups/swarm/`` before overwriting.

    The Swarm Developer/Verifier write whole files; when a planned path collides
    with a real file in an existing repo, this preserves the original so an
    overwrite is never unrecoverable (mirrors the ``.cgx-backups`` convention the
    diff-apply path uses). The first backup for a path wins -- repeated writes in
    one run don't clobber the pristine original. Best-effort; never raises.
    """
    try:
        if not os.path.isfile(full_path):
            return  # brand-new file: nothing to preserve
        import shutil
        backup_path = os.path.join(cwd, ".cgx-backups", "swarm", rel_path)
        if os.path.exists(backup_path):
            return  # keep the earliest (pristine) copy
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(full_path, backup_path)
    except Exception:  # pragma: no cover - backup is best-effort
        pass


def edit_file(path: str, content: str, cwd: str) -> str:
    """Write or overwrite a file with content (backing up an existing original)."""
    full_path = os.path.join(cwd, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        _backup_existing(full_path, cwd, path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def run_python_probe(code: str, cwd: str) -> str:
    """Run a Python snippet in a sandbox REPL to introspect libraries.
    
    Useful for checking if a module, class, or method exists (e.g. using dir() or help()).
    This runs in a temporary virtual environment or subprocess.
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name
        
    try:
        # Run the code using the project's venv python if available, else system python
        python_exe = os.path.join(cwd, ".venv", "bin", "python")
        if not os.path.exists(python_exe):
            python_exe = "python3"
            
        result = subprocess.run(
            [python_exe, temp_path],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20
        )
        return result.stdout or "Success (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Probe timed out after 20 seconds."
    except Exception as e:
        return f"Error executing probe: {e}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def judge_decision(provider: Any, prompt: str) -> tuple:
    """Ask a judge model for an A/B verdict plus a one-line rationale.

    Used by debate mode in both the Tech Lead and Developer. The judge is
    prompted to put the winner letter on the first line and the reason on the
    next; this tolerantly extracts the first ``A``/``B`` seen (defaulting to
    ``A``) and returns ``(letter, reason)`` so callers can record *why* a draft
    won rather than discarding the reasoning. Never raises.
    """
    try:
        res = provider.chat([{"role": "user", "content": prompt}])
        text = str(res.get("content", "")).strip()
    except Exception as e:
        return "A", f"judge error: {e}"
    letter = "A"
    for ch in text:
        if ch.upper() in ("A", "B"):
            letter = ch.upper()
            break
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    reason = lines[1] if len(lines) > 1 else (lines[0] if lines else "")
    return letter, reason[:300]


_UA = "Mozilla/5.0 (compatible; CGX/1.0; +https://github.com/raminmohammadi/CGX)"


def _ddg_real_url(href: str) -> str:
    """Decode a DuckDuckGo redirect (``/l/?uddg=<enc>``) to the real target URL."""
    import urllib.parse
    href = (href or "").strip()
    if "uddg=" in href:
        try:
            qs = urllib.parse.urlparse(href).query
            uddg = urllib.parse.parse_qs(qs).get("uddg", [])
            if uddg:
                return urllib.parse.unquote(uddg[0])
        except Exception:
            pass
    if href.startswith("//"):
        return "https:" + href
    return href


def search_web(query: str) -> str:
    """Web search that returns each result's TITLE, real URL, and snippet.

    Returning the URLs (not just snippets) is what makes ``search_web`` +
    ``fetch_url`` a usable chain: the model searches, picks the best doc URL,
    then fetches the full page to implement against the real API. Best-effort
    scrape of DuckDuckGo's HTML endpoint; degrades to a clear message.
    """
    import re
    import urllib.parse
    import urllib.request

    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            html = response.read(200_000).decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"

    def _strip(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s or "").strip()

    links = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL)
    snippets = re.findall(
        r'class="result__snippet[^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL)
    if not links:
        return "No results found."
    out = []
    for i, (href, title) in enumerate(links[:5]):
        snip = _strip(snippets[i]) if i < len(snippets) else ""
        out.append(f"{i + 1}. {_strip(title)}\n   URL: {_ddg_real_url(href)}"
                   + (f"\n   {snip}" if snip else ""))
    return _screen_untrusted("\n\n".join(out), "web search results")


# Network egress policy (SSRF block + domain allowlist) lives in the shared
# guardrail so BOTH the built-in fetch tools and the MCP path enforce one rule.
# Aliased to the local underscore names the fetch/redirect code already uses.
from cgx.guardrails.net import (  # noqa: E402
    host_allowed as _host_allowed,
    host_is_blocked as _host_is_blocked,
    host_refusal as _fetch_refusal,
)


def _html_to_text(s: str) -> str:
    """Reduce an HTML page to readable text (drop script/style/tags, unescape)."""
    import html as _html
    import re
    s = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<!--.*?-->", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


class _SafeRedirectHandler:
    """Redirect handler that refuses a redirect to a private/loopback host.

    ``fetch_url`` validates the initial host, but urllib follows 3xx redirects
    automatically -- so without this a public URL could 302 to an internal
    target (e.g. cloud metadata at 169.254.169.254). Every hop's host is
    re-checked before it is followed.
    """

    def __new__(cls):
        import urllib.request

        class _Impl(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                import urllib.error
                import urllib.parse
                host = urllib.parse.urlparse(newurl).hostname or ""
                refusal = _fetch_refusal(host)
                if refusal:
                    raise urllib.error.HTTPError(
                        newurl, code, f"blocked redirect: {refusal}", headers, fp)
                return super().redirect_request(
                    req, fp, code, msg, headers, newurl)

        return _Impl()


def _safe_opener():
    """A urllib opener that blocks redirect-based SSRF (see _SafeRedirectHandler)."""
    import urllib.request
    return urllib.request.build_opener(_SafeRedirectHandler())


def _screen_untrusted(text: str, origin: str) -> str:
    """Delegate to the shared guardrail so every untrusted-text path (built-in
    fetch/search AND the MCP tool results) shares one injection policy."""
    from cgx.guardrails.injection import screen_untrusted
    return screen_untrusted(text, origin)


def fetch_url(url: str, max_bytes: int = 40_000, timeout: float = 12.0) -> str:
    """Fetch an http(s) URL and return its readable text (HTML reduced to text).

    Lets the agent READ real API/SDK documentation (found via ``search_web``)
    and implement the actual interface instead of guessing or shipping a stub.
    Safety: only http/https; loopback/private/link-local hosts are refused on
    the initial request AND on every redirect hop (no SSRF into internal
    services); the response is size-capped. Never raises -- returns an error
    string the model can react to.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    u = (url or "").strip()
    parsed = urllib.parse.urlparse(u)
    if parsed.scheme not in ("http", "https"):
        return (f"Error: only http/https URLs may be fetched (got "
                f"{parsed.scheme or 'no'} scheme).")
    refusal = _fetch_refusal(parsed.hostname or "")
    if refusal:
        return f"Error: {refusal}."
    req = urllib.request.Request(u, headers={"User-Agent": _UA})
    try:
        with _safe_opener().open(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} fetching {u}"
    except Exception as e:
        return f"Error fetching {u}: {type(e).__name__}: {e}"
    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    if "html" in ctype or (not ctype and "<html" in text.lower()[:2000]):
        text = _html_to_text(text)
    text = text.strip()
    if truncated:
        text += "\n... [truncated]"
    return _screen_untrusted(text or "(empty response)", u)


# --------------------- registry wiring ---------------------
# Register the model-callable native tools so both swarm loops dispatch through
# one table and their descriptions are auto-injected into the system prompt.
# Handlers share the ``(args, ctx)`` shape; see :mod:`tool_registry`.
from cgx.session.tasks.tool_registry import (  # noqa: E402
    REGISTRY, RiskLevel, ToolContext, ToolSpec)


def _h_run_python_probe(args: Dict[str, Any], ctx: ToolContext) -> str:
    return run_python_probe(str(args.get("code", "")), ctx.root)


def _h_file_skeleton(args: Dict[str, Any], ctx: ToolContext) -> str:
    from cgx.session.tasks.swarm_ground import file_skeleton
    return file_skeleton(str(args.get("path", "")), ctx.root)


def _h_list_symbols(args: Dict[str, Any], ctx: ToolContext) -> str:
    from cgx.session.tasks.swarm_ground import list_symbols
    return str(list_symbols(str(args.get("path", "")), ctx.root))


def _h_query_codebase(args: Dict[str, Any], ctx: ToolContext) -> str:
    if ctx.deps is None:
        return "Error: codebase index not available in this context."
    return query_codebase(str(args.get("query", "")), ctx.deps)


def _h_search_web(args: Dict[str, Any], ctx: ToolContext) -> str:
    return search_web(str(args.get("query", "")))


def _h_fetch_url(args: Dict[str, Any], ctx: ToolContext) -> str:
    return fetch_url(str(args.get("url", "")))


def register_native_tools() -> None:
    """(Re)register the built-in swarm tools on the default registry."""
    REGISTRY.register(ToolSpec(
        name="run_python_probe", risk=RiskLevel.HIGH, arg_hint='{"code": "..."}',
        description="Run a short Python snippet to introspect a library "
                    "(dir(), help(), import checks). Executes code.",
        handler=_h_run_python_probe))
    REGISTRY.register(ToolSpec(
        name="file_skeleton", risk=RiskLevel.LOW, arg_hint='{"path": "..."}',
        description="Show the exact classes/functions a local file defines.",
        handler=_h_file_skeleton))
    REGISTRY.register(ToolSpec(
        name="list_symbols", risk=RiskLevel.LOW, arg_hint='{"path": "..."}',
        description="List the symbols defined in a local file.",
        handler=_h_list_symbols))
    REGISTRY.register(ToolSpec(
        name="query_codebase", risk=RiskLevel.LOW, arg_hint='{"query": "..."}',
        description="Semantic search over the indexed codebase for relevant "
                    "files and snippets.",
        handler=_h_query_codebase))
    REGISTRY.register(ToolSpec(
        name="search_web", risk=RiskLevel.MEDIUM, arg_hint='{"query": "..."}',
        description="Search the web; returns each result's title, URL, and "
                    "snippet. Use it to find the docs URL for a library/SDK, "
                    "then read it with fetch_url.",
        handler=_h_search_web))
    REGISTRY.register(ToolSpec(
        name="fetch_url", risk=RiskLevel.MEDIUM, arg_hint='{"url": "https://..."}',
        description="Fetch an http(s) page (restricted to trusted documentation "
                    "/ package / code-host domains) and return its readable "
                    "text. Use it to READ real API/SDK docs and implement the "
                    "actual interface instead of guessing or writing a stub.",
        handler=_h_fetch_url))


register_native_tools()

# Register the MCP discovery/call tools too. They degrade gracefully when no
# servers are configured or the optional SDK is absent, so registering them
# unconditionally is safe; they are only *advertised* to a role when servers
# exist (see ``mcp_tools_if_configured``).
try:
    from cgx.mcp.manager import register_mcp_tools
    register_mcp_tools()
except Exception:  # pragma: no cover - MCP package optional
    logger.debug("MCP tools not registered", exc_info=True)


def mcp_tools_if_configured() -> tuple:
    """MCP tool names to advertise to the agent when servers are configured.

    Keeping MCP off the advertised list until a server exists means the agent
    only ever sees tools it can actually use, while a single ~/.cgx/mcp.json
    edit makes them appear -- the agent then discovers and calls them via the
    normal tool loop (requirements: MCP-aware + easy to extend).
    """
    try:
        from cgx.mcp.config import enabled_servers
        if enabled_servers():
            return ("mcp_list_servers", "mcp_list_tools", "mcp_call")
    except Exception:  # pragma: no cover
        pass
    return ()
