import subprocess
import os
import json
import logging
from typing import Dict, Any

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

def edit_file(path: str, content: str, cwd: str) -> str:
    """Write or overwrite a file with content."""
    full_path = os.path.join(cwd, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
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


def search_web(query: str) -> str:
    """A simple web search tool for the Tech Lead to fetch API documentation snippets."""
    import urllib.request
    import urllib.parse
    import re
    
    url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    try:
        response = urllib.request.urlopen(req, timeout=10)
        html = response.read().decode('utf-8')
        snippets = re.findall(r'<a class=\"result__snippet[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL)
        clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets]
        return '\\n\\n'.join(clean_snippets[:3]) if clean_snippets else "No relevant snippets found."
    except Exception as e:
        return f"Search failed: {e}"


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
        description="Search the web for API docs / library signatures.",
        handler=_h_search_web))


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
