import subprocess
import os
import json
import logging
from typing import Dict, Any

from cgx.pipeline.auto import run_query_auto
from cgx.session.tasks.base import ExecutorDeps

logger = logging.getLogger(__name__)

def bash_repl(command: str, cwd: str) -> str:
    """Execute a bash command in the given directory and return stdout/stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30  # Don't let it hang forever
        )
        return result.stdout or "Success (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

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

def patch_file(path: str, find: str, replace: str, cwd: str) -> str:
    """Replace a specific block of text in a file."""
    full_path = os.path.join(cwd, path)
    if not os.path.exists(full_path):
        return f"Error: File {path} does not exist."
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if find not in content:
            return f"Error: The 'find' text was not found in {path}. Ensure exact match."
        
        new_content = content.replace(find, replace, 1)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully patched {path}"
    except Exception as e:
        return f"Error patching file: {e}"

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
