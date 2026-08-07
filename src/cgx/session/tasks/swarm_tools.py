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
