

"""Shared helpers for the CGX web UI handlers."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Provider construction moved to the engine (``cgx.answer.provider_factory``)
# so the CLI can build providers without importing ``cgx.webui``. Re-exported
# here for backward compatibility with existing webui importers.
from cgx.answer.provider_factory import (  # noqa: F401
    DEFAULT_OLLAMA_NUM_CTX_CAP,
    _effective_ollama_num_ctx,
    build_provider,
    provider_from_profile_name,
)


def maybe_extract_zip(path: Optional[str]) -> Optional[str]:
    """Extract an uploaded ``.zip`` into a temp dir; return root path."""
    if not path or not os.path.exists(path):
        return None
    tmpdir = tempfile.mkdtemp(prefix="cgx_zip_")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(tmpdir)
    entries = [p for p in Path(tmpdir).iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return tmpdir


def json_safe(obj: Any) -> Any:
    """Best-effort coercion of nested objects into JSON-serialisable form."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def stringify(value: Any) -> str:
    """Render any LLM-returned ``answer_md``-shaped value into a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "markdown", "md"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    if isinstance(value, list):
        return "\n".join(stringify(v) for v in value)
    return str(value)


def diffs_payload(diffs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalise plan diffs into a uniform ``[{file, patch}]`` list.

    The original Gradio surface formatted these as markdown for a single
    blob; the React diff viewer renders one card per file so we return
    structured records instead.
    """
    out: List[Dict[str, str]] = []
    for d in diffs or []:
        if not isinstance(d, dict):
            continue
        out.append({
            "file": str(d.get("file") or d.get("path") or "(unknown)"),
            "patch": str(d.get("patch") or d.get("diff") or ""),
        })
    return out


def report_summary(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Surface only the structured bits of a codegen report.

    The React Plan page renders the report itself; we just pass the
    dict through after filtering to JSON-safe values.
    """
    if not report:
        return None
    return json_safe(report)
