

"""Symbol table context for the agent's working memory.

Builds a compressed map of available symbols from the JSONL records file
(the same artefact used by the retrieval layer) and provides:

  1. :func:`build_symbol_map`   — file → [symbol, …] dictionary
  2. :func:`format_symbol_map`  — compact prompt block injected before
                                  APPLY / FILL_LOGIC tasks so the SLM
                                  knows what is already defined
  3. :func:`fetch_symbol_source` — AST-RAG on demand: returns the exact
                                  source snippet for a symbol when the
                                  retry loop catches a wrong-arguments
                                  failure
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_SYMBOLS_PER_FILE = 20
_MAX_FILES_IN_MAP = 60


def _normalise_path(raw_path: str) -> str:
    """Return a short project-relative path from an absolute chunk_id prefix."""
    # Anchors that mark the start of the project-relative portion.
    for anchor in ("/src/", "/tests/", "/app/", "/backend/", "/frontend/"):
        idx = raw_path.find(anchor)
        if idx >= 0:
            return raw_path[idx + 1:]
    # Fall back to the last two path components so the map stays readable.
    parts = raw_path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else raw_path


def build_symbol_map(records_path: str) -> Dict[str, List[str]]:
    """Build a ``{relative_path: [symbol, …]}`` map from the JSONL records.

    Reads the records file produced by the indexing pipeline.  Each record
    has a ``chunk_id`` in the form ``path::kind::symbol``.  Symbols are
    collected per file and deduplicated; ordering is preserved so callers
    get the symbols in definition order.
    """
    file_symbols: Dict[str, List[str]] = {}
    try:
        with open(records_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = str(rec.get("chunk_id") or "")
                parts = cid.split("::")
                if len(parts) < 3:
                    continue
                raw_path, _kind, symbol = parts[0], parts[1], parts[2]
                if not raw_path or not symbol:
                    continue
                rel = _normalise_path(raw_path)
                syms = file_symbols.setdefault(rel, [])
                if symbol not in syms:
                    syms.append(symbol)
    except Exception as exc:
        logger.warning("symbol_map: failed to read %r: %s", records_path, exc)
    return file_symbols


def format_symbol_map(
    symbol_map: Dict[str, List[str]],
    *,
    max_files: int = _MAX_FILES_IN_MAP,
) -> str:
    """Return a compact prompt block listing available symbols.

    Example output::

        # AVAILABLE CONTEXT (Do not redefine these):
        File: src/db.py -> get_connection(), close_connection()
        File: src/utils.py -> hash_password(str), verify_token(str)

    Caps are applied to keep the injected block small enough that the
    local SLM still has room for the actual instruction.
    """
    if not symbol_map:
        return ""
    lines = ["# AVAILABLE CONTEXT (Do not redefine these):"]
    count = 0
    for fp, syms in symbol_map.items():
        if count >= max_files:
            remaining = len(symbol_map) - max_files
            lines.append(f"  … and {remaining} more file(s) not shown")
            break
        visible = syms[:_MAX_SYMBOLS_PER_FILE]
        suffix = (
            f" +{len(syms) - _MAX_SYMBOLS_PER_FILE} more"
            if len(syms) > _MAX_SYMBOLS_PER_FILE
            else ""
        )
        lines.append(f"File: {fp} -> {', '.join(visible)}{suffix}")
        count += 1
    return "\n".join(lines)


def fetch_symbol_source(records_path: str, symbol_name: str) -> Optional[str]:
    """Return the source text of the first record whose symbol matches ``symbol_name``.

    Used by the retry loop when the model calls an existing function with
    the wrong arguments: the exact source is injected into the re-try
    prompt so the model sees the real signature.

    Returns ``None`` when nothing matches or the records file is unavailable.
    """
    if not records_path or not symbol_name:
        return None
    needle = symbol_name.lower().strip()
    try:
        with open(records_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                cid = str(rec.get("chunk_id") or "")
                parts = cid.split("::")
                sym = parts[2].lower() if len(parts) >= 3 else ""
                # Accept exact match or method-tail match (e.g. "MyClass.foo" → "foo").
                if sym == needle or (sym.endswith("." + needle)):
                    return str(rec.get("text") or "")
    except Exception as exc:
        logger.warning("symbol_map.fetch_symbol_source: error reading %r: %s",
                       records_path, exc)
    return None


def build_symbol_context_prompt(records_path: Optional[str]) -> str:
    """Convenience wrapper: build the map and return the formatted prompt block.

    Returns an empty string when ``records_path`` is ``None`` or the file
    does not exist, so callers can safely include the result without
    checking first.
    """
    if not records_path or not Path(records_path).exists():
        return ""
    sym_map = build_symbol_map(records_path)
    if not sym_map:
        return ""
    return format_symbol_map(sym_map)
