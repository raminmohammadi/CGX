from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from cgx.io.persist import load_indices, load_jsonl
from cgx.answer.providers import LLMProvider
from cgx.answer.intent import detect_intent  # <-- NEW central intent detection
from cgx.retrieval.orchestrator import (
    SYMBOL_STOPWORDS as _SYMBOL_STOPWORDS,
    _extract_symbol_tokens,
)

from networkx.readwrite import json_graph
import networkx as nx  # type: ignore

ALLOWED_CITATION_NOTE = (
    "Cite only chunk_ids that appear in SOURCES. "
    "Return citations as an array of objects: { \"chunk_id\": \"...\" }. "
    "Do not return numbers or invented ids."
)

# Modes whose answers genuinely require a specific code symbol to be present
# in the retrieved sources. Other modes (conceptual ``howto`` / ``overview`` /
# ``change_plan``) can still surface a useful answer even when no single
# named symbol dominates the result set.
_SYMBOL_TARGETED_MODES = frozenset({
    "symbol_explain", "symbol_location", "line_number",
    "callers_list", "callees_list",
})


def _symbol_covers_target(symbol: str, chunk_id: str, target: str) -> bool:
    """Return True when a SOURCE row's symbol/chunk_id covers ``target``.

    Sources carry the chunk_id tail as ``symbol`` (e.g. ``VAE.encode`` for a
    method), so we accept three shapes:
      * exact symbol match (``encode`` == ``encode``)
      * method-tail match (``VAE.encode`` ⊇ ``encode``)
      * literal ``::target`` substring in the chunk_id
    Comparison is case-insensitive to align with ``_find_symbol_rows``.
    """
    if not target:
        return True
    t = target.lower()
    s = (symbol or "").lower()
    if s == t:
        return True
    if "." in s and s.rsplit(".", 1)[-1] == t:
        return True
    cid = chunk_id or ""
    if f"::{target}" in cid or f"::{t}" in cid.lower():
        return True
    return False

# ---------------- utilities ----------------

def _split_chunk_id(cid: str) -> Tuple[str, str, str]:
    parts = str(cid).split("::")
    p = parts[0] if parts else ""
    k = parts[1] if len(parts) > 1 else ""
    s = parts[2] if len(parts) > 2 else ""
    return p, k, s

def _chunk_map(indices: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a map: chunk_id -> row (prefer intent view text)."""
    cmap: Dict[str, Dict[str, Any]] = {}
    views = indices.get("views") or {}
    for name in ["intent", "impl"]:
        vw = views.get(name) or {}
        for r in (vw.get("rows") or []):
            cid = r.get("chunk_id")
            if cid:
                cmap[str(cid)] = r
    return cmap

def _read_readme(project_root: Optional[str]) -> Optional[str]:
    if not project_root:
        return None
    for nm in ["README.md", "Readme.md", "readme.md"]:
        p = Path(project_root) / nm
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return None

def _guess_root(indices: Dict[str, Any]) -> Optional[str]:
    paths: List[str] = []
    for vw in (indices.get("views") or {}).values():
        for r in (vw.get("rows") or [])[:200]:
            p, _, _ = _split_chunk_id(r.get("chunk_id", ""))
            if p and os.path.isabs(p):
                paths.append(p)
    if not paths:
        return None
    try:
        return os.path.commonpath(paths)
    except Exception:
        return str(Path(paths[0]).parent)

def _trim(txt: Optional[str], max_chars: int) -> str:
    if txt is None:
        return ""
    t = str(txt)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."

def _row_signature(row: Dict[str, Any]) -> str:
    """Best-effort signature for a row (intent view typically carries it)."""
    if not isinstance(row, dict):
        return ""
    sig = row.get("signature")
    if isinstance(sig, str) and sig:
        return sig
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if isinstance(meta, dict):
        sig = meta.get("signature")
        if isinstance(sig, str) and sig:
            return sig
    return ""


def _row_lines(row: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Try to extract (start_line, end_line) for a row."""
    if not isinstance(row, dict):
        return None, None
    for k_start, k_end in (("start_line", "end_line"), ("lineno", "end_lineno"), ("line_start", "line_end")):
        s, e = row.get(k_start), row.get(k_end)
        if isinstance(s, int) or isinstance(e, int):
            return (s if isinstance(s, int) else None, e if isinstance(e, int) else None)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if isinstance(meta, dict):
        s, e = meta.get("start_line"), meta.get("end_line")
        if isinstance(s, int) or isinstance(e, int):
            return (s if isinstance(s, int) else None, e if isinstance(e, int) else None)
    return None, None


def _window_text(text: str, focus_terms: List[str], max_chars: int, *, context_lines: int = 8) -> str:
    """Return a focused window of ``text`` centered on the first line matching
    any term in ``focus_terms``.

    When no term matches, falls back to ``_trim(text, max_chars)``. This
    typically reduces SOURCES size 5–10× while preserving the relevant region.
    """
    if not text or not focus_terms:
        return _trim(text, max_chars)
    lines = text.splitlines()
    if not lines:
        return _trim(text, max_chars)
    lc_terms = [t for t in (s.lower() for s in focus_terms) if t]
    hit_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(t in low for t in lc_terms):
            hit_idx = i
            break
    if hit_idx is None:
        return _trim(text, max_chars)
    start = max(0, hit_idx - context_lines)
    end = min(len(lines), hit_idx + context_lines + 1)
    window = "\n".join(lines[start:end])
    if len(window) <= max_chars:
        # Expand outward greedily until we hit the budget.
        while (start > 0 or end < len(lines)) and len(window) < max_chars:
            if start > 0:
                start -= 1
            if end < len(lines):
                end += 1
            window = "\n".join(lines[start:end])
            if len(window) > max_chars:
                break
    return _trim(window, max_chars)


def _as_sources_with_meta(
    hits: List[Dict[str, Any]],
    cmap: Dict[str, Dict[str, Any]],
    max_chunks: int = 24,
    max_chars: int = 900,
    *,
    focus_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Select top hits and attach trimmed text + structured meta for grounding & debug.

    When ``focus_terms`` is non-empty, each chunk's text is windowed around
    the first line matching one of the terms (symbol name, query keywords)
    to reduce prompt size without losing the relevant span.
    """
    out: List[Dict[str, Any]] = []
    for h in hits[:max_chunks]:
        cid = str(h.get("chunk_id"))
        row = cmap.get(cid) or {}
        text = row.get("text", "") if isinstance(row, dict) else ""
        path, kind, symbol = _split_chunk_id(cid)
        prov: Dict[str, Any] = {}
        for k, v in (h or {}).items():
            if k == "chunk_id":
                continue
            if k == "provenance" and isinstance(v, dict):
                prov.update(v)
            else:
                prov[k] = v
        signature = _row_signature(row)
        start_line, end_line = _row_lines(row)
        parent_class = (row.get("parent_class_id") if isinstance(row, dict) else None) or ""
        if focus_terms:
            terms = list(focus_terms)
            if symbol:
                terms.insert(0, symbol)
            body = _window_text(text or "", terms, max_chars)
        else:
            body = _trim(text or "", max_chars)
        out.append({
            "chunk_id": cid,
            "path": path,
            "kind": kind,
            "symbol": symbol,
            "signature": signature,
            "start_line": start_line,
            "end_line": end_line,
            "parent_class_id": parent_class,
            "text": body,
            "hit_meta": prov,
        })
    return out


def _fmt_source(s: Dict[str, Any]) -> str:
    """Render a single source block for the LLM prompt with structured fields."""
    head = f"- {s['chunk_id']} :: {s.get('path','')} :: {s.get('kind','')} :: {s.get('symbol','')}"
    extras: List[str] = []
    sig = s.get("signature") or ""
    if sig:
        extras.append(f"signature={sig}")
    sl, el = s.get("start_line"), s.get("end_line")
    if isinstance(sl, int) or isinstance(el, int):
        extras.append(f"lines={sl if isinstance(sl, int) else '?'}-{el if isinstance(el, int) else '?'}")
    pcid = s.get("parent_class_id") or ""
    if pcid:
        extras.append(f"parent_class={pcid}")
    if extras:
        head += "  [" + ", ".join(extras) + "]"
    body = s.get("text", "") or ""
    return head + "\n  " + body


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first top-level JSON object from `text`.

    Uses brace-balanced scanning (string- and escape-aware) instead of a
    greedy `\\{.*\\}` regex, which can capture unrelated content spanning
    multiple unrelated braces.

    Returns {} when no valid JSON object can be parsed.
    """
    if not isinstance(text, str) or not text:
        return {}
    # Fast path: the whole payload is already JSON (Ollama JSON mode).
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass
                        break
            j += 1
        i = j + 1 if j > i else i + 1
    return {}


_DIFF_FENCE_RE = re.compile(
    r"```(?:diff|patch)?\s*(?:path\s*=\s*(?P<path>[^\s`]+))?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _coerce_answer_text(parsed: Dict[str, Any]) -> str:
    """Best-effort extraction of answer text from an LLM response that obeyed
    JSON mode but ignored the ``answer_md`` key.

    Handles common synonyms (``answer``, ``message``, ``markdown``, ``md``,
    ``text``, ``content``, ``output``, ``response``) and Jupyter MIME bundles
    (``{"data": {"text/markdown" | "text/plain": "..."}}``).
    """
    if not isinstance(parsed, dict):
        return ""
    for k in ("answer", "message", "markdown", "md", "text", "content", "output", "response"):
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            return v
    data = parsed.get("data")
    if isinstance(data, dict):
        for mime in ("text/markdown", "text/plain"):
            v = data.get(mime)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def _parse_plan_freeform(text: str) -> Dict[str, Any]:
    """
    Parse a free-form plan response with ``## Plan`` / ``## Diffs`` sections
    and fenced ```diff path=...``` blocks. Citations are extracted from
    ``[[chunk_id]]`` markers anywhere in the plan body.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    plan_md = ""
    m = re.search(r"##\s*Plan\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        plan_md = m.group(1).strip()
    else:
        # No section header — strip fenced diff blocks and treat the rest as plan.
        plan_md = _DIFF_FENCE_RE.sub("", text).strip()

    diffs: List[Dict[str, str]] = []
    for fm in _DIFF_FENCE_RE.finditer(text):
        body = (fm.group("body") or "").strip("\n")
        if not body:
            continue
        path = (fm.group("path") or "").strip()
        if not path:
            mp = re.search(r"^(?:---|\+\+\+)\s+[ab]/([^\s]+)", body, re.MULTILINE)
            path = mp.group(1) if mp else ""
        diffs.append({"file": path, "patch": body})

    citations = [{"chunk_id": cid} for cid in re.findall(r"\[\[([^\[\]]+)\]\]", text)]
    return {
        "plan_md": plan_md,
        "diffs": diffs,
        "citations": citations,
        "confidence": 0.55 if diffs else 0.4,
    }


# Re-exported for backward compatibility; shared with orchestrator.
_STOPWORDS = _SYMBOL_STOPWORDS

def _symbol_tokens(question: str) -> List[str]:
    """
    Extract candidate symbol tokens from a question, filtering out stopwords.
    Includes tokens inside quotes/backticks and bare identifiers.

    Preserves original-cased quoted tokens (e.g. CamelCase class names) when
    they appear, then appends any extra lowercased bare identifiers that
    survive the shared stopword + min-length filter.
    """
    quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", question or "")
    bare_lc = _extract_symbol_tokens(question or "")
    seen: set[str] = set()
    out: List[str] = []
    for t in quoted:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    for t in bare_lc:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

def _find_symbol_rows(indices: Dict[str, Any], symbol: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Return list of (chunk_id, row, view) that match the symbol by cid/name/text."""
    out: List[Tuple[str, Dict[str, Any], str]] = []
    sym_l = symbol.lower()
    pat_def = re.compile(rf"\b(def|class)\s+{re.escape(symbol)}\b")
    for view in ["intent", "impl"]:
        vw = (indices.get("views") or {}).get(view) or {}
        for r in (vw.get("rows") or []):
            cid = str(r.get("chunk_id", ""))
            name = str(r.get("name", "")).lower()
            text = r.get("text", "") or ""
            if (
                f"::{symbol}" in cid or f"::{sym_l}" in cid.lower()
                or name == sym_l
                or pat_def.search(text) is not None
            ):
                out.append((cid, r, view))
    seen, dedup = set(), []
    for cid, r, view in out:
        if cid not in seen:
            seen.add(cid); dedup.append((cid, r, view))
    return dedup

def _hits_from_records(indices: Dict[str, Any], records_path: Optional[str], symbol: Optional[str]) -> List[Dict[str, Any]]:
    if not records_path or not symbol:
        return []
    try:
        recs = load_jsonl(records_path)
    except Exception:
        return []
    target_ids = set()
    sym_l = symbol.lower()
    for rec in recs:
        nm = str(rec.get("name", "")).lower()
        if nm == sym_l:
            cid = rec.get("id")
            if cid is not None:
                target_ids.add(str(cid))
    if not target_ids:
        return []
    rows_by_cid = {}
    for view in ["intent", "impl"]:
        vw = (indices.get("views") or {}).get(view) or {}
        for r in (vw.get("rows") or []):
            rows_by_cid.setdefault(str(r.get("chunk_id")), []).append((view, r))
    hits: List[Dict[str, Any]] = []
    for cid in target_ids:
        for view_r in rows_by_cid.get(cid, []):
            view, _row = view_r
            hits.append({"chunk_id": cid, "score": 3.0, "view": view})
    return hits

def _sanitize_citations(citations, allowed_ids):
    out = []
    if not isinstance(citations, (list, tuple)):
        return out
    for c in citations:
        if isinstance(c, dict) and "chunk_id" in c and c["chunk_id"] in allowed_ids:
            out.append({"chunk_id": c["chunk_id"]})
        elif isinstance(c, str) and c in allowed_ids:
            out.append({"chunk_id": c})
    seen = set(); dedup = []
    for c in out:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"]); dedup.append(c)
    return dedup

# ---------------- main API ----------------

SYSTEM = (
    "You are a senior codebase assistant. Use ONLY the provided SOURCES to answer. "
    "Cite facts with [[chunk_id]] exactly as provided. Be concise but complete. "
    "If information is missing, say what else is needed rather than inventing details. "
    "Return JSON with keys: answer_md, citations, suggested_changes, confidence (0-1). "
    "Do not include prose outside JSON. "
) + ALLOWED_CITATION_NOTE


# Intent-conditioned system prompts. Each variant keeps the same JSON contract
# (answer_md, citations, suggested_changes, confidence) so downstream parsing
# stays uniform; only the framing and emphasis change.
SYSTEM_PROMPTS: Dict[str, str] = {
    "symbol_explain": (
        "You are a senior code reviewer explaining a specific symbol. "
        "Use ONLY the SOURCES. Structure answer_md as: Purpose, Signature, "
        "Parameters, Returns, Side effects, Key logic (with citations), "
        "Internal dependencies, Typical usage. Cite every non-trivial claim "
        "with [[chunk_id]]. Return JSON keys: answer_md, citations, "
        "suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "howto": (
        "You are a pragmatic guide for using this codebase. Use ONLY the "
        "SOURCES. answer_md should be a short numbered procedure followed by "
        "a minimal code example drawn from SOURCES. Cite each step with "
        "[[chunk_id]]. Return JSON keys: answer_md, citations, "
        "suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "change_plan": (
        "You are a principal engineer drafting a focused change plan. Use "
        "ONLY the SOURCES. answer_md should list: Goal, Affected files, "
        "Step-by-step edits, Tests to add/update, Risks. Cite each affected "
        "location with [[chunk_id]]. Return JSON keys: answer_md, citations, "
        "suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "symbol_location": (
        "You are a precise locator. Use ONLY the SOURCES. answer_md should "
        "list the file paths and line ranges where the symbol is defined or "
        "primarily implemented, one per line, each followed by a one-line "
        "rationale and a [[chunk_id]] citation. Return JSON keys: answer_md, "
        "citations, suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "line_number": (
        "You are a precise locator for edit anchors. Use ONLY the SOURCES. "
        "answer_md should list candidate (file, line_range) edit points with "
        "a one-line justification and a [[chunk_id]] citation each. Return "
        "JSON keys: answer_md, citations, suggested_changes, confidence. "
        "No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "overview": (
        "You are a senior codebase assistant. Use ONLY the SOURCES (and the "
        "optional README lead) to produce a concise repo overview: Purpose, "
        "Major components, How they fit together, Entry points. Cite each "
        "claim with [[chunk_id]]. Return JSON keys: answer_md, citations, "
        "suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
}


def _get_system_prompt(mode: str) -> str:
    """Return the system prompt for ``mode`` with a safe fallback to SYSTEM."""
    return SYSTEM_PROMPTS.get(mode, SYSTEM)

def answer_with_llm(
    index_dir: str,
    records_path: str,
    question: str,
    provider: LLMProvider,
    *,
    top_k: int = 20,
    hits: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Retrieve context from indices/graph and ask the LLM to synthesize a grounded answer.
    """
    indices = load_indices(index_dir)
    _ = load_jsonl(records_path) if records_path else None
    cmap = _chunk_map(indices)

    # Detect intent
    mode = detect_intent(question)

    # --- Improved Target symbol detection ---
    symbols = _symbol_tokens(question)
    target = None
    target_matched = False
    # 1. Prefer the first token that actually exists in the index
    for t in symbols:
        if _find_symbol_rows(indices, t):
            target = t
            target_matched = True
            break
    # 2. If none matched, try reversed order (favor last tokens like "parse_codebase")
    if target is None and symbols:
        for t in reversed(symbols):
            if _find_symbol_rows(indices, t):
                target = t
                target_matched = True
                break
    # 3. As last resort, pick the last token instead of the first. This is a
    #    best-effort focus hint only; the strict coverage gate below is skipped
    #    when target_matched is False so general/module/concept questions still
    #    reach the LLM with retrieval context.
    if target is None and symbols:
        target = symbols[-1]

    # Load graph if needed
    graph_path = Path(index_dir).parent / "graph.json"
    G = None
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges_key = "edges" if isinstance(data, dict) and "edges" in data else "links"
            G = json_graph.node_link_graph(data, edges=edges_key)
        except Exception:
            G = None

    # --- Graph-based answering for callers/callees ---
    if mode in {"callers_list", "callees_list"} and target and G is not None:
        results = []
        seen_nbrs: set[str] = set()
        try:
            target_nodes = [cid for cid, _row, _ in _find_symbol_rows(indices, target)]
            for node in target_nodes:
                if node not in G:
                    continue
                if mode == "callers_list":
                    edges = list(G.in_edges(node, data=True))
                    header = f"Functions that call `{target}`"
                    # neighbor is the source of an inbound edge
                    pairs = [(u, d) for (u, _v, d) in edges]
                else:
                    edges = list(G.out_edges(node, data=True))
                    header = f"Functions called by `{target}`"
                    # neighbor is the target of an outbound edge
                    pairs = [(v, d) for (_u, v, d) in edges]
                for nbr, edata in pairs:
                    # Edge data may itself be a dict-of-dicts (MultiDiGraph).
                    etype: Optional[str] = None
                    if isinstance(edata, dict):
                        if any(isinstance(v, dict) for v in edata.values()):
                            etype = next(
                                (v.get("type") for v in edata.values() if isinstance(v, dict) and v.get("type")),
                                None,
                            )
                        else:
                            etype = edata.get("type")
                    if etype != "calls":
                        continue
                    s = str(nbr)
                    if "::" not in s or s in seen_nbrs:
                        continue
                    seen_nbrs.add(s)
                    results.append({"chunk_id": s, "score": 1.0})
        except Exception:
            results = []
        if results:
            sources = _as_sources_with_meta(results, cmap, max_chunks=40, max_chars=900)
            return {
                "answer_md": header + ":\n\n" + "\n".join(
                    f"- {s['symbol']} ({s['path']})" for s in sources
                ),
                "citations": [{"chunk_id": s["chunk_id"]} for s in sources],
                "suggested_changes": [],
                "confidence": 0.9,
                "debug": {"mode": mode, "target_symbol": target, "graph_used": True, "sources": sources},
            }

    # --- Build/augment hits ---
    forced_hits: List[Dict[str, Any]] = []
    if target:
        for cid, _row, view in _find_symbol_rows(indices, target):
            forced_hits.append({"chunk_id": cid, "score": 2.0, "view": view})
        rec_hits = _hits_from_records(indices, records_path, target)
        seen = {str(h["chunk_id"]) for h in forced_hits}
        for h in rec_hits:
            if str(h["chunk_id"]) not in seen:
                forced_hits.append(h); seen.add(str(h["chunk_id"]))

    base_hits: List[Dict[str, Any]] = []
    if hits:
        base_hits = hits
    else:
        if not forced_hits:
            for name in ["intent", "impl"]:
                vw = (indices.get("views") or {}).get(name) or {}
                for r in (vw.get("rows") or [])[:top_k]:
                    base_hits.append({"chunk_id": r.get("chunk_id"), "score": 1.0, "view": name})

    seen = set()
    merged_hits: List[Dict[str, Any]] = []
    for h in forced_hits + base_hits:
        cid = str(h.get("chunk_id"))
        if cid not in seen:
            seen.add(cid); merged_hits.append(h)

    if not merged_hits:
        return {
            "answer_md": (
                "I couldn't locate matching symbols or chunks for this question in the current index. "
                "Re-index the repo and try again, or provide the file containing the target function/class."
            ),
            "citations": [],
            "suggested_changes": [],
            "confidence": 0.2,
            "debug": {"mode": mode, "target_symbol": target, "sources": [], "hits": []},
        }

    # --- SOURCES for LLM ---
    max_chars = 1400 if mode == "symbol_explain" else 900
    focus_terms: List[str] = []
    if target:
        focus_terms.append(target)
    for t in symbols:
        if t and t not in focus_terms:
            focus_terms.append(t)
    sources = _as_sources_with_meta(
        merged_hits,
        cmap,
        max_chunks=40 if mode == "symbol_explain" else 24,
        max_chars=max_chars,
        focus_terms=focus_terms or None,
    )

    # Require target coverage — only for symbol-targeted modes, and only when
    # target was set by a real index match. Conceptual modes like ``howto`` /
    # ``overview`` / ``change_plan`` extract incidental tokens (e.g. ``encode``
    # from "how to encode images") that are best-effort focus hints, not
    # mandatory symbols, so the gate would wrongly abstain on grounded sources.
    if target and target_matched and mode in _SYMBOL_TARGETED_MODES:
        covers = [
            s for s in sources
            if _symbol_covers_target(s.get("symbol", ""), s.get("chunk_id", ""), target)
        ]
        if not covers:
            return {
                "answer_md": (
                    f"I couldn't find the symbol `{target}` in the indexed chunks. "
                    "Please re-index or verify the symbol name/file."
                ),
                "citations": [],
                "suggested_changes": [],
                "confidence": 0.2,
                "debug": {"mode": mode, "target_symbol": target, "sources": sources, "hits": merged_hits},
            }

    root = _guess_root(indices)
    readme = _read_readme(root)

    context = "QUESTION:\n" + (question or "").strip() + "\n\n"
    if mode == "symbol_explain":
        context += (
            "TASK: Explain the function/class in detail. Cover: purpose, parameters & types (if visible), "
            "return value, side-effects, key branches/logic, dependencies (internal calls), and typical usage. "
            "Ground every claim with a citation.\n\n"
        )
    if readme and mode not in {"symbol_explain"}:
        lead_lines = [ln for ln in readme.splitlines() if ln.strip()][:12]
        context += "README (lead):\n" + "\n".join(lead_lines) + "\n\n"
    if target:
        context += f"TARGET_SYMBOL: {target}\n\n"
    context += "SOURCES:\n" + "\n".join(_fmt_source(s) for s in sources)

    messages = [
        {"role": "system", "content": _get_system_prompt(mode)},
        {"role": "user", "content": context},
    ]

    resp = provider.chat(messages, temperature=0.2)
    content = (resp.get("content") or "").strip()

    parsed: Dict[str, Any] = _extract_json_object(content)

    if not parsed or not isinstance(parsed, dict) or not parsed.get("answer_md"):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Reformat to strict JSON only. "
                                                   "Ensure non-empty 'answer_md' grounded in SOURCES with citations. "
                                                   "Keep the same content; do not add external knowledge."})
        resp2 = provider.chat(messages, temperature=0)
        parsed = _extract_json_object((resp2.get("content") or "")) or {"answer_md": content, "citations": []}

    ans = parsed.get("answer_md")
    if isinstance(ans, dict):
        parsed["answer_md"] = ans.get("content") or ans.get("text") or ans.get("markdown") or ans.get("md") or json.dumps(ans, ensure_ascii=False)
    elif isinstance(ans, list):
        parsed["answer_md"] = "\n".join(str(x) for x in ans)
    elif ans is None:
        parsed["answer_md"] = ""

    # Models in strict JSON mode sometimes obey the JSON contract but ignore
    # the requested key, returning shapes like {"data": {"text/plain": "..."}}
    # or {"text": "..."}. Pull the answer text out of those before abstaining.
    if not parsed["answer_md"].strip():
        coerced = _coerce_answer_text(parsed)
        if coerced.strip():
            parsed["answer_md"] = coerced

    if not parsed["answer_md"].strip():
        parsed["answer_md"] = (
            "The provided SOURCES did not contain enough content to explain this symbol without guessing. "
            "Please re-index or narrow the question to a specific file or snippet."
        )
        parsed.setdefault("citations", [])
        parsed.setdefault("suggested_changes", [])
        parsed["confidence"] = 0.2

    allowed_ids = [s['chunk_id'] for s in sources]
    parsed["citations"] = _sanitize_citations(parsed.get("citations", []), allowed_ids)
    parsed.setdefault("suggested_changes", [])
    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)):
        parsed["confidence"] = 0.6 if parsed["citations"] else 0.4

    parsed["debug"] = {
        "mode": mode,
        "target_symbol": target,
        "sources": sources,
        "hits": merged_hits,
        "readme_included": bool(readme),
    }

    return parsed


def generate_code_plan(
    index_dir: str,
    records_path: str,
    task: str,
    provider: LLMProvider,
    *,
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    chunks_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    top_k_per_view: int = 20,
    project_root: Optional[str] = None,
    self_test: bool = False,
    run_tests: bool = False,
    max_retries: int = 1,
    test_timeout_seconds: float = 120.0,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Use LLM to propose a change plan and diffs (unified patch) grounded in SOURCES.

    Unlike the previous implementation, this routes the task through the same
    hybrid retrieval (semantic + lexical + graph) used by `answer_with_llm`,
    so the SOURCES actually reflect the task description. It also surfaces
    suggested insertion points and impacted files for the planner to use.
    """
    # Lazy import to avoid any potential import cycles at module load.
    from cgx.pipeline.auto import run_query_auto

    # Auto-derive sibling artifacts when not provided.
    out_dir = Path(index_dir).parent
    if chunks_path is None:
        cp = out_dir / "chunks.jsonl"
        chunks_path = str(cp) if cp.exists() else None
    if graph_path is None:
        gp = out_dir / "graph.json"
        graph_path = str(gp) if gp.exists() else None

    retrieval = run_query_auto(
        index_dir=index_dir,
        records_path=records_path,
        query=task or "",
        model_name=model_name,
        chunks_path=chunks_path,
        graph_path=graph_path,
        top_k_per_view=top_k_per_view,
        neighbor_depth=1,
        use_lexical=True,
        embedder=embedder,
    )

    hits = retrieval.get("hits", []) or []
    anchors = retrieval.get("anchors", []) or []
    impact = retrieval.get("impact", []) or []
    top_files = retrieval.get("top_files", []) or []

    indices = load_indices(index_dir)
    cmap = _chunk_map(indices)
    task_focus = _symbol_tokens(task or "")
    sources = _as_sources_with_meta(
        hits, cmap, max_chunks=24, max_chars=900,
        focus_terms=task_focus or None,
    )

    SYSTEM2 = (
        "You are a principal engineer. Propose a step-by-step change plan and unified diffs "
        "to implement the requested change. Use ONLY provided SOURCES and cite with [[chunk_id]]. "
        "When INSERTION_POINTS are provided, prefer them as the locus of new code; when "
        "IMPACTED_FILES are provided, ensure your plan addresses each one. "
        "Return JSON with keys: plan_md, diffs (array of objects: file, patch), citations, confidence. "
        "Do not include prose outside JSON. "
        "STRICT DIFF FORMAT for every 'patch' value:\n"
        "  - Use relative POSIX paths (e.g. 'pkg/mod.py'), never absolute paths.\n"
        "  - For an EDIT to an existing file, start with:\n"
        "      --- a/<path>\n      +++ b/<path>\n      @@ -<old_start>,<old_len> +<new_start>,<new_len> @@\n"
        "    followed by context lines prefixed with ' ', removed lines with '-', added lines with '+'.\n"
        "  - For a NEW file, start with:\n"
        "      --- /dev/null\n      +++ b/<path>\n      @@ -0,0 +1,<N> @@\n"
        "    followed by every line of the new file prefixed with '+'.\n"
        "  - Every diff MUST contain at least one '@@' hunk header. Never emit raw file content without a hunk header.\n"
        "  - Do not duplicate the same path across multiple diff entries.\n"
    ) + ALLOWED_CITATION_NOTE

    parts: List[str] = []
    parts.append("TASK:\n" + (task or "").strip())

    if anchors:
        anchor_lines = []
        for a in anchors[:8]:
            ctype = a.get("container_type", "")
            cid = a.get("container_id", "")
            sc = a.get("score", 0.0)
            anc = a.get("anchors", {}) or {}
            lc = anc.get("likely_caller") or ""
            sn = anc.get("similar_signature_neighbor") or ""
            anchor_lines.append(
                f"- {ctype} {cid} (score={sc:.3f}) "
                f"likely_caller={lc} similar_signature_neighbor={sn}"
            )
        parts.append("INSERTION_POINTS:\n" + "\n".join(anchor_lines))

    if impact:
        impact_lines = []
        for it in impact[:12]:
            f = it.get("file", "")
            sc = it.get("score", 0.0)
            impact_lines.append(f"- {f} (score={sc:.3f})")
        parts.append("IMPACTED_FILES:\n" + "\n".join(impact_lines))
    elif top_files:
        tf_lines = [f"- {tf.get('file','')} (score={tf.get('score',0.0):.3f})" for tf in top_files[:10]]
        parts.append("CANDIDATE_FILES:\n" + "\n".join(tf_lines))

    parts.append("SOURCES:\n" + "\n".join(_fmt_source(s) for s in sources))
    context = "\n\n".join(parts)

    messages = [{"role": "system", "content": SYSTEM2}, {"role": "user", "content": context}]
    out_text = provider.chat(messages, temperature=0.2, force_json=True).get("content", "")
    parsed = _extract_json_object(out_text)
    # Fallback: JSON-mode often mangles unified diffs through backslash escaping
    # on small local models. Retry once in free-form mode and parse fenced blocks.
    if not parsed or not parsed.get("plan_md"):
        free_messages = [
            {"role": "system", "content": (
                "You are a principal engineer. Produce a change plan and unified diffs.\n"
                "Use ONLY provided SOURCES. Cite chunk_ids inline as [[chunk_id]].\n"
                "Format strictly as:\n"
                "## Plan\n<markdown plan>\n\n"
                "## Diffs\nFor each modified file, emit ONE fenced block:\n"
                "```diff path=<relative/path>\n<unified diff>\n```\n"
                "Every unified diff MUST include a hunk header line starting with '@@'.\n"
                "EDIT an existing file (example):\n"
                "```diff path=pkg/mod.py\n"
                "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,3 +1,4 @@\n"
                " def add(a, b):\n     return a + b\n+def mul(a, b):\n+    return a * b\n"
                "```\n"
                "NEW file (example):\n"
                "```diff path=pkg/extra.py\n"
                "--- /dev/null\n+++ b/pkg/extra.py\n@@ -0,0 +1,2 @@\n"
                "+def hello():\n+    return 'hi'\n"
                "```\n"
                "Rules: relative POSIX paths only, one fenced block per file, no duplicates, "
                "no prose between the fenced blocks.\n"
            )},
            {"role": "user", "content": context},
        ]
        free_text = provider.chat(free_messages, temperature=0.2, force_json=False).get("content", "")
        parsed = _parse_plan_freeform(free_text) or {
            "plan_md": free_text, "diffs": [], "citations": [], "confidence": 0.4
        }

    allowed_ids = [s['chunk_id'] for s in sources]
    parsed["citations"] = _sanitize_citations(parsed.get("citations", []), allowed_ids)
    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)):
        parsed["confidence"] = 0.5

    # ---------------- Optional: self-test loop -----------------
    # When the caller asks for validation/testing, we materialize the plan's
    # diffs in memory, run a syntax check, optionally run impacted tests in a
    # sandbox, and retry the LLM at most `max_retries` times with concrete
    # feedback. The final report is attached under parsed["codegen_report"].
    codegen_report: Optional[Dict[str, Any]] = None
    if self_test and project_root:
        try:
            from cgx.codegen.pipeline import validate_and_test, build_retry_feedback
            plan_text = _render_plan_for_validation(parsed)
            report = validate_and_test(
                project_root=project_root,
                plan_text=plan_text,
                run_tests=run_tests,
                timeout_seconds=test_timeout_seconds,
            )
            attempts = 0
            while (
                not report.summary.get("overall_ok")
                and attempts < max(0, int(max_retries))
            ):
                attempts += 1
                feedback = build_retry_feedback(report)
                retry_messages = [
                    {"role": "system", "content": (
                        "You are revising a previous code plan based on validation failures. "
                        "Keep the same goal. Use fenced ```diff path=<relative/path>``` blocks."
                    )},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": _render_plan_for_validation(parsed)},
                    {"role": "user", "content": feedback},
                ]
                retry_text = provider.chat(retry_messages, temperature=0.2, force_json=False).get("content", "")
                revised = _parse_plan_freeform(retry_text)
                if revised and revised.get("diffs"):
                    parsed["plan_md"] = revised.get("plan_md") or parsed.get("plan_md", "")
                    parsed["diffs"] = revised.get("diffs") or parsed.get("diffs", [])
                    plan_text = _render_plan_for_validation(parsed)
                    report = validate_and_test(
                        project_root=project_root,
                        plan_text=plan_text,
                        run_tests=run_tests,
                        timeout_seconds=test_timeout_seconds,
                    )
                else:
                    break
            codegen_report = report.to_dict()
            codegen_report["attempts"] = attempts
        except Exception as e:
            codegen_report = {"error": f"{type(e).__name__}: {e}"}

    parsed["debug"] = {
        "sources": sources,
        "hits": hits,
        "anchors": anchors,
        "impact": impact,
    }
    if codegen_report is not None:
        parsed["codegen_report"] = codegen_report
    return parsed


_CODE_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_]*\s+path\s*=\s*[\"']?(?P<path>[^\s\"'`\n]+)[\"']?\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _content_to_new_file_patch(path: str, content: str) -> str:
    """Convert complete file content into a new-file unified diff (--- /dev/null style)."""
    lines = content.splitlines()
    n = len(lines)
    header = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{n} @@\n"
    body = "\n".join(f"+{line}" for line in lines)
    return header + (body if body else "+")


def _parse_scaffold_freeform(text: str) -> Dict[str, Any]:
    """Parse free-form scaffold response with fenced ``code path=...`` blocks.

    Accepts any language tag followed by ``path=<relative/path>``.
    Citations are extracted from ``[[chunk_id]]`` markers.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    plan_md = ""
    m = re.search(r"##\s*Plan\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        plan_md = m.group(1).strip()
    else:
        plan_md = _CODE_FENCE_RE.sub("", text).strip()

    diffs: List[Dict[str, str]] = []
    for fm in _CODE_FENCE_RE.finditer(text):
        path = (fm.group("path") or "").strip()
        body = (fm.group("body") or "").rstrip("\n")
        if not path or not body:
            continue
        # If the body already looks like a unified diff, use it as-is;
        # otherwise convert the raw content into a new-file diff.
        if "--- " in body or "+++ " in body or "@@" in body:
            diffs.append({"file": path, "patch": body})
        else:
            diffs.append({"file": path, "patch": _content_to_new_file_patch(path, body)})

    citations = [{"chunk_id": cid} for cid in re.findall(r"\[\[([^\[\]]+)\]\]", text)]
    return {
        "plan_md": plan_md,
        "diffs": diffs,
        "citations": citations,
        "confidence": 0.55 if diffs else 0.3,
    }


_SCAFFOLD_SYSTEM = (
    "You are a senior software architect generating a complete new project from scratch.\n\n"
    "Given a project idea, design and implement the full codebase.\n\n"
    "Return strict JSON only:\n"
    "{\n"
    '  "plan_md": "## <Project Name>\\n\\n<architecture overview, directory structure, key design decisions>",\n'
    '  "files": [\n'
    '    {"path": "relative/posix/path", "content": "<complete file content>"},\n'
    "    ...\n"
    "  ],\n"
    '  "confidence": 0.8\n'
    "}\n\n"
    "Requirements:\n"
    "- path: relative POSIX path from the project root (e.g. 'src/main.py')\n"
    "- content: complete, working file content — NOT stubs or placeholders\n"
    "- CRITICAL: If the goal names a specific technology or framework (React, Vue, Angular, "
    "FastAPI, Flask, Django, Express, etc.), you MUST use ONLY that technology. "
    "Do NOT mix in other frameworks or substitute a different one.\n"
    "- Include ALL files needed to run the project:\n"
    "  * Main application code (all modules)\n"
    "  * Configuration file appropriate for the technology "
    "(package.json for JS/TS/React/Vue/Node, requirements.txt for Python, "
    "pyproject.toml for modern Python, Cargo.toml for Rust, go.mod for Go, etc.)\n"
    "  * README.md with setup and usage instructions\n"
    "  * Entry point / main module\n"
    "- For FRONTEND projects (React, Vue, Angular, Svelte, etc.):\n"
    "  * Generate actual component files (e.g. src/App.jsx, src/components/MyComponent.jsx)\n"
    "  * Include index.html (in public/ for CRA, or root for Vite)\n"
    "  * Include src/index.js or src/main.jsx as the React/Vue entry point\n"
    "  * Do NOT generate webpack.config.js, babel.config.js, or other build tooling — "
    "use Create React App (react-scripts) or Vite conventions instead\n"
    "  * Do NOT include conftest.py, requirements.txt, or any Python files\n"
    "- For PYTHON projects only: include a conftest.py at the project root "
    "that adds src/ to sys.path so test imports work without package installation:\n"
    "  conftest.py content: import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))\n"
    "  Test files must import from module names as they exist under src/ "
    "(e.g. 'from main import app', not 'from src.main import app').\n"
    "- Generate production-quality code with proper error handling and type hints where idiomatic.\n"
    "- Do NOT use TODO, FIXME, or placeholder comments — write the real code.\n"
    "- Do not include prose outside JSON.\n"
)

_SCAFFOLD_FREEFORM_SYSTEM = (
    "You are a senior software architect generating a complete new project.\n\n"
    "Format your response as:\n\n"
    "## Plan\n"
    "<architecture overview and directory structure>\n\n"
    "## Files\n\n"
    "For each file, emit ONE fenced block with a path= annotation:\n"
    "```<language> path=<relative/path/to/file>\n"
    "<complete file content>\n"
    "```\n\n"
    "Examples of valid language tags: python, javascript, jsx, tsx, typescript, text, yaml, toml, json\n\n"
    "Rules:\n"
    "- Use any language tag (python, javascript, jsx, tsx, text, yaml, toml, etc.) followed by path=<relative/path>\n"
    "- Generate complete, working code — not stubs\n"
    "- Include README.md and the appropriate dependency file for the technology\n"
    "- Use relative POSIX paths only\n"
    "- CRITICAL: If the goal names a specific technology (React, Vue, FastAPI, etc.), use it exactly. "
    "Do NOT substitute a different technology.\n"
    "- For FRONTEND projects (React, Vue, etc.): generate component files (App.jsx, index.js, etc.), "
    "NOT webpack/babel config files. Do NOT include conftest.py or requirements.txt.\n"
    "- For PYTHON projects only: include a conftest.py at the root that does:\n"
    "  import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))\n"
)


def generate_project_scaffold(
    idea: str,
    provider: Any,
    *,
    project_root: Optional[str] = None,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a complete new project from a plain-language idea.

    Unlike :func:`generate_code_plan`, this function does not need an
    existing index — it generates all project files from scratch using
    the LLM alone.  The returned ``diffs`` list uses ``--- /dev/null``
    new-file unified diffs so the existing ``apply_diffs_to_disk``
    pipeline can write them to ``project_root`` without any special
    handling.

    Parameters
    ----------
    idea:
        Natural-language description of the project to create.
    provider:
        LLM provider instance.
    project_root:
        Destination directory (forwarded to the caller for APPLY; not
        used by this function directly).
    """
    if provider is None:
        return {
            "plan_md": "No LLM provider available; cannot scaffold project.",
            "diffs": [],
            "confidence": 0.0,
        }

    idea_clean = (idea or "").strip()
    goal_clean = (goal or "").strip()
    if goal_clean and goal_clean != idea_clean:
        context = (
            f"ORIGINAL USER GOAL:\n{goal_clean}\n\n"
            f"TASK DESCRIPTION:\n{idea_clean}"
        )
    else:
        context = f"PROJECT IDEA:\n{idea_clean}"
    messages = [
        {"role": "system", "content": _SCAFFOLD_SYSTEM},
        {"role": "user", "content": context},
    ]
    raw = provider.chat(messages, temperature=0.3, force_json=True).get("content", "")
    parsed = _extract_json_object(raw)

    # Fallback: local models often produce free-form text even with JSON mode on.
    if not parsed or not parsed.get("files"):
        ff_messages = [
            {"role": "system", "content": _SCAFFOLD_FREEFORM_SYSTEM},
            {"role": "user", "content": context},
        ]
        ff_text = provider.chat(ff_messages, temperature=0.3, force_json=False).get("content", "")
        parsed = _parse_scaffold_freeform(ff_text) or {
            "plan_md": ff_text, "diffs": [], "confidence": 0.3,
        }
        return parsed

    # Convert {path, content} file list → {file, patch} unified diffs.
    files = parsed.get("files") or []
    diffs: List[Dict[str, str]] = []
    seen: set = set()
    for f in files:
        if not isinstance(f, dict):
            continue
        path = str(f.get("path") or f.get("file") or "").strip()
        content = str(f.get("content") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        diffs.append({"file": path, "patch": _content_to_new_file_patch(path, content)})

    return {
        "plan_md": str(parsed.get("plan_md") or "").strip(),
        "diffs": diffs,
        "citations": [],
        "confidence": float(parsed.get("confidence") or 0.7) if diffs else 0.3,
    }


def _render_plan_for_validation(parsed: Dict[str, Any]) -> str:
    """Render a parsed plan back into fenced-diff form for the validator."""
    parts: List[str] = []
    pm = parsed.get("plan_md")
    if isinstance(pm, str) and pm.strip():
        parts.append("## Plan")
        parts.append(pm.strip())
    diffs = parsed.get("diffs") or []
    if isinstance(diffs, list):
        parts.append("## Diffs")
        for d in diffs:
            if not isinstance(d, dict):
                continue
            path = d.get("file") or d.get("path") or ""
            patch = d.get("patch") or d.get("diff") or ""
            if path and patch:
                parts.append(f"```diff path={path}")
                parts.append(patch.rstrip())
                parts.append("```")
    return "\n".join(parts)
