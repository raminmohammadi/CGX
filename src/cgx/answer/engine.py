

from __future__ import annotations
import logging
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

logger = logging.getLogger(__name__)

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

def _shorten_chunk_refs(text: str, root: Optional[str]) -> str:
    """Rewrite ``[[<chunk_id>]]`` tokens in LLM output to drop the project-root
    prefix so user-visible citations are compact and don't leak ``$HOME``.

    Chunk ids are absolute paths internally (``/home/alice/repo/foo.py::cls::Bar``).
    Citations in ``answer_md`` are rendered to the user verbatim, which makes
    the prefix both noisy and a small privacy leak. Stripping is purely
    cosmetic — ``citations`` and ``debug.sources`` still carry the full ids.
    """
    if not text or not root:
        return text or ""
    prefix = root.rstrip("/") + "/"

    def _sub(m: "re.Match[str]") -> str:
        inner = m.group(1)
        if inner.startswith(prefix):
            inner = inner[len(prefix):]
        return f"[[{inner}]]"

    return re.sub(r"\[\[([^\[\]]+)\]\]", _sub, text)


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
    tier = s.get("tier") or ""
    if tier == "neighbor":
        extras.append("tier=neighbor")
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

    # When the orchestrator surfaced graph-expanded neighbors, switch to the
    # tiered context builder so the prompt spends its budget on full bodies
    # for primary hits and compact stubs for neighbors. Profile-driven
    # budgets keep us off magic numbers per call site.
    _has_neighbors = any(
        int(((h.get("provenance") or {}) if isinstance(h, dict) else {}).get("graph_depth", 0) or 0) >= 1
        for h in merged_hits
    )
    if _has_neighbors:
        from cgx.answer.context_map import build_tiered_context, load_records_by_id
        from cgx.answer.model_caps import get_context_map_budget
        budget = get_context_map_budget(provider)
        sources = build_tiered_context(
            merged_hits, cmap, load_records_by_id(records_path),
            budget=budget, focus_terms=focus_terms or None,
        )
    else:
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

    parsed["answer_md"] = _shorten_chunk_refs(parsed.get("answer_md", ""), root)

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
    skills: Optional[List[str]] = None,
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
    # Tiered context kicks in when the orchestrator surfaced graph neighbors,
    # so plan prompts spend their budget on full bodies for primary hits and
    # compact stubs for graph-expanded neighbors.
    _has_neighbors = any(
        int(((h.get("provenance") or {}) if isinstance(h, dict) else {}).get("graph_depth", 0) or 0) >= 1
        for h in hits
    )
    if _has_neighbors:
        from cgx.answer.context_map import build_tiered_context, load_records_by_id
        from cgx.answer.model_caps import get_context_map_budget
        budget = get_context_map_budget(provider)
        sources = build_tiered_context(
            hits, cmap, load_records_by_id(records_path),
            budget=budget, focus_terms=task_focus or None,
        )
    else:
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

    # Compose skill-specific plan guidance onto SYSTEM2 / the freeform
    # fallback system prompt. Skills are resolved either from the
    # caller-supplied ``skills`` kwarg (Planner-attached) or by detecting
    # them from the task text.
    active_skills = _resolve_skills(skills, task or "")
    try:
        import skills as _sk
        skill_fragment = _sk.compose_plan_prompt(active_skills)
        skill_names_str = ", ".join(s.name for s in active_skills)
    except Exception:  # pragma: no cover - defensive
        skill_fragment = ""
        skill_names_str = ""
    system2 = SYSTEM2
    if skill_fragment:
        system2 = (
            SYSTEM2
            + f"\n\nACTIVE SKILLS: {skill_names_str}\n"
            + "Apply the technology-specific guidance below in addition to "
            + "the rules above.\n\n"
            + skill_fragment
        )

    messages = [{"role": "system", "content": system2}, {"role": "user", "content": context}]
    out_text = provider.chat(messages, temperature=0.2, force_json=True).get("content", "")
    parsed = _extract_json_object(out_text)
    # Fallback: JSON-mode often mangles unified diffs through backslash escaping
    # on small local models. Retry once in free-form mode and parse fenced blocks.
    if not parsed or not parsed.get("plan_md"):
        freeform_system = (
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
        )
        if skill_fragment:
            freeform_system = (
                freeform_system
                + f"\nACTIVE SKILLS: {skill_names_str}\n"
                + skill_fragment
                + "\n"
            )
        free_messages = [
            {"role": "system", "content": freeform_system},
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
    "Path discipline (CRITICAL — read carefully):\n"
    "- The deployment directory IS the project root. Emit paths RELATIVE to it.\n"
    "- NEVER prepend a top-level project folder. WRONG: 'calculator/src/App.jsx', "
    "'my-project/backend/app.py'. RIGHT: 'src/App.jsx', 'backend/app.py'.\n"
    "- Use this canonical layout so sibling scaffold tasks (UI + backend + tests) "
    "share one coherent tree:\n"
    "    src/        — frontend source OR main code for single-language projects\n"
    "    backend/    — Python/Node backend service (only when the project has a "
    "separate backend distinct from the frontend in src/)\n"
    "    tests/      — ALL test files live here (test_*.py for Python, *.test.jsx "
    "or *.test.ts for JS/TS). REQUIRED — every scaffold MUST emit at least one "
    "test file covering its primary logic.\n"
    "    public/     — static assets (index.html, favicons) for frontend projects\n"
    "- All paths must be lowercase-with-underscores or kebab-case. No spaces.\n"
    "Requirements:\n"
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
    "  * At least one real test file under tests/ exercising the main code paths.\n"
    "- For FRONTEND projects (React, Vue, Angular, Svelte, etc.):\n"
    "  * Generate actual component files (e.g. src/App.jsx, src/components/MyComponent.jsx)\n"
    "  * Include public/index.html (for CRA) or index.html at root (for Vite)\n"
    "  * Include src/index.js or src/main.jsx as the React/Vue entry point\n"
    "  * Tests go under tests/ as <Component>.test.jsx using Jest + "
    "@testing-library/react conventions.\n"
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
    "Path discipline (CRITICAL):\n"
    "- Paths are RELATIVE to the project root. NEVER prepend a top-level project "
    "folder. WRONG: 'calculator/src/App.jsx'. RIGHT: 'src/App.jsx'.\n"
    "- Canonical layout shared with sibling tasks: src/ (frontend or main code), "
    "backend/ (Python backend when distinct), tests/ (REQUIRED — at least one "
    "test file per scaffold), public/ (static assets).\n"
    "Rules:\n"
    "- Use any language tag (python, javascript, jsx, tsx, text, yaml, toml, etc.) followed by path=<relative/path>\n"
    "- Generate complete, working code — not stubs\n"
    "- Include README.md and the appropriate dependency file for the technology\n"
    "- Emit at least one real test file under tests/ exercising the main logic.\n"
    "- Use relative POSIX paths only\n"
    "- CRITICAL: If the goal names a specific technology (React, Vue, FastAPI, etc.), use it exactly. "
    "Do NOT substitute a different technology.\n"
    "- For FRONTEND projects (React, Vue, etc.): generate component files (App.jsx, index.js, etc.), "
    "NOT webpack/babel config files. Do NOT include conftest.py or requirements.txt.\n"
    "- For PYTHON projects only: include a conftest.py at the root that does:\n"
    "  import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))\n"
)


def _resolve_skills(skills: Optional[List[str]],
                    goal: str) -> List[Any]:
    """Resolve a ``skills`` kwarg (or detect from ``goal``) to Skill objects.

    When ``skills`` is a non-empty list of names, look them up in the
    registry. Otherwise fall back to running detection over ``goal``.
    Both paths are silent on the ``skills`` package being unavailable.
    """
    try:
        import skills as _sk
    except Exception:  # pragma: no cover - defensive
        return []
    if skills:
        return _sk.skills_by_names(list(skills))
    if goal:
        return _sk.detect_skills(goal)
    return []


_CANONICAL_TOP_DIRS = (
    "src", "backend", "tests", "public", "docs", "scripts",
)


def _normalize_scaffold_path(path: str, existing_files: Optional[List[str]]) -> str:
    """Strip stray top-level project folders the LLM may have prepended.

    The SCAFFOLD prompt forbids paths like ``calculator/src/App.jsx`` but
    weak local models frequently emit them anyway. This collapses the
    first segment to the canonical layout (``src/``, ``backend/``, ...)
    when the LLM prepended a non-canonical root, while leaving paths
    that already begin at the canonical root untouched.

    When ``existing_files`` is supplied we also honour any top-level
    directory a sibling scaffold task has already established, so a
    later task can extend that layout rather than relocating into a
    different parent.
    """
    if not path:
        return path
    # Strip a literal "./" prefix and leading slashes, but NOT a bare
    # leading "." — otherwise dotfiles like ".env.example" / ".gitignore"
    # lose their leading dot and stop being dotfiles on disk.
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if "/" not in p:
        return p
    first, rest = p.split("/", 1)
    first_lc = first.lower()
    if first_lc in _CANONICAL_TOP_DIRS:
        return p
    if existing_files:
        established = {f.split("/", 1)[0].lower()
                       for f in existing_files if "/" in f}
        if first_lc in established:
            return p
    # Drop the inferred project-name prefix; downstream paths win.
    return rest


def _extension_framework_pin(path: str) -> str:
    """Return a per-extension framework constraint block for the single-file
    scaffold system prompt, or ``""`` when the extension has no pin.

    This is the prompt-side companion to :func:`_extension_content_mismatch`:
    the validator catches cross-framework substitutions after generation,
    and this hard-pins the prompt so they're far less likely to occur in
    the first place. It runs regardless of skill detection so a missing or
    typo'd framework name in the goal can't strip the constraint.
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("jsx", "tsx"):
        return (
            f"FILE EXTENSION CONSTRAINT (.{ext}):\n"
            f"- This is a React component file. Write it as JSX/TSX only.\n"
            f"- MUST import from 'react' (e.g. `import React from 'react'` "
            f"or named hook imports) and export a function component.\n"
            f"- Do NOT emit Vue SFC syntax: no `<template>`, no `<script "
            f"setup>`, no `<style scoped>`, no `import ... from 'vue'`.\n"
            f"- Do NOT emit Svelte syntax or a full HTML document.\n"
            f"- Use functional components with hooks (useState, useEffect)."
        )
    if ext == "vue":
        return (
            "FILE EXTENSION CONSTRAINT (.vue):\n"
            "- This is a Vue Single-File Component. Include a `<template>` "
            "block and a `<script setup>` block.\n"
            "- Do NOT import from 'react' and do NOT emit JSX."
        )
    if ext == "svelte":
        return (
            "FILE EXTENSION CONSTRAINT (.svelte):\n"
            "- This is a Svelte component. Use `<script>` + markup + "
            "`<style>` blocks.\n"
            "- Do NOT emit React JSX or Vue SFC syntax."
        )
    return ""


def _extension_content_mismatch(path: str, ext: str, content: str) -> Optional[str]:
    """Return a short error message when ``content`` looks wrong for ``ext``.

    Catches the common cross-framework substitutions a small local model
    makes (Vue SFC under ``.jsx``, React under ``.vue``, plain JS under
    ``.py``). Conservative: only reports on strong signals so we don't
    block valid edge cases.
    """
    if not content:
        return None
    lc = content.lower()
    has_vue_tpl = "<template" in lc
    has_vue_import = "from 'vue'" in lc or 'from "vue"' in lc
    has_react_import = ("from 'react'" in lc or 'from "react"' in lc
                        or "import react" in lc)
    has_react_jsx_export = ("export default function" in lc
                            and ("return (" in lc or "return <" in lc))
    has_svelte_block = "<script" in lc and "</script>" in lc and "<style" in lc

    if ext in ("jsx", "tsx"):
        if has_vue_tpl or has_vue_import:
            return (f"{ext} file contains Vue SFC syntax "
                    "(<template> / import from 'vue').")
        # A .jsx/.tsx file must look like React — at minimum it should
        # mention React or export a component. A 3B model occasionally
        # emits a plain HTML document under .jsx; catch that.
        is_html_doc = lc.lstrip().startswith(("<!doctype", "<html"))
        has_react_signal = (
            has_react_import
            or has_react_jsx_export
            or "export default" in lc
            or "export {" in lc        # named exports: export { App }
            or "return <" in lc        # JSX return statement — strong React indicator
        )
        if is_html_doc or not has_react_signal:
            return (f".{ext} file does not look like React "
                    "(no React import / component export).")
    if ext in ("js", "ts", "mjs"):
        # A small model commonly emits a full Vue SFC into a .js file
        # because nothing in the prompt forbids it. Catch the obvious
        # signal: a <template> block or a Vue import in plain JS.
        if has_vue_tpl or has_vue_import:
            return (f"{ext} file contains Vue SFC syntax "
                    "(<template> / import from 'vue').")
    if ext == "vue":
        if not has_vue_tpl:
            return ".vue file is missing a <template> block."
        if has_react_import and not has_vue_import:
            return ".vue file imports React instead of Vue."
    if ext == "svelte":
        if not has_svelte_block and not has_vue_tpl:
            # Svelte files usually contain a <script> and template-ish HTML.
            if has_react_import:
                return ".svelte file imports React."
    if ext == "py":
        # Reject content that's clearly JS/TS pretending to be Python.
        first = content.lstrip().splitlines()[0] if content.strip() else ""
        if first.startswith(("import {", "const ", "let ", "var ", "function ")):
            return ".py file contains JavaScript-style syntax."
        # Python test files occasionally come back as JavaScript bodies
        # that happen to parse as Python (`x = document.getElementById(...)`).
        # Catch the common JS DOM tokens; restrict to test files so we
        # don't false-positive on legitimate Python modules that mention
        # those strings.
        base = path.rsplit("/", 1)[-1].lower()
        if base.startswith("test_") or base.endswith("_test.py"):
            js_tokens = (
                "document.getelementbyid", "document.queryselector",
                "addeventlistener", ".click()", "console.log",
            )
            if any(tok in lc for tok in js_tokens):
                return ("python test file contains JavaScript DOM calls "
                        "(document.*, addEventListener, …).")
    if ext in ("css", "scss", "sass", "less"):
        # The model occasionally leaks the file's extension as the first
        # token (".css\nbody { … }"). Reject that pattern — a real
        # stylesheet starts with a selector, @rule, or a comment.
        stripped = content.lstrip()
        first_line = stripped.splitlines()[0].strip() if stripped else ""
        if first_line.lower() in (f".{ext}", f".{ext};"):
            return (f".{ext} file starts with a literal '.{ext}' token "
                    "(filename leakage).")
    return None


def generate_project_scaffold(
    idea: str,
    provider: Any,
    *,
    project_root: Optional[str] = None,
    goal: Optional[str] = None,
    skills: Optional[List[str]] = None,
    existing_files: Optional[List[str]] = None,
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
    skills:
        Optional list of skill names (from the Planner's
        ``task.inputs['skills']``). When supplied, the matching skill
        prompt fragments are appended to the scaffold system prompt.
        When omitted, skills are auto-detected from ``goal``/``idea``.
    existing_files:
        Paths produced by sibling SCAFFOLD tasks earlier in the same
        plan. Surfaced to the LLM so it doesn't regenerate them or
        invent a parallel directory tree, and used to normalise stray
        top-level prefixes the LLM may emit.
    """
    if provider is None:
        return {
            "plan_md": "No LLM provider available; cannot scaffold project.",
            "diffs": [],
            "confidence": 0.0,
        }

    idea_clean = (idea or "").strip()
    goal_clean = (goal or "").strip()
    detect_text = goal_clean or idea_clean
    active_skills = _resolve_skills(skills, detect_text)
    try:
        import skills as _sk
        skill_fragment = _sk.compose_scaffold_prompt(active_skills)
        skill_names_str = ", ".join(s.name for s in active_skills)
    except Exception:  # pragma: no cover - defensive
        skill_fragment = ""
        skill_names_str = ""

    json_system = _SCAFFOLD_SYSTEM
    freeform_system = _SCAFFOLD_FREEFORM_SYSTEM
    if skill_fragment:
        # Skills get appended so the generic scaffold rules apply first
        # and the technology-specific layout instructions win on conflict
        # (later fragments override earlier ones in LLM prompting).
        header = (
            f"\n\nACTIVE SKILLS: {skill_names_str}\n"
            "Apply the technology-specific instructions below in addition "
            "to the rules above.\n\n"
        )
        json_system = _SCAFFOLD_SYSTEM + header + skill_fragment
        freeform_system = _SCAFFOLD_FREEFORM_SYSTEM + header + skill_fragment

    parts: List[str] = []
    if goal_clean and goal_clean != idea_clean:
        parts.append(f"ORIGINAL USER GOAL:\n{goal_clean}")
        parts.append(f"TASK DESCRIPTION:\n{idea_clean}")
    else:
        parts.append(f"PROJECT IDEA:\n{idea_clean}")
    if existing_files:
        # Cap the list — local models can't reason over 200+ paths and the
        # context window starts to crowd the actual instructions.
        listed = "\n".join(f"- {p}" for p in list(existing_files)[:60])
        parts.append(
            "EXISTING FILES (already generated by sibling scaffold tasks). "
            "Do NOT regenerate any of these; do NOT relocate them into a "
            "different parent directory; only emit NEW files that complement "
            "this existing tree:\n" + listed
        )
    context = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": json_system},
        {"role": "user", "content": context},
    ]
    raw = provider.chat(messages, temperature=0.3, force_json=True).get("content", "")
    parsed = _extract_json_object(raw)

    # Fallback: local models often produce free-form text even with JSON mode on.
    if not parsed or not parsed.get("files"):
        ff_messages = [
            {"role": "system", "content": freeform_system},
            {"role": "user", "content": context},
        ]
        ff_text = provider.chat(ff_messages, temperature=0.3, force_json=False).get("content", "")
        parsed = _parse_scaffold_freeform(ff_text) or {
            "plan_md": ff_text, "diffs": [], "confidence": 0.3,
        }
        # Normalise freeform diffs through the same path discipline.
        for d in parsed.get("diffs") or []:
            if isinstance(d, dict) and d.get("file"):
                d["file"] = _normalize_scaffold_path(str(d["file"]), existing_files)
        return parsed

    # Convert {path, content} file list → {file, patch} unified diffs.
    files = parsed.get("files") or []
    diffs: List[Dict[str, str]] = []
    seen: set = set()
    existing_set = set(existing_files or [])
    for f in files:
        if not isinstance(f, dict):
            continue
        raw_path = str(f.get("path") or f.get("file") or "").strip()
        content = str(f.get("content") or "")
        path = _normalize_scaffold_path(raw_path, existing_files)
        if not path or path in seen or path in existing_set:
            continue
        seen.add(path)
        diffs.append({"file": path, "patch": _content_to_new_file_patch(path, content)})

    return {
        "plan_md": str(parsed.get("plan_md") or "").strip(),
        "diffs": diffs,
        "citations": [],
        "confidence": float(parsed.get("confidence") or 0.7) if diffs else 0.3,
    }


_MANIFEST_SYSTEM = (
    "You are a senior software architect planning a new project.\n\n"
    "Your job is to OUTPUT ONLY A FILE MANIFEST — paths and one-line descriptions. "
    "Do NOT write any file contents.\n\n"
    "Return strict JSON only:\n"
    "{\n"
    '  "plan_md": "2-4 sentence architecture overview",\n'
    '  "layers": [\n'
    '    {\n'
    '      "name": "core|ui|config|tests",\n'
    '      "files": [\n'
    '        {"path": "src/foo.py", "description": "one-line purpose"}\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Relative POSIX paths only. No top-level project-name prefix "
    "(wrong: calculator/src/App.jsx, right: src/App.jsx).\n"
    "- Group files by layer: core logic, UI, config/packaging, tests.\n"
    "- Test files REQUIRED under tests/.\n"
    "- 3 to 15 files total. Prefer completeness over brevity.\n"
    "- Canonical top-level dirs: src/, backend/, tests/, public/, docs/.\n"
)


def plan_scaffold_manifest(
    idea: str,
    provider: Any,
    *,
    goal: Optional[str] = None,
    skills: Optional[List[str]] = None,
    existing_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a file manifest for a new project — paths and descriptions only, no content.

    This is the lightweight first step of the manifest-first scaffold flow.
    The returned ``layers`` list is consumed by ``loop.py`` to dynamically
    inject one ``SCAFFOLD_FILE`` task per file into the running plan.
    """
    if provider is None:
        return {"plan_md": "No LLM provider available.", "layers": []}

    idea_clean = (idea or "").strip()
    goal_clean = (goal or "").strip()
    detect_text = goal_clean or idea_clean
    active_skills = _resolve_skills(skills, detect_text)
    try:
        import skills as _sk
        skill_fragment = _sk.compose_scaffold_prompt(active_skills)
        skill_names_str = ", ".join(s.name for s in active_skills)
    except Exception:
        skill_fragment = ""
        skill_names_str = ""

    system = _MANIFEST_SYSTEM
    if skill_fragment:
        header = (
            f"\n\nACTIVE SKILLS: {skill_names_str}\n"
            "Apply the technology-specific file layout below.\n\n"
        )
        system = _MANIFEST_SYSTEM + header + skill_fragment

    parts: List[str] = []
    if goal_clean and goal_clean != idea_clean:
        parts.append(f"ORIGINAL USER GOAL:\n{goal_clean}")
        parts.append(f"TASK DESCRIPTION:\n{idea_clean}")
    else:
        parts.append(f"PROJECT IDEA:\n{idea_clean}")
    if existing_files:
        listed = "\n".join(f"- {p}" for p in list(existing_files)[:60])
        parts.append("EXISTING FILES (already planned — do NOT repeat):\n" + listed)

    def _call(user_msg: str) -> Dict[str, Any]:
        # Manifest generation is a structural step validated by a
        # deterministic Judge (required files, layer shape). Sampling
        # variance here turns the same prompt into different file trees
        # across retries and makes pass/fail outcomes flaky, so we pin
        # the temperature to 0.
        resp = provider.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=6000,
            force_json=True,
        )
        if isinstance(resp, dict) and resp.get("error"):
            logger.warning("plan_scaffold_manifest: provider returned error — %s",
                           resp.get("error"))
        raw = (resp or {}).get("content", "") if isinstance(resp, dict) else ""
        return _extract_json_object(raw) or {}

    def _layers_have_files(p: Dict[str, Any]) -> bool:
        ls = p.get("layers")
        if not isinstance(ls, list) or not ls:
            return False
        return any(
            isinstance(l, dict) and (l.get("files") or [])
            for l in ls
        )

    context = "\n\n".join(parts)
    parsed = _call(context)
    # Small models occasionally emit empty layers when given a verbose
    # retry prompt; fall back to a minimal idea-only prompt before
    # surrendering to the empty-layer placeholder.
    if not _layers_have_files(parsed) and goal_clean:
        retry_context = f"PROJECT IDEA:\n{idea_clean[:600]}"
        if existing_files:
            listed = "\n".join(f"- {p}" for p in list(existing_files)[:60])
            retry_context += "\n\nEXISTING FILES (already planned — do NOT repeat):\n" + listed
        parsed = _call(retry_context)
    if not parsed or not isinstance(parsed.get("layers"), list):
        # Fallback: a single generic layer so the flow can still proceed.
        return {
            "plan_md": str(parsed.get("plan_md") or idea_clean),
            "layers": [{"name": "project", "files": []}],
        }
    layers = _normalize_manifest_paths(parsed["layers"])
    layers = _inject_required_manifest_files(
        layers,
        goal=goal_clean or idea_clean,
        skill_names=skills,
    )
    layers = _inject_python_package_inits(layers)
    return {
        "plan_md": str(parsed.get("plan_md") or "").strip(),
        "layers": layers,
    }


def _inject_required_manifest_files(
    layers: List[Any],
    *,
    goal: str = "",
    skill_names: Optional[List[str]] = None,
) -> List[Any]:
    """Ensure required framework files appear in the manifest.

    Small models frequently omit config files like ``package.json`` even
    when the skill prompt instructs them to include it.  Rather than
    burning retries on a deterministic omission, we inject a placeholder
    entry here so the Judge always sees a well-formed manifest.

    Injected files carry a description so the per-file generator knows
    what to produce; the file's actual content is generated later by the
    ``SCAFFOLD_FILE`` capability.
    """
    existing: set = set()
    existing_paths: set = set()
    for lay in layers or []:
        if not isinstance(lay, dict):
            continue
        for f in (lay.get("files") or []):
            if isinstance(f, dict):
                p = str(f.get("path") or "").strip()
                if p:
                    existing.add(p.lower().rsplit("/", 1)[-1])
                    existing_paths.add(p)

    goal_low = (goal or "").lower()
    names_low = {s.lower() for s in (skill_names or [])}
    to_inject: Dict[str, str] = {}  # path → description

    _JS_STACKS = re.compile(
        r"\b(react|vue|svelte|next\.?js|express|angular)\b", re.IGNORECASE
    )
    if _JS_STACKS.search(goal_low) or names_low & {"react", "vue", "svelte", "nextjs", "express"}:
        if "package.json" not in existing:
            to_inject["package.json"] = (
                "npm package manifest: dependencies, devDependencies, and scripts"
            )

    _PY_STACKS = re.compile(
        r"\b(python|fastapi|flask|django)\b", re.IGNORECASE
    )
    _BACKEND_KW = re.compile(r"\b(backend|server|api)\b", re.IGNORECASE)
    if _PY_STACKS.search(goal_low) or names_low & {"python", "fastapi", "flask", "django"}:
        if "requirements.txt" not in existing and "pyproject.toml" not in existing:
            to_inject["requirements.txt"] = "Python package dependencies"
        # The judge checks for at least one .py source file when the goal names
        # Python. Inject an entry module if none exists yet.
        has_py_source = any(p.endswith(".py") for p in existing)
        if not has_py_source and _BACKEND_KW.search(goal_low):
            if re.search(r"\bfastapi\b", goal_low) or "fastapi" in names_low:
                to_inject["backend/main.py"] = "FastAPI application entry point"
            elif re.search(r"\bflask\b", goal_low) or "flask" in names_low:
                to_inject["backend/app.py"] = "Flask application entry point"
            elif re.search(r"\bdjango\b", goal_low) or "django" in names_low:
                to_inject["manage.py"] = "Django management entry point"
            else:
                to_inject["backend/app.py"] = "Python backend entry module"

    # For Python projects that use the src/ layout, inject a root-level
    # conftest.py that prepends src/ to sys.path. Without it, tests under
    # tests/ cannot resolve `from <module> import …` when <module> lives
    # in src/, because pytest's rootdir does not implicitly include src/
    # as a sys.path entry. Generated at scaffold time so the run-tests
    # verify step works without manual setup.
    has_src_py = any(
        p.startswith("src/") and p.endswith(".py")
        for p in existing_paths
    )
    if has_src_py and "conftest.py" not in existing:
        to_inject["conftest.py"] = (
            "pytest bootstrap: prepend src/ to sys.path so tests import "
            "modules by their flat name (e.g. `from foo import bar`)"
        )

    if not to_inject:
        return layers

    # Prefer an existing config/packaging layer; otherwise append one.
    out = list(layers)
    config_layer = next(
        (lay for lay in out
         if isinstance(lay, dict)
         and str(lay.get("name") or "").lower() in ("config", "config/packaging", "packaging")),
        None,
    )
    if config_layer is not None:
        config_layer["files"] = list(config_layer.get("files") or []) + [
            {"path": p, "description": d} for p, d in to_inject.items()
        ]
    else:
        out.append({
            "name": "config",
            "files": [{"path": p, "description": d} for p, d in to_inject.items()],
        })
    return out


def _inject_python_package_inits(layers: List[Any]) -> List[Any]:
    """Ensure every Python source directory has an ``__init__.py``.

    Small models emit ``backend/calculator.py`` but forget the package
    marker, which makes ``from backend.calculator import add`` work only
    under Python 3 namespace-package discovery — and pytest's rootdir
    inference fails on that path when no ``conftest.py`` is present.
    Adding an explicit ``__init__.py`` for every package directory
    turns the layout into regular packages so imports resolve reliably
    regardless of pytest's discovery mode.

    Excludes ``tests/`` and its descendants because pytest convention
    is that test directories are NOT packages (pytest's collector
    handles them via rootdir/conftest, not via ``import tests.…``).
    Also excludes the top-level ``src/`` directory itself: the standard
    "src layout" treats ``src/`` as a sys.path root rather than a
    package, so files inside ``src/`` are imported by their flat module
    name (``from foo import bar``) rather than ``from src.foo import
    bar``. Subpackages under ``src/`` (e.g. ``src/models/``) are still
    regular packages and DO get an ``__init__.py``.
    Files at the project root need no marker either.
    """
    # Collect every directory that contains at least one .py file, plus
    # the set of paths already in the manifest so we don't duplicate.
    py_dirs: set = set()
    existing: set = set()
    for lay in layers or []:
        if not isinstance(lay, dict):
            continue
        for f in (lay.get("files") or []):
            if not isinstance(f, dict):
                continue
            p = str(f.get("path") or "").strip()
            if not p:
                continue
            existing.add(p)
            if not p.endswith(".py"):
                continue
            if "/" not in p:
                continue
            parent = p.rsplit("/", 1)[0]
            # Walk every ancestor directory so nested packages
            # (backend/utils/helpers.py → backend/, backend/utils/)
            # all get markers.
            while parent:
                head = parent.split("/", 1)[0]
                if head == "tests":
                    break
                # Skip the top-level src/ directory: it's a sys.path root
                # in the standard "src layout", not a package. Subpackages
                # under it (src/models/, …) are still added below.
                if parent == "src":
                    break
                py_dirs.add(parent)
                if "/" not in parent:
                    break
                parent = parent.rsplit("/", 1)[0]

    to_inject: List[str] = []
    for d in sorted(py_dirs):
        marker = f"{d}/__init__.py"
        if marker not in existing:
            to_inject.append(marker)

    if not to_inject:
        return layers

    out = list(layers)
    # Prefer adding the markers to the layer that already contains the
    # corresponding .py files so the manifest stays grouped; fall back
    # to a dedicated "packaging" layer when no obvious owner exists.
    by_dir: Dict[str, Dict[str, Any]] = {}
    for lay in out:
        if not isinstance(lay, dict):
            continue
        for f in (lay.get("files") or []):
            if not isinstance(f, dict):
                continue
            p = str(f.get("path") or "")
            if p.endswith(".py") and "/" in p:
                by_dir.setdefault(p.rsplit("/", 1)[0], lay)

    leftover: List[str] = []
    for marker in to_inject:
        owner_dir = marker.rsplit("/", 1)[0]
        host = by_dir.get(owner_dir)
        if host is None:
            leftover.append(marker)
            continue
        host_files = list(host.get("files") or [])
        host_files.append({
            "path": marker,
            "description": f"Package marker for {owner_dir}/.",
        })
        host["files"] = host_files

    if leftover:
        out.append({
            "name": "packaging",
            "files": [
                {"path": m, "description": f"Package marker for {m.rsplit('/', 1)[0]}/."}
                for m in leftover
            ],
        })
    return out


# Framework-convention path overrides: filename basename → canonical path.
# These run after the manifest is parsed so the LLM's intent is preserved
# but config files end up where the toolchain actually expects them.
_CANONICAL_CONFIG_PATHS: Dict[str, str] = {
    "package.json": "package.json",
    "vite.config.js": "vite.config.js",
    "vite.config.ts": "vite.config.ts",
    "next.config.js": "next.config.js",
    "next.config.mjs": "next.config.mjs",
    "next.config.ts": "next.config.ts",
    "tailwind.config.js": "tailwind.config.js",
    "tailwind.config.ts": "tailwind.config.ts",
    "postcss.config.js": "postcss.config.js",
    "tsconfig.json": "tsconfig.json",
    "manage.py": "manage.py",
    "pyproject.toml": "pyproject.toml",
}


def _normalize_manifest_paths(layers: List[Any]) -> List[Any]:
    """Rewrite known-misplaced framework config files to their canonical
    project-root location and de-duplicate paths across layers.
    """
    seen: set = set()
    out: List[Any] = []
    for lay in layers or []:
        if not isinstance(lay, dict):
            out.append(lay)
            continue
        new_files: List[Any] = []
        for f in (lay.get("files") or []):
            if not isinstance(f, dict):
                new_files.append(f)
                continue
            p = str(f.get("path") or "").strip()
            if not p:
                continue
            base = p.rsplit("/", 1)[-1].lower()
            canon = _CANONICAL_CONFIG_PATHS.get(base)
            if canon and p != canon:
                f = {**f, "path": canon}
                p = canon
            if p in seen:
                continue
            seen.add(p)
            new_files.append(f)
        out.append({**lay, "files": new_files})
    return out


_SINGLE_FILE_SYSTEM = (
    "You are a senior software engineer generating EXACTLY ONE source file.\n\n"
    "You will be given:\n"
    "- The project goal\n"
    "- The file path and its purpose\n"
    "- The content of files already generated in this project\n\n"
    "Output the COMPLETE content of the requested file only. "
    "Return strict JSON:\n"
    '{"content": "complete file content as a string"}\n\n'
    "Rules:\n"
    "- Output the full file — no stubs, no placeholders, no ellipsis.\n"
    "- Use imports consistent with what already exists in the project.\n"
    "- Do not repeat or regenerate any already-existing file.\n"
    "- The content MUST be functionally different from every file in "
    "ALREADY GENERATED FILES. Do NOT copy another file's body and rename "
    "the export — write the unique content that fulfils THIS file's "
    "purpose. If the requested purpose duplicates an already-generated "
    "file, return {\"content\": \"\"} instead.\n"
    "- Satisfy the file's stated purpose exactly.\n"
    "Python import discipline (applies to every .py file in this project):\n"
    "- src/ is a sys.path ROOT, NOT a package. There is no src/__init__.py.\n"
    "- Inside files under src/, import sibling modules by their flat name. "
    "RIGHT: `from chat_manager import ChatManager`. "
    "WRONG: `from src.chat_manager import ChatManager`.\n"
    "- Test files under tests/ also import from the same flat module names "
    "(a conftest.py at the project root puts src/ on sys.path). "
    "RIGHT: `from chat_manager import ChatManager`. "
    "WRONG: `from src.chat_manager import ChatManager`.\n"
    "- Subpackages that live under src/ (e.g. src/models/) ARE regular "
    "packages with their own __init__.py and are imported without the "
    "src. prefix: `from models.user import User`, never "
    "`from src.models.user import User`.\n"
    "- Relative imports (`from .foo import bar`) are OK between modules in "
    "the same subpackage, but NEVER use them inside a script that may be "
    "launched directly (streamlit run, python src/app.py, etc.) — those "
    "scripts run as __main__ and have no parent package.\n"
)

_SINGLE_FILE_FREEFORM_SYSTEM = (
    "You are a senior software engineer generating EXACTLY ONE source file.\n\n"
    "Output the complete file content inside a fenced code block with the path:\n"
    "```language path=<relative/path>\n"
    "<full file content>\n"
    "```\n\n"
    "No other files. No explanations outside the fence.\n"
)


# Signature-line regex for the generic (non-Python, non-JSON) summarizer.
# Matches top-level imports, exports, function/class/interface/type decls,
# Python defs, capitalised const bindings (React/Vue components), and
# CommonJS exports. Kept intentionally conservative to avoid leaking full
# function bodies into the "ALREADY GENERATED FILES" prompt block.
_SIG_LINE_RE = re.compile(
    r"^\s*(import\s|from\s+\S+\s+import|export\s|function\s|async\s+function|"
    r"class\s|interface\s|type\s+[A-Z]|const\s+[A-Z_][A-Za-z0-9_]*\s*=|"
    r"def\s|async\s+def\s|module\.exports|@[A-Za-z_])"
)


def _summarize_python(src: str) -> str:
    """Return a Python file's structural skeleton with bodies elided.

    Walks the top-level AST: keeps every ``import``/``from`` line verbatim,
    keeps top-level constant assignments capped at 120 chars, and replaces
    each function/method/class body with ``...`` so the model sees what
    symbols already exist without paying for their implementation tokens.
    """
    import ast as _ast
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return ""
    lines = src.splitlines()

    def _sig(node: Any) -> str:
        start = node.lineno - 1
        body = getattr(node, "body", None)
        end = (body[0].lineno - 1) if body else start + 1
        end = max(end, start + 1)
        return "\n".join(lines[start:end]).rstrip()

    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            out.append(lines[node.lineno - 1].rstrip())
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            out.append(_sig(node) + "\n    ...")
        elif isinstance(node, _ast.ClassDef):
            sig = _sig(node)
            methods: List[str] = []
            for sub in node.body:
                if isinstance(sub, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    msig = _sig(sub)
                    msig_lines = msig.splitlines() or [""]
                    msig_lines[0] = "    " + msig_lines[0].lstrip()
                    methods.append("\n".join(msig_lines) + "\n        ...")
            body = "\n".join(methods) if methods else "    ..."
            out.append(sig + "\n" + body)
        elif isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            ln = lines[node.lineno - 1].rstrip()
            if len(ln) > 120:
                ln = ln[:117] + "..."
            out.append(ln)
    return "\n".join(out)


def _summarize_json(src: str) -> str:
    """Return a compact ``{ "k1", "k2", ... }`` summary of a JSON file."""
    try:
        obj = json.loads(src)
    except Exception:
        return ""
    if isinstance(obj, dict):
        keys = list(obj.keys())[:30]
        rendered = ", ".join(repr(k) for k in keys)
        return "{ " + rendered + (" ... }" if len(obj) > 30 else " }")
    if isinstance(obj, list):
        return f"[ array of {len(obj)} item(s) ]"
    return repr(obj)[:200]


def _summarize_textual(src: str, *, max_lines: int = 25) -> str:
    """Regex-based signature extractor for JS/TS/JSX/TSX/Vue/etc."""
    kept: List[str] = []
    for ln in src.splitlines():
        if _SIG_LINE_RE.match(ln):
            kept.append(ln.rstrip())
            if len(kept) >= max_lines:
                break
    if not kept:
        kept = [ln.rstrip() for ln in src.splitlines()[:6]]
    return "\n".join(kept)


def _summarize_file_for_context(
    path: str, content: str, *, max_chars: int = 800
) -> str:
    """Produce a compact structural summary of a generated file.

    Used in the "ALREADY GENERATED FILES" prompt block when generating
    a new sibling file: the LLM sees the available symbols (imports,
    function / class signatures, top-level constants) without paying
    for the full file body. ``max_chars`` is a hard cap; everything
    beyond it is dropped with a trailing marker.
    """
    if not content:
        return ""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    text = ""
    try:
        if ext == "py":
            text = _summarize_python(content)
        elif ext == "json":
            text = _summarize_json(content)
        else:
            text = _summarize_textual(content)
    except Exception:  # pragma: no cover - defensive
        text = ""
    if not text:
        text = "\n".join(content.splitlines()[:8])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n# ... (summary truncated)"
    return text


def generate_single_scaffold_file(
    path: str,
    description: str,
    provider: Any,
    *,
    layer: str = "",
    existing_files_with_content: Optional[List[Dict[str, str]]] = None,
    goal: str = "",
    skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate the content of a single file in a new-project scaffold.

    Each call generates exactly one file, with the content of all
    previously-generated files provided as context so imports resolve
    correctly. Runs inline syntax validation before returning.

    Returns a dict with keys: ``file``, ``patch``, ``content``,
    ``syntax_ok``, ``confidence``.
    """
    if provider is None:
        return {"file": path, "patch": "", "content": "", "syntax_ok": False, "confidence": 0.0}

    path = _normalize_scaffold_path(path, [f["path"] for f in (existing_files_with_content or [])])

    # Deterministic short-circuit for ``__init__.py`` package markers:
    # these are emitted by ``_inject_python_package_inits`` to make
    # every Python source directory a regular package so pytest can
    # resolve first-party imports without sys.path tricks. The file is
    # content-free by convention; we still need a non-empty body so the
    # Judge's "no content" gate passes, hence a one-line docstring.
    base = path.rsplit("/", 1)[-1]
    if base == "__init__.py":
        owner = path.rsplit("/", 1)[0] if "/" in path else ""
        content = (f'"""Package marker for {owner}/."""\n'
                   if owner else '"""Package marker."""\n')
        patch = _content_to_new_file_patch(path, content)
        return {
            "file": path,
            "patch": patch,
            "content": content,
            "diffs": [{"file": path, "patch": patch}],
            "syntax_ok": True,
            "confidence": 1.0,
        }

    # Deterministic short-circuit for root-level ``conftest.py``: emitted
    # by ``_inject_required_manifest_files`` for Python projects that use
    # the src/ layout. Its job is fixed and one-line — prepend src/ to
    # sys.path — so we generate it without an LLM round-trip to avoid the
    # model writing test stubs into it or omitting the sys.path insert.
    if path == "conftest.py":
        content = (
            '"""pytest bootstrap: make src/ importable as a sys.path root.\n\n'
            "Tests under tests/ import first-party modules by their flat\n"
            "name (e.g. ``from foo import bar`` for ``src/foo.py``). This\n"
            "file runs before collection and prepends src/ to ``sys.path``\n"
            "so those imports resolve without installing the project.\n"
            '"""\n'
            "import os\n"
            "import sys\n"
            "\n"
            "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
            "_SRC = os.path.join(_HERE, \"src\")\n"
            "if os.path.isdir(_SRC) and _SRC not in sys.path:\n"
            "    sys.path.insert(0, _SRC)\n"
        )
        patch = _content_to_new_file_patch(path, content)
        return {
            "file": path,
            "patch": patch,
            "content": content,
            "diffs": [{"file": path, "patch": patch}],
            "syntax_ok": True,
            "confidence": 1.0,
        }

    # Feed the file path into skill detection so an explicit .jsx/.tsx/.vue
    # extension can pull in the matching frontend skill even when the goal
    # text was ambiguous or contained a typo. _JSX_RE / Vue regex match on
    # `\b(?:jsx|tsx)\b` so the dot in ".jsx" forms a word boundary.
    detect_text = " ".join(t for t in (goal, description, path) if t)
    active_skills = _resolve_skills(skills, detect_text)
    try:
        import skills as _sk
        skill_fragment = _sk.compose_scaffold_prompt(active_skills)
        skill_names_str = ", ".join(s.name for s in active_skills)
    except Exception:  # pragma: no cover - defensive
        skill_fragment = ""
        skill_names_str = ""

    system = _SINGLE_FILE_SYSTEM
    if skill_fragment:
        header = (
            f"\n\nACTIVE SKILLS: {skill_names_str}\n"
            "The file you generate MUST follow the technology conventions below "
            "(language, imports, framework idioms). Do NOT substitute a different "
            "framework or language than the one declared for this project.\n\n"
        )
        system = _SINGLE_FILE_SYSTEM + header + skill_fragment

    # Defense-in-depth: hard-pin framework conventions by file extension so
    # the model cannot cross-contaminate frameworks (Vue SFC under .jsx,
    # React under .vue) regardless of whether a skill was detected. The
    # symptom this guards against is the per-file judge rejecting a .jsx
    # file that contains <template> / `import from 'vue'`.
    ext_pin = _extension_framework_pin(path)
    if ext_pin:
        system = system + "\n\n" + ext_pin

    parts: List[str] = []
    if goal:
        parts.append(f"PROJECT GOAL:\n{goal}")
    parts.append(f"FILE TO GENERATE:\nPath: {path}\nPurpose: {description}")
    if layer:
        parts.append(f"Layer: {layer}")
    # Per-call prompt + response budget scaled to the active provider's
    # model context window. Local 8K models get tight caps; cloud
    # models with 200K+ windows get generous ones. See
    # :mod:`cgx.answer.model_caps`.
    from cgx.answer.model_caps import get_summary_budget
    budget = get_summary_budget(provider)

    if existing_files_with_content:
        # Send a *structural summary* of each prior file (imports +
        # function/class signatures with bodies elided) rather than the
        # full source. This keeps the prompt small as the scaffold grows
        # and is what the model actually needs to know: which symbols
        # already exist, not how they are implemented. The full content
        # is still kept in ``existing_files_with_content`` for the
        # downstream duplicate-content guard.
        context_blocks: List[str] = []
        for ef in existing_files_with_content[: budget["max_files"]]:
            ep = ef.get("path", "")
            ec = ef.get("content", "")
            if not ep or not ec:
                continue
            summary = _summarize_file_for_context(
                ep, ec, max_chars=budget["max_chars"],
            )
            if not summary:
                continue
            context_blocks.append(f"### {ep}\n```\n{summary}\n```")
        if context_blocks:
            parts.append(
                "ALREADY GENERATED FILES (do not re-emit these; signatures "
                "shown, bodies elided):\n\n" + "\n\n".join(context_blocks)
            )

    context = "\n\n".join(parts)

    raw = provider.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
        max_tokens=budget["output_tokens"],
        force_json=True,
    ).get("content", "")
    parsed = _extract_json_object(raw)
    content = str(parsed.get("content") or "") if parsed else ""

    if not content:
        # Fallback to freeform. Carry the same skill constraints over.
        ff_system = _SINGLE_FILE_FREEFORM_SYSTEM
        if skill_fragment:
            ff_system = _SINGLE_FILE_FREEFORM_SYSTEM + header + skill_fragment
        ff_raw = provider.chat(
            messages=[
                {"role": "system", "content": ff_system},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            force_json=False,
        ).get("content", "")
        parsed_ff = _parse_scaffold_freeform(ff_raw)
        for d in (parsed_ff.get("diffs") or []):
            if isinstance(d, dict) and d.get("file") == path:
                content = str(d.get("patch") or "")
                break
        if not content and parsed_ff.get("diffs"):
            first = parsed_ff["diffs"][0]
            content = str(first.get("patch") or "")

    # Inline syntax validation.
    syntax_ok = True
    syntax_error: Optional[str] = None
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    # Strip leading/trailing markdown code fences a 3B model occasionally
    # wraps around the file body even when asked for raw content
    # (```python\n…\n```). We only strip when the WHOLE content is
    # wrapped, never an inner fence.
    if content:
        stripped = content.strip()
        if stripped.startswith("```"):
            m = re.match(r"^```[a-zA-Z0-9_+\-]*\s*\n(.*?)\n```\s*$",
                         stripped, re.DOTALL)
            if m:
                content = m.group(1)
    # Reject unified-diff fragments that leak through the freeform parser
    # into the file body (`--- /dev/null`, `+++ b/...`, `@@ ...`).
    if content:
        head = content.lstrip().splitlines()[0] if content.strip() else ""
        if head.startswith(("--- ", "+++ ", "@@ ")):
            syntax_ok = False
            syntax_error = "content is a unified-diff header, not a file body"
    if syntax_ok and ext == "py" and content:
        try:
            import ast as _ast
            _ast.parse(content)
        except SyntaxError as e:
            syntax_ok = False
            syntax_error = str(e)
    elif syntax_ok and ext == "json" and content:
        try:
            import json as _json
            _json.loads(content)
        except Exception as e:
            syntax_ok = False
            syntax_error = str(e)
    elif syntax_ok and ext == "toml" and content:
        try:
            import tomllib as _tomllib
            _tomllib.loads(content)
        except Exception as e:
            syntax_ok = False
            syntax_error = f"TOML parse error: {e}"

    # Extension/content mismatch check: a 3B model frequently emits Vue
    # SFC content under a .jsx path, or vice versa. These heuristics catch
    # the cross-framework mistakes before APPLY writes garbage to disk.
    if content and syntax_ok:
        mismatch = _extension_content_mismatch(path, ext, content)
        if mismatch:
            syntax_ok = False
            syntax_error = mismatch

    # Duplicate-content guard: refuse content that byte-matches a prior
    # file (after normalising whitespace). 3B models frequently rename the
    # export of an already-generated file instead of writing fresh content.
    if content and existing_files_with_content:
        norm_new = "".join(content.split())
        if norm_new:
            for ef in existing_files_with_content:
                ep = ef.get("path", "")
                ec = ef.get("content", "")
                if not ep or not ec or ep == path:
                    continue
                if "".join(ec.split()) == norm_new:
                    syntax_ok = False
                    syntax_error = f"duplicate content of {ep}"
                    content = ""
                    break

    patch = _content_to_new_file_patch(path, content) if content else ""
    result: Dict[str, Any] = {
        "file": path,
        "patch": patch,
        "content": content,
        "diffs": [{"file": path, "patch": patch}] if patch else [],
        "syntax_ok": syntax_ok,
        "confidence": 0.8 if (content and syntax_ok) else 0.3,
    }
    if syntax_error:
        result["syntax_error"] = syntax_error
    return result


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
