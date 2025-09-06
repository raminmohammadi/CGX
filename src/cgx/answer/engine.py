from __future__ import annotations
import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from cgx.io.persist import load_indices, load_jsonl
from cgx.answer.providers import LLMProvider

ALLOWED_CITATION_NOTE = (
    "Cite only chunk_ids that appear in SOURCES. "
    "Return citations as an array of objects: { \"chunk_id\": \"...\" }. "
    "Do not return numbers or invented ids."
)

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

def _as_sources_with_meta(
    hits: List[Dict[str, Any]],
    cmap: Dict[str, Dict[str, Any]],
    max_chunks: int = 24,
    max_chars: int = 900
) -> List[Dict[str, Any]]:
    """Select top hits and attach trimmed text + hit provenance for grounding & debug."""
    out: List[Dict[str, Any]] = []
    for h in hits[:max_chunks]:
        cid = str(h.get("chunk_id"))
        row = cmap.get(cid) or {}
        text = row.get("text", "") if isinstance(row, dict) else ""
        path, kind, symbol = _split_chunk_id(cid)
        # keep score and nested provenance (intent/impl ranks/scores, lexical_count, graph_depth, etc.)
        prov = {}
        for k, v in (h or {}).items():
            if k == "chunk_id":
                continue
            if k == "provenance" and isinstance(v, dict):
                prov.update(v)
            else:
                prov[k] = v
        out.append({
            "chunk_id": cid,
            "path": path,
            "kind": kind,
            "symbol": symbol,
            "text": _trim(text, max_chars),
            "hit_meta": prov,
        })
    return out

def _route(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["overview", "what does this repo", "what is this repo", "high level", "summary"]):
        return "overview"
    if any(k in q for k in ["add", "implement", "feature", "refactor", "plan", "change", "extend"]):
        return "change_plan"
    if any(k in q for k in ["how do i", "how to", "where to"]):
        return "howto"
    if re.search(r"\b(what does|explain|describe)\b.*\b([A-Za-z_][A-Za-z0-9_]*)\b", q):
        return "symbol_explain"
    return "overview"

def _symbol_tokens(question: str) -> List[str]:
    # prefer tokens inside quotes/backticks first
    quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", question or "")
    bare = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question or "")
    seen, out = set(), []
    for t in quoted + bare:
        if t not in seen:
            seen.add(t); out.append(t)
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
    # de-dup by cid while preserving first occurrence (intent tends to carry docstring)
    seen, dedup = set(), []
    for cid, r, view in out:
        if cid not in seen:
            seen.add(cid); dedup.append((cid, r, view))
    return dedup

def _hits_from_records(indices: Dict[str, Any], records_path: Optional[str], symbol: Optional[str]) -> List[Dict[str, Any]]:
    """Use records.jsonl to lock onto exact chunk ids for a symbol name, then map to indexed rows."""
    if not records_path or not symbol:
        return []
    try:
        recs = load_jsonl(records_path)
    except Exception:
        return []
    # collect candidate record ids where name matches (case-insensitive)
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

    # map indices rows by chunk_id for both views
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

def answer_with_llm(
    index_dir: str,
    records_path: str,
    question: str,
    provider: LLMProvider,
    *,
    top_k: int = 20,
    hits: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Retrieve context from indices and ask the LLM to synthesize a grounded answer.
    Returns a payload with `answer_md`, `citations`, and a `debug` section containing the
    SOURCES (with trimmed text and hit provenance) and the raw `hits` list used.
    """
    indices = load_indices(index_dir)
    _ = load_jsonl(records_path) if records_path else None

    cmap = _chunk_map(indices)

    # 1) Determine target symbol (if any)
    symbols = _symbol_tokens(question)
    target = None
    for t in symbols:
        rows_for_t = _find_symbol_rows(indices, t)
        if rows_for_t:
            target = t
            break
    if target is None and symbols:
        target = symbols[0]

    # 2) Build/augment hits: prefer exact-symbol rows; reinforce with records.jsonl mapping
    forced_hits: List[Dict[str, Any]] = []
    if target:
        # from indices rows
        for cid, _row, view in _find_symbol_rows(indices, target):
            forced_hits.append({"chunk_id": cid, "score": 2.0, "view": view})
        # from records.jsonl (exact id match)
        rec_hits = _hits_from_records(indices, records_path, target)
        # merge dedup
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

    # de-dup with priority to forced hits
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
            "debug": {"mode": _route(question), "target_symbol": target, "sources": [], "hits": []},
        }

    # 3) SOURCES with larger window for symbol explanations
    mode = _route(question)
    max_chars = 1400 if mode == "symbol_explain" else 900
    sources = _as_sources_with_meta(merged_hits, cmap, max_chunks=40 if mode == "symbol_explain" else 24, max_chars=max_chars)

    # Require that target is covered in at least one source if we have a target
    if target:
        covers = [s for s in sources if (s.get("symbol") or "").lower() == target.lower() or f"::{target}" in s.get("chunk_id", "")]
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

    def fmt_source(s: Dict[str, Any]) -> str:
        return f"- {s['chunk_id']} :: {s['path']} :: {s['kind']} :: {s['symbol']}\n  {s['text']}"

    context = "QUESTION:\n" + (question or "").strip() + "\n\n"
    if mode == "symbol_explain":
        context += (
            "TASK: Explain the function/class in detail. Cover: purpose, parameters & types (if visible), "
            "return value, side-effects, key branches/logic, dependencies (internal calls), and typical usage. "
            "Ground every claim with a citation.\n\n"
        )
    if readme and mode != "symbol_explain":
        lead_lines = [ln for ln in readme.splitlines() if ln.strip()][:12]
        context += "README (lead):\n" + "\n".join(lead_lines) + "\n\n"
    if target:
        context += f"TARGET_SYMBOL: {target}\n\n"
    context += "SOURCES:\n" + "\n".join(fmt_source(s) for s in sources)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": context},
    ]

    # Ask the model to output JSON
    resp = provider.chat(messages, temperature=0.2)
    content = (resp.get("content") or "").strip()

    # Try to extract JSON
    def extract_json(text: str) -> Dict[str, Any]:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    parsed: Dict[str, Any] = extract_json(content)

    # Retry if empty or missing answer
    if not parsed or not isinstance(parsed, dict) or not parsed.get("answer_md"):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "Reformat to strict JSON only. "
                                                   "Ensure non-empty 'answer_md' grounded in SOURCES with citations. "
                                                   "Keep the same content; do not add external knowledge."})
        resp2 = provider.chat(messages, temperature=0)
        parsed = extract_json((resp2.get("content") or "")) or {"answer_md": content, "citations": []}

    # Normalize answer_md to string
    ans = parsed.get("answer_md")
    if isinstance(ans, dict):
        parsed["answer_md"] = ans.get("content") or ans.get("text") or ans.get("markdown") or ans.get("md") or json.dumps(ans, ensure_ascii=False)
    elif isinstance(ans, list):
        parsed["answer_md"] = "\n".join(str(x) for x in ans)
    elif ans is None:
        parsed["answer_md"] = ""

    # Final guard: if still empty, provide explicit insufficiency
    if not parsed["answer_md"].strip():
        parsed["answer_md"] = (
            "The provided SOURCES did not contain enough content to explain this symbol without guessing. "
            "Please re-index or narrow the question to a specific file or snippet."
        )
        parsed.setdefault("citations", [])
        parsed.setdefault("suggested_changes", [])
        parsed["confidence"] = 0.2

    # Minimal normalization + citation sanitation
    allowed_ids = [s['chunk_id'] for s in sources]
    parsed["citations"] = _sanitize_citations(parsed.get("citations", []), allowed_ids)
    parsed.setdefault("suggested_changes", [])
    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)):
        parsed["confidence"] = 0.6 if parsed["citations"] else 0.4

    # Attach rich debug payload for UI
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
    provider: LLMProvider
) -> Dict[str, Any]:
    """Use LLM to propose a change plan and diffs (unified patch) grounded in SOURCES."""
    indices = load_indices(index_dir)
    _ = load_jsonl(records_path) if records_path else None

    cmap = _chunk_map(indices)
    # Use top impl-view rows for code-centric tasks
    impl_rows = (indices.get("views") or {}).get("impl", {}) or {}
    hits = [
        {"chunk_id": r.get("chunk_id"), "score": 1.0, "view": "impl"}
        for r in (impl_rows.get("rows") or [])[:24]
    ]
    sources = _as_sources_with_meta(hits, cmap, max_chunks=24, max_chars=700)

    SYSTEM2 = (
        "You are a principal engineer. Propose a step-by-step change plan and unified diffs "
        "to implement the requested change. Use ONLY provided SOURCES and cite with [[chunk_id]]. "
        "Return JSON with keys: plan_md, diffs (array of objects: file, patch), citations, confidence. "
        "Do not include prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE

    def fmt_s(s: Dict[str, Any]) -> str:
        return f"- {s['chunk_id']} :: {s['path']} :: {s['kind']} :: {s['symbol']}\n  {s['text']}"

    context = "TASK:\n" + (task or "").strip() + "\n\nSOURCES:\n" + "\n".join(fmt_s(s) for s in sources)
    messages = [{"role": "system", "content": SYSTEM2}, {"role": "user", "content": context}]
    out_text = provider.chat(messages, temperature=0.2).get("content", "")
    try:
        m = re.search(r"\{.*\}", out_text, flags=re.S)
        parsed = json.loads(m.group(0)) if m else {"plan_md": out_text, "diffs": [], "citations": [], "confidence": 0.5}
    except Exception:
        parsed = {"plan_md": out_text, "diffs": [], "citations": [], "confidence": 0.5}

    allowed_ids = [s['chunk_id'] for s in sources]
    parsed["citations"] = _sanitize_citations(parsed.get("citations", []), allowed_ids)
    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)):
        parsed["confidence"] = 0.5

    parsed["debug"] = {"sources": sources, "hits": hits}
    return parsed