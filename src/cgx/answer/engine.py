

from __future__ import annotations
import logging
import os, json, re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from pathlib import Path

from cgx.io.persist import load_indices, load_jsonl
from cgx.answer.providers import LLMProvider
from cgx.answer.schemas import (
    CLARIFY_PATHS_SCHEMA,
    MANIFEST_SCHEMA,
    REPAIR_FILES_SCHEMA,
    validate_json_schema,
)
from cgx.answer.intent import detect_intent  # <-- NEW central intent detection
from cgx.retrieval.orchestrator import (
    SYMBOL_STOPWORDS as _SYMBOL_STOPWORDS,
    _extract_symbol_tokens,
)

from networkx.readwrite import json_graph
import networkx as nx  # type: ignore

from cgx.trace import traced, emit_trace

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
    cosmetic -- ``citations`` and ``debug.sources`` still carry the full ids.
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
    total_chars_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select top hits and attach trimmed text + structured meta for grounding & debug.

    When ``focus_terms`` is non-empty, each chunk's text is windowed around
    the first line matching one of the terms (symbol name, query keywords)
    to reduce prompt size without losing the relevant span.
    """
    out: List[Dict[str, Any]] = []
    total_chars_accum = 0
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
        if total_chars_budget is not None and total_chars_accum + len(body) > total_chars_budget:
            if len(out) >= 3:
                break
            rem = max(100, total_chars_budget - total_chars_accum)
            body = body[:rem]
        total_chars_accum += len(body)
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


def _clean_json_candidate(candidate: str) -> str:
    """Clean common local LLM formatting errors in a JSON string."""
    # Strip trailing commas before closing braces/brackets
    return re.sub(r",\s*([\]}])", r"\1", candidate)


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
    # Fast path: the whole payload is already JSON (Ollama JSON mode or markdown-wrapped).
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    if s.startswith("{") and s.endswith("}"):
        for c_str in (s, _clean_json_candidate(s)):
            try:
                obj = json.loads(c_str)
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
                        for c_str in (candidate, _clean_json_candidate(candidate)):
                            try:
                                obj = json.loads(c_str)
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
        # No section header -- strip fenced diff blocks and treat the rest as plan.
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

# Matches ``[[chunk_id]]`` citation tokens in Markdown text
_INLINE_CITATION_RE = re.compile(r"\[\[([^\[\]]+)\]\]")

def _sanitize_inline_citations(answer_md: str, allowed_ids: Sequence[str]) -> str:
    """Strip or clean inline [[chunk_id]] tokens that are not present in allowed_ids."""
    if not answer_md or not isinstance(answer_md, str):
        return str(answer_md or "")
    allowed_set = set(allowed_ids)
    def _replace_cite(m: re.Match) -> str:
        cid = m.group(1).strip()
        if cid in allowed_set:
            return m.group(0)
        return ""
    cleaned = _INLINE_CITATION_RE.sub(_replace_cite, answer_md)
    cleaned = re.sub(r' +\.', '.', cleaned)
    cleaned = re.sub(r' +,', ',', cleaned)
    return cleaned

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
    "qa": (
        "You are a senior codebase assistant answering a specific question. "
        "Use ONLY the SOURCES (and the optional README lead) to answer the "
        "QUESTION directly. Do NOT impose a fixed section template -- pick "
        "the shape that best fits the question (a short paragraph, a list, "
        "a small table, or a brief code excerpt). Stay focused on what was "
        "asked; do not pivot into a generic repo summary. Cite every "
        "non-trivial claim with [[chunk_id]]. If SOURCES do not cover the "
        "question, say so plainly and name what would be needed. Return JSON "
        "keys: answer_md, citations, suggested_changes, confidence. No prose "
        "outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
    "clarify_paths": (
        "You are a senior codebase assistant responding to an OPEN-ENDED goal "
        "(e.g. 'improve X', 'suggest ways to Y'). The user has not yet picked "
        "a direction, so your job is NOT to commit to a single answer -- it "
        "is to help them choose. Use the SOURCES to ground your suggestions "
        "in this specific codebase, but do not pretend to know which path "
        "they want. Structure answer_md as: (1) a one-sentence restatement "
        "of the goal as you understand it; (2) a numbered list of 3-5 "
        "concrete directions the user could pursue, each with a short "
        "rationale and -- when supported by SOURCES -- a [[chunk_id]] "
        "citation pointing to the component it would touch; (3) a single "
        "follow-up question asking which direction to pursue or what "
        "constraint matters most (accuracy/latency/code-size/etc.). "
        "Do NOT propose code edits and do NOT invent components absent "
        "from SOURCES. Return JSON keys: answer_md, citations, "
        "suggested_changes, confidence. No prose outside JSON. "
    ) + ALLOWED_CITATION_NOTE,
}


def _get_system_prompt(mode: str) -> str:
    """Return the system prompt for ``mode`` with a safe fallback to SYSTEM."""
    return SYSTEM_PROMPTS.get(mode, SYSTEM)


# Streaming variants of the system prompts. JSON mode forces the whole
# response to land as a single payload, which both delays first-token
# emission and triggers the provider's read-timeout on slow local models.
# Streaming asks for plain Markdown with inline ``[[chunk_id]]`` citations
# so tokens can flow continuously and the UI can render incrementally.
SYSTEM_STREAM = (
    "You are a senior codebase assistant. Use ONLY the provided SOURCES to answer. "
    "Cite facts inline with [[chunk_id]] markers exactly as they appear in SOURCES. "
    "Be concise but complete. Reply in plain Markdown -- do NOT wrap your answer in JSON, "
    "do NOT add a heading like '## Answer', and do NOT include external knowledge. "
    "If information is missing, say what else is needed rather than inventing details."
)


SYSTEM_PROMPTS_STREAM: Dict[str, str] = {
    "symbol_explain": (
        "You are a senior code reviewer explaining a specific symbol. Use ONLY the SOURCES. "
        "Reply in plain Markdown structured as: Purpose, Signature, Parameters, Returns, "
        "Side effects, Key logic, Internal dependencies, Typical usage. Cite every "
        "non-trivial claim inline with [[chunk_id]]. Do NOT wrap in JSON."
    ),
    "howto": (
        "You are a pragmatic guide for using this codebase. Use ONLY the SOURCES. "
        "Reply in plain Markdown: a short numbered procedure followed by a minimal code "
        "example drawn from SOURCES. Cite each step with [[chunk_id]]. Do NOT wrap in JSON."
    ),
    "change_plan": (
        "You are a principal engineer drafting a focused change plan. Use ONLY the SOURCES. "
        "Reply in plain Markdown listing: Goal, Affected files, Step-by-step edits, "
        "Tests to add/update, Risks. Cite each affected location with [[chunk_id]]. "
        "Do NOT wrap in JSON."
    ),
    "symbol_location": (
        "You are a precise locator. Use ONLY the SOURCES. Reply in plain Markdown listing "
        "file paths and line ranges where the symbol is defined or primarily implemented, "
        "one per line, each followed by a one-line rationale and a [[chunk_id]] citation."
    ),
    "line_number": (
        "You are a precise locator for edit anchors. Use ONLY the SOURCES. Reply in plain "
        "Markdown listing candidate (file, line_range) edit points with a one-line "
        "justification and a [[chunk_id]] citation each."
    ),
    "overview": (
        "You are a senior codebase assistant. Use ONLY the SOURCES (and the optional README "
        "lead) to produce a concise repo overview in plain Markdown: Purpose, Major "
        "components, How they fit together, Entry points. Cite each claim with [[chunk_id]]. "
        "Do NOT wrap in JSON."
    ),
    "qa": (
        "You are a senior codebase assistant answering a specific question. Use ONLY the "
        "SOURCES (and the optional README lead) to answer the QUESTION directly in plain "
        "Markdown. Do NOT impose a fixed section template (no forced Purpose / Components / "
        "Entry-Points headings); pick the shape that best fits the question -- a short "
        "paragraph, a list, a small table, or a brief code excerpt drawn from SOURCES. "
        "Stay focused on what was asked; do not pivot into a generic repo summary. Cite "
        "every non-trivial claim inline with [[chunk_id]]. If SOURCES do not cover the "
        "question, say so plainly and name what would be needed. Do NOT wrap in JSON."
    ),
    "clarify_paths": (
        "You are a senior codebase assistant responding to an OPEN-ENDED goal (e.g. "
        "'improve X', 'suggest ways to Y'). Your job is NOT to commit to a single answer "
        "-- it is to help the user choose a direction. Use the SOURCES to ground "
        "suggestions in this codebase. Reply in plain Markdown structured as: (1) a "
        "one-sentence restatement of the goal as you understand it; (2) a numbered list "
        "of 3-5 concrete directions, each with a short rationale and -- when supported "
        "by SOURCES -- an inline [[chunk_id]] citation pointing to the component it "
        "would touch; (3) one follow-up question asking which direction to pursue or "
        "what constraint matters most. Do NOT propose code edits and do NOT invent "
        "components absent from SOURCES. Do NOT wrap in JSON."
    ),
}


def _get_stream_system_prompt(mode: str) -> str:
    """Return the markdown-direct system prompt for ``mode`` (streaming path)."""
    return SYSTEM_PROMPTS_STREAM.get(mode, SYSTEM_STREAM)


def _auto_retrieve_hits(
    index_dir: str,
    records_path: str,
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run hybrid retrieval for callers that didn't pre-compute ``hits``.

    Routes through :func:`cgx.pipeline.auto.run_query_auto` -- the same
    semantic+lexical+graph pipeline the web UI's ask path uses -- so the
    SOURCES reflect the question. ``run_query_auto`` auto-selects the embed
    model recorded in the index manifest, so this stays correct for local
    models built with a non-default embedder. Any failure (missing index,
    embedder error) degrades to an empty list so the caller can fall back to
    leading rows rather than crash.
    """
    try:
        from cgx.pipeline.auto import run_query_auto  # lazy: avoid import cycle
        out_dir = Path(index_dir).parent
        cp = out_dir / "chunks.jsonl"
        gp = out_dir / "graph.json"
        retrieval = run_query_auto(
            index_dir=index_dir,
            records_path=records_path,
            query=question or "",
            chunks_path=str(cp) if cp.exists() else None,
            graph_path=str(gp) if gp.exists() else None,
            top_k_per_view=max(int(top_k), 20),
            neighbor_depth=1,
            use_lexical=True,
        )
        return retrieval.get("hits", []) or []
    except Exception as e:
        logger.warning("_auto_retrieve_hits: retrieval failed (%s); "
                       "falling back to leading rows", e)
        return []


def _prepare_answer_request(
    index_dir: str,
    records_path: str,
    question: str,
    provider: LLMProvider,
    *,
    top_k: int = 20,
    hits: Optional[List[Dict[str, Any]]] = None,
    mode_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Shared retrieval + prompt-context prep for sync and streaming answer paths.

    Returns either ``('done', final_result)`` -- when an early-exit case
    fires (no hits, missing target, graph-based callers/callees
    short-circuit) -- or ``('ready', prep)`` where ``prep`` carries the
    materials both :func:`answer_with_llm` and
    :func:`answer_with_llm_stream` need: ``context`` (the user-role
    message body), ``sources``, ``mode``, ``target``, ``target_matched``,
    ``merged_hits``, ``root``, ``readme``.

    ``mode_override`` lets the caller force a specific prompt mode
    (e.g. ``"clarify_paths"`` for exploratory agent goals) instead of
    running :func:`detect_intent` on the question. Unknown modes fall
    back to the default SYSTEM prompt via :func:`_get_system_prompt`.

    Splitting this out keeps the JSON-postprocessing logic in the
    blocking path and the token-streaming logic in the streaming path
    from drifting apart.
    """
    indices = load_indices(index_dir)
    cmap = _chunk_map(indices)

    mode = mode_override if mode_override else detect_intent(question)

    # --- Deterministic endpoint enumeration short-circuit ---
    # "how many / list all API endpoints" is an aggregate over scattered
    # route-decorator chunks, which semantic ranking answers unreliably.
    # Answer it exactly from ``route`` metadata (see parse_codebase.
    # _detect_route). Only short-circuit when endpoints are actually found;
    # otherwise fall through to normal answering so stale/route-less indices
    # still get a reply instead of a confidently-wrong "0 endpoints".
    if mode == "enumerate":
        try:
            from cgx.answer.enumeration import answer_endpoint_enumeration
            recs = load_jsonl(records_path) if records_path else []
            result = answer_endpoint_enumeration(recs, question)
            if int(result.get("debug", {}).get("endpoint_count", 0)) > 0:
                return "done", result
        except Exception as e:
            logger.error("enumerate short-circuit failed: %s", e)
        # Fall through: treat as a normal grounded question.
        mode = "qa"

    # --- Improved Target symbol detection ---
    symbols = _symbol_tokens(question)
    target = None
    target_matched = False
    for t in symbols:
        if _find_symbol_rows(indices, t):
            target = t
            target_matched = True
            break
    if target is None and symbols:
        for t in reversed(symbols):
            if _find_symbol_rows(indices, t):
                target = t
                target_matched = True
                break
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
                    pairs = [(u, d) for (u, _v, d) in edges]
                else:
                    edges = list(G.out_edges(node, data=True))
                    header = f"Functions called by `{target}`"
                    pairs = [(v, d) for (_u, v, d) in edges]
                for nbr, edata in pairs:
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
            return "done", {
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
    elif not forced_hits:
        # No caller-supplied hits and no symbol match. Run real hybrid
        # retrieval (semantic + lexical + graph) so SOURCES reflect the
        # question. The previous fallback grabbed the first ``top_k`` rows
        # of each view -- arbitrary chunks unrelated to the query -- which
        # is the main reason programmatic asks (e.g. the agent's
        # INVESTIGATE step, which passes no ``hits``) returned ungrounded
        # answers. Leading-row selection remains only as a last resort when
        # retrieval yields nothing (missing/empty index).
        base_hits = _auto_retrieve_hits(index_dir, records_path, question, top_k)
        if not base_hits:
            logger.warning("_prepare_answer_request: _auto_retrieve_hits returned no hits for question: %r", question)

    seen = set()
    merged_hits: List[Dict[str, Any]] = []
    for h in forced_hits + base_hits:
        cid = str(h.get("chunk_id"))
        if cid not in seen:
            seen.add(cid); merged_hits.append(h)

    if not merged_hits:
        return "done", {
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

    _has_neighbors = any(
        int(((h.get("provenance") or {}) if isinstance(h, dict) else {}).get("graph_depth", 0) or 0) >= 1
        for h in merged_hits
    )
    from cgx.answer.model_caps import get_context_map_budget
    budget = get_context_map_budget(provider)
    if _has_neighbors:
        from cgx.answer.context_map import build_tiered_context, load_records_by_id
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
            total_chars_budget=budget.get("total_chars") if isinstance(budget, dict) else None,
        )

    if target and target_matched and mode in _SYMBOL_TARGETED_MODES:
        covers = [
            s for s in sources
            if _symbol_covers_target(s.get("symbol", ""), s.get("chunk_id", ""), target)
        ]
        if not covers:
            return "done", {
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

    return "ready", {
        "context": context,
        "sources": sources,
        "mode": mode,
        "target": target,
        "target_matched": target_matched,
        "merged_hits": merged_hits,
        "root": root,
        "readme": readme,
    }


# ---------------- clarify_paths: structured generation ----------------
#
# Small JSON-mode models (e.g. gemma3:4b via Ollama ``format: "json"``)
# routinely collapse the multi-section ``clarify_paths`` ``answer_md``
# contract into a single hallucinated paragraph that ignores SOURCES
# entirely. The cause is structural, not a prompting tweak: a freeform
# ``answer_md`` string is the easiest slot to fill, so the model fills
# only that and skips the requested enumeration. The functions below
# replace single-shot freeform generation with a typed-slot contract:
# pre-clustered CANDIDATES from the indexed sources, an ``options`` array
# with required ``chunk_id`` values drawn from those candidates,
# validation + one retry on insufficient enumeration, and deterministic
# Markdown rendering from the validated structure. The model selects
# from a closed set; it does not invent the list.


def _clarify_candidates_from_sources(
    sources: List[Dict[str, Any]], *, max_candidates: int = 8,
) -> List[Dict[str, Any]]:
    """Group ``sources`` by file path into compact candidate components.

    Retrieval already ordered ``sources`` by relevance, so the first
    occurrence of each path wins. Each candidate exposes the single
    ``chunk_id`` the LLM should cite plus a short list of distinct
    symbol names seen in that file -- enough context for the model to
    write a 1-2 sentence rationale without needing the full code body.
    """
    by_path: Dict[str, Dict[str, Any]] = {}
    for s in sources or []:
        path = str(s.get("path") or "")
        if not path:
            continue
        if path not in by_path:
            by_path[path] = {
                "path": path,
                "chunk_id": str(s.get("chunk_id") or ""),
                "symbols": [],
            }
        sym = str(s.get("symbol") or "").strip()
        if sym and sym not in by_path[path]["symbols"]:
            by_path[path]["symbols"].append(sym)
    return [c for c in by_path.values() if c["chunk_id"]][:max_candidates]


def _render_clarify_markdown(
    restatement: str,
    options: List[Dict[str, Any]],
    follow_up_question: str,
) -> str:
    """Assemble the user-facing ``answer_md`` from structured slots.

    Rendering happens deterministically here -- not inside the LLM --
    so the three-section contract (restatement / numbered options /
    follow-up question) is guaranteed regardless of how flaky the
    model's structured output was.
    """
    parts: List[str] = []
    if restatement:
        parts.append(restatement.strip())
        parts.append("")
    parts.append("**Possible directions:**")
    for i, opt in enumerate(options, 1):
        title = (opt.get("title") or "").strip() or "Investigate this component"
        rationale = (opt.get("rationale") or "").strip()
        line = f"{i}. **{title}**"
        if rationale:
            line += f" — {rationale}"
        cid = (opt.get("chunk_id") or "").strip()
        if cid:
            line += f" [[{cid}]]"
        parts.append(line)
    if follow_up_question:
        parts.append("")
        parts.append(f"_{follow_up_question.strip()}_")
    return "\n".join(parts).strip()


def _validate_clarify_options(
    raw: Any, candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Coerce the LLM's ``options`` payload into a clean, deduped list.

    Drops entries whose ``chunk_id`` is missing or absent from
    ``candidates``, dedupes by chunk_id (so the model can't pad the list
    with the same component twice), and caps at 5. Empty rationales and
    titles are kept so the renderer can fall back to placeholders.
    """
    allowed = {c["chunk_id"] for c in candidates}
    out: List[Dict[str, Any]] = []
    seen: set = set()
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("chunk_id") or "").strip()
        if not cid or cid not in allowed or cid in seen:
            continue
        seen.add(cid)
        out.append({
            "title": str(entry.get("title") or "").strip(),
            "rationale": str(entry.get("rationale") or "").strip(),
            "chunk_id": cid,
        })
        if len(out) >= 5:
            break
    return out


def _clarify_system_prompt() -> str:
    return (
        "You are a senior engineer triaging an OPEN-ENDED user goal "
        "against a specific codebase. Return STRICT JSON with these keys "
        "and NO prose outside JSON:\n"
        "  - restatement: one sentence paraphrasing the goal\n"
        "  - options: array of 3 to 5 objects, each with:\n"
        "      - title: short imperative phrase (4-10 words)\n"
        "      - rationale: 1-2 sentences explaining the direction, "
        "grounded in the chosen CANDIDATE\n"
        "      - chunk_id: the chunk_id of the CANDIDATE this option "
        "touches (verbatim, copied from CANDIDATES)\n"
        "  - follow_up_question: one clarifying question to narrow "
        "the choice (e.g. accuracy vs latency vs code-size)\n"
        "Rules:\n"
        "- options MUST have between 3 and 5 entries; pad with the most "
        "promising candidates when unsure.\n"
        "- Each chunk_id MUST appear verbatim in CANDIDATES below.\n"
        "- Different options SHOULD touch different components when "
        "candidates allow it.\n"
        "- Do NOT invent components absent from CANDIDATES and do NOT "
        "propose code edits."
    )


def _clarify_user_message(
    goal: str,
    candidates: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    retry_hint: Optional[str] = None,
) -> str:
    cand_lines: List[str] = []
    for c in candidates:
        syms = ", ".join(c.get("symbols") or [])
        line = f"- chunk_id={c['chunk_id']}\n    path={c['path']}"
        if syms:
            line += f"\n    symbols=[{syms}]"
        cand_lines.append(line)
    cand_block = "\n".join(cand_lines) if cand_lines else "(no candidates)"
    src_block = "\n".join(_fmt_source(s) for s in (sources or [])[:8])
    parts = [
        f"GOAL:\n{goal.strip()}",
        f"CANDIDATES:\n{cand_block}",
        f"SOURCES (excerpts):\n{src_block}" if src_block else "",
    ]
    if retry_hint:
        parts.append(f"RETRY GUIDANCE:\n{retry_hint.strip()}")
    return "\n\n".join(p for p in parts if p)


def _recover_clarify_options_from_prose(
    content: str, candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Recover structured options from a markdown list if JSON extraction failed."""
    if not content or not candidates:
        return []
    allowed = {c["chunk_id"]: c for c in candidates}
    out: List[Dict[str, Any]] = []
    seen = set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        cites = _INLINE_CITATION_RE.findall(line)
        valid_cid = None
        for cid in cites:
            if cid in allowed and cid not in seen:
                valid_cid = cid
                break
        if not valid_cid:
            continue
        seen.add(valid_cid)
        text = _INLINE_CITATION_RE.sub("", line).strip(" -*#1234567890.:")
        parts = re.split(r"[-—:]", text, maxsplit=1)
        title = parts[0].replace("**", "").replace("*", "").strip() or f"Investigate {valid_cid}"
        rationale = (parts[1].strip() if len(parts) > 1 else text).replace("**", "").replace("*", "").strip()
        out.append({
            "title": title[:60],
            "rationale": rationale,
            "chunk_id": valid_cid,
        })
        if len(out) >= 5:
            break
    return out


def _call_clarify_llm(
    provider: LLMProvider,
    goal: str,
    candidates: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    retry_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Single LLM round-trip for the structured clarify_paths schema.

    Mirrors the reformat-retry pattern used elsewhere in this module so
    a single transient JSON-shape miss doesn't abort the whole flow.
    Returns the parsed object (may be empty on hard failure).
    """
    messages = [
        {"role": "system", "content": _clarify_system_prompt()},
        {"role": "user", "content": _clarify_user_message(
            goal, candidates, sources, retry_hint)},
    ]
    kwargs: Dict[str, Any] = {}
    if getattr(provider, "supports_json_schema", False):
        kwargs["force_json"] = True
        kwargs["json_schema"] = CLARIFY_PATHS_SCHEMA
    resp = provider.chat(messages, temperature=0.2, **kwargs)
    content = (resp.get("content") or "").strip()
    parsed = _extract_json_object(content)
    if isinstance(parsed, dict) and parsed.get("options"):
        return parsed
    recovered = _recover_clarify_options_from_prose(content, candidates)
    if recovered:
        return {
            "restatement": goal,
            "options": recovered,
            "follow_up_question": "Which of these directions would you like to pursue first?",
        }
    if not parsed or not isinstance(parsed, dict):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": (
            "Reformat your previous reply as STRICT JSON exactly matching "
            "the schema (restatement / options[] / follow_up_question). "
            "No prose outside JSON.")})
        resp2 = provider.chat(messages, temperature=0, **kwargs)
        parsed = _extract_json_object((resp2.get("content") or "")) or {}
        if not (isinstance(parsed, dict) and parsed.get("options")):
            recovered2 = _recover_clarify_options_from_prose((resp2.get("content") or ""), candidates)
            if recovered2:
                return {
                    "restatement": goal,
                    "options": recovered2,
                    "follow_up_question": "Which of these directions would you like to pursue first?",
                }
    return parsed if isinstance(parsed, dict) else {}


def _answer_clarify_paths(
    prep: Dict[str, Any],
    question: str,
    provider: LLMProvider,
    root: Optional[str],
) -> Dict[str, Any]:
    """Structured generation path for the ``clarify_paths`` mode.

    Replaces the generic single-shot ``answer_md`` flow when the agent
    forces ``mode_override="clarify_paths"``. The contract is:

    * Cluster ``prep["sources"]`` into candidate components keyed by
      file path; the model picks from this closed set instead of
      inventing items.
    * Call the LLM with a typed-slot schema (``restatement``,
      ``options[]`` with ``chunk_id`` ∈ CANDIDATES, ``follow_up_question``).
    * Validate the returned options; retry once with explicit feedback
      when fewer than three survive validation.
    * Synthesize the remaining slots deterministically from the top
      candidates so the user never sees an empty list when retrieval
      itself succeeded.
    * Render the final ``answer_md`` here -- not in the LLM -- so the
      three-section shape is guaranteed.
    """
    sources: List[Dict[str, Any]] = prep.get("sources") or []
    merged_hits: List[Dict[str, Any]] = prep.get("merged_hits") or []
    candidates = _clarify_candidates_from_sources(sources)

    if not candidates:
        # Retrieval surfaced nothing usable; degrade gracefully rather
        # than asking the model to invent components.
        return {
            "answer_md": (
                "I don't have enough indexed context to suggest concrete "
                "directions for that goal yet. Re-index the repository or "
                "narrow the goal to a specific module/file."),
            "citations": [],
            "suggested_changes": [],
            "confidence": 0.2,
            "debug": {
                "mode": "clarify_paths", "sources": sources,
                "hits": merged_hits, "candidates": [],
                "options_count": 0,
            },
        }

    parsed = _call_clarify_llm(provider, question, candidates, sources)
    options = _validate_clarify_options(parsed.get("options"), candidates)

    if len(options) < 3:
        retry_hint = (
            f"Your previous reply produced {len(options)} valid option(s). "
            f"Return between 3 and 5 options; each option's chunk_id MUST "
            f"appear verbatim in CANDIDATES, and different options should "
            f"touch different components when possible.")
        parsed_retry = _call_clarify_llm(
            provider, question, candidates, sources, retry_hint=retry_hint)
        retry_opts = _validate_clarify_options(
            parsed_retry.get("options"), candidates)
        if len(retry_opts) > len(options):
            options = retry_opts
            parsed = parsed_retry

    # Deterministic backfill from candidates so the user always sees at
    # least three concrete pointers when retrieval worked, even if the
    # model failed twice in a row to fill the slots.
    if len(options) < 3:
        seen = {o.get("chunk_id") for o in options}
        for c in candidates:
            if len(options) >= 3:
                break
            cid = c["chunk_id"]
            if cid in seen:
                continue
            label = (c.get("symbols") or [None])[0]
            if not label:
                label = Path(c["path"]).stem or "this component"
            options.append({
                "title": f"Review {label}",
                "rationale": (
                    f"Investigate `{c['path']}` for changes relevant to "
                    f"the goal."),
                "chunk_id": cid,
            })

    restatement = str(parsed.get("restatement") or "").strip()
    if not restatement:
        restatement = f"Goal restated: {question.strip()}"
    follow_up = str(parsed.get("follow_up_question") or "").strip()
    if not follow_up:
        follow_up = (
            "Which of these directions should we pursue first, or is there "
            "a constraint (accuracy / latency / code-size / risk) that "
            "should drive the choice?")

    answer_md = _render_clarify_markdown(restatement, options, follow_up)
    answer_md = _shorten_chunk_refs(answer_md, root)

    allowed_ids = [s.get("chunk_id") for s in sources]
    citations = _sanitize_citations(
        [{"chunk_id": o["chunk_id"]} for o in options if o.get("chunk_id")],
        allowed_ids,
    )

    return {
        "answer_md": answer_md,
        "citations": citations,
        "suggested_changes": [],
        "confidence": 0.7 if len(options) >= 3 else 0.4,
        "debug": {
            "mode": "clarify_paths",
            "sources": sources,
            "hits": merged_hits,
            "candidates": candidates,
            "options": options,
            "restatement": restatement,
            "follow_up_question": follow_up,
            "options_count": len(options),
            "follow_up_present": True,
        },
    }


@traced("llm")
def answer_with_llm(
    index_dir: str,
    records_path: str,
    question: str,
    provider: LLMProvider,
    *,
    top_k: int = 20,
    hits: Optional[List[Dict[str, Any]]] = None,
    mode_override: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """
    Retrieve context from indices/graph and ask the LLM to synthesize a grounded answer.

    ``mode_override`` lets the agent loop force a specific prompt mode
    (e.g. ``"clarify_paths"``) regardless of the question's surface form.
    Extra kwargs are accepted and ignored so task-level ``inputs`` dicts
    can carry agent-only hints (``goal``, …) without breaking the call.
    """
    kind, payload = _prepare_answer_request(
        index_dir, records_path, question, provider,
        top_k=top_k, hits=hits, mode_override=mode_override,
    )
    if kind == "done":
        return payload

    prep = payload
    mode = prep["mode"]
    target = prep["target"]
    sources = prep["sources"]
    merged_hits = prep["merged_hits"]
    root = prep["root"]
    readme = prep["readme"]

    # ``clarify_paths`` uses a structured-slot contract (typed options[]
    # validated against candidate components) instead of asking the
    # model to freeform a multi-section ``answer_md``. Small models in
    # JSON mode collapse the freeform variant into one paragraph that
    # ignores SOURCES; the structured path renders the final Markdown
    # deterministically from validated slots so the three-section shape
    # is guaranteed.
    if mode == "clarify_paths":
        return _answer_clarify_paths(prep, question, provider, root)

    messages = [
        {"role": "system", "content": _get_system_prompt(mode)},
        {"role": "user", "content": prep["context"]},
    ]

    resp = provider.chat(messages, temperature=0.2)
    content = (resp.get("content") or "").strip()

    parsed: Dict[str, Any] = _extract_json_object(content)

    if not parsed or not isinstance(parsed, dict) or not parsed.get("answer_md"):
        if content and not content.strip().startswith("{"):
            parsed = {"answer_md": content, "citations": []}
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Reformat to strict JSON only. "
                                                       "Ensure non-empty 'answer_md' grounded in SOURCES with citations. "
                                                       "Keep the same content; do not add external knowledge."})
            resp2 = provider.chat(messages, temperature=0)
            parsed = _extract_json_object((resp2.get("content") or "")) or {"answer_md": (resp2.get("content") or content), "citations": []}

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
    if not parsed.get("citations"):
        raw_cites = [{"chunk_id": m.group(1)} for m in _INLINE_CITATION_RE.finditer(parsed.get("answer_md", ""))]
        parsed["citations"] = _sanitize_citations(raw_cites, allowed_ids)
    else:
        parsed["citations"] = _sanitize_citations(parsed.get("citations", []), allowed_ids)
    parsed["answer_md"] = _sanitize_inline_citations(parsed.get("answer_md", ""), allowed_ids)
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


def answer_with_llm_stream(
    index_dir: str,
    records_path: str,
    question: str,
    provider: LLMProvider,
    *,
    top_k: int = 20,
    hits: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    mode_override: Optional[str] = None,
    **_ignored: Any,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Stream the grounded answer as ``(event, data)`` tuples.

    Emits ``("answer_delta", {"delta": "..."})`` for each Markdown token
    produced by ``provider.chat_stream``, and finally a single
    ``("answer", {...})`` event whose payload matches the dict shape
    returned by :func:`answer_with_llm` (``answer_md``, ``citations``,
    ``suggested_changes``, ``confidence``, ``debug``).

    Early-exit cases from :func:`_prepare_answer_request` (no hits,
    missing target, graph-based callers/callees short-circuit) yield only
    a final ``answer`` event with no deltas, matching the blocking path.
    """
    kind, payload = _prepare_answer_request(
        index_dir, records_path, question, provider,
        top_k=top_k, hits=hits, mode_override=mode_override,
    )
    if kind == "done":
        yield "answer", payload
        return

    prep = payload
    mode = prep["mode"]
    target = prep["target"]
    sources = prep["sources"]
    merged_hits = prep["merged_hits"]
    root = prep["root"]
    readme = prep["readme"]

    messages = [
        {"role": "system", "content": _get_stream_system_prompt(mode)},
        {"role": "user", "content": prep["context"]},
    ]

    chunks: List[str] = []
    try:
        for delta in provider.chat_stream(
            messages,
            temperature=float(temperature),
            max_tokens=max_tokens,
        ):
            if not delta:
                continue
            chunks.append(delta)
            yield "answer_delta", {"delta": delta}
    except Exception as e:
        logger.error("answer_with_llm_stream: chat_stream failed: %s", e)
        yield "answer", {
            "answer_md": f"_Stream error: {type(e).__name__}: {e}_",
            "citations": [],
            "suggested_changes": [],
            "confidence": 0.0,
            "debug": {
                "mode": mode, "target_symbol": target,
                "sources": sources, "hits": merged_hits,
                "readme_included": bool(readme),
                "stream_error": f"{type(e).__name__}: {e}",
            },
        }
        return

    answer_md = "".join(chunks).strip()
    if not answer_md:
        answer_md = (
            "The provided SOURCES did not contain enough content to answer without guessing. "
            "Please re-index or narrow the question to a specific file or snippet."
        )

    allowed_ids = [s["chunk_id"] for s in sources]
    raw_cites = [{"chunk_id": m.group(1)} for m in _INLINE_CITATION_RE.finditer(answer_md)]
    citations = _sanitize_citations(raw_cites, allowed_ids)

    answer_md = _shorten_chunk_refs(answer_md, root)

    yield "answer", {
        "answer_md": answer_md,
        "citations": citations,
        "suggested_changes": [],
        "confidence": 0.6 if citations else 0.4,
        "debug": {
            "mode": mode,
            "target_symbol": target,
            "sources": sources,
            "hits": merged_hits,
            "readme_included": bool(readme),
            "streamed": True,
        },
    }




@traced("llm")
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

# A fenced code block carrying no ``path=`` label. :data:`_CODE_FENCE_RE`
# requires the label, so a small model that emits a bare ```json``` fence
# has its only block discarded and the file drops with an empty patch.
# For a single-file request the lone block is unambiguously that file, so
# this looser pattern recovers its body as a last resort.
_ANY_CODE_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_+.\-]*[ \t]*\n(?P<body>.*?)```",
    re.DOTALL,
)


def _first_fenced_block_body(text: str) -> str:
    """Return the body of the first fenced code block, ignoring any path=.

    The single-file freeform fallback asks for exactly one file inside one
    fenced block, but a small model routinely omits the ``path=<...>``
    label the strict parser (:data:`_CODE_FENCE_RE`) requires, so the block
    is discarded and the file drops with a bare empty patch. Since the
    request is for a single file, any lone fenced block is unambiguously
    that file -- recover its body directly. Returns ``""`` when no fenced
    block is present.
    """
    if not isinstance(text, str) or "```" not in text:
        return ""
    m = _ANY_CODE_FENCE_RE.search(text)
    if not m:
        return ""
    return (m.group("body") or "").rstrip("\n")


def _content_to_new_file_patch(path: str, content: str) -> str:
    """Convert complete file content into a new-file unified diff (--- /dev/null style)."""
    lines = content.splitlines()
    n = len(lines)
    header = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{n} @@\n"
    body = "\n".join(f"+{line}" for line in lines)
    return header + (body if body else "+")


def _new_file_body_from_patch(patch: str) -> Optional[str]:
    """Reverse :func:`_content_to_new_file_patch`: recover a new file's body.

    The freeform scaffold parser wraps a raw file body into a ``--- /dev/null``
    new-file diff, but the single-file path needs the plain body back --
    otherwise the diff text leaks in as file content and trips the
    unified-diff-header guard. Returns ``None`` when the patch is not a pure
    new-file addition (context/removal lines, or content before the hunk),
    since such a modification diff cannot be losslessly turned into a body.
    """
    if not patch:
        return None
    seen_hunk = False
    body_lines: List[str] = []
    for ln in patch.splitlines():
        if ln.startswith(("--- ", "+++ ")):
            continue
        if ln.startswith("@@"):
            seen_hunk = True
            continue
        if not seen_hunk:
            return None
        if ln.startswith("+"):
            body_lines.append(ln[1:])
        elif ln.startswith(("-", " ")):
            return None
        else:
            body_lines.append(ln)
    if not seen_hunk:
        return None
    return "\n".join(body_lines)


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
    "Path discipline (CRITICAL -- read carefully):\n"
    "- The deployment directory IS the project root. Emit paths RELATIVE to it.\n"
    "- NEVER prepend a top-level project folder. WRONG: 'calculator/src/App.jsx', "
    "'my-project/backend/app.py'. RIGHT: 'src/App.jsx', 'backend/app.py'.\n"
    "- Use this canonical layout so sibling scaffold tasks (UI + backend + tests) "
    "share one coherent tree:\n"
    "    src/        -- frontend source OR main code for single-language projects\n"
    "    backend/    -- Python/Node backend service (only when the project has a "
    "separate backend distinct from the frontend in src/)\n"
    "    tests/      -- ALL test files live here (test_*.py for Python, *.test.jsx "
    "or *.test.ts for JS/TS). REQUIRED -- every scaffold MUST emit at least one "
    "test file covering its primary logic.\n"
    "    public/     -- static assets (favicons, images) for frontend projects. "
    "The HTML entry point is NOT here -- it is index.html at the project root.\n"
    "- All paths must be lowercase-with-underscores or kebab-case. No spaces.\n"
    "Requirements:\n"
    "- content: complete, working file content -- NOT stubs or placeholders\n"
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
    "  * Use Vite -- it is the only supported frontend toolchain. Include "
    "vite.config.js, and index.html AT THE PROJECT ROOT (never public/index.html: "
    "Vite resolves its entry module from the root file and the build fails "
    "without it). The root index.html must load the script entry with "
    "<script type=\"module\" src=\"/src/main.jsx\"> and contain the mount element.\n"
    "  * Include src/main.jsx (or src/main.tsx) as the React/Vue entry point\n"
    "  * Tests go under tests/ as <Component>.test.jsx using Jest + "
    "@testing-library/react conventions.\n"
    "  * Do NOT generate webpack.config.js, babel.config.js, react-scripts, or any "
    "other build tooling\n"
    "  * Do NOT include conftest.py, requirements.txt, or any Python files\n"
    "- For PYTHON projects only: include a conftest.py at the project root "
    "that adds src/ to sys.path so test imports work without package installation:\n"
    "  conftest.py content: import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))\n"
    "  Test files must import from module names as they exist under src/ "
    "(e.g. 'from main import app', not 'from src.main import app').\n"
    "- Generate production-quality code with proper error handling and type hints where idiomatic.\n"
    "- Do NOT use TODO, FIXME, or placeholder comments -- write the real code.\n"
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
    "backend/ (Python backend when distinct), tests/ (REQUIRED -- at least one "
    "test file per scaffold), public/ (static assets other than the HTML entry "
    "point).\n"
    "Rules:\n"
    "- Use any language tag (python, javascript, jsx, tsx, text, yaml, toml, etc.) followed by path=<relative/path>\n"
    "- Generate complete, working code -- not stubs\n"
    "- Include README.md and the appropriate dependency file for the technology\n"
    "- Emit at least one real test file under tests/ exercising the main logic.\n"
    "- Use relative POSIX paths only\n"
    "- CRITICAL: If the goal names a specific technology (React, Vue, FastAPI, etc.), use it exactly. "
    "Do NOT substitute a different technology.\n"
    "- For FRONTEND projects (React, Vue, etc.): use Vite -- generate component files "
    "(src/App.jsx, src/main.jsx), vite.config.js, and index.html AT THE PROJECT ROOT "
    "(never public/index.html: Vite resolves its entry module from the root file). "
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
    # leading "." -- otherwise dotfiles like ".env.example" / ".gitignore"
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
            f"- Use functional components with hooks (useState, useEffect).\n"
            f"- IMPORTANT: Be precise with relative imports. From `src/main.jsx` (or any `src/` file), "
            f"import sibling components as `./components/...`. For test files under `tests/`, "
            f"remember that `tests/` and `src/` are siblings, so use `../src/components/...`."
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
    if ext == "py":
        return (
            "FILE EXTENSION CONSTRAINT (.py):\n"
            "- This is a Python source file. Write valid Python 3 code.\n"
            "- IMPORTANT: Do NOT execute side-effects (like database initialization, "
            "network requests, or starting a server) at the module level. Put them "
            "inside `if __name__ == '__main__':` or a dedicated startup function. "
            "The file must be safely importable without crashing or hanging.\n"
            "- Do NOT import fixtures from `conftest.py` (e.g. `from conftest import client`). "
            "Pytest automatically injects fixtures into test functions.\n"
            "- Ensure all referenced symbols (e.g. `init_db`) are imported from their correct "
            "modules or defined locally. Do NOT use undefined names."
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
        # A .jsx/.tsx file must look like React -- at minimum it should
        # mention React or export a component. A 3B model occasionally
        # emits a plain HTML document under .jsx; catch that.
        is_html_doc = lc.lstrip().startswith(("<!doctype", "<html"))
        has_react_signal = (
            has_react_import
            or has_react_jsx_export
            or "export default" in lc
            or "export {" in lc        # named exports: export { App }
            or "return <" in lc        # JSX return statement -- strong React indicator
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
        # token (".css\nbody { … }"). Reject that pattern -- a real
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
    existing index -- it generates all project files from scratch using
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
        # Cap the list -- local models can't reason over 200+ paths and the
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
    "Your job is to OUTPUT ONLY A FILE MANIFEST -- paths and one-line descriptions. "
    "Do NOT write any file contents.\n\n"
    "Return strict JSON only:\n"
    "{\n"
    '  "plan_md": "2-4 sentence architecture overview",\n'
    '  "contracts": {\n'
    '    "project_skeleton": "def login_user(token: str) -> bool: pass",\n'
    '    "endpoints": [{"method": "POST", "path": "/api/x",\n'
    '      "request": {"a": "number"}, "response": {"result": "number"},\n'
    '      "description": "one-line purpose"}],\n'
    '    "schemas": [{"name": "Thing", "fields": {"id": "int", "name": "str"}}],\n'
    '    "functions": [{"name": "compute",\n'
    '      "signature": "compute(a: float, b: float) -> float",\n'
    '      "module": "src/core.py"}],\n'
    '    "constants": [{"name": "API_BASE", "value": "/api"}]\n'
    "  },\n"
    '  "layers": [\n'
    '    {\n'
    '      "name": "core|ui|config|tests",\n'
    '      "files": [\n'
    '        {"path": "src/foo.py", "description": "one-line purpose",\n'
    '         "depends_on": ["src/bar.py"]}\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Relative POSIX paths only. No top-level project-name prefix "
    "(wrong: calculator/src/App.jsx, right: src/App.jsx).\n"
    "- Group files strictly into the following layers in order:\n"
    "  1. Layer 1 (No dependencies): Models, Configs, Utils (e.g., src/models.py, backend/config.py).\n"
    "  2. Layer 2 (Depends on L1): Core logic, Auth (e.g., src/core.py, src/auth.py).\n"
    "  3. Layer 3 (Depends on L1 & L2): API Routes, Main App (e.g., backend/main.py).\n"
    "  4. Layer 4 (Depends on everything): Tests (tests/test_*.py).\n"
    "- Test files REQUIRED under tests/.\n"
    "- Optional per file: \"depends_on\" lists sibling manifest paths this "
    "file imports/needs so files generate dependency-first. Every entry "
    "must be spelled exactly as a path listed in this manifest -- never a "
    "package name (wrong: \"react\", \"fastapi\", \"pytest\"), never a "
    "directory, never a file you did not list.\n"
    "- depends_on is a build order and MUST be acyclic. A test depends on "
    "the module it exercises; that module never depends on its test (wrong: "
    "backend/main.py depends_on tests/test_main.py). If two files each seem "
    "to need the other, move the shared piece into a third file both depend "
    "on.\n"
    "- A test must be written in the language of the code it covers, and "
    "depends_on must never cross languages: Python cannot import .jsx/.ts, "
    "and JS cannot import .py. React components are tested by a JS test "
    "(src/App.test.jsx), Python modules by a pytest file "
    "(tests/test_app.py). Wrong: tests/test_main.py depends_on "
    "src/App.jsx. A full-stack project therefore needs BOTH kinds of test, "
    "and a frontend/backend pair talks over HTTP -- never by importing "
    "across the boundary.\n"
    "- contracts (optional, but STRONGLY preferred for any multi-file or "
    "client/server project): declare the shared interfaces every file must "
    "agree on. MUST include a 'project_skeleton' string containing the folder structure and signatures of all files (classes, function names, type hints, docstrings) with 'pass' in the bodies.\n"
    "- 3 to 15 files total. Prefer completeness over brevity.\n"
    "- Canonical top-level dirs: src/, backend/, tests/, public/, docs/.\n"
    "- A frontend manifest uses Vite and MUST list index.html at the project "
    "root (not public/index.html) alongside vite.config.js -- Vite resolves its "
    "entry module from that file, so a manifest without it cannot build.\n"
)


@traced("llm")
def generate_project_skeleton(paths: List[str], provider: Any, goal: str) -> str:
    """Generate a code skeleton for all files before scaffolding begins."""
    if not provider:
        return ""
    system = (
        "You are an API architect. Generate a complete Project Skeleton for the given files.\n"
        "Return a unified string containing the folder structure and ALL file signatures "
        "(classes, function names, type hints, docstrings, AND module-level variables/constants like FastAPI 'app' or configuration objects). "
        "Use 'pass' for all bodies. Do NOT write implementation logic. Return ONLY the skeleton in a python code block."
    )
    user_msg = f"Project Goal:\n{goal}\n\nManifest Paths:\n" + "\n".join(f"- {p}" for p in paths)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    resp = provider.chat(messages=messages, max_tokens=4000, temperature=0.0)
    text = (resp or {}).get("content", "") if isinstance(resp, dict) else ""
    skeleton = _first_fenced_block_body(text) or text
    emit_trace("project_skeleton", skeleton=skeleton)
    return skeleton


def plan_scaffold_manifest(
    idea: str,
    provider: Any,
    *,
    goal: Optional[str] = None,
    skills: Optional[List[str]] = None,
    existing_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a file manifest for a new project -- paths and descriptions only, no content.

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
        parts.append("EXISTING FILES (already planned -- do NOT repeat):\n" + listed)

    # One schema re-ask for the whole planning call: constrained decoding
    # makes violations rare, and a model that misses the shape twice on the
    # same conversation is not going to converge on a third attempt.
    reask_left = [1]

    def _chat(messages: List[Dict[str, str]]) -> str:
        # Manifest generation is a structural step validated by a
        # deterministic Judge (required files, layer shape). Sampling
        # variance here turns the same prompt into different file trees
        # across retries and makes pass/fail outcomes flaky, so we pin
        # the temperature to 0.
        resp = provider.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=6000,
            force_json=True,
            json_schema=MANIFEST_SCHEMA,
        )
        if isinstance(resp, dict) and resp.get("error"):
            logger.warning("plan_scaffold_manifest: provider returned error -- %s",
                           resp.get("error"))
        return (resp or {}).get("content", "") if isinstance(resp, dict) else ""

    def _coerce_manifest_layers(p: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(p, dict):
            return p
        if not p.get("layers") and isinstance(p.get("files"), list) and p["files"]:
            p["layers"] = [{"name": "project", "files": p["files"]}]
        elif isinstance(p.get("layers"), dict) and ("files" in p["layers"] or "name" in p["layers"]):
            p["layers"] = [p["layers"]]
        return p

    def _call(user_msg: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        raw = _chat(messages)
        parsed = _coerce_manifest_layers(_extract_json_object(raw) or {})
        violations = validate_json_schema(parsed, MANIFEST_SCHEMA)
        if violations and reask_left[0]:
            reask_left[0] = 0
            logger.warning("plan_scaffold_manifest: schema violation, "
                           "re-asking -- %s", "; ".join(violations[:4]))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                "Your reply did not match the required schema. Violations:\n- "
                + "\n- ".join(violations[:8])
                + "\n\nReply again with STRICT JSON only, exactly matching the "
                  'schema in the system message: top-level "plan_md" (string), '
                  'optional "contracts" (object), and a non-empty "layers" '
                  'array of {"name", "files": [{"path", "description"}]}. '
                  "No prose outside JSON.")})
            reparsed = _coerce_manifest_layers(_extract_json_object(_chat(messages)) or {})
            if not validate_json_schema(reparsed, MANIFEST_SCHEMA):
                return reparsed
        return parsed

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
            retry_context += "\n\nEXISTING FILES (already planned -- do NOT repeat):\n" + listed
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
    layers = _inject_required_test_file(
        layers,
        goal=goal_clean or idea_clean,
        skill_names=skills,
    )
    layers = _inject_python_package_inits(layers)
    layers = _inject_readme(layers, goal=goal_clean or idea_clean)
    return {
        "plan_md": str(parsed.get("plan_md") or "").strip(),
        "contracts": _normalize_contracts(parsed.get("contracts")),
        "layers": layers,
    }


# Recognised contract categories, in render order. Anything outside this
# set is dropped so a noisy planner reply cannot bloat the per-file prompt.
_CONTRACT_KEYS = ("endpoints", "schemas", "functions", "constants")


def _normalize_contracts(raw: Any) -> Dict[str, Any]:
    """Normalize a planner ``contracts`` block to a clean, bounded dict.

    Keeps only the four recognised interface categories (HTTP endpoints,
    data schemas, shared function signatures, shared constants); each is a
    list of small string-keyed dicts with empty/malformed entries dropped.
    Absent or unusable categories are omitted so an empty block returns
    ``{}`` and callers can skip the section entirely.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _CONTRACT_KEYS:
        items = raw.get(key)
        if not isinstance(items, list):
            continue
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = {
                str(k): v for k, v in item.items()
                if isinstance(k, str) and str(k).strip()
                and isinstance(v, (str, int, float, bool, list, dict))
            }
            if entry:
                cleaned.append(entry)
        if cleaned:
            out[key] = cleaned
    return out


_ENDPOINT_EXTRACT_SYSTEM = (
    "You are an API contract extractor. Given a project goal and its file "
    "manifest (paths + one-line descriptions), OUTPUT ONLY the HTTP endpoints "
    "the frontend client and the backend server must agree on. Do NOT write "
    "file contents and do NOT add files.\n\n"
    "Return strict JSON only:\n"
    "{\n"
    '  "endpoints": [{"method": "POST", "path": "/api/x",\n'
    '    "request": {"a": "number"}, "response": {"result": "number"},\n'
    '    "status": 200, "message": "one-line success message",\n'
    '    "description": "one-line purpose"}]\n'
    "}\n\n"
    "Rules:\n"
    "- request/response keys are the EXACT JSON field names the client sends "
    "and the server reads -- choose one spelling per field and keep it "
    "identical on both sides (do not offer synonyms).\n"
    "- status is the success HTTP status code the handler returns and the "
    "paired test asserts (e.g. 200 for a read, 201 for a create); choose one "
    "and keep it identical on both sides.\n"
    "- message, when the endpoint returns a human-readable success string, is "
    "the EXACT literal the handler emits and the test asserts (omit it when "
    "there is none).\n"
    "- Cover every endpoint the client calls and the server routes; nothing "
    "more.\n"
    "- Use the paths the manifest declares or clearly implies. No prose "
    "outside the JSON object.\n"
)


@traced("llm")
def extract_endpoint_contracts(
    goal: str,
    layers: List[Dict[str, Any]],
    provider: Any,
) -> List[Dict[str, Any]]:
    """Second-pass extractor: derive the HTTP endpoint contracts for a manifest.

    Used when DECOMPOSE detects a cross-language client/server manifest whose
    planner reply omitted the ``endpoints`` contract -- the exact blind spot
    that let ses_4cbf963cdc67435a ship a frontend/backend request-key
    mismatch. One bounded, temperature-0 call over the goal + file manifest;
    returns a normalized list of endpoint dicts (possibly empty). Never
    raises: any provider or parse failure degrades to ``[]`` so the caller
    decides how to fail-close.
    """
    if provider is None:
        return []
    listed: List[str] = []
    for layer in (layers or []):
        for f in (layer.get("files") or []):
            path = str(f.get("path") or "").strip()
            if not path:
                continue
            desc = str(f.get("description") or "").strip()
            listed.append(f"- {path}: {desc}" if desc else f"- {path}")
    if not listed:
        return []
    manifest_txt = "\n".join(listed[:60])
    user = (f"PROJECT GOAL:\n{(goal or '').strip()}\n\n"
            f"FILE MANIFEST:\n{manifest_txt}\n\n"
            "Extract the HTTP endpoint contracts as strict JSON.")
    try:
        resp = provider.chat(
            messages=[
                {"role": "system", "content": _ENDPOINT_EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=1500,
            force_json=True,
        )
        raw = (resp or {}).get("content", "") if isinstance(resp, dict) else ""
    except Exception:  # pragma: no cover - defensive: extractor is best-effort
        logger.exception("extract_endpoint_contracts: provider call failed")
        return []
    parsed = _extract_json_object(raw) or {}
    normalized = _normalize_contracts({"endpoints": parsed.get("endpoints")})
    return normalized.get("endpoints") or []


def _compact_json_fragment(value: Any, max_chars: int = 300) -> str:
    """Render a contract sub-value (schema fields, request body) compactly.

    Returns an empty string for empty/absent values so the caller can omit
    the fragment; non-empty values are single-line JSON (or the raw string)
    clamped to ``max_chars`` so one verbose schema cannot dominate a prompt.
    """
    if value is None or value == "" or value == [] or value == {}:
        return ""
    try:
        if isinstance(value, str):
            s = value.strip()
        else:
            s = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        s = str(value)
    s = s.strip()
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "\u2026"
    return s


def _render_contracts_for_prompt(contracts: Any) -> str:
    """Render a normalized ``contracts`` block as a compact prompt fragment.

    Declares the shared interfaces every file must honour so cross-file
    assumptions (endpoint paths, schema field names, function signatures,
    constant values) are stated once instead of re-derived per file.
    Returns an empty string when there is nothing usable.
    """
    if not isinstance(contracts, dict) or not contracts:
        return ""
    sections: List[str] = []

    endpoints = contracts.get("endpoints")
    if isinstance(endpoints, list) and endpoints:
        lines: List[str] = []
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            path = str(ep.get("path") or "").strip()
            if not path:
                continue
            method = str(ep.get("method") or "").strip().upper()
            head = " ".join(p for p in (method, path) if p)
            tail: List[str] = []
            req = _compact_json_fragment(ep.get("request"))
            resp = _compact_json_fragment(ep.get("response"))
            if req:
                tail.append(f"request={req}")
            if resp:
                tail.append(f"response={resp}")
            status = ep.get("status")
            if isinstance(status, bool):
                status = None
            try:
                status_int = int(status) if status is not None else None
            except (TypeError, ValueError):
                status_int = None
            if status_int is not None:
                tail.append(f"success_status={status_int}")
            message = str(ep.get("message") or "").strip()
            if message:
                tail.append(f"success_message={message!r}")
            desc = str(ep.get("description") or "").strip()
            if desc:
                tail.append(desc)
            lines.append(f"- {head}"
                         + (f" -- {'; '.join(tail)}" if tail else ""))
        if lines:
            sections.append("Endpoints:\n" + "\n".join(lines))

    schemas = contracts.get("schemas")
    if isinstance(schemas, list) and schemas:
        lines = []
        for sc in schemas:
            if not isinstance(sc, dict):
                continue
            name = str(sc.get("name") or "").strip()
            if not name:
                continue
            bits = [b for b in (_compact_json_fragment(sc.get("fields")),
                                str(sc.get("description") or "").strip()) if b]
            lines.append(f"- {name}"
                         + (f": {'; '.join(bits)}" if bits else ""))
        if lines:
            sections.append("Schemas:\n" + "\n".join(lines))

    functions = contracts.get("functions")
    if isinstance(functions, list) and functions:
        lines = []
        for fn in functions:
            if not isinstance(fn, dict):
                continue
            sig = str(fn.get("signature") or fn.get("name") or "").strip()
            if not sig:
                continue
            module = str(fn.get("module") or "").strip()
            lines.append(f"- {sig}" + (f" (in {module})" if module else ""))
        if lines:
            sections.append("Shared functions:\n" + "\n".join(lines))

    constants = contracts.get("constants")
    if isinstance(constants, list) and constants:
        lines = []
        for c in constants:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            val = c.get("value")
            lines.append(f"- {name}"
                         + (f" = {val!r}" if val is not None else ""))
        if lines:
            sections.append("Shared constants:\n" + "\n".join(lines))

    skeleton = contracts.get("project_skeleton")
    if isinstance(skeleton, str) and skeleton.strip():
        sections.append("Project Skeleton:\n" + skeleton.strip())

    if not sections:
        return ""
    return ("PROJECT CONTRACTS (shared interfaces every file MUST honour "
            "exactly -- do not rename paths, fields, or signatures):\n"
            + "\n\n".join(sections))


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


# Test-file path conventions across the stacks CGX scaffolds. A manifest
# already carries tests when any planned path matches one of these: the
# pytest ``test_*.py`` / ``*_test.py`` names or the JS/TS ``*.test.*`` /
# ``*.spec.*`` names.
_TEST_FILE_RE = re.compile(
    r"(?:^|/)test_[^/]+\.py$"
    r"|(?:^|/)[^/]+_test\.py$"
    r"|\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$",
    re.IGNORECASE,
)


def _inject_required_test_file(
    layers: List[Any],
    *,
    goal: str = "",
    skill_names: Optional[List[str]] = None,
) -> List[Any]:
    """Guarantee every greenfield manifest carries at least one test file.

    A scaffold with no tests leaves the ``verify`` step nothing to run, so
    the self-correction loop has no pass/fail signal. Small models often
    skip tests despite the prompt; rather than burn a retry on the Judge's
    required-test check, we inject a stack-appropriate test entry whose
    content the per-file generator fills in later.
    """
    paths: List[str] = []
    for lay in layers or []:
        if not isinstance(lay, dict):
            continue
        for f in (lay.get("files") or []):
            if isinstance(f, dict):
                p = str(f.get("path") or "").strip()
                if p:
                    paths.append(p)
    if any(_TEST_FILE_RE.search(p) for p in paths):
        return layers

    goal_low = (goal or "").lower()
    names_low = {s.lower() for s in (skill_names or [])}
    exts = {"." + p.rsplit(".", 1)[-1].lower() for p in paths if "." in p}

    _JS = re.compile(
        r"\b(react|vue|svelte|next\.?js|express|angular|typescript|javascript|node(?:\.?js)?)\b",
        re.IGNORECASE,
    )
    _PY = re.compile(r"\b(python|fastapi|flask|django)\b", re.IGNORECASE)
    py_stack = (
        bool(_PY.search(goal_low))
        or bool(names_low & {"python", "fastapi", "flask", "django", "python_cli"})
        or ".py" in exts
    )
    js_stack = (
        bool(_JS.search(goal_low))
        or bool(names_low & {"react", "vue", "svelte", "nextjs", "express",
                             "angular", "typescript", "javascript", "node", "nodejs"})
        or bool(exts & {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
    )

    if py_stack or not js_stack:
        # Default to Python: pytest is always available, so the injected
        # test is runnable by ``verify`` even with no toolchain installed.
        entry = {
            "path": "tests/test_smoke.py",
            "description": (
                "pytest smoke test: import the project's primary module(s) "
                "and assert their core behaviour so `verify` has a runnable "
                "pass/fail signal"
            ),
        }
    else:
        ext = ("tsx" if ".tsx" in exts else "ts" if ".ts" in exts
               else "jsx" if ".jsx" in exts else "js")
        entry = {
            "path": f"tests/app.test.{ext}",
            "description": (
                "Vitest/Jest unit test covering the core component/logic so "
                "the build+test step has a real pass/fail signal"
            ),
        }

    out = list(layers)
    tests_layer = next(
        (lay for lay in out
         if isinstance(lay, dict)
         and str(lay.get("name") or "").lower() in ("tests", "test", "testing")),
        None,
    )
    if tests_layer is not None:
        tests_layer["files"] = list(tests_layer.get("files") or []) + [entry]
    else:
        out.append({"name": "tests", "files": [entry]})
    return out


def _inject_python_package_inits(layers: List[Any]) -> List[Any]:
    """Ensure every Python source directory has an ``__init__.py``.

    Small models emit ``backend/calculator.py`` but forget the package
    marker, which makes ``from backend.calculator import add`` work only
    under Python 3 namespace-package discovery -- and pytest's rootdir
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


def _inject_readme(layers: List[Any], *, goal: str = "") -> List[Any]:
    """Guarantee a project ``README.md`` generated LAST of all files.

    Every finished project should ship a top-level README. The manifest
    planner frequently omits it, or places it early where it cannot see
    the files it is meant to describe. We normalise both cases: any
    existing top-level ``README.md`` entry is dropped and a fresh one is
    appended to a trailing ``docs`` layer, so the ``SCAFFOLD_FILE`` task
    for the README runs after every other file and can summarise the
    real, generated project.
    """
    goal_hint = (goal or "").strip()
    description = (
        "Top-level project README in Markdown. Summarise what the project "
        "does, its tech stack, how to install/run it, and how to run the "
        "tests -- grounded strictly in the files already generated. "
        "Do NOT invent files, commands, or dependencies that do not exist."
    )
    if goal_hint:
        description = f"{description} Project goal: {goal_hint}"

    out: List[Any] = []
    for lay in layers or []:
        if not isinstance(lay, dict):
            out.append(lay)
            continue
        kept = [
            f for f in (lay.get("files") or [])
            if not (isinstance(f, dict)
                    and str(f.get("path") or "").strip().lower() == "readme.md")
        ]
        out.append({**lay, "files": kept})

    out.append({
        "name": "docs",
        "files": [{"path": "README.md", "description": description}],
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
            # Canonicalize any dependency hints the same way so a
            # depends_on pointing at a rewritten config file (e.g.
            # src/package.json -> package.json) does not read as dangling
            # to the DECOMPOSE coherence check.
            deps = f.get("depends_on")
            if isinstance(deps, list) and deps:
                canon_deps: List[Any] = []
                for d in deps:
                    ds = str(d or "").strip()
                    if not ds:
                        continue
                    db = ds.rsplit("/", 1)[-1].lower()
                    canon_deps.append(_CANONICAL_CONFIG_PATHS.get(db, ds))
                f = {**f, "depends_on": canon_deps}
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
    "- The project manifest (list of ALL file paths that will exist in this project)\n"
    "- The file path and its purpose\n"
    "- The content of files already generated in this project\n\n"
    "Output the COMPLETE content of the requested file only. "
    "Return strict JSON:\n"
    '{"content": "complete file content as a string"}\n\n'
    "Rules:\n"
    "- Output the full file -- no stubs, no placeholders, no ellipsis.\n"
    "- If you use standard libraries or external packages (e.g., sqlite3, os, React), you MUST explicitly import them at the top of the file. Do not leave undefined names.\n"
    "- Use imports consistent with what already exists in the project.\n"
    "- ONLY import local modules/components if they are explicitly listed in the PROJECT MANIFEST. You are FORBIDDEN from importing ANY local file or component that is not in the PROJECT MANIFEST.\n"
    "- You MUST strictly adhere to the PROJECT CONTRACTS and Project Skeleton. ONLY use symbols, functions, classes, and variables that are explicitly defined there. Do NOT hallucinate undefined variables (e.g. app, API_BASE) if they are not exported by the skeleton.\n"
    "- Assume the project root is the Python path. All imports must be absolute starting from the root directories: src., backend., or tests.. Never use 'import main', use 'from backend.main import router' (or whatever is explicitly in the skeleton).\n"
    "- Pay close attention to relative import paths (e.g., `./` vs `../`).\n"
    "- Do not repeat or regenerate any already-existing file.\n"
    "- The content MUST be functionally different from every file in "
    "ALREADY GENERATED FILES. Do NOT copy another file's body and rename "
    "the export -- write the unique content that fulfils THIS file's "
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
    "launched directly (streamlit run, python src/app.py, etc.) -- those "
    "scripts run as __main__ and have no parent package.\n"
    "Python web-framework test discipline (when the requested file is a "
    "pytest test under tests/ for a Flask / FastAPI / Starlette / Django app):\n"
    "- Exercise the application via the framework's in-process client "
    "(`app.test_client()` for Flask, `TestClient(app)` for FastAPI / "
    "Starlette, `Client()` for Django) -- never bind a real port or use "
    "`requests` / `httpx.Client(base_url='http://localhost:...')` in a test.\n"
    "- For Flask specifically: obtain the client ONLY from the app object, "
    "via `client = app.test_client()` (typically wrapped in a "
    "`@pytest.fixture`). Do NOT import a test client class from werkzeug. "
    "There is NO `werkzeug.test.TestClient` -- that symbol does not exist "
    "and raises AttributeError. `TestClient(app)` is a FastAPI/Starlette "
    "construct imported from `fastapi.testclient` / `starlette.testclient`, "
    "NOT something you import for a Flask app.\n"
    "- Import the application factory or app instance from the project's "
    "own source files (e.g. `from app import app` or `from main import "
    "create_app`); do not stub or reimplement the framework inline.\n"
    "- Tests should be self-contained: no `subprocess.Popen`, no "
    "background threads, no sleep-then-poll patterns.\n"
    "Pytest vs. unittest discipline (applies to ANY file under tests/):\n"
    "- Pick ONE style for the file. If the file uses plain `def test_*` "
    "functions or a class that does NOT inherit from "
    "`unittest.TestCase`, do not call `self.assert*` / `self.assertLogs` "
    "/ `self.assertRaises` / `self.assertEqual` -- those only exist on "
    "`unittest.TestCase` and will raise `AttributeError` at runtime. "
    "Use `assert <expr>`, `pytest.raises(...)`, `caplog`, and other "
    "pytest fixtures instead.\n"
    "- If you genuinely need the `unittest` API, inherit explicitly from "
    "`unittest.TestCase` and use the `self.assert*` helpers consistently "
    "throughout that class; do not mix the two styles.\n"
    "Pytest test-discovery discipline (applies to ANY file under tests/):\n"
    "- Every test MUST be a module-top-level `def test_*` function (or a "
    "method on a top-level `class Test*`). pytest only collects tests at "
    "module scope; a `def test_*` nested inside a @pytest.fixture or any "
    "other function is invisible and yields 'no tests ran' (exit code 5).\n"
    "- A @pytest.fixture provides setup and MUST return or yield a value "
    "for tests to consume via an argument; it MUST NOT be named `test_*` "
    "and MUST NOT define nested test functions inside it.\n"
    "- Emit at least one real, collectable `def test_*` in every test "
    "file you generate.\n"
    "Pytest test-authoring discipline (applies to ANY file under tests/):\n"
    "- Test the ALREADY-GENERATED code by IMPORTING it. NEVER reimplement, "
    "copy, or redefine the application's functions, classes, or the app "
    "object inside the test file -- import them from the project's own "
    "modules and exercise those.\n"
    "- Call ONLY functions, methods, attributes, routes, and keyword "
    "arguments that actually exist in the imported modules. Do NOT invent "
    "APIs, endpoints, or parameters that were never generated.\n"
    "- Every test-function parameter MUST be satisfied by a "
    "@pytest.fixture (in this file or a conftest.py) or by "
    "@pytest.mark.parametrize. Do NOT declare bare parameters pytest "
    "cannot resolve -- an unbacked parameter raises `fixture '<name>' not "
    "found` at collection time (exit code 5) and breaks the whole file.\n"
    "- Use only pytest's built-in marks (skip, skipif, xfail, parametrize, "
    "usefixtures). Do NOT apply a custom @pytest.mark.* unless it is "
    "registered in pytest.ini/pyproject -- unregistered marks fail under "
    "strict-markers.\n"
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


def _unwrap_wrapping_code_fence(content: str) -> str:
    """Strip a markdown code fence a small model wraps the file body in.

    Extends the old whole-content regex (which only matched a clean
    ```lang\\n…\\n``` block) to the shapes that leak a fence onto line 1
    and fail ``ast.parse`` there: a fence with trailing prose after the
    closing ```, an *unclosed* fence, and a short natural-language
    preamble ("Here is the file:") before the opening fence. Only a
    column-0 fence within the first few lines is treated as a wrapper --
    an indented or docstring-embedded fence (any preamble line carrying a
    triple-quote) is left untouched so real Markdown/docstrings survive.
    """
    if not content or "```" not in content:
        return content
    lines = content.splitlines()
    open_idx: Optional[int] = None
    for i, line in enumerate(lines[:4]):
        if line.startswith("```"):
            open_idx = i
            break
    if open_idx is None:
        return content
    preamble = [ln for ln in lines[:open_idx] if ln.strip()]
    if any(("\"\"\"" in ln) or ("'''" in ln) for ln in preamble):
        return content
    body = lines[open_idx + 1:]
    for j, line in enumerate(body):
        if line.startswith("```"):
            body = body[:j]
            break
    return "\n".join(body)


def _format_syntax_error(exc: SyntaxError) -> str:
    """Render a SyntaxError with the offending source line.

    ``str(SyntaxError)`` reports only *where* ("line 1"), never *what* --
    so a leftover fence or prose line yields a bare, unlocalisable
    "invalid syntax" that neither the inline retry nor the router's
    regenerate constraint can act on. Appending ``exc.text`` gives both
    the retry prompt and the constraint the actual bad line to fix.
    """
    base = str(exc)
    text = (getattr(exc, "text", None) or "").strip()
    if not text:
        return base
    return f"{base}; offending line {getattr(exc, 'lineno', '?')}: {text[:120]!r}"


_IMPORT_KW_RE = re.compile(r"\bimport\b")
_FROM_KW_RE = re.compile(r"\bfrom\b")


def _line_joins_imports(line: str) -> bool:
    """True when one physical line carries several import statements.

    A well-formed Python import line is either ``import a, b`` (one
    ``import``, no ``from``) or ``from a import b`` (one of each), so a
    second occurrence of either keyword means separate statements were
    joined. Lines carrying a quote or a semicolon are skipped: a string
    literal can mention the keywords, and ``import a; import b`` is
    unusual but legal. A trailing comment is cut for the same reason.
    The quote guard also takes JS/TS out of scope, where every import
    names a quoted module.
    """
    stmt = line.split("#", 1)[0]
    if not stmt.lstrip().startswith(("import ", "from ")):
        return False
    if any(c in stmt for c in "\"';"):
        return False
    return (len(_IMPORT_KW_RE.findall(stmt)) > 1
            or len(_FROM_KW_RE.findall(stmt)) > 1)


def _looks_newline_collapsed(content: str) -> bool:
    """True when a file body lost the line breaks between statements.

    Asked for strict JSON, some models never emit an escaped newline:
    they join every line of the file with a space, so ``import sqlite3``
    + ``from fastapi import FastAPI`` arrives as ``import sqlite3 from
    fastapi import FastAPI``. The body is then unparseable at line 1, and
    because every recovery path re-asks in JSON mode the *same* encoder
    reproduces it byte for byte -- the regenerate loop cannot converge.
    Recognising the shape lets the caller route around JSON mode instead
    of retrying into it.

    The damage is not always total. The same encoder often collapses only
    the import block and leaves the function bodies below it correctly
    delimited, which reads as an ordinary line-1 syntax error while being
    the identical JSON-mode defect -- so keying solely on a body with no
    newline at all sends exactly the shape that needs freeform back into
    JSON mode. Both are reported here. The length floor keeps genuinely
    one-line files (a single export, a one-line config) out of scope.

    A partially collapsed body is only ever inspected line by line once
    the body as a whole has failed to parse: source that compiles is by
    construction not collapsed, and skipping it keeps prose that merely
    reads like joined imports (a docstring narrating an ``import``) from
    being mistaken for one.
    """
    body = (content or "").strip()
    if not body:
        return False
    if "\n" not in body:
        return len(body) > 120
    import ast as _ast
    try:
        _ast.parse(body)
    except SyntaxError:
        pass
    except (ValueError, MemoryError, RecursionError):
        return False
    else:
        return False
    return any(_line_joins_imports(ln) for ln in body.splitlines())


@traced("llm")
def generate_single_scaffold_file(
    path: str,
    description: str,
    provider: Any,
    *,
    layer: str = "",
    existing_files_with_content: Optional[List[Dict[str, str]]] = None,
    goal: str = "",
    skills: Optional[List[str]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    depends_on: Optional[List[str]] = None,
    contracts: Optional[Dict[str, Any]] = None,
    manifest_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate the content of a single file in a new-project scaffold.

    Each call generates exactly one file. The signatures of previously
    generated files are provided as context so imports resolve correctly;
    when ``depends_on`` is supplied the context digest is scoped to just
    those paths (a file only needs the modules it imports), which bounds
    the prompt at O(depends_on) instead of O(files-so-far). Runs inline
    syntax validation before returning.

    When ``contracts`` is supplied (the WORK_PLAN ``contracts`` block) the
    shared interfaces -- endpoint paths, schema field names, function
    signatures, constants -- are rendered into the prompt so every file
    implements the same declared contract instead of re-deriving it.

    When ``on_token`` is supplied the primary generation is streamed via
    ``provider.chat_stream`` and each raw delta is handed to the callback
    so the caller can surface live progress; the accumulated text is then
    parsed exactly as the non-streamed path, with a non-streaming
    ``provider.chat`` fallback if the stream yields nothing parseable, so
    success rate is never traded for the perceived-speed win.

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
    # the src/ layout. Its job is fixed and one-line -- prepend src/ to
    # sys.path -- so we generate it without an LLM round-trip to avoid the
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

    # Deterministic short-circuit for pure-boilerplate files whose content
    # is fully determined by convention (.gitignore, .dockerignore, ...).
    # Generating these with the LLM spends a full round-trip reproducing a
    # template from memory; emitting a fixed one instead removes those
    # calls from the critical path (P1.3), the same way __init__.py and
    # conftest.py are short-circuited above. Kept narrow to files with no
    # project-specific content.
    trivial = _trivial_boilerplate_content(path)
    if trivial is not None:
        patch = _content_to_new_file_patch(path, trivial)
        return {
            "file": path,
            "patch": patch,
            "content": trivial,
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
    
    emit_trace("scaffold_rules", rules=system)

    # Defense-in-depth: hard-pin framework conventions by file extension so
    # the model cannot cross-contaminate frameworks (Vue SFC under .jsx,
    # React under .vue) regardless of whether a skill was detected. The
    # symptom this guards against is the per-file judge rejecting a .jsx
    # file that contains <template> / `import from 'vue'`.
    ext_pin = _extension_framework_pin(path)
    if ext_pin:
        system = system + "\n\n" + ext_pin

    # README pin: this file is generated last and must document the REAL
    # project. Steer the model away from inventing setup steps or files
    # that were never generated, which is the common failure mode.
    if base.lower() == "readme.md":
        system = system + (
            "\n\nREADME RULES:\n"
            "- Write GitHub-flavoured Markdown for the top-level project README.\n"
            "- Include, in order: a title + one-line summary, a short "
            "description, the tech stack, install/setup steps, how to run "
            "the project, and how to run the tests.\n"
            "- Ground EVERY command, path, and dependency in the ALREADY "
            "GENERATED FILES. If a requirements.txt / package.json exists, "
            "derive install steps from it; never invent files or commands "
            "that were not generated.\n"
            "- Return the Markdown as the JSON \"content\" string. Do not "
            "wrap it in an extra code fence."
        )

    parts: List[str] = []
    if goal:
        parts.append(f"PROJECT GOAL:\n{goal}")
    contract_block = _render_contracts_for_prompt(contracts)
    if contract_block:
        parts.append(contract_block)
    if manifest_paths:
        parts.append("PROJECT MANIFEST (All file paths in this project):\n" + "\n".join(f"- {p}" for p in manifest_paths))
    parts.append(f"FILE TO GENERATE:\nPath: {path}\nPurpose: {description}")
    if layer:
        parts.append(f"Layer: {layer}")
    # Per-call prompt + response budget scaled to the active provider's
    # model context window. Local 8K models get tight caps; cloud
    # models with 200K+ windows get generous ones. See
    # :mod:`cgx.answer.model_caps`.
    from cgx.answer.model_caps import get_scaffold_budget
    budget = get_scaffold_budget(provider)

    if existing_files_with_content:
        # Send a *structural summary* of each prior file (imports +
        # function/class signatures with bodies elided) rather than the
        # full source. This keeps the prompt small as the scaffold grows
        # and is what the model actually needs to know: which symbols
        # already exist, not how they are implemented. The full content
        # is still kept in ``existing_files_with_content`` for the
        # downstream duplicate-content guard.
        #
        # When the manifest declares this file's dependencies, scope the
        # digest to just those paths: importing ``foo`` needs foo's
        # signatures, not every unrelated sibling. This is the dominant
        # prompt-growth lever late in a scaffold. Falls back to all prior
        # files (capped by budget) when no usable dependency set is given.
        dep_set = {d.strip() for d in (depends_on or []) if d and d.strip()}
        digest_pool = existing_files_with_content
        if dep_set:
            scoped = [ef for ef in existing_files_with_content
                      if ef.get("path") in dep_set]
            if scoped:
                digest_pool = scoped
        context_blocks: List[str] = []
        for ef in digest_pool[: budget["max_files"]]:
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

        if path.endswith(".py") and digest_pool:
            sym_index = _module_symbol_index(digest_pool)
            sym_lines = []
            for mod_name, syms in sorted(sym_index.items()):
                if syms and mod_name and not mod_name.endswith(".__init__"):
                    sym_lines.append(f"- {mod_name}: {', '.join(sorted(syms))}")
            if sym_lines:
                parts.append(
                    "AVAILABLE PROJECT MODULE SYMBOLS (when importing from "
                    "project modules, import ONLY these names; do NOT invent symbols):\n"
                    + "\n".join(sym_lines[:30])
                )

    context = "\n\n".join(parts)

    raw = _scaffold_primary_call(
        provider, system, context, budget, on_token)
    parsed = _extract_json_object(raw)
    content = str(parsed.get("content") or "") if parsed else ""
    if not content and raw:
        recovered = _first_fenced_block_body(raw)
        if recovered and not (recovered.strip().startswith("{") and recovered.strip().endswith("}")):
            content = recovered

    # The freeform path is also the escape hatch for a body whose newlines
    # the JSON encoder dropped: a fenced block carries real line breaks, so
    # re-asking there is the one retry that can come back different.
    collapsed = _looks_newline_collapsed(content)
    if not content or collapsed:
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

        def _patch_to_body(entry: Dict[str, Any]) -> str:
            # The freeform parser hands back a unified diff; recover the plain
            # body so it doesn't leak in as a diff-header "file". A modification
            # diff we can't reverse falls through to the header guard.
            p = str(entry.get("patch") or "")
            body = _new_file_body_from_patch(p)
            return body if body is not None else p

        ff_body = ""
        for d in (parsed_ff.get("diffs") or []):
            if isinstance(d, dict) and d.get("file") == path:
                ff_body = _patch_to_body(d)
                break
        if not ff_body and parsed_ff.get("diffs"):
            ff_body = _patch_to_body(parsed_ff["diffs"][0])
        if not ff_body:
            # The strict fence parser needs a ``path=`` label; a small model
            # often omits it, so the block is dropped and no diff is
            # produced at all. For this single-file request the lone block
            # is unambiguously the file -- recover its body directly rather
            # than losing the whole generation to an empty patch.
            ff_body = _first_fenced_block_body(ff_raw)
        # A collapsed body is real content, just unusable: only replace it
        # when the fallback actually came back with line structure.
        if not collapsed or (ff_body and not _looks_newline_collapsed(ff_body)):
            content = ff_body or content

    # Generic empty-body retry: when both the primary call and the freeform
    # fallback came back with no usable content, the file would otherwise
    # drop with a bare "generator returned empty patch". This is a transient
    # generation miss (observed live on manifests such as package.json that
    # then generate fine on the very next attempt), not a content defect, so
    # one hardened JSON-mode retry recovers it in place instead of letting
    # the drop escalate to a re-plan.
    if not content:
        logger.debug(
            "scaffold empty body after primary+freeform; "
            "retrying once: %s", path)
        retry = _regenerate_scaffold_file(
            provider, system, context, budget, _EMPTY_BODY_RETRY_INSTR)
        if retry:
            content = retry

    # Inline syntax validation.
    syntax_ok = True
    syntax_error: Optional[str] = None
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    # Strip a markdown code fence a small model wraps the file body in.
    # Covers a trailing-prose or unclosed fence and a short prose preamble
    # in addition to the clean whole-wrap case -- an unstripped leading
    # fence otherwise fails ast.parse on line 1 with a bare, unactionable
    # "invalid syntax" the retry can never localise.
    if content:
        content = _unwrap_wrapping_code_fence(content)
    # Reject unified-diff fragments that leak through the freeform parser
    # into the file body (`--- /dev/null`, `+++ b/...`, `@@ ...`).
    if content:
        head = content.lstrip().splitlines()[0] if content.strip() else ""
        if head.startswith(("--- ", "+++ ", "@@ ")):
            syntax_ok = False
            syntax_error = "content is a unified-diff header, not a file body"
    if syntax_ok and ext == "py" and content:
        import ast as _ast
        try:
            _ast.parse(content)
        except SyntaxError as e:
            # One targeted retry with the exact SyntaxError surfaced --
            # including the offending source line so the fix (and the
            # router's regenerate constraint on failure) is localised, not
            # a bare "invalid syntax (line 1)". Broken Python that slips
            # through here is silently dropped by APPLY's own syntax gate,
            # which can leave the project without its entry point or its
            # only test file (VERIFY then reports "no tests located" and
            # the loop declares a false success).
            syntax_msg = _format_syntax_error(e)
            retry = _syntax_repair_retry(
                provider, path=path, lang="Python", error=syntax_msg,
                broken=content, budget=budget)
            retry_ok = False
            if retry:
                try:
                    _ast.parse(retry)
                    retry_ok = True
                except SyntaxError:
                    retry_ok = False
            if retry_ok:
                content = retry
            else:
                syntax_ok = False
                syntax_error = syntax_msg
    elif syntax_ok and ext == "json" and content:
        import json as _json
        try:
            _json.loads(content)
        except Exception as e:
            # One targeted retry with the exact parse error surfaced --
            # symmetric with the .py path above. A malformed data file
            # (e.g. users.json the app reads at startup) is otherwise
            # silently dropped by APPLY's syntax gate, so the app boots
            # against a missing file and SMOKE/VERIFY fail downstream.
            retry = _syntax_repair_retry(
                provider, path=path, lang="JSON", error=str(e),
                broken=content, budget=budget)
            retry_ok = False
            if retry:
                try:
                    _json.loads(retry)
                    retry_ok = True
                except Exception:
                    retry_ok = False
            if retry_ok:
                content = retry
            else:
                syntax_ok = False
                syntax_error = str(e)
    elif syntax_ok and ext == "toml" and content:
        try:
            import tomllib as _tomllib
            _tomllib.loads(content)
        except Exception as e:
            syntax_ok = False
            syntax_error = f"TOML parse error: {e}"
    elif syntax_ok and ext in _JS_TS_GRAMMAR_BY_EXT and content:
        # Symmetric with the .py/.json gates above, for the JS/TS/JSX/Vue
        # family. A frontend file that fails to parse (unbalanced JSX, a
        # dangling brace) is otherwise silently dropped by APPLY's syntax
        # gate, leaving the app without its entry component -- the
        # build-smoke then fails downstream with no way to self-correct in
        # SCAFFOLD. Degrades to a no-op when the tree-sitter grammar is
        # unavailable (validate_js_ts_source returns ok=True with a skip).
        from cgx.codegen.validate import validate_js_ts_source
        _grammar = _JS_TS_GRAMMAR_BY_EXT[ext]
        diag = validate_js_ts_source(path, content, _grammar)
        if not diag.ok:
            retry = _syntax_repair_retry(
                provider, path=path, lang=_grammar, error=diag.error,
                broken=content, budget=budget)
            if retry and validate_js_ts_source(path, retry, _grammar).ok:
                content = retry
            else:
                syntax_ok = False
                syntax_error = diag.error

    # requirements.txt content gate: symmetric with the JSON/TOML gates
    # above. There is no stdlib parser for pip's requirements format, but
    # the live failure mode is severe: a model pasted a Python module's
    # source into requirements.txt and pip tolerated enough of it that
    # the venv provisioned against a corrupted manifest. Validate
    # line-by-line against a plausible PEP-508-ish specifier shape with
    # one targeted repair retry. requirements.txt is foundational -- if it
    # is dropped, BOOTSTRAP_ENV misdetects a node-only project and the
    # Python venv is never provisioned, so downstream recovery is
    # structurally impossible. When the model retry also fails, salvage the
    # manifest deterministically (strip non-specifier lines, backfill from
    # real imports) rather than failing the file into a drop.
    if syntax_ok and content and _is_requirements_txt_path(path):
        req_err = _requirements_content_error(content)
        if req_err:
            retry = _syntax_repair_retry(
                provider, path=path, lang="pip requirements",
                error=req_err, broken=content, budget=budget)
            if retry and not _requirements_content_error(retry):
                content = retry
            else:
                content = _deterministic_requirements_repair(
                    content, existing_files_with_content)

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

    # First-party symbol-consistency gate: a regenerated file (typically
    # a test) frequently imports a symbol from an already-generated
    # project module that the module never actually defines (e.g.
    # `from auth import generate_jwt` when auth.py has no `generate_jwt`).
    # API_CHECK would flag it as a hallucinated attribute, but the
    # regenerate loop can't fix it without knowing what the module really
    # exports. Retry once with the real symbol inventory; otherwise fail
    # the file so APPLY drops it rather than persisting a broken import.
    if content and syntax_ok and ext == "py" and existing_files_with_content:
        sym_index = _module_symbol_index(existing_files_with_content)
        violations = _first_party_symbol_violations(content, sym_index)
        if violations:
            retry = _regenerate_scaffold_file(
                provider, system, context, budget,
                _symbol_retry_instruction(violations))
            retry_ok = False
            if retry:
                import ast as _ast
                try:
                    _ast.parse(retry)
                    retry_ok = not _first_party_symbol_violations(
                        retry, sym_index)
                except SyntaxError:
                    retry_ok = False
            if retry_ok:
                content = retry
            else:
                first = violations[0]
                syntax_ok = False
                avail = ", ".join(first.get("available") or []) or "(nothing importable)"
                syntax_error = (
                    f"imports undefined first-party symbol(s) "
                    f"{first['missing']} from module '{first['module']}'. "
                    f"Available symbols in '{first['module']}' are: [{avail}]")
                content = ""

    # Undefined-name gate: a file can parse cleanly, import only modules
    # that really exist, and still die the instant anything touches it
    # because it uses a name nothing ever bound -- `class Operation(str,
    # enum.Enum)` with no `import enum` (live failure: pytest aborted at
    # collection with "NameError: name 'enum' is not defined", taking the
    # whole suite with it via the conftest chain). Every other gate here
    # judges imports that are present; none can see a name that is
    # absent. Retry once naming the unbound names, then fail the file so
    # APPLY drops it rather than persisting a module that cannot import.
    if content and syntax_ok and ext == "py":
        unbound = _undefined_module_names(content)
        if unbound:
            retry = _regenerate_scaffold_file(
                provider, system, context, budget,
                _undefined_name_retry_instruction(unbound))
            retry_ok = False
            if retry:
                import ast as _ast
                try:
                    _ast.parse(retry)
                    retry_ok = not _undefined_module_names(retry)
                except SyntaxError:
                    retry_ok = False
            if retry_ok:
                content = retry
            else:
                syntax_ok = False
                syntax_error = (
                    f"uses undefined name(s) {unbound}: never imported, "
                    "assigned, or defined anywhere in this file. YOU MUST IMPORT IT AT THE TOP OF THE FILE. Regenerate the file with the missing imports added.")
                content = ""

    # Test-collectability gate: a pytest module that parses cleanly but
    # defines no module-top-level `def test_*` collects zero tests (pytest
    # exit 5) and stalls the verify->repair->regenerate loop -- the single
    # most common greenfield failure, where the model reimplements the app
    # under the test path instead of testing it. Retry once with a
    # hardened, path-specific instruction so the empty suite never reaches
    # the (budget-limited) repair loop.
    if (content and syntax_ok and ext == "py"
            and _is_pytest_test_path(path)
            and not _has_collectable_pytest_test(content)):
        retry = _regenerate_scaffold_file(
            provider, system, context, budget, _TEST_RETRY_INSTR)
        if retry and _has_collectable_pytest_test(retry):
            content = retry
        else:
            syntax_ok = False
            syntax_error = (
                "no collectable pytest test: define at least one "
                "top-level `def test_*` function (pytest exit 5)")

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


def _is_pytest_test_path(path: str) -> bool:
    """True when ``path`` is a Python file pytest would collect as a test.

    Matches pytest's default naming convention: a ``.py`` file whose
    basename is ``test_*.py`` or ``*_test.py``. Files under ``tests/``
    that do not follow the convention (helpers, fixtures modules) are not
    collected by pytest, so they are excluded here too.
    """
    p = path.strip().lower()
    if not p.endswith(".py"):
        return False
    base = p.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith("_test.py")


def _is_requirements_txt_path(path: str) -> bool:
    """True when ``path`` is a pip requirements file (requirements*.txt)."""
    base = path.strip().lower().rsplit("/", 1)[-1]
    return base.startswith("requirements") and base.endswith(".txt")


# One plausible pip-requirements line: an include/option flag, or a PEP
# 508-ish specifier -- distribution name, optional extras, then an
# optional version-specifier list / direct-URL reference / environment
# marker. Deliberately permissive about versions; its job is rejecting
# source code masquerading as a requirement, not full PEP 508 parsing.
_REQ_SPEC_OPS = r"(?:===|==|!=|<=|>=|~=|<|>)"
_REQUIREMENT_LINE_RE = re.compile(
    r"^(?:"
    r"-(?:r|c|e)\s+\S+"                                   # -r/-c/-e refs
    r"|--?[A-Za-z][A-Za-z0-9-]*(?:[= ]\s*\S+)?"           # pip options
    r"|[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"        # dist name
    r"(?:\[[A-Za-z0-9,._\s-]*\])?"                        # extras
    r"(?:\s*@\s*\S+"                                      # direct URL
    r"|\s*" + _REQ_SPEC_OPS + r"\s*[A-Za-z0-9.*+!_-]+"    # version spec
    r"(?:\s*,\s*" + _REQ_SPEC_OPS + r"\s*[A-Za-z0-9.*+!_-]+)*)?"
    r"(?:\s*;.*)?"                                        # env marker
    r")$"
)


def _requirements_content_error(content: str) -> Optional[str]:
    """Return a diagnostic when ``content`` is not a requirements file.

    Checks every non-empty, non-comment line against
    :data:`_REQUIREMENT_LINE_RE`; ``None`` means every line is a
    plausible requirement specifier. The message lists the first few
    offending lines so the repair retry (and the router's regenerate
    constraint on failure) is concrete.
    """
    bad: List[str] = []
    for i, raw in enumerate(content.splitlines(), start=1):
        line = re.split(r"\s#", raw)[0].strip()
        if not line or line.startswith("#"):
            continue
        if not _REQUIREMENT_LINE_RE.match(line):
            bad.append(f"line {i}: {line[:80]!r}")
            if len(bad) >= 5:
                break
    if not bad:
        return None
    return ("not a valid pip requirements file; offending line(s): "
            + "; ".join(bad))


def _synthesize_requirements_from_imports(
        existing_files_with_content: Optional[List[Dict[str, str]]],
) -> List[str]:
    """Best-effort requirement lines from the generated .py files' imports.

    Scans every already-generated Python file for its third-party import
    roots, drops stdlib and first-party (project-local) roots, and maps the
    survivors to their PyPI distribution names via the same
    :data:`~cgx.codegen.env_manager._IMPORT_TO_PYPI` table the dynamic
    installer uses. Used only as a backfill when a corrupted manifest has
    no salvageable specifier line, so the venv still gets the obvious
    dependencies instead of an empty file.
    """
    from cgx.codegen.env_manager import (
        _IMPORT_TO_PYPI,
        _NAMESPACE_ROOTS,
        _STDLIB_TOP,
        _extract_imports_python,
    )
    first_party: set = set()
    imports: set = set()
    for ef in existing_files_with_content or []:
        ep = (ef.get("path") or "").strip()
        ec = ef.get("content") or ""
        if not ep.endswith(".py") or not ec:
            continue
        parts = [p for p in ep.split("/") if p]
        if parts:
            first_party.add(parts[0][:-3] if parts[0].endswith(".py")
                            else parts[0])
            first_party.add(parts[-1][:-3])
        imports |= _extract_imports_python(ec)
    dotted_roots = {n.split(".", 1)[0] for n in imports if "." in n}
    dists: set = set()
    for name in imports:
        if name in _NAMESPACE_ROOTS and name in dotted_roots:
            continue
        root = name.split(".")[0]
        if root.lower().replace("-", "_") in _STDLIB_TOP:
            continue
        if root in first_party:
            continue
        if name in _IMPORT_TO_PYPI:
            dists.add(_IMPORT_TO_PYPI[name])
        elif "." in name:
            continue
        else:
            dists.add(name)
    return sorted(dists)


def _deterministic_requirements_repair(
        content: str,
        existing_files_with_content: Optional[List[Dict[str, str]]],
) -> str:
    """Salvage a corrupted requirements.txt into a valid one, no model call.

    Keeps every comment/blank line and every line that already parses as a
    plausible specifier (:data:`_REQUIREMENT_LINE_RE`), dropping the rest --
    typically Python source a weak model pasted into the manifest. When no
    specifier line survives, backfills from the third-party imports the
    generated ``.py`` files actually use. The result is guaranteed to
    satisfy :func:`_requirements_content_error`, so requirements.txt is
    never dropped for a content fault (which would misdetect a node-only
    project and skip Python venv provisioning entirely).
    """
    kept: List[str] = []
    kept_specs = False
    for raw in content.splitlines():
        line = re.split(r"\s#", raw)[0].strip()
        if not line or line.startswith("#"):
            kept.append(raw)
            continue
        if _REQUIREMENT_LINE_RE.match(line):
            kept.append(raw)
            kept_specs = True
    if not kept_specs:
        synth = _synthesize_requirements_from_imports(
            existing_files_with_content)
        if synth:
            kept.append("# synthesised from project imports "
                        "(original manifest was not a valid requirements file)")
            kept.extend(synth)
    text = "\n".join(kept).strip()
    return text + "\n" if text else "# no third-party dependencies detected\n"


# ES-module / CommonJS import specifier extractors. Deliberately narrow:
# they only need to recover the *module specifier* string so the caller can
# decide whether it is a bare (external) package. Both grammars are matched
# because a weak model mixes ``import``/``require`` freely in the same file.
_JS_IMPORT_FROM_RE = re.compile(
    r"""(?:import|export)\b[^;'"]*?\bfrom\s*['"]([^'"]+)['"]""")
_JS_BARE_IMPORT_RE = re.compile(r"""\bimport\s*['"]([^'"]+)['"]""")
_JS_REQUIRE_RE = re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)""")

# Node core modules that must never be added to package.json. Not
# exhaustive, but covers the builtins a scaffold model actually reaches for.
_NODE_BUILTINS = frozenset({
    "assert", "buffer", "child_process", "cluster", "console", "crypto",
    "dgram", "dns", "events", "fs", "http", "http2", "https", "net", "os",
    "path", "perf_hooks", "process", "punycode", "querystring", "readline",
    "stream", "string_decoder", "timers", "tls", "tty", "url", "util", "v8",
    "vm", "worker_threads", "zlib",
})


def _js_package_name_from_specifier(spec: str) -> Optional[str]:
    """Return the installable npm package name for an import specifier.

    ``None`` for relative (``./`` / ``../``), absolute, alias (``@/`` /
    bare ``@``-with-no-scope-path), builtin, and ``node:`` specifiers --
    none of which map to a package.json dependency. A scoped package keeps
    its ``@scope/name`` root; an unscoped subpath (``react-dom/client``) is
    reduced to its package root (``react-dom``).
    """
    s = (spec or "").strip()
    if not s or s.startswith((".", "/")):
        return None
    if s.startswith("node:"):
        return None
    if s.startswith("@"):
        parts = s.split("/")
        if len(parts) < 2 or not parts[0][1:] or not parts[1]:
            return None
        return f"{parts[0]}/{parts[1]}"
    root = s.split("/", 1)[0]
    if not root or root in _NODE_BUILTINS:
        return None
    return root


def _js_external_imports(content: str) -> List[str]:
    """External npm package names imported/required by one JS/TS source."""
    found: List[str] = []
    for rx in (_JS_IMPORT_FROM_RE, _JS_BARE_IMPORT_RE, _JS_REQUIRE_RE):
        for spec in rx.findall(content or ""):
            name = _js_package_name_from_specifier(spec)
            if name and name not in found:
                found.append(name)
    return found


def _js_relative_imports(content: str) -> List[str]:
    """Relative import/require specifiers (``./`` / ``../``) in one source.

    The counterpart to :func:`_js_external_imports`: instead of the bare
    package names that map to package.json dependencies, it returns the
    first-party specifiers a bundler resolves against sibling files
    (``./index.css``, ``../App.jsx``). Order-preserving and de-duplicated;
    absolute (``/``) and alias (``@/``) specifiers are excluded because
    they are not path-relative to the importer. Callers resolve each one
    against the importer's directory to check the target was generated.
    """
    found: List[str] = []
    for rx in (_JS_IMPORT_FROM_RE, _JS_BARE_IMPORT_RE, _JS_REQUIRE_RE):
        for spec in rx.findall(content or ""):
            s = (spec or "").strip()
            if s.startswith(".") and s not in found:
                found.append(s)
    return found


def _deterministic_package_json_repair(
        content: str,
        js_files_with_content: Optional[List[Dict[str, str]]],
) -> Optional[str]:
    """Add npm deps that generated JS/TS source imports but package.json omits.

    Symmetric with :func:`_deterministic_requirements_repair`: a weak model
    routinely imports a runtime package (``axios``) in a component while
    leaving it out of ``package.json``, so the build resolves nothing and
    VERIFY fails with an unrecoverable red. Scans every generated JS/TS file
    for external (bare) imports, and adds any not already present under
    ``dependencies``/``devDependencies``/``peerDependencies`` to
    ``dependencies`` with a permissive ``"*"`` range (the lockfile pins the
    concrete version at install time). Returns the rewritten JSON text, or
    ``None`` when the manifest is unparseable or nothing needs adding, so
    the caller leaves the original diff untouched.
    """
    import json as _json
    try:
        pkg = _json.loads(content)
    except Exception:
        return None
    if not isinstance(pkg, dict):
        return None

    declared: set = set()
    for key in ("dependencies", "devDependencies", "peerDependencies",
                "optionalDependencies"):
        section = pkg.get(key)
        if isinstance(section, dict):
            declared.update(str(k) for k in section)

    wanted: List[str] = []
    for entry in (js_files_with_content or []):
        path = str(entry.get("path") or "")
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext not in _JS_TS_GRAMMAR_BY_EXT:
            continue
        for name in _js_external_imports(str(entry.get("content") or "")):
            if name not in declared and name not in wanted:
                wanted.append(name)

    if not wanted:
        return None

    deps = pkg.get("dependencies")
    if not isinstance(deps, dict):
        deps = {}
    for name in wanted:
        deps[name] = "*"
    pkg["dependencies"] = deps
    return _json.dumps(pkg, indent=2) + "\n"


def _has_collectable_pytest_test(content: str) -> bool:
    """True when ``content`` defines at least one pytest-collectable test.

    A test is collectable when it is a module-top-level function named
    ``test`` / ``test_*`` (sync or async) or a ``test_*`` method on a
    top-level ``class Test*``. Returns ``False`` on any syntax error --
    an unparseable module collects nothing either.
    """
    import ast as _ast

    def _is_test_name(name: str) -> bool:
        return name == "test" or name.startswith("test_")

    try:
        tree = _ast.parse(content)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if _is_test_name(node.name):
                return True
        elif isinstance(node, _ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if (isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                        and _is_test_name(item.name)):
                    return True
    return False


_TEST_RETRY_INSTR = (
    "\n\nCRITICAL RETRY -- YOUR PREVIOUS ATTEMPT WAS REJECTED:\n"
    "The file you produced contained NO pytest-collectable tests, so "
    "pytest collected 0 items (exit code 5). Do NOT reimplement or copy "
    "the application code into this file. Instead:\n"
    "- IMPORT the code under test from the already-generated modules.\n"
    "- Write at least THREE module-top-level `def test_*` functions, "
    "each with real `assert` statements exercising a distinct behaviour.\n"
    "- Do NOT add an `if __name__ == '__main__'` block or any "
    "application logic."
)


_EMPTY_BODY_RETRY_INSTR = (
    "\n\nCRITICAL RETRY -- YOUR PREVIOUS ATTEMPT RETURNED NO FILE CONTENT:\n"
    "The prior response contained no usable file body (empty or "
    "unparseable). Return the COMPLETE content of this one file as the "
    "JSON `content` string -- real newlines between every line, no "
    "markdown fences, no prose, no empty result."
)


def _module_symbol_index(
        existing_files_with_content: List[Dict[str, str]],
) -> Dict[str, Optional[set]]:
    """Map each already-generated Python module to its exported symbols.

    Keys are the import names another file could plausibly use -- the
    module basename (``auth``), the full dotted path (``backend.auth``),
    and the root-stripped path (``auth`` for ``src/auth.py``). The value
    is the set of top-level names the module defines (functions,
    classes, assignments, and imported aliases, which are all reachable
    as module attributes), or ``None`` when the module does a
    ``from x import *`` and its surface can't be determined statically.
    """
    import ast as _ast
    index: Dict[str, Optional[set]] = {}
    for ef in existing_files_with_content or []:
        ep = (ef.get("path") or "").strip()
        ec = ef.get("content") or ""
        if not ep or not ec or not ep.endswith(".py"):
            continue
        try:
            tree = _ast.parse(ec)
        except SyntaxError:
            continue
        symbols: set = set()
        star = False
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                 _ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, _ast.Assign):
                for t in node.targets:
                    if isinstance(t, _ast.Name):
                        symbols.add(t.id)
            elif isinstance(node, _ast.AnnAssign):
                if isinstance(node.target, _ast.Name):
                    symbols.add(node.target.id)
            elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
                for a in node.names:
                    if a.name == "*":
                        star = True
                        continue
                    symbols.add(a.asname or a.name.split(".")[0])
        segs = [s for s in ep[:-3].split("/") if s and s != "__init__"]
        names = set()
        if segs:
            names.add(segs[-1])
            names.add(".".join(segs))
            if len(segs) > 1:
                names.add(".".join(segs[1:]))
        for nm in names:
            if star:
                index[nm] = None
            elif index.get(nm) is not None or nm not in index:
                index[nm] = (index.get(nm) or set()) | symbols
    return index


def _first_party_symbol_violations(
        content: str, index: Dict[str, Optional[set]],
) -> List[Dict[str, Any]]:
    """Return ``from <module> import <name>`` uses that don't resolve.

    Only ``from`` imports of a first-party module present in ``index``
    (with a known symbol surface) are checked; absolute third-party
    imports, relative imports, and star-exporting modules are skipped.
    """
    import ast as _ast
    try:
        tree = _ast.parse(content)
    except SyntaxError:
        return []
    out: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, _ast.ImportFrom):
            continue
        if node.level:
            continue
        mod = node.module or ""
        if mod not in index:
            continue
        avail = index[mod]
        if avail is None:
            continue
        missing = [a.name for a in node.names
                   if a.name != "*" and a.name not in avail]
        if missing:
            out.append({"module": mod, "missing": missing,
                        "available": sorted(avail)})
    return out


def _symbol_retry_instruction(violations: List[Dict[str, Any]]) -> str:
    """Build the hardened retry prompt listing each module's real API."""
    parts = []
    for v in violations:
        avail = ", ".join(v["available"]) or "(nothing importable)"
        parts.append(
            f"- module '{v['module']}' does NOT define "
            f"{', '.join(v['missing'])}; it only defines: {avail}")
    return (
        "\n\nCRITICAL RETRY -- YOUR PREVIOUS ATTEMPT WAS REJECTED:\n"
        "You imported first-party symbols that the already-generated "
        "modules do not define:\n" + "\n".join(parts) + "\n"
        "Return the COMPLETE, corrected file. Import ONLY symbols that "
        "actually exist in those modules (listed above) and call their "
        "real API. Do NOT invent names or add markdown fences.")


# Module attributes Python provides implicitly. Absent from
# ``dir(builtins)``, so they must be seeded explicitly or the
# undefined-name gate would flag ordinary ``if __name__ == "__main__"``
# blocks and ``__all__`` declarations.
_IMPLICIT_MODULE_NAMES = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__", "__dict__",
})


def _undefined_module_names(source: str) -> List[str]:
    """Return names the module loads but never binds, in first-use order.

    A deliberately over-generous binding collector: every name bound
    *anywhere* in the file (any import, assignment, parameter, ``def`` /
    ``class``, comprehension or ``except`` target, ``global`` /
    ``nonlocal`` declaration) counts as bound for the whole file, so
    scoping subtleties can only ever make this abstain, never
    false-positive. What is left is a name that no scope could possibly
    supply -- a guaranteed ``NameError`` the moment that line executes.

    Abstains entirely (returns ``[]``) on an unparsable file or a
    ``from x import *``, whose bindings are not knowable statically.
    """
    import ast as _ast
    import builtins as _builtins

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return []

    bound = set(dir(_builtins)) | set(_IMPLICIT_MODULE_NAMES)
    used: List[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return []
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                               _ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, _ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (_ast.Global, _ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, _ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, _ast.Name):
            if isinstance(node.ctx, _ast.Store):
                bound.add(node.id)
            elif isinstance(node.ctx, _ast.Load):
                used.append(node.id)

    out: List[str] = []
    for name in used:
        if name not in bound and name not in out:
            out.append(name)
    return out


def _undefined_name_retry_instruction(names: List[str]) -> str:
    """Build the hardened retry prompt listing every unbound name."""
    listed = ", ".join(repr(n) for n in names)
    return (
        "\n\nCRITICAL RETRY -- YOUR PREVIOUS ATTEMPT WAS REJECTED:\n"
        f"The file used the name(s) {listed} without ever binding them: "
        "they were not imported, not assigned, and not defined anywhere "
        "in the file, so it raises NameError as soon as it is imported.\n"
        "Return the COMPLETE, corrected file. Every module, class, "
        "function and constant the file references must be either "
        "imported at the top of the file or defined in it -- if you use "
        "'enum.Enum' you must 'import enum'. Do NOT add markdown fences.")


# Extension -> tree-sitter grammar name for the JS/TS/Vue family. Mirrors
# ``cgx.codegen.validate._JS_TS_LANGS`` (plus ``vue``) so the inline scaffold
# syntax gate covers exactly the frontend files APPLY's own gate would
# otherwise silently drop. The grammar name doubles as the retry-instruction
# language label.
_JS_TS_GRAMMAR_BY_EXT = {
    "js": "javascript", "jsx": "javascript",
    "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "mts": "typescript", "cts": "typescript",
    "tsx": "tsx", "vue": "vue",
}


_SYNTAX_FIX_SYSTEM = (
    "You are a code-repair tool. You are given exactly ONE source file that "
    "failed to parse, together with the parser's error. Fix ONLY the syntax "
    "so the file parses cleanly; preserve the code's intent, structure, and "
    "every identifier. Do not add, remove, or rename any functionality.\n\n"
    "Return strict JSON only:\n"
    '{"content": "<the complete corrected file>"}\n'
    "The content MUST be the whole file: no markdown fences, no commentary, "
    "no ellipsis, no unified-diff markers."
)

_SYNTAX_FIX_FREEFORM_SYSTEM = (
    "You are a code-repair tool. You are given exactly ONE source file that "
    "failed to parse, together with the parser's error. Fix ONLY the syntax "
    "so the file parses cleanly; preserve the code's intent, structure, and "
    "every identifier. Do not add, remove, or rename any functionality.\n\n"
    "Return the COMPLETE corrected file as raw text inside a single fenced "
    "code block, with REAL line breaks -- one statement per line, correctly "
    "indented. No commentary, no ellipsis, no unified-diff markers."
)


def _syntax_fix_call(
        provider: Any, user: str, budget: Dict[str, Any],
        *, force_json: bool) -> str:
    """One syntax-repair round-trip. Returns the fence-stripped body."""
    try:
        raw = provider.chat(
            messages=[
                {"role": "system",
                 "content": (_SYNTAX_FIX_SYSTEM if force_json
                             else _SYNTAX_FIX_FREEFORM_SYSTEM)},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=budget["output_tokens"],
            force_json=force_json,
        ).get("content", "")
    except Exception:  # pragma: no cover - defensive: provider hiccup
        return ""
    if force_json:
        parsed = _extract_json_object(raw)
        out = str(parsed.get("content") or "") if parsed else ""
    else:
        out = str(raw or "")
    if not out.strip():
        return ""
    return _unwrap_wrapping_code_fence(out.strip()) if "```" in out else out


def _syntax_repair_retry(
        provider: Any, *, path: str, lang: str, error: str,
        broken: str, budget: Dict[str, Any]) -> str:
    """Targeted syntax fix: re-ask with only the broken file + the error.

    Unlike :func:`_regenerate_scaffold_file`, which resends the full
    scaffold context (goal + every prior file's signature digest), this
    ships just the offending file and the parser's message. A syntax error
    is local -- the model needs the broken bytes, not the rest of the
    project -- so the smaller prompt is both cheaper (it drops the
    O(files) digest block that dominates a late-layer retry) and more
    focused. Pinned to temperature 0 for a deterministic correction.
    Returns the fence-stripped ``content`` string, or ``""`` on any
    provider/parse failure.

    A body whose newlines the JSON encoder dropped
    (:func:`_looks_newline_collapsed`) is asked for in freeform first: a
    fenced block carries real line breaks, whereas a JSON-mode retry goes
    back through the encoder that caused the damage and returns the same
    single line, so the file can never recover.
    """
    user = (
        f"File: {path}\n"
        f"Language: {lang}\n"
        f"The file below is not valid {lang}. The parser reported:\n"
        f"  {error}\n\n"
        "Return the COMPLETE corrected file. Balance all "
        "quotes/brackets/parentheses, close every string (including "
        "triple-quoted ones), and terminate every block header with the "
        "correct delimiter.\n\n"
        f"BROKEN FILE:\n{broken}"
    )
    collapsed = _looks_newline_collapsed(broken)
    if collapsed:
        user += (
            "\n\nNOTE: every line break in the file above was lost -- the "
            "whole body arrived as one physical line. Restore the line "
            "structure: one statement per line, with correct indentation."
        )
    for force_json in ((False, True) if collapsed else (True,)):
        out = _syntax_fix_call(provider, user, budget, force_json=force_json)
        if out and not _looks_newline_collapsed(out):
            return out
    return ""


_GITIGNORE_TEMPLATE = (
    "# Python\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    "*.egg-info/\n"
    ".eggs/\n"
    "build/\n"
    "dist/\n"
    ".pytest_cache/\n"
    ".mypy_cache/\n"
    ".ruff_cache/\n"
    ".coverage\n"
    "htmlcov/\n"
    ".venv/\n"
    "venv/\n"
    "env/\n\n"
    "# Node\n"
    "node_modules/\n"
    "npm-debug.log*\n"
    "yarn-debug.log*\n"
    "yarn-error.log*\n"
    ".pnpm-debug.log*\n\n"
    "# Env / secrets\n"
    ".env\n"
    ".env.local\n"
    ".env.*.local\n\n"
    "# Editor / OS\n"
    ".DS_Store\n"
    ".idea/\n"
    ".vscode/\n"
    "*.swp\n"
)

_DOCKERIGNORE_TEMPLATE = (
    ".git\n"
    ".gitignore\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    ".venv/\n"
    "venv/\n"
    "node_modules/\n"
    "dist/\n"
    "build/\n"
    ".env\n"
    ".pytest_cache/\n"
    ".mypy_cache/\n"
)

_GITATTRIBUTES_TEMPLATE = "* text=auto eol=lf\n"

_EDITORCONFIG_TEMPLATE = (
    "root = true\n\n"
    "[*]\n"
    "charset = utf-8\n"
    "end_of_line = lf\n"
    "insert_final_newline = true\n"
    "trim_trailing_whitespace = true\n"
    "indent_style = space\n"
    "indent_size = 4\n\n"
    "[*.{js,jsx,ts,tsx,json,yml,yaml,css,html,vue}]\n"
    "indent_size = 2\n"
)

# Basename -> fixed template for pure-boilerplate files that carry no
# project-specific content. Deterministic generation removes one LLM
# round-trip per matched file from the SCAFFOLD critical path (P1.3).
_TRIVIAL_BOILERPLATE: Dict[str, str] = {
    ".gitignore": _GITIGNORE_TEMPLATE,
    ".dockerignore": _DOCKERIGNORE_TEMPLATE,
    ".gitattributes": _GITATTRIBUTES_TEMPLATE,
    ".editorconfig": _EDITORCONFIG_TEMPLATE,
}


def _trivial_boilerplate_content(path: str) -> Optional[str]:
    """Return fixed content for a pure-boilerplate file, else ``None``.

    Matches on the basename only, so ``.gitignore`` resolves whether it is
    at the repo root or nested. Non-boilerplate paths return ``None`` and
    fall through to normal LLM generation.
    """
    base = path.rsplit("/", 1)[-1]
    return _TRIVIAL_BOILERPLATE.get(base)


# Whole-file generation grows its output cap on a length-cap stop up to
# this many times (each step doubles ``num_predict``), so an occasional
# large file that overruns the tier ceiling still completes instead of
# being truncated and dropped. Bounded so a runaway generation cannot spin.
_MAX_TRUNCATION_RETRIES = 2


def _response_finish_was_length(resp: Any) -> bool:
    """True when a provider stopped because the output-token cap was hit.

    Recognises the length-cap stop across the three provider response
    shapes carried in ``resp["raw"]``: Ollama ``done_reason == "length"``,
    OpenAI-compatible ``choices[0].finish_reason == "length"`` and Gemini
    ``candidates[0].finishReason == "MAX_TOKENS"``. A whole-file body cut
    at the cap is otherwise indistinguishable from a complete one and gets
    silently dropped by the syntax gate.
    """
    if not isinstance(resp, dict):
        return False
    raw = resp.get("raw")
    if not isinstance(raw, dict):
        return False
    if str(raw.get("done_reason") or "").lower() == "length":
        return True
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        if str(choices[0].get("finish_reason") or "").lower() == "length":
            return True
    candidates = raw.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        if str(candidates[0].get("finishReason") or "").upper() == "MAX_TOKENS":
            return True
    return False


def _blocking_scaffold_call(
        provider: Any, messages: List[Dict[str, str]],
        max_tokens: int) -> str:
    """Blocking JSON-mode generation that grows on a truncated response.

    A single ``provider.chat`` cut off at ``num_predict`` returns a partial
    file body whose truncated JSON fails to parse (or parses to a
    syntactically broken body) and is then silently dropped downstream. When
    the provider reports a length-cap stop, re-issue with a doubled budget
    -- up to :data:`_MAX_TRUNCATION_RETRIES` times -- so a large file is
    generated to completion instead of lost.
    """
    resp = provider.chat(
        messages=messages, temperature=0.2,
        max_tokens=max_tokens, force_json=True)
    attempts = 0
    while (_response_finish_was_length(resp)
           and attempts < _MAX_TRUNCATION_RETRIES):
        max_tokens *= 2
        attempts += 1
        resp = provider.chat(
            messages=messages, temperature=0.2,
            max_tokens=max_tokens, force_json=True)
    return resp.get("content", "") if isinstance(resp, dict) else ""


def _scaffold_primary_call(
        provider: Any, system: str, context: str,
        budget: Dict[str, Any],
        on_token: Optional[Callable[[str], None]]) -> str:
    """Return the raw primary-generation response for one scaffold file.

    Without ``on_token`` this is a single JSON-mode ``provider.chat`` --
    identical to the legacy call. With ``on_token`` the same request is
    streamed through ``provider.chat_stream`` (still JSON-mode, so the
    accumulated text parses exactly like the blocking response) and each
    delta is forwarded to the callback for live progress. Streaming is
    purely additive: on any exception, or when the stream produces nothing
    parseable, it falls back to the blocking ``chat`` so the file's success
    rate is unchanged.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]
    max_tokens = budget["output_tokens"]
    if on_token is not None and getattr(provider, "stream_json_capable", False):
        chunks: List[str] = []
        try:
            for delta in provider.chat_stream(
                    messages, temperature=0.2, max_tokens=max_tokens,
                    force_json=True):
                if not delta:
                    continue
                chunks.append(delta)
                try:
                    on_token(delta)
                except Exception:  # pragma: no cover - callback is best-effort
                    pass
        except Exception:  # pragma: no cover - defensive: stream hiccup
            chunks = []
        raw = "".join(chunks)
        parsed = _extract_json_object(raw)
        if parsed and str(parsed.get("content") or ""):
            return raw
        # Stream failed or produced unparseable text (a mid-JSON truncation
        # looks exactly like this) -- fall back to the reliable blocking
        # call, which additionally grows its budget on a length-cap stop so
        # a truncated body is regenerated to completion instead of dropped.
        # Trace the empty-stream outcome: a streamed call leaves no
        # prompt/response preview in agent.log, so without this an empty
        # body is invisible and cannot be told apart from a parse miss.
        logger.debug(
            "scaffold stream produced no usable content "
            "(raw_chars=%d); falling back to blocking call", len(raw))
    return _blocking_scaffold_call(provider, messages, max_tokens)


def _regenerate_scaffold_file(
        provider: Any, base_system: str, context: str,
        budget: Dict[str, Any], instruction: str) -> str:
    """Re-request a single scaffold file with a hardened instruction.

    Reuses the original context (goal + already-generated files) with an
    appended, failure-specific ``instruction`` so a second attempt can
    correct the problem the first attempt shipped (a syntax error, or a
    test file with no collectable tests). Returns the fence-stripped
    ``content`` string, or ``""`` on any provider/parse failure.
    """
    hardened = base_system + instruction
    try:
        raw = provider.chat(
            messages=[
                {"role": "system", "content": hardened},
                {"role": "user", "content": context},
            ],
            temperature=0.1,
            max_tokens=budget["output_tokens"],
            force_json=True,
        ).get("content", "")
    except Exception:  # pragma: no cover - defensive: provider hiccup
        return ""
    parsed = _extract_json_object(raw)
    out = str(parsed.get("content") or "") if parsed else ""
    if out:
        stripped = out.strip()
        if stripped.startswith("```"):
            m = re.match(r"^```[a-zA-Z0-9_+\-]*\s*\n(.*?)\n```\s*$",
                         stripped, re.DOTALL)
            if m:
                out = m.group(1)
    return out


_LOGIC_REPAIR_SYSTEM = (
    "You are a senior software engineer repairing a project whose automated "
    "tests FAIL.\n\n"
    "You will be given:\n"
    "- The project goal\n"
    "- The failing test output (pytest / test-runner)\n"
    "- The CURRENT complete contents of the most relevant source and test "
    "files\n\n"
    "Diagnose the failure and return corrected COMPLETE file contents for "
    "ONLY the files you change. Return strict JSON:\n"
    '{"files": [{"path": "<relative/path>", "content": "<complete new '
    'file>"}]}\n\n'
    "Rules:\n"
    "- Prefer fixing the SOURCE code so the EXISTING tests pass. Only edit a "
    "test file when the test itself is clearly wrong (asserts behaviour the "
    "goal never asked for).\n"
    "- Output the COMPLETE file content for every file you return -- no "
    "stubs, placeholders, ellipsis, or unified-diff markers.\n"
    "- Only return files that appear in the provided file list, and only "
    "those you actually modified. Do NOT invent new files.\n"
    "- Keep the change as small as the failure requires and consistent with "
    "the existing imports and style.\n"
    "- If you cannot determine a fix from the information given, return "
    '{"files": []}.\n'
)


def _validate_repair_source(path: str, content: str) -> Optional[str]:
    """Return ``None`` when ``content`` parses, else a short error string.

    Mirrors the per-language syntax gate the scaffold single-file path
    applies (Python via ``ast``, JSON via ``json``, the JS/TS/Vue family
    via tree-sitter), so a repaired file that does not parse is rejected
    before it ever reaches APPLY's own gate. Unknown extensions pass.
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "py":
        import ast as _ast
        try:
            _ast.parse(content)
        except SyntaxError as e:
            return str(e)
    elif ext == "json":
        import json as _json
        try:
            _json.loads(content)
        except Exception as e:
            return str(e)
    elif ext in _JS_TS_GRAMMAR_BY_EXT:
        from cgx.codegen.validate import validate_js_ts_source
        diag = validate_js_ts_source(path, content, _JS_TS_GRAMMAR_BY_EXT[ext])
        if not diag.ok:
            return diag.error
    return None


def _extract_repair_files_from_fences(
    raw: str, known: Dict[str, str]
) -> List[Dict[str, str]]:
    """Recover repaired files from markdown fences when JSON parsing fails."""
    if not isinstance(raw, str) or "```" not in raw or not known:
        return []

    entries: List[Dict[str, str]] = []
    seen_paths: set = set()

    # 1. Match code blocks with explicit path= header
    for m in _CODE_FENCE_RE.finditer(raw):
        path = m.group("path").strip()
        body = (m.group("body") or "").rstrip("\n")
        for kp in known:
            if kp not in seen_paths and (kp == path or kp.endswith("/" + path) or path.endswith("/" + kp)):
                entries.append({"path": kp, "content": body})
                seen_paths.add(kp)
                break

    if entries:
        return entries

    # 2. Match code blocks with filename comment on first line
    all_blocks = [
        (m.group("body") or "").rstrip("\n")
        for m in _ANY_CODE_FENCE_RE.finditer(raw)
    ]
    for body in all_blocks:
        if not body:
            continue
        first_line = body.splitlines()[0].strip()
        for kp in known:
            if kp in seen_paths:
                continue
            base = kp.split("/")[-1]
            if (
                kp in first_line or base in first_line
            ) and any(first_line.startswith(pfx) for pfx in ("#", "//", "/*", "*-", "--")):
                entries.append({"path": kp, "content": body})
                seen_paths.add(kp)
                break

    if entries:
        return entries

    # 3. Unambiguous single-file fallback
    if len(known) == 1 and len(all_blocks) == 1:
        body = all_blocks[0]
        kp = list(known.keys())[0]
        if body and not (body.strip().startswith("{") and "files" in body):
            return [{"path": kp, "content": body}]

    return []


def generate_repair_files(
        provider: Any, *,
        goal: str,
        failure_text: str,
        files: List[Dict[str, str]],
        max_files: int = 8,
        localized_files: Optional[List[str]] = None) -> Dict[str, str]:
    """Propose corrected file contents for a logic/assertion failure.

    ``files`` is a list of ``{"path", "content"}`` for the on-disk
    source/test files most relevant to the failure. ``localized_files``
    (optional) names the subset the failure traceback pointed at; those
    blocks are flagged and called out in the prompt so the model starts
    from the failing frames instead of re-deriving the culprit. Returns
    ``{path: new_content}`` for each file the model rewrote whose new
    content differs from the original and passes
    :func:`_validate_repair_source`. Returns an empty mapping on any
    provider/parse failure, when the model declined (``{"files": []}``),
    or when no candidate file was supplied -- the caller then falls back
    to the regenerate path.
    """
    localized = {str(p).strip() for p in (localized_files or []) if str(p).strip()}
    known: Dict[str, str] = {}
    blocks: List[str] = []
    for f in files[:max_files]:
        p = str(f.get("path") or "").strip()
        c = f.get("content")
        if not p or not isinstance(c, str):
            continue
        known[p] = c
        marker = " (traceback points here)" if p in localized else ""
        blocks.append(f"### {p}{marker}\n```\n{c}\n```")
    if not known:
        return {}
    from cgx.answer.model_caps import get_summary_budget
    budget = get_summary_budget(provider)
    shown_localized = [p for p in known if p in localized]
    localized_note = ""
    if shown_localized:
        localized_note = (
            "TRACEBACK LOCALIZATION: the failure traceback flows through "
            + ", ".join(shown_localized)
            + " -- start your diagnosis there.\n\n")
    context = (
        f"PROJECT GOAL:\n{goal}\n\n"
        f"FAILING TEST OUTPUT:\n{failure_text}\n\n"
        + localized_note
        + "CURRENT FILES:\n\n" + "\n\n".join(blocks)
    )
    messages = [
        {"role": "system", "content": _LOGIC_REPAIR_SYSTEM},
        {"role": "user", "content": context},
    ]
    try:
        raw = provider.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=budget["output_tokens"],
            force_json=True,
            json_schema=REPAIR_FILES_SCHEMA,
        ).get("content", "")
    except Exception:  # pragma: no cover - defensive: provider hiccup
        return {}
    parsed = _extract_json_object(raw)
    entries = parsed.get("files") if parsed else None
    if not isinstance(entries, list) or not entries:
        fence_entries = _extract_repair_files_from_fences(raw, known)
        if fence_entries:
            entries = fence_entries
            parsed = {"files": fence_entries}
    violations = validate_json_schema(parsed or {}, REPAIR_FILES_SCHEMA)
    if violations:
        # One bounded re-ask: fold the concrete violations back so the model
        # fixes the shape; a second miss falls through to the empty mapping
        # and the caller's regenerate path.
        logger.warning("generate_repair_files: schema violation, "
                       "re-asking -- %s", "; ".join(violations[:4]))
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            "Your reply did not match the required schema. Violations:\n- "
            + "\n- ".join(violations[:8])
            + "\n\nReply again with STRICT JSON only: "
              '{"files": [{"path": "<known path>", "content": "<complete '
              'corrected file>"}]}. Use {"files": []} to decline. '
              "No prose outside JSON.")})
        try:
            raw = provider.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=budget["output_tokens"],
                force_json=True,
                json_schema=REPAIR_FILES_SCHEMA,
            ).get("content", "")
        except Exception:  # pragma: no cover - defensive: provider hiccup
            return {}
        parsed = _extract_json_object(raw)
        entries = parsed.get("files") if parsed else None
        if not isinstance(entries, list) or not entries:
            fence_entries = _extract_repair_files_from_fences(raw, known)
            if fence_entries:
                entries = fence_entries
                parsed = {"files": fence_entries}
    entries = parsed.get("files") if parsed else None
    if not isinstance(entries, list):
        return {}
    out: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        p = str(entry.get("path") or "").strip()
        c = entry.get("content")
        if p not in known or not isinstance(c, str) or not c.strip():
            continue
        stripped = c.strip()
        if stripped.startswith("```"):
            m = re.match(r"^```[a-zA-Z0-9_+\-]*\s*\n(.*?)\n```\s*$",
                         stripped, re.DOTALL)
            if m:
                c = m.group(1)
        if c == known[p]:
            continue
        if _validate_repair_source(p, c) is not None:
            continue
        out[p] = c
    return out


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
