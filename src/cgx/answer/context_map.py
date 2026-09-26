

"""Tiered SLM context builder ("Code Map") for the answer pipeline.

The retrieval orchestrator surfaces two kinds of hits in its top-K list:

1. **Primary hits** -- chunks that matched semantically or lexically (or were
   the seeds for graph expansion). The LLM needs the full code body for
   these to ground its answer.
2. **Graph neighbors** -- chunks discovered by walking the call/import graph
   one or more hops from a primary hit. They typically don't need a full
   body in the prompt; a compact stub (``signature + doc_first_sentence +
   class_name``) is enough for the model to understand the structural
   relationship without burning prompt budget.

``build_tiered_context`` splits a single hit list into those two tiers, sizes
each tier against a budget provided by :func:`cgx.answer.model_caps.get_context_map_budget`,
and returns a list of source dicts in the same shape as
:func:`cgx.answer.engine._as_sources_with_meta` returns -- plus a ``tier`` key
so :func:`cgx.answer.engine._fmt_source` can render a hint to the model.

Assignment is deterministic and query-aware (:func:`decide_visibility`): each
hit is banded full / long / short / hide by its relevance rank for the current
query, so a strong hit gets a full body and a weak one a compact stub or is
hidden -- without ever being deleted from state. This is the query-aware
compression the paper contrasts with compaction. ``classify_hits`` (the older
graph-depth primary/neighbor split) is retained for callers that still want it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from cgx.io.persist import load_jsonl


def load_records_by_id(records_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load a records.jsonl file and key it by ``id``.

    Returns an empty dict when ``records_path`` is falsy or unreadable; the
    caller treats the absence of records as "no enrichment available" rather
    than as an error condition.
    """
    if not records_path:
        return {}
    try:
        recs = load_jsonl(records_path)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if isinstance(r, dict):
            rid = r.get("id")
            if rid:
                out[str(rid)] = r
    return out


def classify_hits(
    hits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``hits`` into ``(primary, neighbors)`` by ``provenance.graph_depth``."""
    primary: List[Dict[str, Any]] = []
    neighbors: List[Dict[str, Any]] = []
    for h in hits or []:
        prov = h.get("provenance") if isinstance(h, dict) else None
        depth = 0
        if isinstance(prov, dict):
            d = prov.get("graph_depth")
            if isinstance(d, (int, float)):
                depth = int(d)
        if depth >= 1:
            neighbors.append(h)
        else:
            primary.append(h)
    return primary, neighbors


def format_neighbor_stub(record: Optional[Dict[str, Any]], symbol: str) -> str:
    """Compose a neighbor stub from record fields.

    Format: ``[class.]symbol(signature) -- doc_first_sentence`` with each
    component dropped silently when missing. Falls back to the symbol alone
    when no enrichment is available.
    """
    rec = record or {}
    sig = str(rec.get("signature") or "").strip()
    doc1 = str(rec.get("doc_first_sentence") or "").strip()
    cls = str(rec.get("class_name") or "").strip()
    name = str(rec.get("name") or symbol or "").strip()

    head = f"{cls}.{name}" if cls and name else name
    if sig:
        # Signatures already include the parameter list; if the parser stored
        # them as bare ``(a, b)`` we still want a readable head.
        head = f"{head}{sig}" if sig.startswith("(") else f"{head} :: {sig}"
    if doc1:
        head = f"{head} -- {doc1}" if head else doc1
    return head


def _split_chunk_id(cid: str) -> Tuple[str, str, str]:
    parts = str(cid).split("::")
    p = parts[0] if parts else ""
    k = parts[1] if len(parts) > 1 else ""
    s = parts[2] if len(parts) > 2 else ""
    return p, k, s


def _provenance_of(h: Dict[str, Any]) -> Dict[str, Any]:
    prov: Dict[str, Any] = {}
    for k, v in (h or {}).items():
        if k == "chunk_id":
            continue
        if k == "provenance" and isinstance(v, dict):
            prov.update(v)
        else:
            prov[k] = v
    return prov


# The visibility ladder (JEV Point #1), most-visible first.
_VISIBILITY_LEVELS: Tuple[str, ...] = ("full", "long", "short", "hide")


def _relevance_signals(h: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """Extract (strong, deep, demoted) provenance bumps for a hit.

    ``strong``  -- a symbol match or a top-2 intent rank: bump UP one band.
    ``deep``    -- a graph neighbor two-plus hops out: bump DOWN one band.
    ``demoted`` -- scope-penalized (wrong src/tests half): bump DOWN one band.
    """
    prov = h.get("provenance") if isinstance(h, dict) else None
    prov = prov if isinstance(prov, dict) else {}
    intent_rank = prov.get("intent_rank")
    strong = bool(prov.get("symbol_match")) or (
        isinstance(intent_rank, (int, float))
        and not isinstance(intent_rank, bool) and int(intent_rank) <= 2)
    depth = prov.get("graph_depth")
    deep = (isinstance(depth, (int, float)) and not isinstance(depth, bool)
            and int(depth) >= 2)
    demoted = bool(prov.get("scope_demoted"))
    return strong, deep, demoted


def decide_visibility(
    hits: List[Dict[str, Any]],
    *,
    budget: Dict[str, int],
) -> List[Tuple[Dict[str, Any], str]]:
    """Assign each hit a visibility level (full / long / short / hide) for THIS query.

    Deterministic and model-free -- the correct local-first default: a per-chunk
    LLM call would reintroduce exactly the cost the paper warns about. Bands by
    QUANTILE over the RRF hit order (top ~20% full, next ~30% long, next ~30%
    short, rest hide); a quantile, not an absolute score cut, because RRF /
    reranker scores are batch-relative. A strong signal (symbol match or a
    top-2 intent rank) bumps a hit up one band; a deep graph neighbor
    (``graph_depth >= 2``) or a scope-demoted hit bumps it down one. The rank-1
    hit is always ``full`` so the top result is never summarized away, and the
    number of ``full`` chunks is capped at ``budget['primary_max']`` (overflow
    demotes to ``long``). Returns ``(hit, level)`` pairs in the input order.
    """
    kept = [h for h in (hits or []) if isinstance(h, dict)]
    n = len(kept)
    if n == 0:
        return []
    full_cut = max(1, math.ceil(0.20 * n))
    long_cut = max(full_cut, math.ceil(0.50 * n))
    short_cut = max(long_cut, math.ceil(0.80 * n))
    primary_max = int(budget.get("primary_max", n) or n)

    out: List[Tuple[Dict[str, Any], str]] = []
    full_used = 0
    for i, h in enumerate(kept):
        if i < full_cut:
            band = 0
        elif i < long_cut:
            band = 1
        elif i < short_cut:
            band = 2
        else:
            band = 3
        strong, deep, demoted = _relevance_signals(h)
        if strong:
            band -= 1
        if deep or demoted:
            band += 1
        band = max(0, min(3, band))
        if i == 0:  # rank-1 is never summarized away
            band = 0
        if band == 0:
            if full_used >= primary_max:
                band = 1  # demote overflow full -> long
            else:
                full_used += 1
        out.append((h, _VISIBILITY_LEVELS[band]))
    return out


def build_tiered_context(
    hits: List[Dict[str, Any]],
    cmap: Dict[str, Dict[str, Any]],
    records_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    budget: Dict[str, int],
    focus_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return source dicts ordered most-visible first (full -> long -> short).

    Parameters
    ----------
    hits : list of dict
        Retrieval hits with ``chunk_id`` and optional ``provenance.graph_depth``.
    cmap : dict
        Mapping ``chunk_id -> row`` from :func:`cgx.answer.engine._chunk_map`.
        Used to fetch the full chunk body for primary sources.
    records_by_id : dict or None
        Mapping ``chunk_id -> record`` from :func:`load_records_by_id`. Used
        to fetch ``signature``, ``doc_first_sentence``, and ``class_name`` for
        neighbor stubs and for primary signature metadata. When ``None`` the
        builder falls back to whatever the view row already carries.
    budget : dict
        Output of :func:`cgx.answer.model_caps.get_context_map_budget`.
    focus_terms : list of str or None
        Terms to centre the primary window text on; forwarded to
        :func:`cgx.answer.engine._as_sources_with_meta`.

    Returns
    -------
    list of dict
        Each item carries the same keys as ``_as_sources_with_meta`` plus a
        ``tier`` field -- one of ``"full"`` / ``"long"`` / ``"short"`` from the
        visibility ladder (:func:`decide_visibility`). The list is ordered
        most-visible first and truncated to fit ``budget['total_chars']``.
    """
    # Lazy import: engine.py imports this module, so a top-level import would
    # create a cycle at module load. The functions we use here are pure.
    from cgx.answer.engine import _as_sources_with_meta

    records_by_id = records_by_id or {}
    leveled = decide_visibility(hits, budget=budget)
    full_hits = [h for (h, lvl) in leveled if lvl == "full"]
    long_hits = [h for (h, lvl) in leveled if lvl == "long"]
    short_hits = [h for (h, lvl) in leveled if lvl == "short"]
    # ``hide`` hits stay in state but are invisible for THIS query.

    full_chars = int(budget.get("primary_chars", 0))
    long_chars = int(budget.get("long_chars", 0)) or max(1, full_chars // 2)
    short_chars = int(budget.get("neighbor_chars", 0))
    neighbor_max = int(budget.get("neighbor_max", len(short_hits)))

    def _bodied(level_hits: List[Dict[str, Any]], max_chars: int,
                tier: str) -> List[Dict[str, Any]]:
        srcs = _as_sources_with_meta(
            level_hits, cmap, max_chunks=len(level_hits),
            max_chars=max_chars, focus_terms=focus_terms)
        for s in srcs:
            s["tier"] = tier
            # Backfill signature from the full record when the corpus row
            # didn't carry one.
            if not s.get("signature"):
                rec = records_by_id.get(str(s.get("chunk_id"))) or {}
                sig = rec.get("signature")
                if isinstance(sig, str) and sig:
                    s["signature"] = sig
        return srcs

    full_sources = _bodied(full_hits, full_chars, "full")
    long_sources = _bodied(long_hits, long_chars, "long")

    short_sources: List[Dict[str, Any]] = []
    for h in short_hits[:neighbor_max]:
        cid = str(h.get("chunk_id"))
        path, kind, symbol = _split_chunk_id(cid)
        row = cmap.get(cid) or {}
        rec = records_by_id.get(cid) or {}
        stub = format_neighbor_stub(rec, symbol)
        if not stub:
            # Last-resort fallback: trim the view text so the chunk stays
            # visible even without enrichment.
            text = row.get("text", "") if isinstance(row, dict) else ""
            stub = (text or "")[:short_chars]
        elif len(stub) > short_chars > 0:
            stub = stub[: short_chars - 3] + "..."
        short_sources.append({
            "chunk_id": cid,
            "path": path,
            "kind": kind,
            "symbol": symbol,
            "signature": rec.get("signature") or "",
            "start_line": rec.get("start_line"),
            "end_line": rec.get("end_line"),
            "parent_class_id": rec.get("parent_class_id") or "",
            "text": stub,
            "hit_meta": _provenance_of(h),
            "tier": "short",
        })

    # Enforce the total-chars ceiling deterministically. Bodies are ordered
    # most-visible first (full -> long -> short), so dropping trailing items
    # once the cumulative length would exceed the cap drops the
    # lowest-relevance content first.
    ordered = full_sources + long_sources + short_sources
    total_cap = int(budget.get("total_chars", 0))
    if total_cap <= 0:
        return ordered
    out: List[Dict[str, Any]] = []
    used = 0
    for s in ordered:
        body = s.get("text", "") or ""
        if used + len(body) > total_cap and out:
            break
        out.append(s)
        used += len(body)
    return out
