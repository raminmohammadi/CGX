

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

The classifier rule is deterministic: a hit with
``provenance.graph_depth >= 1`` is a neighbor, anything else is primary.
The depth annotation is set by ``HybridRetriever._expand_multi_hop`` in
:mod:`cgx.retrieval.orchestrator`.
"""

from __future__ import annotations

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


def build_tiered_context(
    hits: List[Dict[str, Any]],
    cmap: Dict[str, Dict[str, Any]],
    records_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    budget: Dict[str, int],
    focus_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return source dicts ordered as ``primary…`` then ``neighbor…``.

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
        ``tier`` field (``"primary"`` or ``"neighbor"``). The list is ordered
        primary-first and truncated to fit ``budget['total_chars']``.
    """
    # Lazy import: engine.py imports this module, so a top-level import would
    # create a cycle at module load. The functions we use here are pure.
    from cgx.answer.engine import _as_sources_with_meta

    primary, neighbors = classify_hits(hits)
    records_by_id = records_by_id or {}

    primary_sources: List[Dict[str, Any]] = _as_sources_with_meta(
        primary,
        cmap,
        max_chunks=int(budget.get("primary_max", 0)),
        max_chars=int(budget.get("primary_chars", 0)),
        focus_terms=focus_terms,
    )
    for s in primary_sources:
        s["tier"] = "primary"
        # Backfill signature from the full record when the corpus row didn't
        # carry one; neighbors-as-primary still benefit from the richer head.
        if not s.get("signature"):
            rec = records_by_id.get(str(s.get("chunk_id"))) or {}
            sig = rec.get("signature")
            if isinstance(sig, str) and sig:
                s["signature"] = sig

    neighbor_sources: List[Dict[str, Any]] = []
    neighbor_max = int(budget.get("neighbor_max", 0))
    neighbor_chars = int(budget.get("neighbor_chars", 0))
    for h in neighbors[:neighbor_max]:
        cid = str(h.get("chunk_id"))
        path, kind, symbol = _split_chunk_id(cid)
        row = cmap.get(cid) or {}
        rec = records_by_id.get(cid) or {}
        stub = format_neighbor_stub(rec, symbol)
        if not stub:
            # Last-resort fallback: trim the view text. Keeps the neighbor
            # visible even when no enrichment is available.
            text = row.get("text", "") if isinstance(row, dict) else ""
            stub = (text or "")[:neighbor_chars]
        elif len(stub) > neighbor_chars > 0:
            stub = stub[: neighbor_chars - 3] + "..."
        neighbor_sources.append({
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
            "tier": "neighbor",
        })

    # Enforce the total-chars ceiling deterministically: walk in order
    # (primary first, then neighbors) and drop trailing items once the
    # cumulative body length would exceed the cap.
    total_cap = int(budget.get("total_chars", 0))
    ordered = primary_sources + neighbor_sources
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
