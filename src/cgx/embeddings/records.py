

from __future__ import annotations

"""
S4 -- Deterministic record builder and two-view embedding corpus.

This module is ADDITIVE and safe: it does not modify existing parse/graph code or
assume any embedding model. It turns your parsed chunks (+ optional graph) into:

1) Canonical index records (one per chunk) with the exact, deterministic fields
   you outlined (identity, graph anchors, semantics, metrics, search helpers).
2) A flat embedding corpus with two rows per chunk (view='intent' and 'impl'),
   containing the text to embed and basic accounting (tokens estimate, mapping).

Primary entrypoints:
- make_index_records(chunks, G=None, ...)
- prepare_embedding_corpus(records, which=('intent','impl'))

Embedding Context
-----------------
- The records built here feed directly into **embedding backends**.
- Each row from `prepare_embedding_corpus` is consumed by
  `cgx.embeddings.build.build_embeddings`, which turns `text` into dense vectors.
- To add a new model (e.g., Gemma), extend `build_embeddings` in `build.py`.
"""

from cgx.logging_setup import get_logger
from typing import Any, Dict, List, Sequence
from cgx.embeddings.helpers import (
    _safe_get,
    _neighbors_summary,
    _lexical_helpers,
    _estimate_tokens,
    _attribute_roots_read,
    _imports_full,
    _parent_class_id,
    _defines_children_ids,
    _calls_degree,
    _calls_out_ids,
    _normalize_raises,
)

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # we guard for None at call sites

from .views import (
    build_intent_view,
    build_implementation_view,
)

logger = get_logger(__name__)


# Schema version for records and persisted index manifests.
# Bumped whenever the record/chunk/lexical-helper shape or tokenizer changes
# in a way that requires re-indexing. Readers should reject or rebuild caches
# whose manifest schema_version is older than this value.
#
# v2: Symmetric sub-word tokenizer (cgx.retrieval.tokenize) -- camelCase /
#     PascalCase / snake_case identifiers now expand into their sub-words on
#     BOTH the indexer side (_split_tokens -> lexical_helpers.ngrams_*) and
#     the query side (_tokenize_lc, _extract_symbol_tokens). Records from
#     v1 indices will under-match for partial-name queries and should be
#     rebuilt.
# v3: Line/column anchors persisted per record. Each record now carries
#     start_line / end_line / col_offset (mirrored from the parser chunk)
#     so downstream consumers (suggest_insertion_points, ast_insert) can
#     splice code without re-walking the AST. v2 indices lack these
#     fields; readers should rebuild.
SCHEMA_VERSION = 3


# ---------------------------
# Public API
# ---------------------------

def make_index_records(
    chunks: List[Dict[str, Any]],
    G=None,
    *,
    topk_callees: int = 10,
    normalize_impl: bool = False,
    strip_literals: bool = False,
    neighbors_cap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Build canonical, deterministic **index records** for each chunk.

    This DOES NOT compute embeddings (no model assumptions).
    It includes the two text views (intent, impl) and all metadata you specified.

    Returns a NEW list of records. Original chunks are not modified.

    Record schema (key fields)
    --------------------------
    - Identity & context:
        id, type, name, file, class_name, signature,
        docstring (full), doc_first_sentence, module_path (if present)
    - Graph anchors:
        parent_file_id, parent_class_id,
        defines_children_ids,
        calls_out_ids (internal only),
        calls_out_unresolved (names),
        calls_in_count, calls_out_count,
        neighbors_summary ([(edge_type, neighbor_id)] limited)
    - Code semantics:
        imports_used (full module names),
        attributes_used_root_reads,
        instance_attributes_written (name, lineno, value_preview, source),
        raises, decorators, method_kind, is_async, is_generator, returns_annotation
    - Metrics:
        metrics (as in chunk.meta.metrics)
    - Views:
        view_intent (string), view_impl (string)
    - Search helpers:
        lexical_helpers (lowercased fields + ngrams),
        tokens_estimate: {"intent": int, "impl": int}

    Parameters
    ----------
    chunks : list[dict]
        Output of parse_codebase (S1/S2/S3).
    G : nx.(Multi)DiGraph or None
        Knowledge graph; used for anchors and degrees. If None, those fields are best-effort.
    topk_callees : int
        Number of callee names used in the intent card (passed into views).
    normalize_impl : bool
        Normalize whitespace in implementation view deterministically.
    strip_literals : bool
        Replace string/numeric literals with <STR>/<NUM> in implementation view.
    neighbors_cap : int
        Cap for neighbors_summary.

    Returns
    -------
    list[dict]
        Deterministic records, one per chunk.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"make_index_records: 'chunks' must be a list, got {type(chunks)}")

    records: List[Dict[str, Any]] = []
    seen_ids = set()

    for ch in chunks:
        # --- Validation ---
        if not isinstance(ch, dict):
            logger.error("make_index_records: skipping non-dict chunk %r", ch)
            continue

        cid = ch.get("id")
        ctype = ch.get("type")

        if not cid or not ctype:
            logger.error("make_index_records: skipping invalid chunk (missing id/type): %r", ch)
            continue

        if cid in seen_ids:
            logger.warning("make_index_records: duplicate chunk id %s skipped", cid)
            continue
        seen_ids.add(cid)

        try:
            meta = ch.get("meta") or {}

            # Views (deterministic; do not mutate original chunk)
            v_intent = build_intent_view(ch, G=G, topk_callees=topk_callees)
            v_impl = build_implementation_view(
                ch,
                all_chunks=chunks,
                normalize=normalize_impl,
                strip_literals=strip_literals,
            )

            # Graph-derived
            parent_file_id = ch.get("file")
            parent_class_id = _parent_class_id(ch)
            defines_children = _defines_children_ids(G, cid, limit=neighbors_cap)
            calls_internal, calls_unresolved = _calls_out_ids(G, cid)
            cin, cout = _calls_degree(G, cid)
            neigh = _neighbors_summary(G, cid, max_n=neighbors_cap)

            # Semantics
            imports_used = _imports_full(meta)
            attr_reads_roots = _attribute_roots_read(meta)
            inst_attr_written = meta.get("instance_attributes") or []
            raises_list = _normalize_raises(meta)

            # Identity & doc
            doc_full = meta.get("docstring")
            doc_first = (
                _safe_get(meta, "doc_parsed.summary")
                or (doc_full.splitlines()[0].strip()
                    if isinstance(doc_full, str) and doc_full.strip()
                    else "")
            )

            # Search helpers
            lex = _lexical_helpers(ch)
            tok_intent = _estimate_tokens(v_intent)
            tok_impl = _estimate_tokens(v_impl)

            rec: Dict[str, Any] = {
                # schema version (see SCHEMA_VERSION at module top)
                "schema_version": SCHEMA_VERSION,
                # identity & context
                "id": cid,
                "type": ctype,
                "name": ch.get("name"),
                "file": ch.get("file"),
                "class_name": meta.get("class_name"),
                "signature": meta.get("signature"),
                "docstring": doc_full,
                "doc_first_sentence": doc_first,
                "module_path": ch.get("module_path"),  # for file chunks (S2)

                # location anchors (v3): zero when the chunk source lacks AST
                # span info (e.g. malformed nodes). Consumers should treat 0
                # as "unknown" and fall back to AST walks.
                "start_line": int(ch.get("start_line") or 0),
                "end_line": int(ch.get("end_line") or 0),
                "col_offset": int(ch.get("col_offset") or 0),

                # graph anchors
                "parent_file_id": parent_file_id,
                "parent_class_id": parent_class_id,
                "defines_children_ids": defines_children,
                "calls_out_ids": calls_internal,
                "calls_out_unresolved": calls_unresolved,
                "calls_in_count": cin,
                "calls_out_count": cout,
                "neighbors_summary": neigh,

                # semantics
                "imports_used": imports_used,
                "attributes_used_root_reads": attr_reads_roots,
                "instance_attributes_written": inst_attr_written,
                "raises": raises_list,
                "decorators": meta.get("decorators"),
                "method_kind": meta.get("method_kind"),
                "is_async": bool(meta.get("is_async", False)),
                "is_generator": bool(meta.get("is_generator", False)),
                "returns_annotation": meta.get("returns_annotation"),

                # metrics
                "metrics": meta.get("metrics"),

                # views (text)
                "view_intent": v_intent,
                "view_impl": v_impl,

                # search helpers
                "lexical_helpers": lex,
                "tokens_estimate": {"intent": tok_intent, "impl": tok_impl},

                # vectors (placeholders; you will fill later if desired)
                "vec_intent": None,
                "vec_impl": None,
            }

            records.append(rec)

        except Exception as e:
            logger.error("make_index_records: failed on chunk %s: %s", cid, e)

    return records


def prepare_embedding_corpus(
    records: List[Dict[str, Any]],
    *,
    which: Sequence[str] = ("intent", "impl"),
) -> List[Dict[str, Any]]:
    """
    Flatten index records into a corpus suitable for embedding/indexing.

    Each row:
      {
        "chunk_id": <id>,
        "view": "intent" | "impl",
        "text": <string>,
        "tokens_estimate": <int>,
        # convenient echo of identity for downstream tooling (optional)
        "type": <chunk type>,
        "name": <symbol name>,
        "file": <file path>,
      }

    This function DOES NOT call any model and keeps ordering stable: records order
    × the 'which' order.

    Parameters
    ----------
    records : list[dict]
        Produced by make_index_records.
    which : sequence of str
        Subset of {"intent","impl"} to include and in what order.

    Returns
    -------
    list[dict]
    """
    allowed = {"intent", "impl"}
    for w in which:
        if w not in allowed:
            raise ValueError(
                f"prepare_embedding_corpus: unsupported view '{w}'. Use any of {sorted(allowed)}."
            )

    corpus: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            logger.error(
                "prepare_embedding_corpus: BAD RECORD at index=%s type=%s value=%r",
                idx, type(rec), rec
            )
            continue

        try:
            token_est = rec.get("tokens_estimate", {})
            for w in which:
                # normalize token_est
                if isinstance(token_est, dict):
                    tok = token_est.get(w, 0)
                elif isinstance(token_est, (int, float)):
                    tok = token_est
                else:
                    tok = 0

                corpus.append(
                    {
                        "chunk_id": rec.get("id"),
                        "view": w,
                        "text": rec.get(f"view_{w}", "") or "",
                        "tokens_estimate": int(tok),
                        "type": rec.get("type"),
                        "name": rec.get("name"),
                        "file": rec.get("file"),
                    }
                )
        except Exception as e:
            logger.exception(
                "prepare_embedding_corpus: FAILED on record index=%s type=%s rec=%r (%s)",
                idx, type(rec), rec, e
            )

    logger.info(
        "prepare_embedding_corpus: built %d rows from %d records",
        len(corpus), len(records)
    )
    return corpus
