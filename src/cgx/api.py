# src/cgx/api.py
from __future__ import annotations

"""
Public, high-level API entrypoints for cgx.

These functions are **non-invasive wrappers** that stitch together your existing
building blocks. They catch/log errors and return structured results without
changing any of your current modules.

Main entrypoints:
- parse(project_root) -> (chunks, calls)
- build_graph(chunks, calls) -> G
- build_records(chunks, G=None) -> records
- build_indexes(records, embed_cfg, faiss_cfg) -> {"two_view": tv, "lexical": lex, "meta": {...}}
- hybrid_search(project_root, query, embed_cfg, faiss_cfg, hybrid_cfg) -> result dict
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from .logging_setup import get_logger
from .config import EmbeddingConfig, FaissConfig, HybridSearchConfig

logger = get_logger(__name__)


# --------- tolerant imports (no hard coupling) ---------
def _import_parse_codebase():
    try:
        from cgx.parse.ast_parser import parse_codebase  # your S1 location
        return parse_codebase
    except Exception:
        pass
    try:
        from cgx.parse import parse_codebase
        return parse_codebase
    except Exception:
        return None

def _import_build_graph():
    try:
        from cgx.graph.knowledge import build_knowledge_graph
        return build_knowledge_graph
    except Exception:
        pass
    try:
        from cgx.graph import build_knowledge_graph
        return build_knowledge_graph
    except Exception:
        return None

def _import_records():
    try:
        from cgx.embeddings.records import make_index_records
        return make_index_records
    except Exception:
        return None

def _import_two_view_index():
    try:
        from cgx.retrieval.index import TwoViewIndex
        return TwoViewIndex
    except Exception:
        return None

def _import_lexical_index():
    try:
        from cgx.retrieval.lexical import LexicalIndex
        return LexicalIndex
    except Exception:
        return None

def _import_hybrid():
    try:
        from cgx.retrieval.hybrid import HybridRetriever, HybridConfig  # if you keep a local HybridConfig
        return HybridRetriever, HybridConfig
    except Exception:
        return None, None


# --------- simple embedder loader (no model assumption) ---------
def _load_embedder(model_name: str):
    """
    Return an object with .encode(list[str]) -> np.ndarray[float32].
    Tries Sentence-Transformers, then HF Transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        class _ST:
            def encode(self, texts: List[str]):
                import numpy as np
                vecs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
                return vecs.astype("float32", copy=False)
        logger.info("api: embedder=sentence-transformers (%s)", model_name)
        return _ST()
    except Exception:
        pass

    from transformers import AutoTokenizer, AutoModel
    import torch, numpy as np
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()

    def _encode(texts: List[str]):
        with torch.no_grad():
            t = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = mdl(**t)
            hidden = out.last_hidden_state
            mask = t["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return vec.cpu().numpy().astype("float32", copy=False)

    class _HF:
        def encode(self, texts: List[str]): return _encode(texts)

    logger.info("api: embedder=transformers (%s)", model_name)
    return _HF()


# --------- public orchestration wrappers ---------
def parse(project_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse the codebase rooted at `project_root`.

    Returns:
        (chunks, calls)

    Raises:
        RuntimeError if parser is not available or parsing fails.
    """
    parse_codebase = _import_parse_codebase()
    if parse_codebase is None:
        msg = "api.parse: parse_codebase not found (cgx.parse.ast_parser or cgx.parse)."
        logger.error(msg)
        raise RuntimeError(msg)

    try:
        chunks, calls = parse_codebase(project_root)
        logger.info("api.parse: chunks=%d calls=%d", len(chunks), len(calls))
        return chunks, calls
    except Exception as e:
        logger.exception("api.parse: failed on %r", project_root)
        raise RuntimeError(f"parse failed for {project_root}: {e}") from e


def build_graph(chunks: List[Dict[str, Any]], calls: List[Dict[str, Any]]):
    """
    Build a knowledge graph from chunks + calls.

    Returns:
        A networkx (Multi)DiGraph, or None if the builder is not available.
    """
    builder = _import_build_graph()
    if builder is None:
        logger.warning("api.build_graph: graph builder not found; returning None")
        return None
    try:
        G = builder(chunks, calls)
        logger.info("api.build_graph: nodes=%d edges=%d", len(G.nodes), len(G.edges))
        return G
    except Exception as e:
        logger.exception("api.build_graph: failed")
        return None


def build_records(chunks: List[Dict[str, Any]], G=None) -> List[Dict[str, Any]]:
    """
    Build canonical index records from chunks (+ optional graph).

    Returns:
        records (list of dict) suitable for two-view embedding / lexical index.
    """
    make_index_records = _import_records()
    if make_index_records is None:
        msg = "api.build_records: make_index_records not found (cgx.embeddings.records)."
        logger.error(msg)
        raise RuntimeError(msg)
    try:
        recs = make_index_records(chunks, G=G)
        logger.info("api.build_records: %d records", len(recs))
        return recs
    except Exception as e:
        logger.exception("api.build_records: failed")
        raise RuntimeError(f"build_records failed: {e}") from e


def build_indexes(
    records: List[Dict[str, Any]],
    *,
    embed_cfg: Optional[EmbeddingConfig] = None,
    faiss_cfg: Optional[FaissConfig] = None,
) -> Dict[str, Any]:
    """
    Build the two-view ANN index and the lexical index.

    Returns:
        {"two_view": TwoViewIndex, "lexical": LexicalIndex, "meta": {...}}

    Notes:
        - This function only wires the objects; it does not assume a specific model.
    """
    embed_cfg = embed_cfg or EmbeddingConfig()
    faiss_cfg = faiss_cfg or FaissConfig()

    TwoViewIndex = _import_two_view_index()
    LexicalIndex = _import_lexical_index()
    if TwoViewIndex is None:
        raise RuntimeError("api.build_indexes: TwoViewIndex not found (cgx.retrieval.index).")
    if LexicalIndex is None:
        raise RuntimeError("api.build_indexes: LexicalIndex not found (cgx.retrieval.lexical).")

    try:
        embedder = _load_embedder(embed_cfg.model_name)
        tv = TwoViewIndex.from_records(
            records,
            embedder=embedder,
            # pass through FAISS config if your TwoViewIndex accepts it
            faiss_metric=faiss_cfg.metric,
            faiss_index=faiss_cfg.index,
            nlist=faiss_cfg.nlist,
            nprobe=faiss_cfg.nprobe,
            M=faiss_cfg.M,
            efConstruction=faiss_cfg.efConstruction,
            efSearch=faiss_cfg.efSearch,
            use_gpu=faiss_cfg.use_gpu,
        )
        lex = LexicalIndex.from_records(records)
        logger.info("api.build_indexes: two_view and lexical indexes ready")
        return {"two_view": tv, "lexical": lex, "meta": {"embedding_model": embed_cfg.model_name}}
    except Exception as e:
        logger.exception("api.build_indexes: failed")
        raise RuntimeError(f"build_indexes failed: {e}") from e


def hybrid_search(
    project_root: str,
    query: str,
    *,
    embed_cfg: Optional[EmbeddingConfig] = None,
    faiss_cfg: Optional[FaissConfig] = None,
    hybrid_cfg: Optional[HybridSearchConfig] = None,
) -> Dict[str, Any]:
    """
    Full end-to-end hybrid search:
        parse → (optional) graph → records → indexes → hybrid.RRF

    Returns:
        Result dict as produced by your HybridRetriever.search().
    """
    embed_cfg = embed_cfg or EmbeddingConfig()
    faiss_cfg = faiss_cfg or FaissConfig()
    hybrid_cfg = hybrid_cfg or HybridSearchConfig()

    # 1) parse
    chunks, calls = parse(project_root)

    # 2) graph (optional)
    G = build_graph(chunks, calls) if hybrid_cfg.build_graph else None

    # 3) records
    records = build_records(chunks, G=G)

    # 4) indexes
    idx = build_indexes(records, embed_cfg=embed_cfg, faiss_cfg=faiss_cfg)
    tv, lex = idx["two_view"], idx["lexical"]

    # 5) hybrid orchestration
    HybridRetriever, HybridConfig = _import_hybrid()
    if HybridRetriever is None:
        raise RuntimeError("api.hybrid_search: HybridRetriever not found (cgx.retrieval.hybrid).")

    # prefer your native HybridConfig if available; otherwise reuse values from HybridSearchConfig
    cfg = (HybridConfig(k_intent=hybrid_cfg.k_intent,
                        k_impl=hybrid_cfg.k_impl,
                        k_lex=hybrid_cfg.k_lex,
                        expand_top_n=hybrid_cfg.expand_top_n,
                        expand_per_seed=hybrid_cfg.expand_per_seed,
                        rrf_k=hybrid_cfg.rrf_k,
                        top_k_chunks=hybrid_cfg.top_k)
           if HybridConfig else hybrid_cfg)

    try:
        embedder = _load_embedder(embed_cfg.model_name)
        retr = HybridRetriever(tv_index=tv, records=records, lexical_index=lex, G=G)
        result = retr.search(query, embedder=embedder, cfg=cfg)
        logger.info("api.hybrid_search: ok (query=%r)", query)
        return result
    except Exception as e:
        logger.exception("api.hybrid_search: failed for query=%r", query)
        raise RuntimeError(f"hybrid_search failed: {e}") from e
