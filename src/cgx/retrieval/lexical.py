

from __future__ import annotations

"""
Deterministic lexical index over S4 records (BM25-lite, no external deps).

Input:
  - S4 records with record["lexical_helpers"] carrying:
      - "ngrams_1", "ngrams_2" (lowercased tokens)
      - lowercased fields like name_lc/file_lc/signature_lc (optional)

Usage:
  idx = LexicalIndex.from_records(records)
  hits = idx.search("database reconnect backoff", top_k=50)  # -> [{chunk_id, score, rank}, ...]

Purely additive; does not read/modify any existing indices or models.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import os
import re
import logging

from cgx.retrieval.tokenize import tokenize_text

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _tokenize_lc(q: str) -> List[str]:
    # Symmetric with embeddings.helpers._split_tokens: identifier-aware
    # camelCase / snake_case sub-word expansion so a query like "reconnect"
    # can hit a chunk whose name is ``databaseReconnect``.
    return tokenize_text(q, min_len=1)


@dataclass
class _Posting:
    tf: int


class LexicalIndex:
    def __init__(self) -> None:
        self.df: Dict[str, int] = {}                       # term -> doc freq
        self.postings: Dict[str, Dict[str, _Posting]] = {} # term -> {chunk_id -> posting}
        self.doc_len: Dict[str, int] = {}                  # chunk_id -> length in tokens
        self.N: int = 0
        self.avgdl: float = 0.0

        # params (BM25-ish)
        self.k1: float = 1.2
        self.b: float = 0.75

    # ---------- build ----------

    @classmethod
    def from_records(cls, records: List[Dict]) -> "LexicalIndex":
        self = cls()
        valid = 0

        for rec in records:
            if not isinstance(rec, dict):
                logger.warning("LexicalIndex.from_records: skipping non-dict record %r", type(rec))
                continue

            cid = rec.get("id")
            if not isinstance(cid, str) or not cid:
                continue

            lex = rec.get("lexical_helpers") or {}
            toks: List[str] = []
            toks.extend(lex.get("ngrams_1") or [])
            # light boost to bigrams by duplicating once (stable & deterministic)
            bigrams = lex.get("ngrams_2") or []
            toks.extend(bigrams)
            toks.extend(bigrams)  # duplicate to act as a 2x weight

            dl = 0
            seen_in_doc = set()
            for t in toks:
                if not t:
                    continue
                dl += 1
                self.postings.setdefault(t, {}).setdefault(cid, _Posting(tf=0)).tf += 1
                if t not in seen_in_doc:
                    self.df[t] = self.df.get(t, 0) + 1
                    seen_in_doc.add(t)

            self.doc_len[cid] = dl
            self.N += 1
            valid += 1

        self.avgdl = (sum(self.doc_len.values()) / self.N) if self.N else 0.0
        logger.info("LexicalIndex built over %d valid records (N=%d docs)", valid, self.N)
        return self

    # ---------- scoring ----------

    def _idf(self, term: str) -> float:
        # BM25 idf variant; add-1 smoothing for determinism
        n = self.df.get(term, 0)
        if n <= 0 or self.N == 0:
            return 0.0
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score_doc(self, cid: str, q_terms: List[str]) -> float:
        dl = max(1, self.doc_len.get(cid, 1))
        K = self.k1 * ((1 - self.b) + self.b * (dl / (self.avgdl or 1.0)))
        score = 0.0
        for t in q_terms:
            p = self.postings.get(t, {}).get(cid)
            if not p:
                continue
            idf = self._idf(t)
            tf = p.tf
            score += idf * ((tf * (self.k1 + 1)) / (tf + K))
        return score

    # ---------- query ----------

    def search(self, query: str, *, top_k: int = 50) -> List[Dict]:
        q_terms = _tokenize_lc(query)
        # gather candidate set from union of postings
        cands = set()
        for t in q_terms:
            if t in self.postings:
                cands.update(self.postings[t].keys())

        scored: List[Tuple[str, float]] = []
        for cid in cands:
            s = self._score_doc(cid, q_terms)
            if s > 0.0:
                scored.append((cid, s))

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        out: List[Dict] = []
        for i, (cid, s) in enumerate(scored[:top_k], start=1):
            out.append({"chunk_id": cid, "score": float(s), "rank": i})
        return out


# ---------------------------
# Path-keyed cache
# ---------------------------

# Module-level cache: (abs_path, mtime_ns, size_bytes, schema_version) -> LexicalIndex.
# Bounded size with simple FIFO eviction; queries are read-only so this is
# safe across the FastAPI app or any in-process re-use. ``schema_version`` is
# included in the key so a tokenizer/record-shape bump (see
# ``cgx.embeddings.records.SCHEMA_VERSION``) automatically invalidates any
# previously cached index even if the records.jsonl file path is reused.
_LEX_CACHE: "Dict[Tuple[str, int, int, int], LexicalIndex]" = {}
_LEX_CACHE_MAX = 4


def _records_schema_version(records: List[Dict]) -> int:
    for r in records:
        if isinstance(r, dict):
            v = r.get("schema_version")
            if isinstance(v, int):
                return v
    return 0


def _cache_key(path: str, schema_version: int) -> Optional[Tuple[str, int, int, int]]:
    try:
        ap = os.path.abspath(path)
        st = os.stat(ap)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        return (ap, mtime_ns, int(st.st_size), int(schema_version))
    except Exception:
        return None


def get_cached_lexical_index(records_path: str, records: List[Dict]) -> LexicalIndex:
    """
    Return a LexicalIndex for the given records, reusing a cached instance
    when the underlying file (path + mtime + size + schema_version) is unchanged.

    Falls back to a fresh build (without caching) when the path is invalid.
    """
    sv = _records_schema_version(records)
    key = _cache_key(records_path, sv) if records_path else None
    if key is not None:
        idx = _LEX_CACHE.get(key)
        if idx is not None:
            return idx
    idx = LexicalIndex.from_records(records)
    if key is not None:
        if len(_LEX_CACHE) >= _LEX_CACHE_MAX:
            try:
                _LEX_CACHE.pop(next(iter(_LEX_CACHE)))
            except StopIteration:
                pass
        _LEX_CACHE[key] = idx
    return idx
