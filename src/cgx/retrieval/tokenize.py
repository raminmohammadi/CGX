

from __future__ import annotations

"""
Symmetric sub-word tokenizer for identifiers.

Used by BOTH the indexer (embeddings.helpers._split_tokens, which builds the
lexical_helpers.ngrams_* posting tokens) and the querier (retrieval.lexical.
_tokenize_lc, orchestrator._extract_symbol_tokens). Keeping the two sides
identical is what makes a query like "reconnect" actually hit a function
named ``databaseReconnect`` -- otherwise BM25 sees ``databasereconnect`` on
the index side and ``reconnect`` on the query side and yields zero overlap.

Splitting rules (in order, all case-insensitive at output):
  1. Run extraction: pull contiguous ``[A-Za-z0-9_]+`` runs out of raw text.
  2. Underscore split: ``parse_input_args`` -> ``parse``, ``input``, ``args``.
  3. Camel/Pascal split:
        ``fooBar``          -> ``foo``, ``Bar``
        ``HTTPSConnection`` -> ``HTTPS``, ``Connection``  (acronym then word)
  4. Lower-case the resulting pieces; drop empty strings.

Tokens are NEVER deduplicated by ``split_identifier`` itself (callers may
want to count occurrences); ``expand_with_subwords`` is the dedup helper.
"""

import re
from typing import Iterable, List

# Contiguous identifier-like runs. Punctuation between runs is treated as
# a separator (matches the previous behaviour of ``_split_tokens``).
_WORD_RUN = re.compile(r"[A-Za-z0-9_]+")

# Fixed-width camel/Pascal boundaries. Both lookbehinds are 1 char wide so
# Python's stdlib ``re`` accepts them (variable-width lookbehind is unsupported).
#   (?<=[a-z0-9])(?=[A-Z])      -> "fooBar"   -> "foo|Bar"
#   (?<=[A-Z])(?=[A-Z][a-z])    -> "HTTPSConn" -> "HTTPS|Conn"
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(name: str) -> List[str]:
    """Split a raw identifier-like token into lower-cased sub-words.

    Empty input returns ``[]``. Non-string input is coerced via ``str``.
    Single-letter sub-words ARE preserved here so callers can apply their
    own minimum-length filter (the indexer keeps single chars; the query
    side filters via ``min_len``).
    """
    if not name:
        return []
    s = str(name)
    out: List[str] = []
    for run in _WORD_RUN.findall(s):
        for piece in run.split("_"):
            if not piece:
                continue
            for sub in _CAMEL_SPLIT.split(piece):
                if sub:
                    out.append(sub.lower())
    return out


def expand_with_subwords(tokens: Iterable[str], *, min_len: int = 1) -> List[str]:
    """Return ``tokens`` + their sub-word splits, deduped, order-stable.

    The original (lower-cased) token is emitted first, followed by its
    sub-word splits in left-to-right order. The first occurrence wins,
    later duplicates are dropped. Sub-words shorter than ``min_len`` are
    skipped, but the *original* token is always kept once -- the caller
    already decided it was worth including.
    """
    seen: set = set()
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        lt = str(t).lower()
        if lt and lt not in seen:
            seen.add(lt)
            out.append(lt)
        for sub in split_identifier(t):
            if len(sub) < min_len:
                continue
            if sub not in seen:
                seen.add(sub)
                out.append(sub)
    return out


def tokenize_text(text: str, *, min_len: int = 1) -> List[str]:
    """One-shot helper: split free-form text into sub-word-aware tokens.

    Equivalent to ``expand_with_subwords(_WORD_RUN.findall(text), ...)`` but
    without an intermediate Python list when the caller only needs the final
    flat token stream (the common case for BM25 / lexical-helpers builders).
    """
    if not text:
        return []
    return expand_with_subwords(_WORD_RUN.findall(str(text)), min_len=min_len)
