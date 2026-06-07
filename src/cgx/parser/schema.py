"""Type definitions pinning the shape of parser output records.

These TypedDicts document -- but do not enforce at runtime -- the chunk and
call-relation dicts emitted by :func:`cgx.parser.parse_codebase.parse_codebase`
and consumed by the embeddings/retrieval/codegen layers.

They exist so that:

* alternative ``BaseParser`` implementations have a single source of truth
  for the keys they must populate;
* downstream readers can opt into static type checking by annotating
  variables with these aliases without introducing a runtime dependency;
* schema changes (governed by ``cgx.embeddings.records.SCHEMA_VERSION``)
  remain reviewable in one place.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 fallback, unused in practice
    from typing_extensions import TypedDict  # type: ignore[assignment]


ChunkType = Literal["file", "class", "function", "method", "lambda"]
"""Discriminator for the ``type`` field of a code chunk."""


class CodeChunk(TypedDict, total=False):
    """A single parsed entity (file / class / function / method / lambda).

    The keys mirror the dict-literals emitted in
    :mod:`cgx.parser.parse_codebase`. ``total=False`` accepts the fact that
    ``meta`` schemas vary per chunk type; the always-populated identity
    fields are the first six keys (``id`` through ``code``) plus the line
    triple introduced in schema v3.
    """

    id: str
    type: ChunkType
    name: str
    file: str
    module_path: str
    code: str
    start_line: int
    end_line: int
    col_offset: int
    meta: Dict[str, Any]


class CallRelation(TypedDict, total=False):
    """A single resolved or unresolved call site emitted by the parser."""

    caller_id: str
    callee_name: Optional[str]
    callee_fullname: Optional[str]
    lineno: Optional[int]


# Result tuple of any BaseParser.parse_file()/parse_codebase() call.
ParseResult = "tuple[List[CodeChunk], List[CallRelation]]"


__all__ = [
    "CodeChunk",
    "CallRelation",
    "ChunkType",
    "ParseResult",
]
