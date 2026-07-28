"""Markdown / reStructuredText parser implementing the :class:`BaseParser` seam.

Standalone documentation (``README.md``, ``docs/*.md``, design notes) carries
the *intent* and rationale that source code alone omits. This parser makes that
prose first-class in the index: it splits a doc file on its headings and emits
one chunk per section, mirroring the ``(chunks, call_relations)`` contract the
project walker aggregates. Docs have no call graph, so ``call_relations`` is
always empty.

Every emitted chunk is stamped ``meta['source_kind'] = 'doc'`` so downstream
retrieval/answer layers can distinguish (and, if desired, down-weight) prose
against code. Unlike the tree-sitter parsers this needs no optional dependency,
so it is always registered.

Heading detection covers ATX (``# Title``) and setext (a text line underlined
with ``===`` or ``---``), which together cover Markdown and the common RST
title forms. Content before the first heading becomes a preamble section.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from cgx.parser.base import BaseParser

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^(=+|-+)\s*$")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "section"


def _first_paragraph(body: str) -> str:
    out: List[str] = []
    for line in body.splitlines():
        if line.strip():
            out.append(line.strip())
        elif out:
            break
    return " ".join(out)


class _Section:
    __slots__ = ("level", "title", "start", "end", "lines")

    def __init__(self, level: int, title: str, start: int) -> None:
        self.level = level
        self.title = title
        self.start = start
        self.end = start
        self.lines: List[str] = []


def _split_sections(lines: List[str], filename: str) -> List[_Section]:
    """Split ``lines`` into heading-delimited sections (1-based line numbers)."""
    sections: List[_Section] = []
    current = _Section(0, filename, 1)
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        atx = _ATX_RE.match(raw)
        # setext: a non-blank text line directly followed by an underline rule
        setext_level: Optional[int] = None
        if not atx and raw.strip() and i + 1 < n and _SETEXT_RE.match(lines[i + 1]):
            setext_level = 1 if lines[i + 1].lstrip().startswith("=") else 2
        if atx or setext_level is not None:
            current.end = i  # previous section ends on the line before this heading
            sections.append(current)
            if atx:
                level, title = len(atx.group(1)), atx.group(2).strip()
                current = _Section(level, title or "section", i + 1)
                current.lines.append(raw)
            else:
                title = raw.strip()
                current = _Section(int(setext_level), title, i + 1)
                current.lines.append(raw)
                current.lines.append(lines[i + 1])
                i += 1
        else:
            current.lines.append(raw)
        i += 1
    current.end = n
    sections.append(current)
    return sections


class MarkdownParser(BaseParser):
    """Parser for Markdown / RST docs; chunks by heading, no call relations."""

    extensions: Tuple[str, ...] = (".md", ".markdown", ".mdx", ".rst")

    def parse_file(
        self,
        filepath: str,
        source_code: str,
        project_root: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not source_code or not source_code.strip():
            return [], []

        lines = source_code.splitlines()
        basename = os.path.basename(filepath)
        sections = [s for s in _split_sections(lines, basename)
                    if any(ln.strip() for ln in s.lines)]
        if not sections:
            return [], []

        chunks: List[Dict[str, Any]] = []
        titles = [s.title for s in sections if s.level > 0] or [basename]
        preamble = next((s for s in sections if s.level == 0), None)
        intro = _first_paragraph("\n".join(preamble.lines)) if preamble else \
            _first_paragraph("\n".join(sections[0].lines))

        # File-level chunk: a first-class 'file' node for aggregation / repo map.
        chunks.append({
            "id": filepath, "type": "file", "name": basename, "file": filepath,
            "module_path": None,
            "code": "# " + basename + "\n" + "\n".join(f"- {t}" for t in titles),
            "start_line": 1, "end_line": max(1, len(lines)), "col_offset": 0,
            "meta": {"docstring": intro, "members": titles,
                     "metrics": {"n_loc": len(lines)}, "source_kind": "doc"},
        })

        # One 'doc' chunk per section.
        seen: Dict[str, int] = {}
        for sec in sections:
            body = "\n".join(sec.lines).strip()
            if not body:
                continue
            slug = _slugify(sec.title)
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = f"{slug}-{seen[slug]}"
            chunks.append({
                "id": f"{filepath}::doc::{slug}", "type": "doc",
                "name": sec.title, "file": filepath, "module_path": None,
                "code": body, "start_line": sec.start,
                "end_line": max(sec.start, sec.end), "col_offset": 0,
                "meta": {"docstring": _first_paragraph(body) or sec.title,
                         "source_kind": "doc", "heading_level": sec.level,
                         "metrics": {"n_loc": len(sec.lines)}},
            })
        return chunks, []


__all__ = ["MarkdownParser"]
