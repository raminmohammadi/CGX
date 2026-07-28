"""JavaScript / TypeScript parsers implementing the :class:`BaseParser` seam.

These are thin :class:`~cgx.parser.treesitter_base.TreeSitterParser` subclasses
that bind a file-extension set to a tree-sitter grammar. They only become
active when the optional ``tree_sitter_language_pack`` dependency is installed
(see the ``parsers`` extra); otherwise the registry skips them and the walker
falls back to ignoring those files, exactly as before multi-language support.

The JavaScript and TypeScript grammars share the same node vocabulary for the
constructs CGX indexes (functions, arrow functions, classes, methods, calls),
so the shared defaults on ``TreeSitterParser`` cover both. The ``.tsx`` variant
needs the dedicated ``tsx`` grammar to parse JSX inside TypeScript.
"""

from __future__ import annotations

from typing import Tuple

from cgx.parser.treesitter_base import TreeSitterParser


class JavaScriptParser(TreeSitterParser):
    """Parser for JavaScript sources (incl. JSX and ESM/CJS variants)."""

    language: str = "javascript"
    extensions: Tuple[str, ...] = (".js", ".jsx", ".mjs", ".cjs")


class TypeScriptParser(TreeSitterParser):
    """Parser for ``.ts`` / ``.mts`` / ``.cts`` TypeScript sources."""

    language: str = "typescript"
    extensions: Tuple[str, ...] = (".ts", ".mts", ".cts")


class TSXParser(TreeSitterParser):
    """Parser for ``.tsx`` sources (TypeScript + JSX).

    Uses the dedicated ``tsx`` grammar; the plain ``typescript`` grammar
    cannot parse JSX syntax.
    """

    language: str = "tsx"
    extensions: Tuple[str, ...] = (".tsx",)


__all__ = ["JavaScriptParser", "TypeScriptParser", "TSXParser"]
