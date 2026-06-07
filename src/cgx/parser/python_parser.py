"""Python AST parser implementing the :class:`BaseParser` seam.

The actual AST walking logic lives in
:func:`cgx.parser.parse_codebase._parse_python_module` so the seam stays
shallow and the in-repo implementation has one canonical location. This
module simply adapts that helper to the ``BaseParser`` contract so the
project walker can dispatch on file extension.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from cgx.parser.base import BaseParser


class PythonASTParser(BaseParser):
    """Parser for ``.py`` files using the standard-library :mod:`ast` module.

    The class is intentionally minimal: it owns the extension registration
    and forwards each file to the closure-based implementation in
    :mod:`cgx.parser.parse_codebase`. That keeps the historical record
    shapes (which are validated by snapshot tests) byte-identical with
    no behavioral drift.
    """

    extensions: Tuple[str, ...] = (".py",)

    def parse_file(
        self,
        filepath: str,
        source_code: str,
        project_root: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Imported lazily to break the import cycle: parse_codebase
        # builds the parser registry at module import time.
        from cgx.parser.parse_codebase import _parse_python_module

        return _parse_python_module(filepath, source_code, project_root)


__all__ = ["PythonASTParser"]
