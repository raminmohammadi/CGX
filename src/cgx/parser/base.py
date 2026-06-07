"""Abstract base for per-language parsers.

A ``BaseParser`` consumes a single source file and yields the same
``(chunks, call_relations)`` tuple that :func:`parse_codebase` aggregates
across an entire project. The project-level walker dispatches files to
parsers based on their extension (see ``cgx.parser.parse_codebase``).

Only the Python parser is registered today; this seam exists to keep the
extension point explicit without committing to multi-language work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseParser(ABC):
    """Contract for any per-file source parser.

    Subclasses declare the file extensions they handle via
    :attr:`extensions` (lowercase, leading dot, e.g. ``(".py",)``) and
    implement :meth:`parse_file`. The walker holds a single instance per
    registered parser, so implementations should be safe to reuse across
    files but must not retain per-file state between calls.
    """

    #: Lowercased file extensions handled by this parser. Each entry must
    #: include the leading dot.
    extensions: Tuple[str, ...] = ()

    @abstractmethod
    def parse_file(
        self,
        filepath: str,
        source_code: str,
        project_root: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse one source file.

        Parameters
        ----------
        filepath:
            Absolute (or repo-relative, as the walker passes it) path to
            the source file. Used verbatim in chunk ``id`` and ``file``
            fields.
        source_code:
            Full text of the file, already decoded as UTF-8 with errors
            ignored by the walker.
        project_root:
            Absolute project root used to compute deterministic
            ``module_path`` values via
            :func:`cgx.parser.module_path.compute_module_path`.

        Returns
        -------
        tuple
            ``(chunks, call_relations)`` shaped like the per-file slice
            of :func:`cgx.parser.parse_codebase.parse_codebase`'s output.
            Implementations must return empty lists on unrecoverable
            parse errors rather than raising; the walker logs and skips.
        """
        raise NotImplementedError


__all__ = ["BaseParser"]
