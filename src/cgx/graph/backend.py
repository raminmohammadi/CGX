"""CodeGraphBackend -- thin typed facade over the nx ops used at query time.

Retrieval (``cgx.retrieval.orchestrator``) and record assembly
(``cgx.embeddings.helpers``) only consume a small, stable subset of the
NetworkX MultiDiGraph API:

  * node membership (``nid in G``),
  * forward/backward adjacency (``successors`` / ``predecessors``),
  * edge attribute lookup (with the multi-edge dict-of-keyed-dicts shape
    that ``MultiDiGraph`` exposes via ``G[u][v]``),
  * node attribute lookup (``G.nodes[nid]``),
  * bounded BFS distances (``single_source_shortest_path_length``),
  * an undirected one-hop neighbor list.

This module wraps that subset behind ``CodeGraphBackend`` so call sites
have a single, typed surface to depend on. The wrapper holds the raw
graph by reference -- there is no copy and no new dependency. Graph
construction, visualization, and persistence keep using ``networkx``
directly, which is why ``cgx.graph.build_graph`` is untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # callers guard via wrap(None) -> None


def _flatten_edge_attrs(ed: Any) -> Dict[str, Any]:
    """Normalize ``G[u][v]`` into a single attribute dict.

    A ``MultiDiGraph`` stores edges as ``{key: {attr: val, ...}, ...}``;
    a plain ``DiGraph`` stores them as ``{attr: val, ...}``. The outer
    container is often an ``AtlasView``, not a plain ``dict``, so we
    duck-type via ``.values()``. The multi shape is detected by checking
    whether *all* values themselves expose ``.get``/``.values`` (i.e. are
    nested attribute dicts). ``{}`` is returned for missing / opaque
    shapes.
    """
    if ed is None:
        return {}
    try:
        values = list(ed.values())
    except AttributeError:
        return {}
    if not values:
        return {}
    # MultiDiGraph multi-edge shape: every value is itself an attribute mapping.
    if all(hasattr(v, "get") and hasattr(v, "values") for v in values):
        first = values[0]
        try:
            return dict(first)
        except Exception:
            return {}
    # Plain DiGraph: ed is the attribute mapping itself.
    try:
        return dict(ed)
    except Exception:
        return {}


class CodeGraphBackend:
    """Thin facade over the small set of nx ops used by retrieval + embeddings."""

    __slots__ = ("_G",)

    def __init__(self, G: Any) -> None:
        self._G = G

    @classmethod
    def wrap(cls, G: Any) -> "Optional[CodeGraphBackend]":
        """Return a backend around ``G`` (or ``None`` when ``G`` is None).

        Idempotent: passing an existing ``CodeGraphBackend`` returns it
        unchanged. This is the canonical entry point -- every caller that
        accepts ``G: Any`` should funnel through it before reading.
        """
        if G is None:
            return None
        if isinstance(G, cls):
            return G
        return cls(G)

    @property
    def raw(self) -> Any:
        """Escape hatch: the underlying nx graph (read-only by convention)."""
        return self._G

    def has_node(self, nid: str) -> bool:
        try:
            return nid in self._G
        except Exception:
            return False

    def successors(self, nid: str) -> List[str]:
        try:
            return list(self._G.successors(nid))
        except Exception:
            return []

    def predecessors(self, nid: str) -> List[str]:
        try:
            return list(self._G.predecessors(nid))
        except Exception:
            return []

    def undirected_neighbors(self, nid: str) -> List[str]:
        """Forward + backward one-hop neighborhood, deduplicated by order."""
        seen: Dict[str, None] = {}
        for n in self.successors(nid):
            seen.setdefault(n, None)
        for n in self.predecessors(nid):
            seen.setdefault(n, None)
        return list(seen.keys())

    def edge_attrs(self, u: str, v: str) -> Dict[str, Any]:
        try:
            return _flatten_edge_attrs(self._G[u][v])
        except Exception:
            return {}

    def node_attrs(self, nid: str) -> Dict[str, Any]:
        try:
            attrs = self._G.nodes[nid]
            return dict(attrs) if attrs is not None else {}
        except Exception:
            return {}

    def bfs_distances(self, source: str, *, cutoff: int) -> Dict[str, int]:
        """Single-source BFS distances out to ``cutoff`` hops.

        Uses :func:`nx.single_source_shortest_path_length` when available
        (the canonical fast path) and falls back to a pure-Python BFS for
        environments where networkx is missing.
        """
        if not self.has_node(source):
            return {}
        if nx is not None:
            try:
                return dict(nx.single_source_shortest_path_length(
                    self._G, source, cutoff=int(cutoff)
                ))
            except Exception:
                pass
        out: Dict[str, int] = {source: 0}
        frontier: List = [(source, 0)]
        while frontier:
            nid, d = frontier.pop(0)
            if d >= int(cutoff):
                continue
            for nb in self.successors(nid):
                if nb not in out:
                    out[nb] = d + 1
                    frontier.append((nb, d + 1))
        return out


__all__ = ["CodeGraphBackend", "_flatten_edge_attrs"]
