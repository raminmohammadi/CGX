

import os
import re
import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import matplotlib.pyplot as plt

from cgx.graph.aggregation import project_graph_for_visualization


# ---------------------------
# Small utilities
# ---------------------------
def _get_attr(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get nested attributes using dot-paths, e.g. "meta.metrics.n_calls".
    """
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _satisfies(val: Any, op: str, rhs: Any) -> bool:
    """
    Simple comparison helpers for where-clauses.
    Supported ops: ==, !=, >, >=, <, <=, in, contains, startswith, endswith, regex
    """
    try:
        if op == "==":          return val == rhs
        if op == "!=":          return val != rhs
        if op == ">":           return val > rhs
        if op == ">=":          return val >= rhs
        if op == "<":           return val < rhs
        if op == "<=":          return val <= rhs
        if op == "in":          return val in rhs
        if op == "contains":    return (rhs in val) if isinstance(val, (str, list, tuple, set)) else False
        if op == "startswith":  return str(val).startswith(str(rhs))
        if op == "endswith":    return str(val).endswith(str(rhs))
        if op == "regex":       return re.search(rhs, str(val)) is not None
    except Exception:
        return False
    return False


def _passes_where(data: Dict[str, Any], where: Optional[Sequence[Tuple[str, str, Any]]]) -> bool:
    """
    where: list of (attr_path, op, value)
      Example: [("type","in",{"function","method"}), ("meta.metrics.n_calls",">",0)]
    """
    if not where:
        return True
    for path, op, rhs in where:
        if not _satisfies(_get_attr(data, path), op, rhs):
            return False
    return True

def _normalize(values, lo, hi, clamp_zero=False):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return [(lo+hi)/2 for _ in values]

    mn, mx = min(filtered), max(filtered)
    if math.isclose(mn, mx):
        mid = (lo + hi) / 2.0
        return [mid for _ in values]

    out = []
    for v in values:
        if v is None:
            out.append((lo+hi)/2)   # fallback if missing
        else:
            if clamp_zero and v < 0:
                v = 0
            norm = (v - mn) / (mx - mn)
            out.append(lo + norm * (hi - lo))
    return out



def _label_from_template(node_id: str, data: Dict[str, Any], template: str, truncate: Optional[int]) -> str:
    """
    Build a label from a template string, exposing top-level fields and meta.* as flattened keys.
    Supported keys (examples):
      {id}, {name}, {type}, {file}, {signature}, {class_name}, {docstring}, {metrics_n_calls}, ...
    Any missing key becomes empty.
    """
    # flatten a small, safe view
    flat = {
        "id": node_id,
        "name": data.get("name", ""),
        "type": data.get("type", ""),
        "file": data.get("file", ""),
        "signature": _get_attr(data, "signature", ""),
        "class_name": _get_attr(data, "class_name", ""),
        "docstring": _get_attr(data, "docstring", ""),
        "returns_annotation": _get_attr(data, "returns_annotation", ""),
        "method_kind": _get_attr(data, "method_kind", ""),
    }
    # include common metrics if present
    for k in ("n_loc", "n_params", "n_returns", "n_yields", "n_branches", "n_calls"):
        flat[f"metrics_{k}"] = _get_attr(data, f"metrics.{k}", "")

    # Best-effort format
    try:
        s = template.format(**flat)
    except KeyError:
        # Fallback: replace unknown fields with empty
        s = re.sub(r"\{[^}]+\}", "", template)

    if truncate and len(s) > truncate:
        s = s[: truncate - 1] + "…"
    return s or node_id


# ---------------------------
# Main visualizer
# ---------------------------
def visualize_subgraph(
    G: nx.DiGraph,
    center_node: Optional[str] = None,
    depth: int = 1,
    relation_types: Optional[Iterable[str]] = None,
    *,
    # Node/edge filtering by metadata (applied after hop expansion, center node always kept)
    node_where: Optional[Sequence[Tuple[str, str, Any]]] = None,
    edge_where: Optional[Sequence[Tuple[str, str, Any]]] = None,
    # Visual encodings
    label_template: str = "{name}\n{signature}",
    label_truncate: Optional[int] = 60,
    show_edge_labels: bool = True,
    edge_label_key: str = "type",
    size_by: Optional[str] = None,                 # e.g., "meta.metrics.n_loc" or "metrics.n_loc" (promoted)
    size_range: Tuple[float, float] = (800, 3000), # node size range
    width_by: Optional[str] = None,                # e.g., "meta.metrics.n_calls"
    width_range: Tuple[float, float] = (0.8, 3.5), # edge width range
    layout: str = "spring",                        # "spring" | "kamada_kawai" | "shell" | "circular"
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 9),
    alpha: float = 0.9,
    arrows: bool = True,
):
    """
    Visualize a metadata-aware subgraph around a specified node in a directed graph.

    This function takes a graph `G` (typically built from parsed source code or 
    dependency information) and generates a visualization of the neighborhood 
    surrounding a `center_node`, allowing the user to explore classes, functions, 
    methods, attributes, and their relationships.

    Parameters
    ----------
    G : nx.DiGraph
        The input directed graph containing nodes and edges with metadata.
        Each node should have attributes like `type`, `name`, and optionally 
        nested `meta` fields (e.g., `metrics.n_loc`). Each edge should have a `type`.

    center_node : str, optional
        The identifier of the node to visualize the subgraph around. 
        Must match an existing node key in `G`. If `None` or not found, 
        the function prints a warning and returns without plotting.

    depth : int, default=1
        The BFS expansion depth from `center_node`. Larger values include 
        nodes further away in the call/definition graph.

    relation_types : Iterable[str], optional
        If provided, restricts traversal and visualization to only edges 
        whose `type` attribute is in this list. Example values:
        - `"defines"`: class defines method
        - `"calls"`: function calls another function
        - `"reads_attr"` / `"writes_attr"`: attribute accesses

    node_where : list of (attr_path, op, value), optional
        Post-filter applied to nodes after subgraph expansion. Keeps the 
        `center_node` regardless. Each tuple defines a filter rule:
        - `attr_path` : dot path into node attributes (e.g. `"type"`, `"meta.metrics.n_loc"`)
        - `op`        : comparison operator, e.g. `"=="`, `"!="`, `"in"`, `"not in"`, `">"`, `"<"`
        - `value`     : target value or set
        Example: `[("type","in",{"function","method"})]` keeps only functions/methods.

    edge_where : list of (attr_path, op, value), optional
        Filter applied to edges. Same format as `node_where` but applied to edge attributes.

    label_template : str, default="{name}\n{signature}"
        Format string for node labels. Can use node fields like:
        - `{name}`: function or class name
        - `{signature}`: function signature if available
        - `{type}`: node type ("class", "function", etc.)
        - `{metrics_n_calls}`: flattened numeric metric fields
        Supports truncation with `label_truncate`.

    label_truncate : int, optional, default=60
        Maximum number of characters in labels. Longer labels are truncated.

    show_edge_labels : bool, default=True
        Whether to draw edge labels (e.g., edge type).

    edge_label_key : str, default="type"
        The edge attribute key to use for labeling edges. Typical: `"type"`.

    size_by : str, optional
        Node attribute path (supports `"meta."` prefixes) to scale node sizes.
        Example: `"metrics.n_loc"` or `"meta.metrics.n_loc"`.
        If missing or non-numeric, a default node size is used.

    size_range : Tuple[float, float], default=(800, 3000)
        Range of node sizes (min, max). Used if `size_by` is given.

    width_by : str, optional
        Edge attribute path used to scale edge widths (numeric).

    width_range : Tuple[float, float], default=(0.8, 3.5)
        Range of edge widths (min, max). Used if `width_by` is given.

    layout : str, default="spring"
        Layout algorithm for graph visualization. Options:
        - `"spring"`: force-directed spring layout
        - `"kamada_kawai"`: Kamada-Kawai layout
        - `"shell"`: circular shell layout
        - `"circular"`: circular layout

    seed : int, default=42
        Random seed for deterministic layout.

    figsize : Tuple[int, int], default=(12, 9)
        Matplotlib figure size (width, height).

    alpha : float, default=0.9
        Transparency level of nodes (0=fully transparent, 1=opaque).

    arrows : bool, default=True
        Whether to draw arrows on directed edges.

    Behavior
    --------
    - Expands subgraph around `center_node` up to given `depth`.
    - Applies optional edge and node filters.
    - Visual encodings: node color by type, node size by attribute, 
      edge width by attribute, labels by template.
    - Ambiguous edges (marked with `ambiguous=True`) are drawn dashed.
    - If no nodes remain after filtering, prints a message and returns.

    Returns
    -------
    None
        This function does not return a value. It produces a Matplotlib plot 
        of the subgraph.

    Examples
    --------
    Basic visualization:
    >>> visualize_subgraph(G, center_node="path/to/file.py::function::foo")

    With relation filters and size scaling:
    >>> visualize_subgraph(
    ...     G,
    ...     center_node="path/to/file.py::class::MyClass",
    ...     depth=2,
    ...     relation_types=["defines", "calls"],
    ...     size_by="metrics.n_loc",
    ...     width_by="lineno",
    ...     label_template="{name}\n{signature}",
    ...     layout="kamada_kawai"
    ... )

    Node filtering (only show functions and methods):
    >>> visualize_subgraph(
    ...     G,
    ...     center_node="path/to/file.py::method::MyClass.my_method",
    ...     node_where=[("type","in",{"function","method"})]
    ... )

    Edge filtering (exclude "uses_module" edges):
    >>> visualize_subgraph(
    ...     G,
    ...     center_node="path/to/file.py::function::helper",
    ...     edge_where=[("type","!=","uses_module")]
    ... )
    """

    if center_node is None or center_node not in G:
        print("Node not found in graph.")
        return

    if isinstance(G, nx.MultiDiGraph):
        G = project_graph_for_visualization(G)  # aggregate to a simple DiGraph

    # NEW: allow a single string like "calls"
    if isinstance(relation_types, str):
        relation_types = {relation_types}
        
    # ---------------------------
    # 1) BFS to collect nodes within depth, honoring relation_types/edge_where
    # ---------------------------
    nodes_kept = {center_node}
    q = deque([(center_node, 0)])
    while q:
        nid, d = q.popleft()
        if d >= depth:
            continue

        # Outgoing
        for succ in G.successors(nid):
            edata = G[nid][succ]
            if relation_types and edata.get("type") not in relation_types:
                continue
            if edge_where and not _passes_where(edata, edge_where):
                continue
            if succ not in nodes_kept:
                nodes_kept.add(succ)
                q.append((succ, d + 1))

        # Incoming
        for pred in G.predecessors(nid):
            edata = G[pred][nid]
            if relation_types and edata.get("type") not in relation_types:
                continue
            if edge_where and not _passes_where(edata, edge_where):
                continue
            if pred not in nodes_kept:
                nodes_kept.add(pred)
                q.append((pred, d + 1))

    subG = G.subgraph(nodes_kept).copy()

    # ---------------------------
    # 2) Apply node_where post-filter (keep center always)
    # ---------------------------
    if node_where:
        for n in list(subG.nodes()):
            if n == center_node:
                continue
            if not _passes_where(subG.nodes[n], node_where):
                subG.remove_node(n)

    if subG.number_of_nodes() == 0:
        print("No nodes to visualize after filtering.")
        return

    # ---------------------------
    # 3) Visual encodings
    # ---------------------------
    # Node colors by type (extend as needed)
    palette = {
        "file":        "#9ecae1",
        "class":       "#a1d99b",
        "function":    "#fff7bc",
        "method":      "#fdd0a2",
        "lambda":      "#e5f5e0",
        "module":      "#c6dbef",
        "attribute":   "#dadaeb",
        "exception":   "#fcbba1",
        "unresolved":  "#fdae6b",
        None:          "#dddddd",
    }

    node_colors, node_sizes, node_labels = [], [], {}
    # Precompute sizes if required
    size_values = []
    if size_by:
        for n, data in subG.nodes(data=True):
            # Try top-level, then meta.*
            v = _get_attr(data, size_by.replace("meta.", "")) if size_by.startswith("meta.") else _get_attr(data, size_by)
            if v is None:
                v = _get_attr(data, f"meta.{size_by}")  # fallback if user omitted `meta.`
            try:
                size_values.append(float(v) if v is not None else None)
            except Exception:
                size_values.append(None)
        node_sizes = _normalize(size_values, size_range[0], size_range[1])
    else:
        node_sizes = [1600 for _ in subG.nodes()]

    # Labels
    for i, (n, data) in enumerate(subG.nodes(data=True)):
        t = data.get("type")
        node_colors.append(palette.get(t, palette[None]))
        node_labels[n] = _label_from_template(n, data, label_template, label_truncate)

    # Edge widths
    edge_widths = []
    if width_by:
        wvals = []
        for u, v, ed in subG.edges(data=True):
            try:
                wvals.append(float(_get_attr(ed, width_by)))
            except Exception:
                wvals.append(None)
        edge_widths = _normalize(wvals, width_range[0], width_range[1], clamp_zero=True)
    else:
        edge_widths = [1.6 for _ in subG.edges()]

    # Edge styles (e.g., dashed for ambiguous call resolution)
    ambiguous_edges = [(u, v) for u, v, ed in subG.edges(data=True) if ed.get("ambiguous")]
    normal_edges     = [(u, v) for u, v, ed in subG.edges(data=True) if not ed.get("ambiguous")]

    # Edge labels
    edge_labels = {}
    if show_edge_labels and edge_label_key:
        for u, v, ed in subG.edges(data=True):
            edge_labels[(u, v)] = str(ed.get(edge_label_key, ""))

    # ---------------------------
    # 4) Layout & draw
    # ---------------------------
    if layout == "spring":
        pos = nx.spring_layout(subG, seed=seed, k=None)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subG)
    elif layout == "shell":
        pos = nx.shell_layout(subG)
    elif layout == "circular":
        pos = nx.circular_layout(subG)
    else:
        pos = nx.spring_layout(subG, seed=seed)

    plt.figure(figsize=figsize)

    # Draw normal edges
    nx.draw_networkx_edges(
        subG, pos,
        edgelist=normal_edges if normal_edges else subG.edges(),
        width=[edge_widths[list(subG.edges()).index(e)] for e in (normal_edges if normal_edges else subG.edges())],
        alpha=0.8,
        arrows=arrows,
    )
    # Draw ambiguous edges dashed
    if ambiguous_edges:
        nx.draw_networkx_edges(
            subG, pos, edgelist=ambiguous_edges,
            style="dashed", width=[edge_widths[list(subG.edges()).index(e)] for e in ambiguous_edges],
            alpha=0.9, arrows=arrows,
        )

    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=alpha,
        linewidths=0.8,
        edgecolors="#444444",
    )
    nx.draw_networkx_labels(subG, pos, labels=node_labels, font_size=8)

    if show_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=7)

    title_rel = f" types={sorted(set(d.get('type') for _,_,d in subG.edges(data=True)))}" if subG.number_of_edges() else ""
    plt.title(f"Subgraph around\n{center_node}\n(depth={depth}){title_rel}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    
def query_graph(G: nx.Graph, query: str) -> Optional[List[str]]:
    """
    Query and visualize a repository knowledge graph.

    This function interprets a small set of natural-language patterns and executes the
    corresponding NetworkX traversals and (optionally) a visualization via `visualize_subgraph`.
    It is SAFE to call with either a `nx.DiGraph` or a `nx.MultiDiGraph`. When given a
    MultiDiGraph, it attempts to aggregate parallel edges to a DiGraph using
    `project_graph_for_visualization`. If that helper is not available, a minimal
    internal fallback aggregation is used.

    Parameters
    ----------
    G : nx.Graph
        The repository knowledge graph. May be a DiGraph or MultiDiGraph produced by
        `build_knowledge_graph`.
    query : str
        Natural-language query. Case-insensitive. Supports optional flags.

    Supported Query Patterns
    ------------------------
    Calls / Callees:
      - "show all functions called by <FUNC>"
      - "show callees of <FUNC>"
      - "calls of <FUNC>"

    Callers:
      - "what calls <FUNC>"
      - "who calls <FUNC>"
      - "callers of <FUNC>"

    Class methods:
      - "visualize methods of <CLASS>"
      - "show methods of <CLASS>"

    Modules (imports):
      - "show modules used by <FUNC|CLASS>"
      - "modules of <FUNC>"
      - "imports of <FUNC>"

    Attributes:
      - "show attributes read by <FUNC|METHOD>"
      - "show attributes written by <FUNC|METHOD>"

    Exceptions:
      - "what does <FUNC> raise"
      - "raises of <FUNC>"

    Generic neighbors:
      - "neighbors of <NODE>"
      - "visualize neighbors of <NODE>"

    Optional Flags (can appear anywhere)
    ------------------------------------
      - depth <N>      or  depth=<N>        (default: 1)
      - internal       (keep edges with internal == True)
      - external       (keep edges with internal == False)

    Returns
    -------
    list[str] | None
        A list of matching neighbor node IDs for recognized queries. Returns None if
        the query cannot be resolved or an error occurs (errors are printed and suppressed).

    Error Handling
    --------------
    - Validates input types; prints an informative message and returns None on misuse.
    - If graph projection/aggregation fails, falls back to a minimal in-function aggregator.
    - Visualization errors are caught and reported without crashing the process.

    Examples
    --------
    >>> query_graph(G, "show modules used by parse convert_full_file")
    >>> query_graph(G, "what calls load_config depth 2")
    >>> query_graph(G, "visualize methods of MyClass")
    """
    import re

    # ---------- Basic validation ----------
    if not isinstance(query, str):
        print("query_graph: 'query' must be a string.")
        return None
    if not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        print("query_graph: 'G' must be a networkx DiGraph or MultiDiGraph.")
        return None

    # ---------- Normalize & extract flags ----------
    raw = query.strip()
    q = raw.lower()

    # depth flag
    depth = 1
    try:
        m_depth = re.search(r"\bdepth\s*=?\s*(\d+)\b", q)
        if m_depth:
            depth = int(m_depth.group(1))
            q = q[:m_depth.start()] + q[m_depth.end():]
    except Exception as e:
        print(f"query_graph: could not parse depth; defaulting to 1 ({e}).")
        depth = 1

    # internal/external flags (mutually exclusive; 'internal' wins if both present)
    internal_only = bool(re.search(r"\binternal(?:\s+only)?\b", q))
    external_only = (not internal_only) and bool(re.search(r"\bexternal(?:\s+only)?\b", q))
    q = re.sub(r"\b(internal(?:\s+only)?|external(?:\s+only)?)\b", "", q).strip()

    # ---------- Ensure DiGraph view for filtering/visuals ----------
    def _fallback_aggregate(M: nx.MultiDiGraph) -> nx.DiGraph:
        """Minimal in-place aggregator used only if projection helper is unavailable."""
        S = nx.DiGraph()
        S.add_nodes_from(M.nodes(data=True))
        for u, v in M.edges():
            edicts = list(M[u][v].values())
            # Compute minimal rollups we rely on in visuals/filters
            types = [ed.get("type") for ed in edicts]
            call_sites_count = sum(1 for t in types if t == "calls")
            ambiguous = any(ed.get("ambiguous") for ed in edicts)
            # Aggregate 'internal' by callee node type (project entity => internal True)
            callee_type = M.nodes[v].get("type")
            agg_internal = callee_type in {"function", "method", "lambda"}
            label_compact = f"calls ×{call_sites_count}" if call_sites_count else ",".join(sorted({t for t in types if t}))
            attrs = {
                "type": "calls" if "calls" in types else (sorted({t for t in types if t})[0] if any(types) else ""),
                "call_sites_count": call_sites_count,
                "ambiguous": ambiguous,
                "internal": agg_internal,
                "label_compact": label_compact,
            }
            if S.has_edge(u, v):
                # merge if needed
                prev = S[u][v]
                attrs["call_sites_count"] += prev.get("call_sites_count", 0)
                attrs["ambiguous"] = prev.get("ambiguous", False) or attrs["ambiguous"]
                attrs["internal"] = prev.get("internal", False) or attrs["internal"]
                if prev.get("type") == "calls":
                    attrs["type"] = "calls"
            S.add_edge(u, v, **attrs)
        return S

    Gq: nx.DiGraph
    try:
        if isinstance(G, nx.MultiDiGraph):
            Gq = project_graph_for_visualization(G)
        else:
            Gq = G.copy()
    except Exception as e:
        print(f"query_graph: graph projection failed ({e}); using fallback aggregation.")
        try:
            Gq = _fallback_aggregate(G) if isinstance(G, nx.MultiDiGraph) else G.copy()
        except Exception as e2:
            print(f"query_graph: unable to create a queryable graph view ({e2}).")
            return None

    # ---------- Helpers ----------
    def _edge_pairs_out(u: str):
        try:
            for v in Gq.successors(u):
                yield v, Gq[u][v]
        except Exception:
            return

    def _edge_pairs_in(v: str):
        try:
            for u in Gq.predecessors(v):
                yield u, Gq[u][v]
        except Exception:
            return

    def _edge_passes(ed: Dict[str, Any], etype: Optional[str] = None) -> bool:
        try:
            if etype and ed.get("type") != etype:
                return False
            if internal_only and ed.get("internal") is not True:
                return False
            if external_only and ed.get("internal") is not False:
                return False
            return True
        except Exception:
            return False

    def _tokenize(name_str: str) -> List[str]:
        try:
            name_str = name_str.strip().strip('"\'')
            name_str = re.sub(r"\s+", " ", name_str)
            return [t for t in re.split(r"[ ,]+", name_str) if t]
        except Exception:
            return [name_str]

    def _resolve_node_smart(name_str: str, preferred_types: Optional[Iterable[str]] = None) -> Optional[str]:
        """
        Robust resolver for multi-token targets like "parse convert_full_file" or "MyClass.process".
        Prefers exact matches on the last token, then scores candidates by token coverage.
        Returns the best node id or None.
        """
        try:
            tokens = _tokenize(name_str)
            if not tokens:
                return None
            last = tokens[-1].lower()
            prefset = set(preferred_types) if preferred_types else None

            # exact name via helper
            if "find_nodes_by_name" in globals():
                try:
                    cands = find_nodes_by_name(Gq, last, type_in=prefset)  # type: ignore[name-defined]
                    if cands:
                        return cands[0]
                except Exception:
                    pass

            # exact on any token
            if "find_nodes_by_name" in globals():
                for tok in tokens:
                    try:
                        cands = find_nodes_by_name(Gq, tok.lower(), type_in=prefset)  # type: ignore[name-defined]
                        if cands:
                            return cands[0]
                    except Exception:
                        pass

            # dotted hint like Class.method or file.py
            dotted = next((tok for tok in tokens if "." in tok), None)
            if dotted:
                dotl = dotted.lower()
                for nid, data in Gq.nodes(data=True):
                    t = data.get("type")
                    if prefset and t not in prefset:
                        continue
                    name = str(data.get("name", "")).lower()
                    if name == dotl or dotl in name or dotl in nid.lower():
                        return nid

            # score-based ranking over all nodes
            def score_node(nid: str, data: Dict[str, Any]) -> float:
                scr = 0.0
                name = str(data.get("name", "")).lower()
                idl = nid.lower()
                t = data.get("type")
                if name == last:
                    scr += 2.0
                for tok in tokens:
                    tl = tok.lower()
                    if tl in name:
                        scr += 1.0
                    if tl in idl:
                        scr += 0.5
                if prefset and t in prefset:
                    scr += 0.5
                return scr

            pool: List[Tuple[float, str]] = []
            for nid, data in Gq.nodes(data=True):
                t = data.get("type")
                if preferred_types and t not in (prefset or set()):
                    # allow strong id match to survive filtering
                    if last in str(nid).lower():
                        pool.append((score_node(nid, data), nid))
                else:
                    pool.append((score_node(nid, data), nid))

            pool = [(s, n) for s, n in pool if s > 0]
            if not pool:
                return None

            def tie_rank(nid: str) -> Tuple[int, int]:
                t = Gq.nodes[nid].get("type")
                pri = {"function": 0, "method": 0, "lambda": 0, "class": 1, "file": 2,
                       "module": 3, "attribute": 4, "exception": 5}.get(t, 10)
                return (pri, len(str(Gq.nodes[nid].get("name", ""))))
            pool.sort(key=lambda sn: (-sn[0], *tie_rank(sn[1])))
            return pool[0][1]
        except Exception as e:
            print(f"query_graph: name resolution failed for '{name_str}' ({e}).")
            return None

    def _visualize(node: str, rel_types: Optional[Iterable[str]], node_filter: Optional[Sequence[Tuple[str, str, Any]]] = None) -> None:
        try:
            edge_where = []
            if internal_only: edge_where.append(("internal", "==", True))
            if external_only: edge_where.append(("internal", "==", False))
            visualize_subgraph(
                Gq,
                center_node=node,
                depth=depth,
                relation_types=rel_types,
                node_where=node_filter,
                edge_where=edge_where if edge_where else None,
                edge_label_key="label_compact",
                width_by="call_sites_count",
            )
        except Exception as e:
            print(f"query_graph: visualization failed ({e}).")

    # ---------- Pattern routing ----------
    try:
        # CALLEES
        m = re.match(r"^(?:show|visualize)\s+(?:all\s+)?(?:functions?\s+)?(?:called\s+by|callees\s+of|calls\s+of)\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types={"function", "method", "lambda"})
            if not node:
                print(f"No function/method found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="calls")]
            print(f"Functions called by {node}: {succs}")
            _visualize(node, rel_types=["calls"], node_filter=[("type", "in", {"function", "method", "lambda"})])
            return succs

        # CALLERS
        m = re.match(r"^(?:what|who)\s+calls\s+(.+)$", q) or re.match(r"^callers\s+of\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types={"function", "method", "lambda"})
            if not node:
                print(f"No function/method found matching '{name}'")
                return None
            preds = [p for p, ed in _edge_pairs_in(node) if _edge_passes(ed, etype="calls")]
            print(f"{node} is called by: {preds}")
            _visualize(node, rel_types=["calls"], node_filter=[("type", "in", {"function", "method", "lambda"})])
            return preds

        # CLASS METHODS
        m = re.match(r"^(?:visualize|show)\s+methods?\s+of\s+(.+)$", q)
        if m:
            classname = m.group(1).strip()
            node = _resolve_node_smart(classname, preferred_types={"class"})
            if not node:
                print(f"No class found matching '{classname}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="defines") and Gq.nodes[s].get("type") == "method"]
            print(f"Methods of {classname}: {succs}")
            _visualize(node, rel_types=["defines"], node_filter=[("type", "in", {"class", "method"})])
            return succs

        # MODULES USED (imports)
        m = re.match(r"^(?:show|visualize)\s+(?:modules?|imports)\s+(?:used\s+by|of)\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types={"function", "method", "class"})
            if not node:
                print(f"No node found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="uses_module")]
            print(f"Modules used by {node}: {succs}")
            _visualize(node, rel_types=["uses_module"], node_filter=[("type", "in", {"function", "method", "class", "module"})])
            return succs

        # ATTRIBUTES READ
        m = re.match(r"^(?:show|visualize)\s+attributes?\s+read\s+by\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types={"method", "function"})
            if not node:
                print(f"No function/method found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="reads_attr")]
            print(f"Attributes read by {node}: {succs}")
            _visualize(node, rel_types=["reads_attr"], node_filter=[("type", "in", {"method", "function", "attribute"})])
            return succs

        # ATTRIBUTES WRITTEN
        m = re.match(r"^(?:show|visualize)\s+attributes?\s+written\s+by\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types={"method", "function"})
            if not node:
                print(f"No function/method found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="writes_attr")]
            print(f"Attributes written by {node}: {succs}")
            _visualize(node, rel_types=["writes_attr"], node_filter=[("type", "in", {"method", "function", "attribute"})])
            return succs

        # RAISES
        m = re.match(r"^(?:what\s+does\s+(.+)\s+raise|raises\s+of\s+(.+))$", q)
        if m:
            name = (m.group(1) or m.group(2)).strip()
            node = _resolve_node_smart(name, preferred_types={"function", "method"})
            if not node:
                print(f"No function/method found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype="raises")]
            print(f"Exceptions raised by {node}: {succs}")
            _visualize(node, rel_types=["raises"], node_filter=[("type", "in", {"function", "method", "exception"})])
            return succs

        # NEIGHBORS (generic)
        m = re.match(r"^(?:neighbors\s+of|visualize\s+neighbors\s+of)\s+(.+)$", q)
        if m:
            name = m.group(1).strip()
            node = _resolve_node_smart(name, preferred_types=None)
            if not node:
                print(f"No node found matching '{name}'")
                return None
            succs = [s for s, ed in _edge_pairs_out(node) if _edge_passes(ed, etype=None)]
            preds = [p for p, ed in _edge_pairs_in(node) if _edge_passes(ed, etype=None)]
            neigh = sorted(set(succs + preds))
            print(f"Neighbors of {node}: {neigh}")
            _visualize(node, rel_types=None)
            return neigh

    except Exception as e:
        print(f"query_graph: unexpected error while processing query '{raw}' ({e}).")
        return None

    # ---------- No match ----------
    print("⚠️ Query not recognized. Try:\n"
          "  'show modules used by parse convert_full_file'\n"
          "  'show callees of main depth 2 internal'\n"
          "  'what calls load_config depth=2'\n"
          "  'visualize methods of MyClass'\n"
          "  'neighbors of Database'")
    return None
    
def find_nodes_by_name(G, name, type_in=None, class_name=None, regex=False):
    """
    Find node IDs in the graph by name only.

    Args:
        G: networkx DiGraph
        name: str, the name to match (e.g. "load_config", "__init__", "SqlDB")
        type_in: optional set of node types to filter (e.g. {"function","method","class"})
        class_name: optional class name if you're looking for a method
        regex: if True, `name` is treated as a regex pattern

    Returns:
        list of matching node IDs
    """
    matches = []
    for nid, data in G.nodes(data=True):
        ntype = data.get("type")
        nname = data.get("name")

        if type_in and ntype not in type_in:
            continue

        if regex:
            import re
            if not re.search(name, str(nname)):
                continue
        else:
            if nname != name:
                continue

        if class_name:
            # For methods, check class context
            if data.get("class_name") == class_name:
                matches.append(nid)
                continue
            # Or parse it out of the ID suffix (::method::<Class>.<func>)
            if "::method::" in nid:
                qual = nid.split("::method::",1)[1]
                if qual.split(".",1)[0] == class_name:
                    matches.append(nid)
                continue
        else:
            matches.append(nid)

    return matches



def project_callees(G: nx.DiGraph, node_id: str):
    """
    Return IDs of in-project callees (functions/methods/lambdas) for the given node.
    """
    out, seen = [], set()
    for _, v, ed in G.out_edges(node_id, data=True):
        if ed.get("type") == "calls" and ed.get("internal") is True:
            t = G.nodes[v].get("type")
            if t in {"function", "method", "lambda"} and v not in seen:
                seen.add(v)
                out.append(v)
    return out

def visualize_internal_calls(G: nx.DiGraph, center_node: str, depth: int = 1, **kwargs):
    """
    Convenience wrapper to visualize only internal (project) calls.
    """
    if isinstance(G, nx.MultiDiGraph):          # <<< NEW
        G = project_graph_for_visualization(G)  # <<< NEW
    return visualize_subgraph(
        G,
        center_node=center_node,
        depth=depth,
        relation_types="calls",
        edge_where=[("internal", "==", True)],
        node_where=[("type", "in", {"function", "method", "lambda"})],
        **kwargs
    )

