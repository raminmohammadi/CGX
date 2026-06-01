from cgx.logging_setup import get_logger
import networkx as nx

# -------------------------------
# Logger
# -------------------------------
logger = get_logger(__name__)

# -------------------------------
# Aggregation for Visualization
# -------------------------------

def aggregate_multiedges_for_viz(G: nx.Graph) -> nx.DiGraph:
    """
    Aggregate parallel edges in a MultiDiGraph for visualization.

    This function collapses multiple edges between the same nodes
    into a single edge with aggregated attributes, making the graph
    easier to visualize.

    Rules:
      - Primary edge type = "calls" if present, else the most frequent type
      - Edge attributes aggregated:
          * edge_types (all distinct types)
          * edge_count (total edges merged)
          * call_sites_count (number of "calls" edges)
          * ambiguous (True if any edge was ambiguous)
          * internal (True if callee is a project entity: function/method/lambda)
          * any_internal / any_external
          * internal_count / external_count
          * linenos, callee_fullnames, aliases (deduplicated lists)
          * label_compact (short label like "calls ×3")

    Args:
        G (nx.Graph): The original knowledge graph. May be a MultiDiGraph
                      (with multiple edges) or a DiGraph.

    Returns:
        nx.DiGraph: A simplified DiGraph with one aggregated edge per (u, v).

    Example:
        >>> G = init_graph()
        >>> G.add_node("a"); G.add_node("b")
        >>> G.add_edge("a", "b", type="calls", lineno=10)
        >>> G.add_edge("a", "b", type="calls", lineno=20)
        >>> S = aggregate_multiedges_for_viz(G)
        >>> list(S.edges(data=True))
        [('a', 'b', {'type': 'calls',
                     'edge_types': ['calls'],
                     'edge_count': 2,
                     'call_sites_count': 2,
                     'ambiguous': False,
                     'internal': False,
                     'any_internal': False,
                     'any_external': False,
                     'internal_count': 0,
                     'external_count': 0,
                     'linenos': [10, 20],
                     'callee_fullnames': [],
                     'aliases': [],
                     'label_compact': 'calls ×2'})]
    """

    if isinstance(G, nx.DiGraph) and not isinstance(G, nx.MultiDiGraph):
        return G.copy()

    S = nx.DiGraph()
    S.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges():
        edicts = list(G[u][v].values()) if isinstance(G, nx.MultiDiGraph) else [G[u][v]]
        types = []
        ambiguous_any = False
        any_internal = any_external = False
        internal_count = external_count = 0
        linenos, callee_fullnames, aliases = set(), set(), set()

        for ed in edicts:
            et = ed.get("type"); types.append(et)
            if ed.get("ambiguous"): ambiguous_any = True
            internal = ed.get("internal")
            if internal is True:  any_internal, internal_count = True, internal_count + 1
            if internal is False: any_external, external_count = True, external_count + 1
            ln = ed.get("lineno");          ln is not None and linenos.add(ln)
            cf = ed.get("callee_fullname"); cf and callee_fullnames.add(cf)
            al = ed.get("alias");           al and aliases.add(al)

        edge_types = sorted({t for t in types if t})
        if "calls" in edge_types:
            primary_type = "calls"
        else:
            freq = {}
            for t in types:
                if not t: continue
                freq[t] = freq.get(t, 0) + 1
            primary_type = sorted([t for t, c in freq.items() if c == max(freq.values())])[0] if freq else ""

        call_sites_count = sum(1 for t in types if t == "calls")
        edge_count = len(edicts)

        # ★ set aggregated ‘internal’: true iff the callee is a project entity
        callee_type = G.nodes[v].get("type")
        agg_internal = callee_type in {"function", "method", "lambda"}

        label_compact = f"calls ×{call_sites_count}" if call_sites_count else (",".join(edge_types) if edge_types else "")

        attrs = {
            "type": primary_type,
            "edge_types": edge_types,
            "edge_count": edge_count,
            "call_sites_count": call_sites_count,
            "ambiguous": ambiguous_any,
            "internal": agg_internal,                # ★ important for your edge_where
            "any_internal": any_internal,
            "any_external": any_external,
            "internal_count": internal_count,
            "external_count": external_count,
            "linenos": sorted(linenos),
            "callee_fullnames": sorted(callee_fullnames),
            "aliases": sorted(aliases),
            "label_compact": label_compact,
        }

        if S.has_edge(u, v):
            prev = S[u][v]
            attrs["edge_types"]         = sorted(set(prev.get("edge_types", [])) | set(attrs["edge_types"]))
            attrs["edge_count"]        += prev.get("edge_count", 0)
            attrs["call_sites_count"]  += prev.get("call_sites_count", 0)
            attrs["ambiguous"]          = prev.get("ambiguous", False) or attrs["ambiguous"]
            attrs["internal"]           = prev.get("internal", False) or attrs["internal"]
            attrs["any_internal"]       = prev.get("any_internal", False) or attrs["any_internal"]
            attrs["any_external"]       = prev.get("any_external", False) or attrs["any_external"]
            attrs["internal_count"]    += prev.get("internal_count", 0)
            attrs["external_count"]    += prev.get("external_count", 0)
            attrs["linenos"]            = sorted(set(prev.get("linenos", [])) | set(attrs["linenos"]))
            attrs["callee_fullnames"]   = sorted(set(prev.get("callee_fullnames", [])) | set(attrs["callee_fullnames"]))
            attrs["aliases"]            = sorted(set(prev.get("aliases", [])) | set(attrs["aliases"]))
            if prev.get("type") == "calls" or attrs["type"] == "calls":
                attrs["type"] = "calls"

        S.add_edge(u, v, **attrs)

    return S


def project_graph_for_visualization(G: nx.Graph) -> nx.DiGraph:
    """
    Project a knowledge graph into a visualization-friendly form.

    Args:
        G (nx.Graph): Original knowledge graph.

    Returns:
        nx.DiGraph: Aggregated graph suitable for visualization.

    Example:
        >>> G = init_graph()
        >>> G.add_node("a"); G.add_node("b")
        >>> G.add_edge("a", "b", type="calls")
        >>> S = project_graph_for_visualization(G)
        >>> isinstance(S, nx.DiGraph)
        True
    """
    return aggregate_multiedges_for_viz(G)