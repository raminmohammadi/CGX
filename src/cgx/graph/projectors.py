
# -------------------------------
# Aggregation for Visualization (NEW)
# -------------------------------

def aggregate_multiedges_for_viz(G: nx.Graph) -> nx.DiGraph:
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
    return aggregate_multiedges_for_viz(G)