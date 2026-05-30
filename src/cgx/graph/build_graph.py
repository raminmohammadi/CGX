import os
from cgx.logging_setup import get_logger
import networkx as nx

# -------------------------------
# Logger
# -------------------------------
logger = get_logger(__name__)


# -------------------------------
# Utilities & Normalizers
# -------------------------------

def base_name(path: str):
    """
    Extract the base name of a given file path.

    Args:
        path (str): The file path.

    Returns:
        str: The base name of the file path, or the original path if None.

    Example:
        >>> base_name("/home/user/project/main.py")
        'main.py'
    """
    return os.path.basename(path) if path else path



def node_kind(ch: dict):
    """
    Determine the node kind (function, method, class, etc.) based on chunk data.

    Special handling:
      - Functions with '::method::' in their id are treated as methods.
      - Classes with '::class::' in their id are treated as classes.

    Args:
        ch (dict): The chunk dictionary containing 'id' and 'type'.

    Returns:
        str: The normalized node type ("method", "class", or the original type).

    Example:
        >>> node_kind({"id": "file.py::method::Class.method", "type": "function"})
        'method'
    """
    t = ch.get("type")
    if t == "function" and "::method::" in ch.get("id", ""):
        return "method"
    elif t == "class" and "::class::" in ch.get("id", ""):
        return "class"
    return t

def meta_of(ch: dict):
    """
    Safely extract metadata from a chunk.

    Args:
        ch (dict): The chunk dictionary.

    Returns:
        dict: The 'meta' field if present, otherwise an empty dict.

    Example:
        >>> meta_of({"meta": {"docstring": "example"}})
        {'docstring': 'example'}
    """
    return ch.get("meta") or {}

def parse_method_triplet_from_id(cid: str):
    """
    Parse a method triplet (file, class, method) from an id string.

    Args:
        cid (str): A chunk id formatted like '<file>::method::<Class>.<method>'.

    Returns:
        tuple[str, str, str]: The parsed (file, class, method),
                              or (None, None, None) if parsing fails.

    Example:
        >>> parse_method_triplet_from_id("file.py::method::MyClass.my_method")
        ('file.py', 'MyClass', 'my_method')
    """

    try:
        pre, qual = cid.split("::method::", 1)
        cls, meth = qual.split(".", 1)
        return pre, cls, meth
    except Exception:
        return None, None, None

def normalize_imports_used(meta: dict):
    """
    Normalize the 'imports_used' field into iterable tuples.

    Accepts:
      - dict: alias -> full_module
      - list: full_module strings
      - None/missing

    Args:
        meta (dict): Metadata dictionary.

    Returns:
        list[tuple[str|None, str]]: Normalized imports.

    Example:
        >>> normalize_imports_used({"imports_used": {"np": "numpy"}})
        [('np', 'numpy')]
    """
    imports = meta.get("imports_used")
    if not imports:
        return []
    if isinstance(imports, dict):
        out = []
        for alias, full in imports.items():
            if not full:
                continue
            try:
                out.append((alias, str(full)))
            except Exception:
                continue
        return out
    if isinstance(imports, list):
        out = []
        for full in imports:
            if not full:
                continue
            try:
                out.append((None, str(full)))
            except Exception:
                continue
        return out
    logger.debug("Unexpected imports_used shape: %r", type(imports))
    return []

def normalize_attr_reads(meta: dict):
    """
    Normalize attribute read operations.

    Accepts:
      - meta['attributes_used']: list[str]
      - meta['reads']: list[str]

    Args:
        meta (dict): Metadata dictionary.

    Returns:
        list[str]: List of dotted attribute paths.

    Example:
        >>> normalize_attr_reads({"reads": ["self.user.name"]})
        ['self.user.name']
    """
    reads = meta.get("attributes_used") or meta.get("reads") or []
    if not isinstance(reads, list):
        logger.debug("Unexpected reads shape: %r", type(reads))
        return []
    out = []
    for r in reads:
        try:
            if r:
                out.append(str(r))
        except Exception:
            continue
    return out

def normalize_attr_writes(meta: dict):
    """
    Normalize attribute write operations.

    Accepts:
      - meta['instance_attributes']: list[dict]
      - meta['writes']: list[str] (fallback)

    Args:
        meta (dict): Metadata dictionary.

    Returns:
        list[dict]: Each dict includes
            { name, inferred_type, value_preview, source, lineno }.

    Example:
        >>> normalize_attr_writes({"writes": ["user"]})
        [{'name': 'user', 'inferred_type': None, 'value_preview': None,
          'source': None, 'lineno': None}]
    """
    writes = meta.get("instance_attributes")
    out = []
    if isinstance(writes, list):
        for it in writes:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            if not name:
                continue
            out.append({
                "name": str(name),
                "inferred_type": it.get("inferred_type"),
                "value_preview": it.get("value_preview"),
                "source": it.get("source"),
                "lineno": it.get("lineno"),
            })
        return out

    # Fallback: limited case from meta['writes'] as list[str]
    writes2 = meta.get("writes")
    if isinstance(writes2, list):
        for w in writes2:
            try:
                if w:
                    out.append({"name": str(w), "inferred_type": None,
                                "value_preview": None, "source": None, "lineno": None})
            except Exception:
                continue
    return out


def normalize_raises(meta: dict):
    """
    Normalize exceptions raised.

    Accepts:
      - meta['raises']: list[str] or list[dict{name: ...}]

    Args:
        meta (dict): Metadata dictionary.

    Returns:
        list[str]: Exception names.

    Example:
        >>> normalize_raises({"raises": ["ValueError", {"name": "KeyError"}]})
        ['ValueError', 'KeyError']
    """
    raises = meta.get("raises") or []
    out = []
    if isinstance(raises, list):
        for r in raises:
            if isinstance(r, str):
                if r:
                    out.append(r)
            elif isinstance(r, dict):
                name = r.get("name")
                if name:
                    out.append(str(name))
            # else ignore
    else:
        logger.debug("Unexpected raises shape: %r", type(raises))
    return out


# -------------------------------
# Validation & Initialization
# -------------------------------

def validate_inputs(chunks, calls):
    """
    Validate that inputs are lists.

    Args:
        chunks (list): List of code chunks.
        calls (list): List of call relations.

    Raises:
        TypeError: If inputs are not lists.

    Example:
        >>> validate_inputs([], [])
        # No error
    """
    if not isinstance(chunks, list):
        raise TypeError(f"Expected chunks to be a list, got {type(chunks)}")
    if not isinstance(calls, list):
        raise TypeError(f"Expected calls to be a list, got {type(calls)}")


# def init_graph():
#     return nx.DiGraph()

def init_graph():
    """
    Initialize the knowledge graph.

    Returns:
        nx.MultiDiGraph: A new directed multigraph.

    Example:
        >>> G = init_graph()
        >>> isinstance(G, nx.MultiDiGraph)
        True
    """
    # Authoritative store should keep one edge per callsite
    return nx.MultiDiGraph()


# -------------------------------
# Index Structures
# -------------------------------

def init_indices():
    """
    Initialize index structures for graph building.

    Returns:
        dict: Dictionary of indices:
            - id_to_chunk
            - file_set
            - funcs_by_name
            - funcs_by_file_and_name
            - methods_by_class_and_name

    Example:
        >>> indices = init_indices()
        >>> sorted(indices.keys())
        ['file_set', 'funcs_by_file_and_name', 'funcs_by_name',
         'id_to_chunk', 'methods_by_class_and_name']
    """
    return {
        "id_to_chunk": {},                  # id -> chunk
        "file_set": set(),                  # {file}
        "funcs_by_name": {},                # name -> [ids]
        "funcs_by_file_and_name": {},       # (file, name) -> [ids]
        "methods_by_class_and_name": {},    # (file, class, method) -> id
    }


# -------------------------------
# Graph Builders
# -------------------------------

def add_file_nodes(G, chunks, indices):
    """
    Add file nodes to the graph.

    Args:
        G (nx.Graph): The graph.
        chunks (list): List of code chunks.
        indices (dict): Index structures.

    Side effects:
        - Adds 'file' nodes to G.
        - Updates indices['file_set'].

    Example:
        >>> G, indices = init_graph(), init_indices()
        >>> add_file_nodes(G, [{"file": "main.py"}], indices)
        >>> "main.py" in G
        True
    """
    for ch in chunks:
        try:
            f = ch["file"]
            indices["file_set"].add(f)
        except KeyError:
            logger.warning("Chunk missing file field, skipped: %s", ch)
            continue

    for f in indices["file_set"]:
        G.add_node(f, type="file", name=base_name(f))


def add_code_nodes(G, chunks, indices):
    """
    Add code nodes (files, classes, functions, methods, lambdas).

    Args:
        G (nx.Graph): The graph.
        chunks (list): List of code chunks.
        indices (dict): Index structures.

    Side effects:
        - Adds nodes with attributes.
        - Adds 'defines' edges (file -> entity).
        - Updates indices.

    Example:
        >>> G, indices = init_graph(), init_indices()
        >>> ch = {"id": "f.py::class::C", "file": "f.py", "type": "class", "name": "C"}
        >>> add_code_nodes(G, [ch], indices)
        >>> "f.py::class::C" in G
        True
    """
    for ch in chunks:
        try:
            cid = ch["id"]
            cfile = ch["file"]
            kind = node_kind(ch) # TODO: Only checked for methods and class otherwise returns default
            cname = ch.get("name", "")
            meta = meta_of(ch)
        except Exception as e:
            logger.warning("Malformed chunk skipped: %s (%s)", ch, e)
            continue

        if not cid or not cfile or not kind:
            logger.warning("Incomplete chunk skipped: %s", ch)
            continue

        node_attrs = {
            "type": kind,
            "name": cname,
            "file": cfile,
            "code": ch.get("code"),
        }

        # ---- NEW: enrich 'file' nodes with docstring/module_path/members/metrics
        if kind == "file":
            node_attrs.update({
                "module_path": ch.get("module_path"),
                "docstring": meta.get("docstring"),
                "members": meta.get("members"),
                "metrics": meta.get("metrics"),
            })

        if kind == "class":
            node_attrs.update({
                "decorators": meta.get("decorators"),
                "bases": meta.get("bases"),
                "keywords": meta.get("keywords"),
                "docstring": meta.get("docstring"),
                "doc_parsed": meta.get("doc_parsed"),
                "is_dataclass": meta.get("is_dataclass"),
                "dataclass_fields": meta.get("dataclass_fields"),
                # additive; may be None
                "enclosing_class": meta.get("enclosing_class"),
            })

        if kind in ("function", "method"):
            node_attrs.update({
                "decorators": meta.get("decorators"),
                "method_kind": meta.get("method_kind"),
                "is_async": meta.get("is_async"),
                "is_method": meta.get("is_method"),
                "class_name": meta.get("class_name"),
                "signature": meta.get("signature"),
                "parameters": meta.get("parameters"),
                "returns_annotation": meta.get("returns_annotation"),
                "docstring": meta.get("docstring"),
                "doc_parsed": meta.get("doc_parsed"),
                "is_generator": meta.get("is_generator"),
                "metrics": meta.get("metrics"),
            })

        G.add_node(cid, **node_attrs)
        indices["id_to_chunk"][cid] = ch

        # file -> entity (defines) — but skip for file chunks to avoid self-loop
        if kind != "file" and cfile in G:
            G.add_edge(cfile, cid, type="defines")

        # Indexing for resolution (unchanged)
        if kind in ("function", "method", "lambda"):
            indices["funcs_by_name"].setdefault(cname, []).append(cid)
            indices["funcs_by_file_and_name"].setdefault((cfile, cname), []).append(cid)

            if kind == "method":
                f2, cls2, meth2 = parse_method_triplet_from_id(cid)
                if f2 and cls2 and meth2:
                    indices["methods_by_class_and_name"][(f2, cls2, meth2)] = cid
                else:
                    cls_fallback = meta.get("class_name")
                    if cls_fallback:
                        indices["methods_by_class_and_name"][(cfile, cls_fallback, cname)] = cid


def add_defines_edges(G, chunks):
    """
    Add 'defines' edges for relationships:
      - class -> method
      - class -> nested class
      - function/method -> lambda
    
    Note: doesn’t need to add anything for non-nested classes — they’re already connected at the file level.
    
    Args:
        G (nx.Graph): The graph.
        chunks (list): List of code chunks.

    Example:
        >>> G, indices = init_graph(), init_indices()
        >>> ch = {"id": "f.py::method::C.m", "file": "f.py", "type": "function"}
        >>> G.add_node("f.py::class::C", type="class")
        >>> add_defines_edges(G, [ch])
        >>> list(G.edges("f.py::class::C"))
        [('f.py::class::C', 'f.py::method::C.m')]
    """
    # class -> method
    for ch in chunks:
        try:
            if "::method::" in ch.get("id", ""):
                cfile = ch["file"]
                qual = ch["id"].split("::method::", 1)[1]
                class_name = qual.split(".", 1)[0]
                class_id = f"{cfile}::class::{class_name}"
                if class_id in G:
                    G.add_edge(class_id, ch["id"], type="defines")
        except Exception as e:
            logger.debug("Skipping class->method link: %s (%s)", ch, e)

    # class -> nested class
    for ch in chunks:
        try:
            if ch.get("type") == "class":
                enclosing_cls = meta_of(ch).get("enclosing_class") or ch.get("parent_class")
                if enclosing_cls:
                    parent_id = f"{ch['file']}::class::{enclosing_cls}"
                    if parent_id in G:
                        G.add_edge(parent_id, ch["id"], type="defines")
        except Exception as e:
            logger.debug("Skipping nested class link: %s (%s)", ch, e)

    # function/method -> lambda
    for ch in chunks:
        try:
            if ch.get("type") == "lambda":
                enc = meta_of(ch).get("enclosing") or ch.get("parent_function") or ch.get("parent_method")
                if enc and enc in G:
                    G.add_edge(enc, ch["id"], type="defines")
        except Exception as e:
            logger.debug("Skipping lambda link: %s (%s)", ch, e)


# -------------------------------
# Calls Resolution & Edges
# -------------------------------


def resolve_callee_candidates(rel: dict, indices: dict, G: nx.DiGraph):
    """
    Resolve possible callee nodes for a call relation.

    Resolution order:
      1. self.method / super().method within same class & file
      2. Same-file function/method by name
      3. Global functions/methods by name
      4. Dotted fullname -> module::<prefix>
      5. unresolved::<name>

    Args:
        rel (dict): Call relation with caller_id, callee_name, callee_fullname, lineno.
        indices (dict): Index structures.
        G (nx.Graph): Knowledge graph.

    Returns:
        tuple[list[str], dict]: (candidate target IDs, edge attributes).

    Example:
        >>> indices = init_indices()
        >>> G = init_graph()
        >>> rel = {"caller_id": "c1", "callee_name": "foo"}
        >>> resolve_callee_candidates(rel, indices, G)
        (['unresolved::foo'], {'lineno': None, 'callee_fullname': None})
    """
    caller_id = rel.get("caller_id")
    callee_name = rel.get("callee_name")
    callee_full = rel.get("callee_fullname")
    lineno = rel.get("lineno")

    attrs = {"lineno": lineno, "callee_fullname": callee_full}

    if caller_id not in indices["id_to_chunk"] or not callee_name:
        return [], attrs

    caller = indices["id_to_chunk"][caller_id]
    caller_file = caller.get("file")
    caller_k = node_kind(caller)
    caller_cls = meta_of(caller).get("class_name") if caller_k == "method" else None

    # Case 1: self.method / super().method → same-class method resolve
    if caller_cls and callee_full:
        if callee_full.startswith("self.") or callee_full.startswith("super()."):
            m = indices["methods_by_class_and_name"].get((caller_file, caller_cls, callee_name))
            if m:
                return [m], attrs

    # Case 2: same-file functions/methods
    cands = list(indices["funcs_by_file_and_name"].get((caller_file, callee_name), []))
    if cands:
        return cands, attrs

    # Case 3: global functions/methods
    cands = list(indices["funcs_by_name"].get(callee_name, []))
    if cands:
        return cands, attrs

    # Case 4: dotted fullname -> module::<prefix>
    if callee_full and "." in callee_full:
        mod_hint = callee_full.rsplit(".", 1)[0]
        mod_node = f"module::{mod_hint}"
        if mod_node not in G:
            G.add_node(mod_node, type="module", name=mod_hint)
        return [mod_node], attrs

    # Case 5: unresolved
    unresolved = f"unresolved::{callee_name}"
    if unresolved not in G:
        G.add_node(unresolved, type="unresolved", name=callee_name)
    return [unresolved], attrs


def add_calls_edges(G, calls, indices):
    """
    Add 'calls' edges to connect caller and callee nodes in the graph.

    Functions, methods, and lambdas must already have been added
    as nodes by `add_code_nodes`. This function resolves callees
    (using `resolve_callee_candidates`) and adds the appropriate
    call edges:

      - caller -> callee (type="calls")
      - with attributes: ambiguous, internal, lineno, callee_fullname

    If the callee cannot be resolved to an existing node, this
    function may create placeholder nodes:
      - module::<name> for external modules
      - unresolved::<name> for unknown callees

    Args:
        G (nx.Graph): Knowledge graph.
        calls (list): List of call relations (dicts with caller_id, callee_name, etc.).
        indices (dict): Index structures from `add_code_nodes`.

    Example:
        >>> G, indices = init_graph(), init_indices()
        >>> G.add_node("f.py::func::foo", type="function")
        >>> rel = {"caller_id": "f.py::func::foo", "callee_name": "bar"}
        >>> add_calls_edges(G, [rel], indices)
        >>> any(G.edges[e]["type"] == "calls" for e in G.edges)
        True
    """
    for rel in calls:
        try:
            targets, attrs = resolve_callee_candidates(rel, indices, G)
            caller = rel.get("caller_id")
            if not caller or caller not in G:
                continue

            # ambiguous if >1 internal targets
            internal_targets = [t for t in targets if isinstance(t, str) and t in indices["id_to_chunk"]]
            ambiguous = len(internal_targets) > 1

            for tgt in targets:
                if tgt in G:
                    tgt_type = G.nodes[tgt].get("type")
                    internal = tgt_type in {"function", "method", "lambda"}
                    G.add_edge(
                        caller, tgt,
                        type="calls",
                        ambiguous=ambiguous,
                        internal=internal,
                        **attrs
                    )
        except Exception as e:
            logger.warning("Failed to add call edge: %s (%s)", rel, e)


# -------------------------------
# Modules, Attributes, Exceptions
# -------------------------------


def add_module_edges(G, chunks):
    """
    Add 'uses_module' edges from code entities to module nodes.

    Code nodes (functions, classes, files, etc.) must already have
    been added by `add_code_nodes`. For each chunk, this function
    inspects `meta['imports_used']` (normalized via
    `normalize_imports_used`) and:

      - Ensures a module node exists with id "module::<full_name>"
      - Adds an edge from the code entity to the module node:
            code_entity -> module::<full_name>
        with attributes {type="uses_module", alias=<import alias or None>}

    Args:
        G (nx.Graph): Knowledge graph.
        chunks (list): List of code chunks.

    Example:
        >>> G = init_graph()
        >>> G.add_node("f.py::func::foo", type="function")
        >>> ch = {"id": "f.py::func::foo", "meta": {"imports_used": {"np": "numpy"}}}
        >>> add_module_edges(G, [ch])
        >>> list(G.edges("f.py::func::foo"))
        [('f.py::func::foo', 'module::numpy')]
    """

    for ch in chunks:
        try:
            cid = ch["id"]
            if cid not in G:
                continue
            meta = meta_of(ch)
            for alias, full in normalize_imports_used(meta):
                mod_node = f"module::{full}"
                if mod_node not in G:
                    G.add_node(mod_node, type="module", name=full)
                # Parity: edge carries alias only; no internal flag
                G.add_edge(cid, mod_node, type="uses_module", alias=alias)
        except Exception as e:
            logger.warning("Failed to add module edges for %s: %s", ch.get("id"), e)


def add_attr_and_exception_edges(G, chunks):
    """
    Add edges for attribute accesses (reads/writes) and exceptions raised.

    Functions and methods must already have been added as nodes by
    `add_code_nodes`. This function enriches them with edges to
    attribute or exception nodes:

      - Reads:
          self.attr -> adds 'reads_attr' edge to an attribute node
      - Writes:
          self.attr assignment -> adds 'writes_attr' edge with metadata
      - Raises:
          exception type -> adds 'raises' edge to an exception node

    Attribute and exception nodes are created here on demand if they
    do not already exist.

    Args:
        G (nx.Graph): Knowledge graph.
        chunks (list): List of code chunks.

    Example:
        >>> G = init_graph()
        >>> G.add_node("f.py::method::C.m", type="method")
        >>> ch = {
        ...   "id": "f.py::method::C.m", "file": "f.py", "type": "function",
        ...   "meta": {"class_name": "C",
        ...            "reads": ["self.x"],
        ...            "writes": ["y"],
        ...            "raises": ["ValueError"]}
        ... }
        >>> add_attr_and_exception_edges(G, [ch])
        >>> sorted((u, v, d["type"]) for u, v, d in G.edges(data=True))
        [('f.py::method::C.m', 'exception::ValueError', 'raises'),
         ('f.py::method::C.m', 'f.py::attr::C.x', 'reads_attr'),
         ('f.py::method::C.m', 'f.py::attr::C.y', 'writes_attr')]
    """
    for ch in chunks:
        try:
            cid = ch["id"]
            if cid not in G:
                continue
            kind = node_kind(ch)
            if kind not in ("function", "method"):
                continue

            cfile = ch["file"]
            meta = meta_of(ch)
            cls = meta.get("class_name") if kind == "method" else None

            # Reads (only 'self.*' roots become attributes)
            for dotted in normalize_attr_reads(meta):
                if cls and isinstance(dotted, str) and dotted.startswith("self."):
                    try:
                        root = dotted.split(".", 1)[1]
                        attr_root = root.split(".", 1)[0]
                        attr_id = f"{cfile}::attr::{cls}.{attr_root}"
                        if attr_id not in G:
                            G.add_node(attr_id, type="attribute", name=f"{cls}.{attr_root}", file=cfile, class_name=cls)
                        G.add_edge(cid, attr_id, type="reads_attr", detail=dotted)
                    except Exception:
                        # Keep going—one malformed dotted path shouldn't kill others
                        continue

            # Writes
            for it in normalize_attr_writes(meta):
                name = it.get("name")
                if cls and name:
                    attr_id = f"{cfile}::attr::{cls}.{name}"
                    if attr_id not in G:
                        G.add_node(attr_id, type="attribute", name=f"{cls}.{name}", file=cfile, class_name=cls)
                    G.add_edge(
                        cid, attr_id, type="writes_attr",
                        inferred_type=it.get("inferred_type"),
                        value_preview=it.get("value_preview"),
                        source=it.get("source"),
                        lineno=it.get("lineno"),
                    )

            # Raises
            for exc in normalize_raises(meta):
                exc_node = f"exception::{exc}"
                if exc_node not in G:
                    G.add_node(exc_node, type="exception", name=exc)
                G.add_edge(cid, exc_node, type="raises")
        except Exception as e:
            logger.warning("Failed to enrich node %s: %s", ch.get("id"), e)


# -------------------------------
# Orchestration
# -------------------------------

def build_knowledge_graph(chunks, calls):
    """
    Build the full knowledge graph.

    Steps:
      - Validate inputs
      - Initialize graph and indices
      - Add file nodes
      - Add code nodes
      - Add defines edges
      - Add calls edges
      - Add module edges
      - Add attribute/exception edges

    Args:
        chunks (list): List of code chunks.
        calls (list): List of call relations.

    Returns:
        nx.MultiDiGraph: The built knowledge graph.

    Example:
        >>> chunks = [{"id": "f.py::func::foo", "file": "f.py", "type": "function", "name": "foo"}]
        >>> calls = []
        >>> G = build_knowledge_graph(chunks, calls)
        >>> "f.py::func::foo" in G
        True
    """
    validate_inputs(chunks, calls)
    G = init_graph()
    indices = init_indices()

    add_file_nodes(G, chunks, indices)
    # adds nodes (files, classes, functions, methods, lambdas) + file -> entity edges.
    add_code_nodes(G, chunks, indices)
    # adds extra edges that can’t be inferred from file-level containment: class → method, class → nested class, function/method → lambda
    add_defines_edges(G, chunks)
    add_calls_edges(G, calls, indices)
    add_module_edges(G, chunks)
    add_attr_and_exception_edges(G, chunks)

    return G


