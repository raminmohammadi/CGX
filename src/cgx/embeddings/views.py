

from __future__ import annotations

"""
Deterministic, auditable text views for each chunk:

- Intent view (NL-friendly, compact card)
- Implementation view (code-ish)

These functions are **purely additive** and do not alter existing behavior unless
you choose to attach the views back onto your chunk dicts.

Usage (non-destructive):
    cards = [build_intent_view(ch, G) for ch in chunks]
    impls = [build_implementation_view(ch, normalize=False, strip_literals=False) for ch in chunks]

Or, to attach (additive keys):
    enriched = attach_views_to_chunks(chunks, G, topk_callees=10, normalize_impl=False, strip_literals=False)

All outputs are deterministic and derived purely from AST/graph metadata you already emit.

Embedding Context
-----------------
The `view_intent` and `view_impl` strings produced here are the **exact inputs**
to embedding models (e.g., BGE, Jina, Gemma) when constructing FAISS indices.

The actual embedding step is delegated to:
    `cgx.embeddings.build.build_embeddings(model_name, ...)`

So if you want to add support for a new embedding model (e.g., Gemma), update
**build.py**. This file only prepares the text to embed.
"""

import ast
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # optional; we handle None safely

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_f)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------
# Small utilities (deterministic helpers)
# ---------------------------

def _first_sentence(text: Optional[str]) -> str:
    """
    Extract the first non-empty line/sentence from a docstring-like text.
    Deterministic and defensive: returns "" if missing.
    """
    if not text:
        return ""
    # Prefer first non-empty line
    
    # TODO: For now return the full string, later maybe only first line or a summary
    # for line in text.splitlines():
    s = text.strip()
    if s:
        # Trim trailing sentence terminators conservatively
        return s
    return ""


def _doc_first_sentence(meta: Dict[str, Any]) -> str:
    """
    Use parsed docstring summary when available, else fallback to first line of raw docstring.
    """
    try:
        parsed = meta.get("doc_parsed") or {}
        summ = parsed.get("summary")
        if isinstance(summ, str) and summ.strip():
            return summ.strip()
    except Exception:
        pass
    raw = meta.get("docstring") or ""
    return _first_sentence(raw)


def _attribute_roots_read(meta: Dict[str, Any]) -> List[str]:
    """
    Unique roots for `self.foo.*` -> 'foo'. Sorted for determinism.

    Canonical home for this helper; :mod:`cgx.embeddings.helpers` re-exports
    it for back-compatibility with the original import path used by
    :mod:`cgx.embeddings.records`.
    """
    roots = set()
    try:
        reads = meta.get("attributes_used") or meta.get("reads") or []
        for dotted in reads if isinstance(reads, list) else []:
            if isinstance(dotted, str) and dotted.startswith("self."):
                try:
                    after = dotted.split("self.", 1)[1]
                    root = after.split(".", 1)[0]
                    if root:
                        roots.add(root)
                except Exception:
                    continue
    except Exception:
        pass
    return sorted(roots)


def _attribute_roots_written(meta: Dict[str, Any]) -> List[str]:
    """
    Unique instance attribute roots written (from instance_attributes).
    """
    roots = set()
    try:
        writes = meta.get("instance_attributes") or []
        if isinstance(writes, list):
            for it in writes:
                if isinstance(it, dict):
                    nm = it.get("name")
                    if nm:
                        roots.add(str(nm))
    except Exception:
        pass
    # Fallback: some pipelines might carry a plain list in 'writes'
    if not roots:
        try:
            writes2 = meta.get("writes") or []
            for w in writes2 if isinstance(writes2, list) else []:
                if isinstance(w, str) and w:
                    roots.add(w)
        except Exception:
            pass
    return sorted(roots)


def _imports_fullnames(meta: Dict[str, Any]) -> List[str]:
    """
    Return sorted list of full import module names (deduped).
    Accepts meta['imports_used'] as dict alias->full or list of full strings.
    """
    fulls = set()
    imps = meta.get("imports_used")
    if isinstance(imps, dict):
        for full in imps.values():
            if isinstance(full, str) and full:
                fulls.add(full)
    elif isinstance(imps, list):
        for full in imps:
            if isinstance(full, str) and full:
                fulls.add(full)
    return sorted(fulls)


def _raises_list(meta: Dict[str, Any]) -> List[str]:
    """
    Normalize raises to list[str].
    """
    out: List[str] = []
    rs = meta.get("raises") or []
    if isinstance(rs, list):
        for r in rs:
            if isinstance(r, str) and r:
                out.append(r)
            elif isinstance(r, dict):
                nm = r.get("name")
                if nm:
                    out.append(str(nm))
    return sorted(set(out))


def _topk_callee_names(meta: Dict[str, Any], k: int = 10) -> List[str]:
    """
    Deterministic top-K callee names by frequency, then alphabetical as tie-breaker.
    """
    freq: Dict[str, int] = {}
    calls = meta.get("calls_detailed") or []
    if isinstance(calls, list):
        for c in calls:
            if isinstance(c, dict):
                nm = c.get("callee_name")
                if isinstance(nm, str) and nm:
                    freq[nm] = freq.get(nm, 0) + 1
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [nm for nm, _ in items[:k]]


def _called_by_count(node_id: str, G) -> int:
    """
    Count inbound call edges to this node. If G is a MultiDiGraph, we count edges of type='calls'.
    If G is None or the node doesn't exist, returns 0.
    """
    try:
        if G is None or node_id not in G:
            return 0
        # Support DiGraph or MultiDiGraph with aggregated attributes
        cnt = 0
        # Prefer predecessor iteration for clarity
        for u in G.predecessors(node_id):
            edata = G[u][node_id]
            # MultiDiGraph => dict of dicts; DiGraph => single dict
            if isinstance(edata, dict) and any(isinstance(v, dict) for v in edata.values()):
                # MultiDiGraph path
                for ed in edata.values():
                    if ed.get("type") == "calls":
                        cnt += 1
            else:
                if edata.get("type") == "calls":
                    cnt += 1
        return int(cnt)
    except Exception:
        return 0


def _metrics_meta(meta: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], bool, bool]:
    """
    Pick standard metrics for the card: n_loc, n_params, async, generator.
    """
    try:
        m = meta.get("metrics") or {}
        n_loc = m.get("n_loc")
        n_params = m.get("n_params")
        is_async = bool(meta.get("is_async", False))
        is_gen = bool(meta.get("is_generator", False))
        return n_loc, n_params, is_async, is_gen
    except Exception:
        return None, None, False, False


# ---------------------------
# Intent view
# ---------------------------

def _is_method_chunk(ctype: str, cid: str) -> bool:
    """Python methods are emitted with ``type='function'`` and a ``::method::``
    id segment; treat both encodings as methods so the class name is never lost."""
    return ctype == "method" or (ctype == "function" and "::method::" in str(cid))


def _location_label(chunk: Dict[str, Any]) -> str:
    """Prefer the clean dotted ``module_path`` (e.g. ``net.client``) over the
    absolute filesystem path so the embedded text carries a portable, semantic
    location instead of machine/directory tokens. Falls back to the file's
    basename (never the full path)."""
    mp = chunk.get("module_path")
    if isinstance(mp, str) and mp.strip():
        return mp.strip()
    f = str(chunk.get("file") or "")
    return f.replace("\\", "/").rsplit("/", 1)[-1]


def _structured_doc_lines(meta: Dict[str, Any]) -> List[str]:
    """Render the parsed docstring (params/returns/raises) as short NL lines.

    Includes the signal the old card discarded (it kept only the first docstring
    sentence). Degrades to nothing when ``doc_parsed`` is absent."""
    dp = meta.get("doc_parsed")
    if not isinstance(dp, dict):
        return []
    out: List[str] = []
    params = dp.get("params")
    if isinstance(params, list) and params:
        rendered = []
        for it in params:
            if isinstance(it, dict):
                nm = str(it.get("name") or "").strip()
                desc = str(it.get("description") or it.get("desc") or "").strip()
                if nm and desc:
                    rendered.append(f"{nm}: {desc}")
                elif nm:
                    rendered.append(nm)
            elif isinstance(it, str) and it.strip():
                rendered.append(it.strip())
        if rendered:
            out.append("parameters: " + "; ".join(rendered))
    returns = dp.get("returns")
    if isinstance(returns, str) and returns.strip():
        out.append("returns: " + returns.strip())
    raises = dp.get("raises")
    if isinstance(raises, list) and raises:
        names = []
        for it in raises:
            if isinstance(it, dict):
                nm = str(it.get("name") or it.get("type") or "").strip()
                desc = str(it.get("description") or it.get("desc") or "").strip()
                names.append(f"{nm}: {desc}".strip(": ").strip() if desc else nm)
            elif isinstance(it, str):
                names.append(it.strip())
        names = [n for n in names if n]
        if names:
            out.append("raises: " + "; ".join(names))
    return out


def build_intent_view(chunk: Dict[str, Any], G=None, *, topk_callees: int = 10) -> str:
    """
    Build the natural-language "intent" text for a chunk, deterministically.

    This view is the primary **semantic search** input embedded by
    ``build_embeddings`` (see ``cgx/embeddings/build.py``). It is written as
    readable prose -- an identity line, the FULL docstring, structured
    params/returns/raises, and compact behavioural signals (calls, uses,
    route) -- rather than a fixed ``key: value`` metadata card.

    The old card embedded a rigid scaffold (``type:/symbol:/file:/class:/
    called_by_count:/metrics:``) shared verbatim across every chunk, plus the
    absolute file path twice; those constant tokens dominated the vector and
    pulled all chunks toward a common point (embedding-space collapse), while
    the real signal (a one-line docstring summary) was a fraction of the text.
    This builder drops the scaffold and the absolute path, keeps only lines
    that carry per-chunk meaning, and never emits an all-boilerplate card.
    """
    try:
        ctype = chunk.get("type", "")
        cid = chunk.get("id", "")
        name = chunk.get("name", "") or ""
        meta = chunk.get("meta") or {}

        is_method = _is_method_chunk(ctype, cid)
        kind = "method" if is_method else (ctype or "symbol")
        class_name = (meta.get("class_name") or "") if is_method else ""
        signature = meta.get("signature") or "" if ctype in {"method", "function"} else ""
        location = _location_label(chunk)

        # deterministic comma join
        def cj(xs: Iterable[str]) -> str:
            try:
                return ", ".join([x for x in xs if isinstance(x, str) and x])
            except Exception:
                return ""

        lines: List[str] = []

        # 1) Human-readable identity line: "<kind> <Class.name><sig> in <module>"
        qual = f"{class_name}.{name}" if (is_method and class_name) else name
        head = f"{kind} {qual}".rstrip()
        if signature:
            head += signature if signature.startswith("(") else f" {signature}"
        if location:
            head += f" in {location}"
        lines.append(head)

        # 2) HTTP route (endpoint queries have text to match against)
        route = meta.get("route")
        if isinstance(route, dict):
            methods = "/".join(m for m in (route.get("methods") or []) if isinstance(m, str))
            path = str(route.get("path") or "").strip()
            if methods or path:
                lines.append(f"HTTP endpoint {methods} {path}".strip())

        # 3) Decorators (skip the trivial/no-op case)
        decs = [str(d) for d in (meta.get("decorators") or []) if d]
        if decs:
            lines.append("decorators: " + cj(decs))

        # 4) FULL docstring, then structured params/returns/raises
        doc = meta.get("docstring")
        if isinstance(doc, str) and doc.strip():
            lines.append(doc.strip())
        lines.extend(_structured_doc_lines(meta))

        # 5) Return annotation when the docstring did not describe returns
        ret = meta.get("returns_annotation")
        if isinstance(ret, str) and ret.strip() and not any(l.startswith("returns:") for l in lines):
            lines.append(f"returns: {ret.strip()}")

        # 6) Compact behavioural signals (only when non-empty -- no scaffold)
        calls_out = _topk_callee_names(meta, k=topk_callees)
        if calls_out:
            lines.append("calls: " + cj(calls_out))
        imports = _imports_fullnames(meta)
        if imports:
            lines.append("uses: " + cj(imports))
        raises = _raises_list(meta)
        if raises:
            lines.append("raises: " + cj(raises))

        # 7) Class/file containers: surface members so a container query matches.
        if ctype == "class":
            bases = [str(b) for b in (meta.get("bases") or []) if b]
            if bases:
                lines.append("inherits: " + cj(bases))

        return "\n".join(lines)
    except Exception as e:
        logger.error("build_intent_view failed for chunk %r: %s", chunk.get("id"), e)
        # Return a minimal, safe NL line (no absolute path).
        nm = chunk.get("name", "") or ""
        return f"{chunk.get('type','symbol')} {nm}".strip()


# ---------------------------
# Implementation view (code-ish)
# ---------------------------

_STR_RE = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', re.S)

def _strip_string_literals(s: str) -> str:
    """
    Replace string literals with <STR> without altering layout drastically.
    """
    try:
        return _STR_RE.sub("<STR>", s)
    except Exception:
        return s

def _strip_numeric_literals(s: str) -> str:
    """
    Replace integers/floats with <NUM>. Keep identifiers intact.
    """
    try:
        # Conservative numeric pattern: integers and floats outside identifiers
        return re.sub(r"(?<![A-Za-z_])(?:\d+(?:\.\d+)?)(?![A-Za-z_])", "<NUM>", s)
    except Exception:
        return s

def _normalize_whitespace(s: str) -> str:
    """
    Strip trailing spaces and collapse excessive blank lines deterministically.
    """
    try:
        # strip trailing spaces
        s = "\n".join([ln.rstrip() for ln in s.splitlines()])
        # collapse >=3 consecutive blank lines to exactly 2
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip("\n")
    except Exception:
        return s


def _class_member_signatures(class_chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Collect method signatures belonging to this class from sibling method chunks.
    """
    sigs: List[str] = []
    try:
        file = class_chunk.get("file")
        cls = class_chunk.get("name")
        for ch in all_chunks:
            if not isinstance(ch, dict):
                continue
            if ch.get("type") == "function" and "::method::" in ch.get("id",""):
                # your function nodes represent methods when their id contains ::method::
                cid = ch.get("id","")
                try:
                    suffix = cid.split("::method::",1)[1]
                    cls_name = suffix.split(".",1)[0]
                except Exception:
                    cls_name = None
                if cls_name == cls and ch.get("file") == file:
                    sig = ch.get("meta",{}).get("signature") or "()"
                    sigs.append(f"def {ch.get('name','')}{sig}")
    except Exception:
        pass
    return sorted(set(sigs))


def build_implementation_view(
    chunk: Dict[str, Any],
    *,
    all_chunks: Optional[List[Dict[str, Any]]] = None,
    normalize: bool = False,
    strip_literals: bool = False,
) -> str:
    """
    Build the "implementation" text:

    - function/method: the function body text as emitted in chunk['code'].
    - class: class docstring + member signatures (not the entire file body).
    - file: module docstring + top-level member signatures (from meta.members).

    Options:
      - normalize=True       -> normalize whitespace deterministically.
      - strip_literals=True  -> replace string & numeric literals with <STR>/<NUM>.

    Notes:
      - This does not attempt to reformat code; it keeps your original spans (from AST).
      - all_chunks is only needed for class member signatures (to look up methods).
    """
    try:
        ctype = chunk.get("type", "")
        code_text = chunk.get("code", "") or ""
        meta = chunk.get("meta") or {}

        if ctype in {"function", "method", "lambda"}:
            text = code_text

        elif ctype == "class":
            # class view: docstring + member signatures
            doc = meta.get("docstring") or ""
            parts: List[str] = []
            if doc:
                parts.append('"""' + doc.replace('"""', r'\"\"\"') + '"""')
            sigs: List[str] = []
            if all_chunks is not None:
                sigs = _class_member_signatures(chunk, all_chunks)
            # If no all_chunks provided, we still return doc-only view (deterministic)
            parts.extend(sigs)
            text = "\n".join(parts)

        elif ctype == "file":
            # from S2 we already store a compact stub for files
            text = code_text

        else:
            # Fallback: return 'code' as-is
            text = code_text

        if strip_literals:
            text = _strip_string_literals(text)
            text = _strip_numeric_literals(text)
        if normalize:
            text = _normalize_whitespace(text)

        return text
    except Exception as e:
        logger.error("build_implementation_view failed for chunk %r: %s", chunk.get("id"), e)
        return chunk.get("code", "") or ""


# ---------------------------
# Attacher (optional, additive)
# ---------------------------

def attach_views_to_chunks(
    chunks: List[Dict[str, Any]],
    G=None,
    *,
    topk_callees: int = 10,
    normalize_impl: bool = False,
    strip_literals: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return a **new list** of chunks where each chunk dict is shallow-copied and
    augmented with:
        - 'view_intent': str
        - 'view_impl' : str

    This function is **non-destructive**: the original list and dicts are left untouched.

    Parameters
    ----------
    chunks : list[dict]
        Output of parse_codebase (now including file/class/function/method/lambda).
    G : networkx graph or None
        Your knowledge graph; used only for called_by_count (safe to omit).
    topk_callees : int
        Number of callee names to include in the intent card.
    normalize_impl : bool
        Apply deterministic whitespace normalization to impl view.
    strip_literals : bool
        Replace string/numeric literals with <STR>/<NUM> in impl view.

    Returns
    -------
    list[dict]
        New chunk dicts with 'view_intent' and 'view_impl' keys added.
    """
    enriched: List[Dict[str, Any]] = []
    for ch in chunks:
        try:
            ch2 = dict(ch)  # shallow copy to remain non-destructive
            card = build_intent_view(ch, G=G, topk_callees=topk_callees)
            impl = build_implementation_view(
                ch, all_chunks=chunks, normalize=normalize_impl, strip_literals=strip_literals
            )
            ch2["view_intent"] = card
            ch2["view_impl"] = impl
            enriched.append(ch2)
        except Exception as e:
            logger.warning("attach_views_to_chunks: failed for %r: %s", ch.get("id"), e)
            # still append original to preserve alignment
            enriched.append(ch)
    return enriched
