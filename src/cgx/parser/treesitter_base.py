"""Tree-sitter-backed multi-language parser behind the :class:`BaseParser` seam.

Grammars are soft-imported via ``tree_sitter_language_pack`` (an optional
dependency, see the ``parsers`` extra in ``pyproject.toml``). When the package
is absent the concrete subclasses register nothing and the project walker keeps
its historical behavior of skipping files it has no parser for. When present,
each subclass declares a grammar ``language`` name plus the node-type sets that
map that grammar onto CGX's canonical chunk shape (see ``cgx.parser.schema``).

The emitted ``(chunks, call_relations)`` tuple mirrors the Python parser's shape
byte-for-byte in its identity fields so every downstream consumer (records,
embeddings, retrieval, codegen) works unchanged. As with the Python parser,
methods are emitted with ``type == "function"`` and an ``id`` segment of
``::method::`` -- consumers discriminate on the id/meta, not the chunk type.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from cgx.parser.base import BaseParser
from cgx.parser.module_path import compute_module_path
from cgx.logging_setup import get_logger

logger = get_logger("treesitter")

# Cache one tree-sitter Parser per grammar name; remember failures so a missing
# optional dependency is probed at most once per language per process.
_PARSER_CACHE: Dict[str, Any] = {}
_UNAVAILABLE: set = set()


def _get_ts_parser(language: str) -> Optional[Any]:
    """Return a cached tree-sitter ``Parser`` for ``language`` or ``None``.

    ``None`` means the optional ``tree_sitter_language_pack`` dependency (or
    the specific grammar) is unavailable; callers degrade to an empty parse.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    if language in _UNAVAILABLE:
        return None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(language)
    except Exception as exc:  # pragma: no cover - depends on optional install
        logger.debug("tree-sitter unavailable for %r: %s", language, exc)
        _UNAVAILABLE.add(language)
        return None
    _PARSER_CACHE[language] = parser
    return parser


def treesitter_available(language: str) -> bool:
    """True when a tree-sitter grammar for ``language`` can be loaded."""
    return _get_ts_parser(language) is not None


class TreeSitterParser(BaseParser):
    """Generic tree-sitter parser configured per language via class attrs.

    Subclasses set :attr:`language` (the grammar name understood by
    ``tree_sitter_language_pack``) and, when a grammar names its nodes
    differently, override the ``*_nodes`` tuples below.
    """

    #: Grammar name passed to ``get_parser`` (e.g. ``"javascript"``).
    language: str = ""

    # Node-type vocabulary. Defaults target the JavaScript/TypeScript family;
    # other languages override as needed.
    function_nodes: Tuple[str, ...] = (
        "function_declaration", "generator_function_declaration",
        "function_signature",
    )
    class_nodes: Tuple[str, ...] = (
        "class_declaration", "abstract_class_declaration",
        "interface_declaration", "enum_declaration",
    )
    method_nodes: Tuple[str, ...] = (
        "method_definition", "method_signature", "abstract_method_signature",
    )
    #: Values of a ``variable_declarator`` that should be lifted to a named
    #: function chunk (``const f = () => ...`` / ``const f = function () ...``).
    func_value_nodes: Tuple[str, ...] = (
        "arrow_function", "function_expression", "function",
        "generator_function",
    )
    call_nodes: Tuple[str, ...] = ("call_expression", "new_expression")

    def parse_file(
        self,
        filepath: str,
        source_code: str,
        project_root: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        parser = _get_ts_parser(self.language)
        if parser is None:
            return [], []
        try:
            tree = parser.parse(source_code.encode("utf-8", errors="ignore"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tree-sitter parse failed for %s: %s", filepath, exc)
            return [], []
        walker = _Walker(self, filepath, project_root, source_code)
        try:
            walker.run(tree.root_node)
        except Exception as exc:
            logger.error("tree-sitter walk failed for %s: %s", filepath, exc)
        return walker.chunks, walker.calls


class _Walker:
    """Depth-first traversal that emits chunks + call relations for one file."""

    def __init__(self, spec: TreeSitterParser, filepath: str,
                 project_root: str, source: str) -> None:
        self.spec = spec
        self.filepath = filepath
        self.source = source
        self.chunks: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self.module_path = compute_module_path(project_root, filepath)

    # -- small node helpers ------------------------------------------------
    @staticmethod
    def _text(node: Any) -> str:
        try:
            return node.text.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _name(self, node: Any) -> Optional[str]:
        n = node.child_by_field_name("name")
        return self._text(n) if n is not None else None

    @staticmethod
    def _span(node: Any) -> Tuple[int, int, int]:
        return (node.start_point[0] + 1, node.end_point[0] + 1,
                node.start_point[1])

    def run(self, root: Any) -> None:
        self._emit_file_chunk(root)
        self._walk(root, [], [], None)

    # -- file chunk --------------------------------------------------------
    def _emit_file_chunk(self, root: Any) -> None:
        loc = len(self.source.splitlines()) or 1
        members = self._collect_members(root)
        self.chunks.append({
            "id": self.filepath,
            "type": "file",
            "name": os.path.basename(self.filepath),
            "file": self.filepath,
            "module_path": self.module_path,
            "code": self._stub(members),
            "start_line": 1,
            "end_line": int(loc),
            "col_offset": 0,
            "meta": {
                "docstring": None,
                "members": members,
                "metrics": {"n_loc": loc},
            },
        })

    def _collect_members(self, root: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Best-effort top-level member summary (mirrors the Python shape)."""
        out: Dict[str, List[Dict[str, Any]]] = {
            "functions": [], "classes": [], "imports": [], "globals": []}
        for child in root.children:
            t = child.type
            if t in self.spec.class_nodes:
                nm = self._name(child)
                if nm:
                    out["classes"].append({"name": nm})
            elif t in self.spec.function_nodes:
                nm = self._name(child)
                if nm:
                    out["functions"].append({"name": nm})
            elif t in ("import_statement", "import"):
                out["imports"].append({"name": self._text(child)})
            elif t in ("lexical_declaration", "variable_declaration"):
                for d in child.children:
                    if d.type != "variable_declarator":
                        continue
                    val = d.child_by_field_name("value")
                    nm = self._name(d)
                    if nm and val is not None and val.type in self.spec.func_value_nodes:
                        out["functions"].append({"name": nm})
        return out

    @staticmethod
    def _stub(members: Dict[str, List[Dict[str, Any]]]) -> str:
        parts: List[str] = []
        for imp in members.get("imports", []):
            parts.append(str(imp.get("name") or ""))
        for cl in members.get("classes", []):
            parts.append(f"class {cl.get('name')}")
        for fn in members.get("functions", []):
            parts.append(f"function {fn.get('name')}")
        return "\n".join(p for p in parts if p)

    # -- traversal ---------------------------------------------------------
    def _walk(self, node: Any, class_stack: List[str],
              func_stack: List[str], cur_fid: Optional[str]) -> None:
        t = node.type
        if t in self.spec.class_nodes:
            name = self._name(node) or "<anon>"
            self._emit_class(node, class_stack, name)
            body = node.child_by_field_name("body")
            children = body.children if body is not None else node.children
            for c in children:
                self._walk(c, class_stack + [name], func_stack, None)
            return
        if t in self.spec.method_nodes:
            name = self._name(node) or "<anon>"
            cls = ".".join(class_stack)
            qual = f"{cls}.{name}" if cls else name
            fid = f"{self.filepath}::method::{qual}"
            self._emit_func(node, fid, name, class_name=cls or None)
            self._walk_body(node, class_stack, func_stack + [name], fid)
            return
        if t in self.spec.function_nodes:
            name = self._name(node) or "<anon>"
            if func_stack:
                fid = (f"{self.filepath}::function::"
                       f"{'.'.join(func_stack)}.{name}")
            else:
                fid = f"{self.filepath}::function::{name}"
            self._emit_func(node, fid, name, class_name=None)
            self._walk_body(node, class_stack, func_stack + [name], fid)
            return
        if t == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in self.spec.func_value_nodes:
                name = self._name(node) or "<anon>"
                if func_stack:
                    fid = (f"{self.filepath}::function::"
                           f"{'.'.join(func_stack)}.{name}")
                else:
                    fid = f"{self.filepath}::function::{name}"
                self._emit_func(value, fid, name, class_name=None,
                                name_override=name)
                self._walk_body(value, class_stack, func_stack + [name], fid)
                return
        if t in self.spec.call_nodes and cur_fid:
            self._emit_call(node, cur_fid)
        for c in node.children:
            self._walk(c, class_stack, func_stack, cur_fid)

    def _walk_body(self, node: Any, class_stack: List[str],
                   func_stack: List[str], cur_fid: str) -> None:
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, class_stack, func_stack, cur_fid)
        else:
            for c in node.children:
                self._walk(c, class_stack, func_stack, cur_fid)

    # -- emit helpers ------------------------------------------------------
    def _emit_class(self, node: Any, class_stack: List[str],
                    name: str) -> None:
        qual = ".".join(class_stack + [name])
        start, end, col = self._span(node)
        bases: List[str] = []
        heritage = None
        for c in node.children:
            if c.type in ("class_heritage", "extends_clause"):
                heritage = c
                break
        if heritage is not None:
            for c in heritage.children:
                if c.type in ("identifier", "type_identifier"):
                    bases.append(self._text(c))
        self.chunks.append({
            "id": f"{self.filepath}::class::{qual}",
            "type": "class",
            "name": name,
            "file": self.filepath,
            "module_path": self.module_path,
            "code": self._text(node),
            "start_line": start,
            "end_line": end,
            "col_offset": col,
            "meta": {
                "decorators": [],
                "bases": bases,
                "keywords": {},
                "docstring": None,
                "doc_parsed": None,
                "is_dataclass": False,
                "dataclass_fields": [],
                "enclosing_class": class_stack[-1] if class_stack else None,
            },
        })

    def _emit_func(self, node: Any, fid: str, name: str, *,
                   class_name: Optional[str],
                   name_override: Optional[str] = None) -> None:
        start, end, col = self._span(node)
        params = node.child_by_field_name("parameters")
        signature = self._text(params) if params is not None else "()"
        child_types = [c.type for c in node.children]
        is_async = "async" in child_types
        is_generator = ("*" in child_types
                        or node.type.startswith("generator"))
        self.chunks.append({
            "id": fid,
            "type": "function",
            "name": name_override or name,
            "file": self.filepath,
            "module_path": self.module_path,
            "code": self._text(node),
            "start_line": start,
            "end_line": end,
            "col_offset": col,
            "meta": {
                "decorators": [],
                "method_kind": self._method_kind(node, name, class_name),
                "is_async": bool(is_async),
                "is_method": class_name is not None,
                "class_name": class_name,
                "signature": signature,
                "parameters": self._params(params),
                "returns_annotation": self._return_type(node),
                "docstring": None,
                "doc_parsed": None,
                "is_generator": bool(is_generator),
                "raises": [],
                "instance_attributes": [],
                "imports_used": {},
                "calls_detailed": [],
                "metrics": {
                    "n_loc": max(1, end - start + 1),
                    "n_params": len(self._params(params)),
                    "n_returns": 0,
                    "n_yields": 0,
                    "n_branches": 0,
                    "n_calls": 0,
                },
            },
        })

    @staticmethod
    def _method_kind(node: Any, name: str,
                     class_name: Optional[str]) -> str:
        if class_name is None:
            return "function"
        types = [c.type for c in node.children]
        if name == "constructor":
            return "constructor"
        if "get" in types:
            return "property"
        if "set" in types:
            return "property_accessor"
        if "static" in types:
            return "staticmethod"
        return "instance"

    def _params(self, params: Any) -> List[Dict[str, Any]]:
        if params is None:
            return []
        out: List[Dict[str, Any]] = []
        for c in params.children:
            if c.type in ("(", ")", ","):
                continue
            nm = c.child_by_field_name("name") if hasattr(c, "child_by_field_name") else None
            if nm is not None:
                out.append({"name": self._text(nm)})
            elif c.type == "identifier":
                out.append({"name": self._text(c)})
            elif c.type == "rest_pattern":
                out.append({"name": self._text(c)})
        return out

    def _return_type(self, node: Any) -> Optional[str]:
        rt = node.child_by_field_name("return_type")
        if rt is not None:
            return self._text(rt)
        for c in node.children:
            if c.type == "type_annotation":
                return self._text(c)
        return None

    def _emit_call(self, node: Any, cur_fid: str) -> None:
        if node.type == "new_expression":
            fn = node.child_by_field_name("constructor")
        else:
            fn = node.child_by_field_name("function")
        if fn is None:
            for c in node.children:
                if c.type in ("identifier", "member_expression"):
                    fn = c
                    break
        if fn is None:
            return
        full, name = self._callee(fn)
        if name:
            self.calls.append({
                "caller_id": cur_fid,
                "callee_name": name,
                "callee_fullname": full,
                "lineno": node.start_point[0] + 1,
            })

    def _callee(self, fn: Any) -> Tuple[Optional[str], Optional[str]]:
        if fn.type == "identifier":
            t = self._text(fn)
            return t, t
        if fn.type == "member_expression":
            prop = fn.child_by_field_name("property")
            obj = fn.child_by_field_name("object")
            name = self._text(prop) if prop is not None else None
            full = None
            if obj is not None and name:
                full = f"{self._text(obj)}.{name}"
            return full or name, name
        text = self._text(fn)
        return (text or None), (text.split(".")[-1] if text else None)


__all__ = ["TreeSitterParser", "treesitter_available", "_get_ts_parser"]
