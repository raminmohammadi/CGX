

# src/cgx/ast/parse_codebase.py
from __future__ import annotations

import ast
import fnmatch
import io
import os
import tokenize
from typing import Any, Dict, Iterable, List, Optional, Tuple
from cgx.parser.module_path import compute_module_path
from cgx.logging_setup import get_logger

logger = get_logger("parser")

# Parser dispatch registry — populated at module import time once the
# concrete parsers below have been defined (see ``_PARSER_REGISTRY``).
# Each key is a lowercase file extension including the leading dot.
_PARSER_REGISTRY: "Dict[str, Any]" = {}

# Defaults: keep the indexer cheap & safe on arbitrary trusted-but-noisy repos.
# Override with the CGX_PARSER_MAX_FILE_BYTES env var or the `max_file_bytes`
# argument to parse_codebase().
DEFAULT_MAX_FILE_BYTES = 1_000_000  # 1 MB

# Directory names that are essentially always noise for code indexing. Kept as
# basename matches so they apply at any depth in the tree.
DEFAULT_IGNORE_DIRS = (
    ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "node_modules",
    "build", "dist", ".eggs", "site-packages",
    ".idea", ".vscode",
)

# Glob patterns (gitignore-style) applied to repo-relative paths.
DEFAULT_IGNORE_GLOBS = (
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib",
    "*.egg-info", "*.egg-info/**",
    ".DS_Store",
)


def _load_gitignore_patterns(project_root: str) -> List[str]:
    """Read .gitignore at the project root and return a normalized pattern list."""
    pats: List[str] = []
    gi = os.path.join(project_root, ".gitignore")
    if not os.path.isfile(gi):
        return pats
    try:
        with open(gi, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue
                if ln.startswith("!"):
                    # Negations are not supported; safer to skip than mis-handle.
                    continue
                pats.append(ln.lstrip("/"))
    except Exception:
        pass
    return pats


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    """fnmatch-style match against repo-relative POSIX paths and basenames."""
    rp = rel_path.replace(os.sep, "/")
    name = rp.rsplit("/", 1)[-1]
    for p in patterns:
        pat = p.rstrip("/").replace(os.sep, "/")
        if not pat:
            continue
        if fnmatch.fnmatch(rp, pat) or fnmatch.fnmatch(name, pat):
            return True
        # Match nested entries when pattern is dir-like (no glob chars).
        if "*" not in pat and "?" not in pat and "[" not in pat:
            if rp == pat or rp.startswith(pat + "/"):
                return True
    return False


# ---------- AST helpers ----------


def _unparse(node: Optional[ast.AST]) -> Optional[str]:
    """Safely unparse an AST node into its string representation."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return getattr(node, "id", None) or getattr(node, "attr", None) or type(node).__name__


def _class_signature(node: ast.ClassDef) -> str:
    """Build a ``class Name(Base1, Base2)`` style signature from a ClassDef."""
    try:
        bases = [_unparse(b) for b in getattr(node, "bases", [])] or []
        base_seg = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{base_seg}"
    except Exception:
        return f"class {getattr(node, 'name', 'Unknown')}"


def _signature_str(args: ast.arguments) -> str:
    """Render an ast.arguments node as a Python-style ``(...)`` signature."""
    parts: List[str] = []
    try:
        po = getattr(args, "posonlyargs", [])
        for a in po:
            seg = a.arg + (f": {_unparse(a.annotation)}" if a.annotation else "")
            parts.append(seg)
        if po:
            parts.append("/")

        reg = list(args.args)
        ndef = len(args.defaults or [])
        for i, a in enumerate(reg):
            ann = f": {_unparse(a.annotation)}" if a.annotation else ""
            if ndef and i >= len(reg) - ndef:
                j = i - (len(reg) - ndef)
                parts.append(f"{a.arg}{ann}={_unparse(args.defaults[j])}")
            else:
                parts.append(f"{a.arg}{ann}")

        if args.vararg:
            a = args.vararg
            parts.append(f"*{a.arg}" + (f": {_unparse(a.annotation)}" if a.annotation else ""))
        elif args.kwonlyargs:
            parts.append("*")

        for a, d in zip(args.kwonlyargs, args.kw_defaults or [None] * len(args.kwonlyargs)):
            seg = a.arg + (f": {_unparse(a.annotation)}" if a.annotation else "")
            if d is not None:
                seg += f"={_unparse(d)}"
            parts.append(seg)

        if args.kwarg:
            a = args.kwarg
            parts.append(f"**{a.arg}" + (f": {_unparse(a.annotation)}" if a.annotation else ""))

        return "(" + ", ".join(parts) + ")"
    except Exception:
        return "()"


def _get_source(source: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        return seg if seg is not None else ""
    except Exception:
        return ""


def _dotted_attr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        left = _dotted_attr(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return _unparse(node)


def _serialize_arg(a: ast.arg) -> Dict[str, Any]:
    return {
        "name": a.arg,
        "annotation": _unparse(getattr(a, "annotation", None)),
        "type_comment": getattr(a, "type_comment", None) if hasattr(a, "type_comment") else None,
    }


def _param_list(args: ast.arguments) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    for a in getattr(args, "posonlyargs", []):
        params.append({**_serialize_arg(a), "kind": "posonly", "default": None})

    reg = list(args.args)
    ndef = len(args.defaults or [])
    for i, a in enumerate(reg):
        default = None
        if ndef and i >= len(reg) - ndef:
            j = i - (len(reg) - ndef)
            default = _unparse(args.defaults[j])
        params.append({**_serialize_arg(a), "kind": "pos_or_kw", "default": default})

    if args.vararg:
        params.append({**_serialize_arg(args.vararg), "kind": "vararg", "default": None})

    kwdefs = args.kw_defaults or []
    for i, a in enumerate(args.kwonlyargs):
        default = _unparse(kwdefs[i]) if i < len(kwdefs) and kwdefs[i] is not None else None
        params.append({**_serialize_arg(a), "kind": "kwonly", "default": default})

    if args.kwarg:
        params.append({**_serialize_arg(args.kwarg), "kind": "kwarg", "default": None})

    return params


def _infer_type(node: Optional[ast.AST]) -> Optional[str]:
    try:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                return "NoneType"
            return type(v).__name__
        if isinstance(node, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(node, (ast.Tuple, ast.GeneratorExp)):
            return "tuple"
        if isinstance(node, (ast.Set, ast.SetComp)):
            return "set"
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(node, ast.Call):
            fn = _dotted_attr(node.func) or _unparse(node.func) or "call"
            return f"{fn}()"
        if isinstance(node, ast.Name):
            return f"Symbol:{node.id}"
        if isinstance(node, ast.Attribute):
            return f"Attr:{_dotted_attr(node)}"
        return type(node).__name__
    except Exception:
        return None


def _value_preview(node: Optional[ast.AST], maxlen: int = 160) -> Optional[str]:
    try:
        if node is None:
            return None
        s = _unparse(node) or ""
        if s and len(s) > maxlen:
            return s[: maxlen - 3] + "..."
        return s or None
    except Exception:
        return None


def _comments_by_line(source: str) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                result.setdefault(tok.start[0], []).append(tok.string)
    except Exception:
        pass
    return result


def _comments_in_span(cmap: Dict[int, List[str]], start: int, end: int) -> List[str]:
    out: List[str] = []
    for ln in range(start, end + 1):
        if ln in cmap:
            out.extend(cmap[ln])
    return out


def _parse_docstring(docstring: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a Python docstring into ``summary/params/returns/yields/raises/notes``."""
    if not docstring:
        return None
    try:
        lines = [ln.rstrip() for ln in docstring.splitlines()]
        sections: Dict[str, Any] = {
            "summary": None, "params": [], "returns": None,
            "yields": None, "raises": [], "notes": [],
        }
        for ln in lines:
            if ln.strip():
                sections["summary"] = ln.strip()
                break
        state: Optional[str] = None
        buf: List[str] = []

        def flush():
            nonlocal buf, state
            text = "\n".join(buf).strip()
            if not text:
                buf = []
                return
            if state in ("Args", "Parameters"):
                for raw in text.splitlines():
                    if not raw.strip():
                        continue
                    name, type_, desc = None, None, raw.strip()
                    if ":" in raw:
                        head, desc = raw.split(":", 1)
                        head = head.strip()
                        desc = desc.strip()
                        if "(" in head and head.endswith(")"):
                            try:
                                name = head[: head.index("(")].strip()
                                type_ = head[head.index("(") + 1 : -1].strip()
                            except Exception:
                                name = head
                        else:
                            name = head
                    sections["params"].append({"name": name or raw.strip(), "type": type_, "desc": desc})
            elif state == "Returns":
                sections["returns"] = text
            elif state == "Yields":
                sections["yields"] = text
            elif state == "Raises":
                for raw in text.splitlines():
                    if not raw.strip():
                        continue
                    sections["raises"].append(raw.strip())
            elif state == "Notes":
                sections["notes"].append(text)
            buf = []

        for ln in lines:
            h = ln.strip()
            if h in ("Args:", "Parameters:", "Returns:", "Yields:", "Raises:", "Notes:"):
                flush()
                state = h[:-1]
                continue
            if state:
                buf.append(ln)
        flush()
        return sections
    except Exception:
        return None


def _collect_top_level_members(tree: ast.AST, source: str) -> Dict[str, List[Dict[str, Any]]]:
    """Collect deterministic summaries of top-level members of a file."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "functions": [], "classes": [], "imports": [], "globals": []
    }
    try:
        for n in getattr(tree, "body", []):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out["functions"].append({
                    "name": n.name,
                    "signature": _signature_str(n.args),
                    "docstring": ast.get_docstring(n),
                })
            elif isinstance(n, ast.ClassDef):
                out["classes"].append({
                    "name": n.name,
                    "signature": _class_signature(n),
                    "docstring": ast.get_docstring(n),
                })
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                src = _get_source(source, n)
                if src:
                    out["imports"].append(src.strip())
            elif isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
                out["globals"].append({
                    "name": n.targets[0].id,
                    "value": _value_preview(n.value),
                    "annotation": None,
                })
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out["globals"].append({
                    "name": n.target.id,
                    "value": _value_preview(n.value),
                    "annotation": _unparse(n.annotation),
                })
    except Exception:
        pass
    return out


def _build_file_code_stub(module_doc: Optional[str], members: Dict[str, List[Dict[str, Any]]]) -> str:
    """Deterministic, compact text summary of a file (docstring + imports + stubs)."""
    parts: List[str] = []
    if module_doc:
        parts.append('"""' + module_doc.replace('"""', r'\"\"\"') + '"""')

    for imp in members.get("imports", []):
        parts.append(imp)

    for g in members.get("globals", []):
        ann = f": {g['annotation']}" if g.get("annotation") else ""
        val = f" = {g['value']}" if g.get("value") else ""
        parts.append(f"{g['name']}{ann}{val}")

    for f in members.get("functions", []):
        parts.append(f"def {f['name']}{f['signature']}: ...")

    for c in members.get("classes", []):
        sig = c["signature"]  # already 'class Name(Base, ...)'
        parts.append(f"{sig}: ...")

    return "\n".join(parts)


def _parse_python_module(
    filepath: str,
    source_code: str,
    project_root: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a single Python source file into chunks and call relations.

    Worker for the ``.py`` parser seam (see
    :class:`cgx.parser.python_parser.PythonASTParser`). The project walker
    in :func:`parse_codebase` reads, prunes and forwards each file's
    in-memory source here. This function is responsible only for the
    per-file slice: it emits the file/module chunk, runs the AST visitor
    that produces class / function / method / lambda chunks, and returns
    ``(code_chunks, call_relations)`` for the single file.

    Cross-file post-processing (call-relation deduplication and reverse-
    edge aggregation onto chunk metadata) lives in :func:`parse_codebase`.

    Returns empty lists if ``source_code`` is unparseable; callers should
    not treat that as an error.
    """

    # Storage for extracted code entities and call relations
    code_chunks: List[Dict[str, Any]] = []
    call_relations: List[Dict[str, Any]] = []

    # ---------- visitor ----------
    class CodeVisitor(ast.NodeVisitor):
        """
        AST visitor that traverses a Python module and extracts structured
        information about classes, functions, methods, lambdas, and their metadata.

        Attributes
        ----------
        filename : str
            Absolute path of the file being parsed.
        module_path : str
            Dotted import path of the file relative to project root.
        source : str
            Raw source code of the file.
        comments_map : dict[int, list[str]]
            Maps line numbers to associated comment strings.
        current_func_id : str | None
            ID of the function currently being visited.
        current_class_name : str | None
            Name of the class currently being visited (for nested class context).
        func_meta : dict[str, dict]
            Collected metadata for each function/method keyed by ID.
        func_index : dict[str, int]
            Index of function/method chunks inside `code_chunks`.
        import_alias : dict[str, str]
            Mapping from local import alias → fully qualified name.
        star_imports : list[str]
            List of modules imported with `from x import *`.
        """
        def __init__(self, filename: str, module_path: str, source: str):
            """
            Initialize the visitor with file path, module path, and source code.
            """
            super().__init__()
            self.filename = filename
            self.module_path = module_path
            self.source = source
            self.comments_map = _comments_by_line(source)
            self.current_func_id: Optional[str] = None
            self.current_class_name: Optional[str] = None
            self.func_meta: Dict[str, Dict[str, Any]] = {}
            self.func_index: Dict[str, int] = {}
            self.import_alias: Dict[str, str] = {}
            self.star_imports: List[str] = []
            # Stack of enclosing function names (non-method only) so nested
            # functions get a qualified ID like "outer.inner" instead of just
            # "inner", preventing duplicate chunk IDs when identically-named
            # helpers are defined inside multiple different test functions.
            self._func_name_stack: List[str] = []

        # -------- imports --------
        def visit_Import(self, node: ast.Import):
            """
            Capture `import x [as y]` statements.

            Updates `import_alias` to map alias → original module.
            """
            for alias in node.names:
                asname = alias.asname or alias.name
                self.import_alias[asname] = alias.name
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom):
            """
            Capture `from x import y [as z]` and star-imports.

            Updates `import_alias` and `star_imports`.
            """
            mod = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    self.star_imports.append(mod)
                else:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    asname = alias.asname or alias.name
                    self.import_alias[asname] = full
            self.generic_visit(node)

        # -------- classes --------
        def visit_ClassDef(self, node: ast.ClassDef):
            """
            Visit a class definition and emit a class chunk.

            - Captures class decorators, bases, keywords, docstring.
            - Marks dataclass fields if decorated with @dataclass.
            - Tracks enclosing class context for nested classes.
            - Visits all methods and nested classes recursively.
            """
            class_id = f"{self.filename}::class::{node.name}"
            # Get source code segment of the source that generated node.
            class_code = _get_source(self.source, node) 
            # unparse an AST node decorator_list into its string representations.
            decorators = [_unparse(d) for d in node.decorator_list]

            # Capture enclosing class deterministically (for nested classes)
            enclosing = self.current_class_name

            meta: Dict[str, Any] = {
                "decorators": decorators,
                "bases": [_unparse(b) for b in node.bases],
                "keywords": {(kw.arg or ""): _unparse(kw.value) for kw in getattr(node, "keywords", [])}
                if getattr(node, "keywords", None)
                else {},
                "docstring": ast.get_docstring(node),
                "doc_parsed": _parse_docstring(ast.get_docstring(node)), #Parse a Python docstring into structured sections.
                "is_dataclass": any("dataclass" in (d or "") for d in decorators),
                "dataclass_fields": [],  # filled if dataclass with AnnAssigns
                "enclosing_class": enclosing,  # ★ NEW
            }
            if meta["is_dataclass"]:
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        meta["dataclass_fields"].append(
                            {
                                "name": stmt.target.id,
                                "annotation": _unparse(stmt.annotation),
                                "default": _value_preview(stmt.value),
                            }
                        )

            idx = len(code_chunks)
            code_chunks.append(
                {
                    "id": class_id,
                    "type": "class",
                    "name": node.name,
                    "file": self.filename,
                    "module_path": self.module_path,  # ★ NEW
                    "code": class_code,
                    "start_line": int(getattr(node, "lineno", 0) or 0),
                    "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                    "col_offset": int(getattr(node, "col_offset", 0) or 0),
                    "meta": meta,
                }
            )

            prev_class = self.current_class_name
            self.current_class_name = node.name

            # methods & nested
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._visit_function_like(child, is_method=True)
                elif isinstance(child, ast.ClassDef):
                    # Nested class
                    self.visit(child)
                else:
                    self.visit(child)

            self.current_class_name = prev_class

        # -------- functions --------
        def visit_FunctionDef(self, node: ast.FunctionDef):
            """
            Visit a synchronous function definition (not a method).
            """
            self._visit_function_like(node, is_method=False)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            """
            Visit an asynchronous function definition (not a method).
            """
            self._visit_function_like(node, is_method=False, is_async=True)

        def _detect_method_kind(self, decorators: List[str]) -> str:
            """
            Infer whether a function inside a class is a property,
            staticmethod, classmethod, or instance method.
            """
            if any(d.endswith(".setter") or d.endswith(".deleter") for d in decorators):
                return "property_accessor"
            if any(d.endswith(".getter") or d == "property" for d in decorators):
                return "property"
            if any(d.endswith("staticmethod") or d == "staticmethod" for d in decorators):
                return "staticmethod"
            if any(d.endswith("classmethod") or d == "classmethod" for d in decorators):
                return "classmethod"
            return "instance"

        def _visit_function_like(self, node, is_method: bool, is_async: bool = False):
            """
            Common handler for FunctionDef and AsyncFunctionDef.

            - Builds function/method ID, code, and metadata.
            - Captures signature, decorators, parameters, return annotation,
            docstring, and comments.
            - Initializes per-function metadata in `func_meta`.
            - Visits body recursively and finalizes metrics.
            """
            effective_is_method = is_method or (self.current_class_name is not None)
            qual = (
                f"{self.current_class_name}.{node.name}"
                if effective_is_method and self.current_class_name
                else node.name
            )
            if effective_is_method:
                func_id = f"{self.filename}::method::{qual}"
            elif self._func_name_stack:
                # Nested function inside another function: qualify with the
                # enclosing function path so identically-named helpers in
                # different test functions get distinct chunk IDs.
                func_id = f"{self.filename}::function::{'.'.join(self._func_name_stack)}.{node.name}"
            else:
                func_id = f"{self.filename}::function::{node.name}"

            func_code = _get_source(self.source, node)
            decorators = [_unparse(d) for d in node.decorator_list]
            method_kind = self._detect_method_kind(decorators)
            doc = ast.get_docstring(node)
            doc_parsed = _parse_docstring(doc)

            meta: Dict[str, Any] = {
                "decorators": decorators,
                "method_kind": method_kind,
                "is_async": bool(is_async or isinstance(node, ast.AsyncFunctionDef)),
                "is_method": bool(effective_is_method),
                "class_name": self.current_class_name if effective_is_method else None,
                "signature": _signature_str(node.args),
                "parameters": _param_list(node.args),
                "args_struct": {
                    "posonlyargs": [_serialize_arg(x) for x in getattr(node.args, "posonlyargs", [])],
                    "args": [_serialize_arg(x) for x in node.args.args],
                    "vararg": _serialize_arg(node.args.vararg) if node.args.vararg else None,
                    "kwonlyargs": [_serialize_arg(x) for x in node.args.kwonlyargs],
                    "kw_defaults": [_unparse(x) for x in node.args.kw_defaults] if node.args.kw_defaults else [],
                    "kwarg": _serialize_arg(node.args.kwarg) if node.args.kwarg else None,
                    "defaults": [_unparse(x) for x in node.args.defaults] if node.args.defaults else [],
                },
                "returns_annotation": _unparse(getattr(node, "returns", None))
                if getattr(node, "returns", None)
                else None,
                "type_comment": getattr(node, "type_comment", None),
                "docstring": doc,
                "doc_parsed": doc_parsed,
                "is_generator": False,
                "return_values": [],
                "yield_values": [],
                "raises": [],
                "exceptions_handled": [],
                "attributes_used": set(),
                "names_used": set(),
                "imports_used": {},
                "lambda_ids": [],
                "calls_detailed": [],
                "local_vars": [],
                "instance_attributes": [],
                "metrics": {
                    "n_loc": (getattr(node, "end_lineno", node.lineno) - node.lineno + 1),
                    "n_params": len(_param_list(node.args)),
                    "n_returns": 0,
                    "n_yields": 0,
                    "n_branches": 0,
                    "n_calls": 0,
                },
                "comments": _comments_in_span(
                    self.comments_map,
                    getattr(node, "lineno", 0),
                    getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                ),
            }

            idx = len(code_chunks)
            code_chunks.append(
                {
                    "id": func_id,
                    "type": "function",
                    "name": node.name,
                    "file": self.filename,
                    "module_path": self.module_path,  # ★ NEW
                    "code": func_code,
                    "start_line": int(getattr(node, "lineno", 0) or 0),
                    "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                    "col_offset": int(getattr(node, "col_offset", 0) or 0),
                    "meta": {},  # fill after visit
                }
            )

            self.func_index[func_id] = idx
            self.func_meta[func_id] = meta

            if not effective_is_method:
                self._func_name_stack.append(node.name)
            prev_func = self.current_func_id
            self.current_func_id = func_id
            try:
                self.generic_visit(node)
            finally:
                self.current_func_id = prev_func
                if not effective_is_method:
                    self._func_name_stack.pop()

            # finalize meta
            meta["attributes_used"] = sorted(meta["attributes_used"])
            meta["names_used"] = sorted(meta["names_used"])

            # imports used: intersect names/attributes with aliases
            used: Dict[str, str] = {}
            base_candidates = set(a.split(".", 1)[0] for a in meta["attributes_used"]) | set(meta["names_used"])
            for alias, full in self.import_alias.items():
                if alias in base_candidates:
                    used[alias] = full
            meta["imports_used"] = used
            meta["metrics"]["n_calls"] = len(meta["calls_detailed"])

            code_chunks[idx]["meta"] = meta

        # -------- lambdas --------
        def visit_Lambda(self, node: ast.Lambda):
            """
            Visit a lambda expression.

            - Emits a lambda chunk with synthetic ID.
            - Captures arguments, body, enclosing function, and comments.
            - Links lambda ID into parent function metadata.
            """
            name = f"lambda@L{getattr(node, 'lineno', 0)}c{getattr(node, 'col_offset', 0)}"
            lam_id = f"{self.filename}::lambda::L{getattr(node, 'lineno', 0)}c{getattr(node, 'col_offset', 0)}"
            meta = {
                "args": _param_list(node.args),
                "body": _unparse(node.body),
                "enclosing": self.current_func_id,
                "comments": _comments_in_span(
                    self.comments_map,
                    getattr(node, "lineno", 0),
                    getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                ),
            }
            code_chunks.append(
                {
                    "id": lam_id,
                    "type": "lambda",
                    "name": name,
                    "file": self.filename,
                    "module_path": self.module_path,  # ★ NEW
                    "code": _get_source(self.source, node),
                    "start_line": int(getattr(node, "lineno", 0) or 0),
                    "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                    "col_offset": int(getattr(node, "col_offset", 0) or 0),
                    "meta": meta,
                }
            )
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["lambda_ids"].append(lam_id)
            self.generic_visit(node)

        # -------- returns / yields / exceptions --------
        def visit_Return(self, node: ast.Return):
            """
            Record return expressions inside current function and increment metrics.
            """

            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["return_values"].append(_unparse(node.value))
                self.func_meta[self.current_func_id]["metrics"]["n_returns"] += 1
            self.generic_visit(node)

        def visit_Yield(self, node: ast.Yield):
            """
            Record yield expressions and mark current function as a generator.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["is_generator"] = True
                self.func_meta[self.current_func_id]["yield_values"].append(_unparse(node.value))
                self.func_meta[self.current_func_id]["metrics"]["n_yields"] += 1
            self.generic_visit(node)

        def visit_YieldFrom(self, node: ast.YieldFrom):
            """
            Record yield-from expressions and mark current function as a generator.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["is_generator"] = True
                val = f"from { _unparse(node.value) }"
                self.func_meta[self.current_func_id]["yield_values"].append(val)
                self.func_meta[self.current_func_id]["metrics"]["n_yields"] += 1
            self.generic_visit(node)

        def visit_Raise(self, node: ast.Raise):
            """
            Record raise statements inside current function.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["raises"].append(_unparse(node.exc))
            self.generic_visit(node)

        def visit_Try(self, node: ast.Try):
            """
            Record try/except blocks, exceptions handled, and branch metric.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["metrics"]["n_branches"] += 1
                handled = []
                for h in node.handlers:
                    handled.append(_unparse(h.type) or "Exception")
                self.func_meta[self.current_func_id]["exceptions_handled"].extend(handled)
            self.generic_visit(node)

        # -------- control flow metrics --------
        def visit_If(self, node: ast.If):
            """
            Increment branch count metric for if-statements.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["metrics"]["n_branches"] += 1
            self.generic_visit(node)

        def visit_For(self, node: ast.For):
            """
            Increment branch count metric for for-loops.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["metrics"]["n_branches"] += 1
            self.generic_visit(node)

        def visit_While(self, node: ast.While):
            """
            Increment branch count metric for while-loops.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["metrics"]["n_branches"] += 1
            self.generic_visit(node)

        def visit_With(self, node: ast.With):
            """
            Increment branch count metric for with-statements.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["metrics"]["n_branches"] += 1
            self.generic_visit(node)

        # -------- variable & attribute tracking --------
        def _record_local(
            self,
            name: str,
            annotation: Optional[str],
            value: Optional[ast.AST],
            type_comment: Optional[str],
            lineno: int,
        ):
            """
            Record assignment to a local variable in current function.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["local_vars"].append(
                    {
                        "name": name,
                        "annotation": annotation,
                        "inferred_type": _infer_type(value),
                        "value_preview": _value_preview(value),
                        "type_comment": type_comment,
                        "lineno": lineno,
                    }
                )

        def _record_instance_attr(
            self,
            name: str,
            annotation: Optional[str],
            value: Optional[ast.AST],
            lineno: int,
            source: Optional[str] = None,
        ):
            """
            Record assignment to self.<attr> inside a method (instance attribute).
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["instance_attributes"].append(
                    {
                        "name": name,
                        "source": source,
                        "annotation": annotation,
                        "inferred_type": _infer_type(value),
                        "value_preview": _value_preview(value),
                        "lineno": lineno,
                    }
                )

        def visit_Assign(self, node: ast.Assign):
            """
            Visit assignment statements.
            - Distinguishes between instance attributes, locals, and tuple/list targets.
            """
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    src = None
                    if isinstance(node.value, ast.Name):
                        src = f"param: {node.value.id}"
                    self._record_instance_attr(
                        tgt.attr, None, node.value, getattr(node, "lineno", 0), source=src
                    )
                elif isinstance(tgt, ast.Name):
                    self._record_local(
                        tgt.id, None, node.value, getattr(node, "type_comment", None), getattr(node, "lineno", 0)
                    )
                elif isinstance(tgt, (ast.Tuple, ast.List)):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            self._record_local(
                                elt.id, None, None, getattr(node, "type_comment", None), getattr(node, "lineno", 0)
                            )
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            """
            Visit annotated assignment statements.
            - Records instance attributes or locals with type annotations.
            """
            ann = _unparse(node.annotation)
            if isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
                self._record_instance_attr(node.target.attr, ann, node.value, getattr(node, "lineno", 0))
            elif isinstance(node.target, ast.Name):
                self._record_local(
                    node.target.id, ann, node.value, getattr(node, "type_comment", None), getattr(node, "lineno", 0)
                )
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign):
            """
            Visit augmented assignments (+=, -=, etc.).
            - Records updates to locals or instance attributes.
            """
            tgt = node.target
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                self._record_instance_attr(tgt.attr, None, None, getattr(node, "lineno", 0))
            elif isinstance(tgt, ast.Name):
                self._record_local(tgt.id, None, None, None, getattr(node, "lineno", 0))
            self.generic_visit(node)

        # -------- names / attributes --------
        def visit_Attribute(self, node: ast.Attribute):
            """
            Record attribute usage inside current function.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                dotted = _dotted_attr(node)
                if dotted:
                    self.func_meta[self.current_func_id]["attributes_used"].add(dotted)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            """
            Record variable name usage inside current function.
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                self.func_meta[self.current_func_id]["names_used"].add(node.id)
            self.generic_visit(node)

        # -------- calls --------
        def visit_Call(self, node: ast.Call):
            """
            Visit function/method calls.

            - Records call site details (callee, args, kwargs, lineno).
            - Adds call relation entry to global `call_relations`.
            - Updates function metadata (`calls_detailed`, metrics).
            """
            if self.current_func_id and self.current_func_id in self.func_meta:
                callee_full = _dotted_attr(node.func) or None
                if isinstance(node.func, ast.Name):
                    callee_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee_name = node.func.attr
                elif callee_full:
                    callee_name = callee_full.split(".")[-1]
                else:
                    callee_name = None

                has_starargs = any(isinstance(a, ast.Starred) for a in node.args)
                has_kwargs = any(kw.arg is None for kw in node.keywords)

                self.func_meta[self.current_func_id]["calls_detailed"].append(
                    {
                        "callee_fullname": callee_full,
                        "callee_name": callee_name,
                        "args": [_unparse(a) for a in node.args],
                        "keywords": {(kw.arg if kw.arg is not None else "**"): _unparse(kw.value) for kw in node.keywords},
                        "has_starargs": has_starargs,
                        "has_kwargs": has_kwargs,
                        "lineno": getattr(node, "lineno", None),
                    }
                )

                if callee_name:
                    call_relations.append(
                        {
                            "caller_id": self.current_func_id,
                            "callee_name": callee_name,
                            "callee_fullname": callee_full,
                            "lineno": getattr(node, "lineno", None),
                        }
                    )
            self.generic_visit(node)

    # ---------- parse + emit file chunk + run visitor ----------
    try:
        tree = ast.parse(source_code, filename=filepath)
    except SyntaxError:
        return code_chunks, call_relations
    except Exception as e:
        logger.warning("Failed to parse %s: %s", filepath, e)
        return code_chunks, call_relations

    module_path = compute_module_path(project_root, filepath)

    # ---- file/module chunk ----
    try:
        module_doc = ast.get_docstring(tree)
    except Exception:
        module_doc = None
    members = _collect_top_level_members(tree, source_code)
    file_code_stub = _build_file_code_stub(module_doc, members)
    try:
        _file_loc = len(source_code.splitlines())
        code_chunks.append({
            "id": filepath,
            "type": "file",
            "name": os.path.basename(filepath),
            "file": filepath,
            "module_path": module_path,
            "code": file_code_stub,
            "start_line": 1,
            "end_line": int(_file_loc),
            "col_offset": 0,
            "meta": {
                "docstring": module_doc,
                "members": members,
                "metrics": {
                    "n_loc": _file_loc
                }
            }
        })
    except Exception as e:
        logger.warning("Failed to emit file chunk for %s: %s", filepath, e)

    # ---- class/function/method/lambda extraction ----
    visitor = CodeVisitor(filepath, module_path, source_code)
    try:
        visitor.visit(tree)
    except Exception as e:
        logger.error("AST visit failed for %s: %s", filepath, e)

    return code_chunks, call_relations


def parse_codebase(
    project_root: str,
    *,
    ignore_patterns: Optional[List[str]] = None,
    max_file_bytes: Optional[int] = None,
    follow_symlinks: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse and analyze an entire codebase into structured entities and call relations.

    Recursively traverses ``project_root``, dispatching each file to the
    parser registered for its extension in :data:`_PARSER_REGISTRY` (today
    only ``.py`` via :class:`PythonASTParser`). Per-file results are
    concatenated and the cross-file post-processing — call-relation
    deduplication and reverse-edge aggregation onto chunk metadata —
    runs at the end.

    Parameters
    ----------
    project_root:
        Repository root to walk.
    ignore_patterns:
        Extra gitignore-style globs (applied alongside the project's own
        ``.gitignore`` and ``DEFAULT_IGNORE_GLOBS``).
    max_file_bytes:
        Soft cap on file size; files above the cap are skipped with a
        warning. Defaults to the ``CGX_PARSER_MAX_FILE_BYTES`` env var
        (or ``DEFAULT_MAX_FILE_BYTES`` if unset).
    follow_symlinks:
        Whether ``os.walk`` should descend into symlinked directories
        and whether symlinked files should be parsed.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(code_chunks, deduped_call_relations)`` with the per-chunk
        shape pinned by :mod:`cgx.parser.schema` and the schema version
        tracked by ``cgx.embeddings.records.SCHEMA_VERSION``.
    """
    code_chunks: List[Dict[str, Any]] = []
    call_relations: List[Dict[str, Any]] = []

    # Resolve safety knobs (args > env > defaults).
    if max_file_bytes is None:
        try:
            max_file_bytes = int(os.environ.get("CGX_PARSER_MAX_FILE_BYTES", "") or DEFAULT_MAX_FILE_BYTES)
        except Exception:
            max_file_bytes = DEFAULT_MAX_FILE_BYTES
    user_globs = list(ignore_patterns or [])
    gitignore_globs = _load_gitignore_patterns(project_root)
    all_globs = list(DEFAULT_IGNORE_GLOBS) + gitignore_globs + user_globs
    abs_root = os.path.abspath(project_root)

    def _rel(p: str) -> str:
        try:
            return os.path.relpath(p, abs_root)
        except Exception:
            return p

    for root, dirs, files in os.walk(project_root, followlinks=follow_symlinks):
        # Prune ignored directories in-place to avoid descending into them.
        pruned: List[str] = []
        for d in list(dirs):
            if d in DEFAULT_IGNORE_DIRS:
                continue
            rel_d = _rel(os.path.join(root, d))
            if _matches_any(rel_d, all_globs):
                continue
            pruned.append(d)
        dirs[:] = pruned

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            parser = _PARSER_REGISTRY.get(ext)
            if parser is None:
                continue
            filepath = os.path.join(root, fname)
            rel_fp = _rel(filepath)
            if _matches_any(rel_fp, all_globs):
                continue
            try:
                if not follow_symlinks and os.path.islink(filepath):
                    continue
                st = os.stat(filepath)
                if max_file_bytes and st.st_size > max_file_bytes:
                    logger.warning(
                        "Skipping %s: size %d bytes exceeds max_file_bytes=%d",
                        rel_fp, st.st_size, max_file_bytes,
                    )
                    continue
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()
            except Exception as e:
                logger.warning("Skipping unreadable file %s: %s", filepath, e)
                continue

            try:
                file_chunks, file_calls = parser.parse_file(filepath, source_code, project_root)
            except Exception as e:
                logger.error("Parser failed for %s: %s", filepath, e)
                continue
            code_chunks.extend(file_chunks)
            call_relations.extend(file_calls)

    # Deduplicate call relations
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for cr in call_relations:
        key = (cr.get("caller_id"), cr.get("callee_name"), cr.get("callee_fullname"), cr.get("lineno"))
        if key not in seen:
            seen.add(key)
            deduped.append(cr)

    # ---- compute reverse edges & topK calls_out ----
    calls_out_map: Dict[str, List[str]] = {}
    calls_in_count: Dict[str, int] = {}

    for cr in deduped:
        caller = cr["caller_id"]
        callee = cr.get("callee_fullname") or cr.get("callee_name")
        if not callee:
            continue
        calls_out_map.setdefault(caller, []).append(callee)
        calls_in_count[callee] = calls_in_count.get(callee, 0) + 1

    # attach to function/method chunks
    for ch in code_chunks:
        if ch["type"] in ("function", "lambda"):
            cid = ch["id"]
            meta = ch.get("meta", {})
            calls_out = sorted(calls_out_map.get(cid, []))
            meta["calls_out_top"] = calls_out[:10]
            meta["called_by_count"] = calls_in_count.get(cid, 0)
            ch["meta"] = meta

    return code_chunks, deduped


# Register concrete parsers. Imported here (rather than at module top)
# so the registry is populated lazily and cannot create an import cycle
# with cgx.parser.python_parser, which imports back from this module.
def _register_default_parsers() -> None:
    from cgx.parser.python_parser import PythonASTParser

    inst = PythonASTParser()
    for ext in inst.extensions:
        _PARSER_REGISTRY[ext] = inst


_register_default_parsers()
