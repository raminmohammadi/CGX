

"""AST-anchored code insertion.

Bridges :func:`cgx.retrieval.orchestrator.suggest_insertion_points` (which
yields a ``container_id`` plus optional sibling anchors) with the existing
:class:`cgx.codegen.diff_apply.PatchResult` shape consumed by the rest of
the codegen pipeline (validate / disk_apply / test_runner).

Given a target container (file or class) and a Python code snippet, the
planner re-parses the target file with the stdlib :mod:`ast` module,
locates the requested anchor sibling (a top-level ``FunctionDef`` /
``ClassDef`` for a file container, or a child method for a class
container), and splices the snippet into the right line range with
indentation matched to the container. The output is a ``PatchResult``
whose ``new_content`` can flow directly through
:func:`cgx.codegen.validate.validate_patch_results`,
:func:`cgx.codegen.disk_apply.apply_diffs_to_disk` (via
:func:`build_unified_diff`), or :func:`cgx.codegen.pipeline.validate_and_test`.

This module is intentionally additive: it does not modify any existing
function signature, never writes to disk, and only depends on the
standard library so it can run in every CGX deployment.
"""

from __future__ import annotations

import ast
import difflib
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from cgx.codegen.diff_apply import PatchResult

logger = logging.getLogger(__name__)


# Container ids emitted by suggest_insertion_points carry the parser's
# absolute filename. Class containers have a "::class::<Name>" suffix.
_CLASS_SEP = "::class::"

# Sibling anchor chunk ids are of the form
#   "<abs_path>::function::<qual>" or "<abs_path>::method::<qual>".
# The qual is either "name" for top-level or "ClassName.name" for methods.
_FUNC_SEP = "::function::"
_METHOD_SEP = "::method::"


@dataclass
class AstInsertSpec:
    """Declarative description of an AST-anchored insertion.

    Attributes
    ----------
    rel_path:
        Project-relative POSIX path of the file to modify. The file does
        not need to exist yet — when absent the spec is treated as a
        new-file create, mirroring :func:`apply_diffs_in_memory`'s
        ``is_new_file`` behaviour.
    code:
        Python source snippet to insert. May contain multiple top-level
        defs; whitespace is :func:`textwrap.dedent`-normalised before
        re-indenting for the destination container.
    class_name:
        When set, the insertion targets the named ``ClassDef`` inside
        ``rel_path``. Otherwise the insertion is at module level.
    anchor_symbol:
        Optional sibling name (``function`` / ``method`` / ``class``)
        after which to splice. For class containers this is matched
        against the class body's direct children. When the anchor cannot
        be resolved, behaviour falls back to ``append`` placement.
    dedupe:
        When True (default), top-level defs from ``code`` whose ``name``
        already exists in the target container are skipped silently.
    anchor_loc:
        Optional line-anchor metadata carried by the v3 record schema
        (``{"start_line", "end_line", "indent_col"}``) describing the
        sibling pointed to by ``anchor_symbol``. When present and the
        ``end_line`` falls inside the target file, the planner uses it
        directly to compute the splice position and indent — avoiding a
        second AST walk for the anchor. Otherwise the AST-walk fallback
        in :func:`_module_insert_after_line` /
        :func:`_class_insert_after_line` is used.
    """

    rel_path: str
    code: str
    class_name: Optional[str] = None
    anchor_symbol: Optional[str] = None
    dedupe: bool = True
    anchor_loc: Optional[dict] = None


@dataclass
class _ParsedSnippet:
    """Internal: top-level defs extracted from the user-supplied snippet."""

    defs: List[ast.stmt] = field(default_factory=list)
    other: List[ast.stmt] = field(default_factory=list)

    @property
    def top_level_names(self) -> List[str]:
        out: List[str] = []
        for n in self.defs:
            name = getattr(n, "name", None)
            if isinstance(name, str):
                out.append(name)
        return out


def _read_text(abs_path: Path) -> Optional[str]:
    try:
        return abs_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("ast_insert: failed to read %s: %s", abs_path, exc)
        return None


def _parse_snippet(
    code: str,
) -> Tuple[Optional[_ParsedSnippet], Optional[str], str]:
    """Dedent and ``ast.parse`` the snippet.

    Returns ``(parsed, error_str, dedented_source)``. ``dedented_source``
    is the exact text that was parsed and is later sliced via
    :func:`ast.get_source_segment` so the user's comments and blank
    lines survive the splice intact.
    """
    if not code or not code.strip():
        return None, "ast_insert: empty snippet", ""
    dedented = textwrap.dedent(code).strip("\n")
    try:
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        return None, f"ast_insert: snippet SyntaxError: {exc}", dedented
    parsed = _ParsedSnippet()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parsed.defs.append(node)
        else:
            parsed.other.append(node)
    return parsed, None, dedented


def _is_def_node(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _name_of(node: ast.AST) -> Optional[str]:
    return getattr(node, "name", None) if _is_def_node(node) else None



def _existing_names_in_module(tree: ast.Module) -> List[str]:
    return [n for n in (_name_of(b) for b in tree.body) if n]


def _existing_names_in_class(cls: ast.ClassDef) -> List[str]:
    return [n for n in (_name_of(b) for b in cls.body) if n]


def _find_class(tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    """Locate a top-level class by name. Nested classes are not targeted."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _detect_body_indent(body: List[ast.stmt], default: int = 4) -> int:
    """Return the column offset shared by the first statement in *body*."""
    for stmt in body:
        col = getattr(stmt, "col_offset", None)
        if isinstance(col, int) and col > 0:
            return col
    return default


def _splice_lines(
    source: str,
    insert_after_line: int,
    new_block: str,
    *,
    blank_lines_before: int = 1,
) -> str:
    """Insert *new_block* after the 1-indexed line ``insert_after_line``.

    ``insert_after_line == 0`` means "prepend at the very top".
    The block is wrapped with blank-line padding so the result reads
    cleanly when the anchor is mid-file. Preserves the source's trailing
    newline policy.
    """
    had_final_newline = source.endswith("\n")
    lines = source.splitlines()
    block_lines = new_block.splitlines()
    if insert_after_line > 0:
        block_lines = [""] * blank_lines_before + block_lines
    if insert_after_line < len(lines):
        block_lines = block_lines + [""]
    before = lines[:insert_after_line]
    after = lines[insert_after_line:]
    out = before + block_lines + after
    text = "\n".join(out)
    if had_final_newline and not text.endswith("\n"):
        text += "\n"
    return text


def _indent_block(text: str, indent: int) -> str:
    """Indent every non-empty line of *text* by *indent* spaces."""
    prefix = " " * indent
    out_lines: List[str] = []
    for line in text.splitlines():
        if line:
            out_lines.append(prefix + line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _filter_dupes(
    parsed: _ParsedSnippet, existing_names: List[str]
) -> Tuple[_ParsedSnippet, List[str]]:
    """Drop snippet defs whose name already exists. Returns (filtered, dropped)."""
    existing = set(existing_names)
    keep_defs: List[ast.stmt] = []
    dropped: List[str] = []
    for node in parsed.defs:
        name = _name_of(node)
        if name and name in existing:
            dropped.append(name)
        else:
            keep_defs.append(node)
    return _ParsedSnippet(defs=keep_defs, other=list(parsed.other)), dropped


def _leading_comments(source_lines: List[str], lineno: int) -> str:
    """Return a contiguous block of ``# ...`` lines immediately above ``lineno``.

    ``lineno`` is 1-indexed (matching :mod:`ast`). The walk stops at the
    first blank or non-comment line. Returned text has no trailing newline.
    """
    out: List[str] = []
    i = lineno - 2
    while i >= 0:
        stripped = source_lines[i].lstrip()
        if stripped.startswith("#"):
            out.append(source_lines[i])
            i -= 1
            continue
        break
    out.reverse()
    return "\n".join(out)


def _snippet_to_text(parsed: _ParsedSnippet, dedented_source: str = "") -> str:
    """Re-emit the kept top-level defs (and any preamble statements).

    Uses :func:`ast.get_source_segment` against ``dedented_source`` when
    available so comments and exact formatting are preserved. Falls back
    to :func:`ast.unparse` for nodes whose source segment cannot be
    recovered (e.g. synthetically-built nodes). Leading ``#`` comment
    blocks directly above a def are stitched back in since they are not
    part of the AST node itself.
    """
    pieces: List[str] = []
    source_lines = dedented_source.splitlines() if dedented_source else []

    def _segment_or_unparse(stmt: ast.stmt) -> str:
        if dedented_source:
            try:
                seg = ast.get_source_segment(dedented_source, stmt)
            except Exception:
                seg = None
            if seg:
                start = int(getattr(stmt, "lineno", 0) or 0)
                # Decorators are part of FunctionDef/ClassDef but ``lineno``
                # already points at the first decorator line, so leading
                # comments above the decorator chain are still outside.
                lead = _leading_comments(source_lines, start) if start else ""
                return f"{lead}\n{seg}" if lead else seg
        return ast.unparse(stmt)

    for stmt in parsed.other:
        pieces.append(_segment_or_unparse(stmt))
    for stmt in parsed.defs:
        pieces.append(_segment_or_unparse(stmt))
    return "\n\n".join(p.rstrip() for p in pieces if p.strip())



def _module_insert_after_line(
    tree: ast.Module, anchor_symbol: Optional[str]
) -> int:
    """Return the 1-indexed line after which a module-level insertion sits.

    When ``anchor_symbol`` matches a top-level def, returns its
    ``end_lineno``. Otherwise returns the position past the last
    statement (i.e. end-of-file equivalent).
    """
    if anchor_symbol:
        for node in tree.body:
            if _is_def_node(node) and _name_of(node) == anchor_symbol:
                end = getattr(node, "end_lineno", None)
                if isinstance(end, int):
                    return end
    if tree.body:
        last = tree.body[-1]
        end = getattr(last, "end_lineno", None)
        if isinstance(end, int):
            return end
    return 0


def _class_insert_after_line(
    cls: ast.ClassDef, anchor_symbol: Optional[str]
) -> Tuple[int, bool]:
    """Return (insert_after_line, is_inside_class_body).

    For a class with an empty body (``class Foo: ...``) we still return a
    valid line: the class' own ``end_lineno``. ``is_inside_class_body``
    is True when the splice point is between body members and False when
    appending at end-of-class — relevant for whether the trailing blank
    line is needed.
    """
    if anchor_symbol:
        for node in cls.body:
            if _is_def_node(node) and _name_of(node) == anchor_symbol:
                end = getattr(node, "end_lineno", None)
                if isinstance(end, int):
                    return end, True
    if cls.body:
        last = cls.body[-1]
        end = getattr(last, "end_lineno", None)
        if isinstance(end, int):
            return end, False
    # Empty class body: insert right after the class header line.
    return int(getattr(cls, "lineno", 1)), False


def _make_new_file_result(rel_path: str, code: str) -> PatchResult:
    """Materialize a new-file PatchResult from a raw snippet."""
    parsed, err, dedented = _parse_snippet(code)
    if err is not None:
        return PatchResult(path=rel_path, ok=False, is_new_file=True, error=err)
    text = _snippet_to_text(parsed, dedented) if parsed else code.strip()
    if not text.endswith("\n"):
        text += "\n"
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return PatchResult(
            path=rel_path, ok=False, is_new_file=True,
            error=f"ast_insert: new-file snippet failed reparse: {exc}",
        )
    return PatchResult(
        path=rel_path, ok=True, new_content=text,
        original_content=None, is_new_file=True,
    )


def plan_ast_insertion(
    project_root: str,
    spec: AstInsertSpec,
    *,
    file_text: Optional[str] = None,
) -> PatchResult:
    """Plan an AST-anchored insertion and return a :class:`PatchResult`.

    The function is read-only with respect to the filesystem: it reads
    ``project_root / spec.rel_path`` (or accepts in-memory ``file_text``)
    and returns a ``PatchResult`` whose ``new_content`` is the patched
    file. Pass the result to :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`
    (via :func:`build_unified_diff`) or to
    :func:`cgx.codegen.validate.validate_patch_results` to plug into the
    rest of the codegen pipeline.

    Non-Python paths are rejected with ``ok=False`` so callers can fall
    back to a diff-based path; this mirrors the rest of the validate
    layer which only understands Python / JSON / YAML.
    """
    rel = spec.rel_path.replace("\\", "/").lstrip("/")
    if not rel.endswith(".py"):
        return PatchResult(
            path=rel, ok=False,
            error="ast_insert: not a Python file (only .py supported)",
        )
    abs_path = Path(project_root) / rel
    original = file_text if file_text is not None else _read_text(abs_path)
    if original is None:
        # New file path: still validate the snippet via reparse.
        if spec.class_name:
            return PatchResult(
                path=rel, ok=False, is_new_file=True,
                error=(
                    "ast_insert: class_name target requires an existing file; "
                    f"{rel} does not exist"
                ),
            )
        return _make_new_file_result(rel, spec.code)
    try:
        tree = ast.parse(original, filename=rel)
    except SyntaxError as exc:
        return PatchResult(
            path=rel, ok=False, original_content=original,
            error=f"ast_insert: target file SyntaxError: {exc}",
        )
    parsed, err, dedented = _parse_snippet(spec.code)
    if err is not None or parsed is None:
        return PatchResult(
            path=rel, ok=False, original_content=original,
            error=err or "ast_insert: snippet did not parse",
        )
    return _plan_into_existing(rel, original, tree, spec, parsed, dedented)



def _coerce_anchor_loc(loc: Optional[dict], total_lines: int) -> Optional[Tuple[int, int]]:
    """Validate ``loc`` against the target file, returning ``(end_line, indent_col)``.

    Returns ``None`` when ``loc`` is missing, malformed, or points outside
    the file's line range. ``indent_col`` is clamped to ``>= 0``. The
    caller is responsible for deciding whether to use the indent or
    re-detect from the surrounding AST.
    """
    if not isinstance(loc, dict):
        return None
    try:
        end_line = int(loc.get("end_line") or 0)
        indent_col = int(loc.get("indent_col") or 0)
    except (TypeError, ValueError):
        return None
    if end_line <= 0 or end_line > total_lines:
        return None
    return end_line, max(0, indent_col)


def _plan_into_existing(
    rel: str,
    original: str,
    tree: ast.Module,
    spec: AstInsertSpec,
    parsed: _ParsedSnippet,
    dedented_source: str,
) -> PatchResult:
    """Splice ``parsed`` into ``original`` according to ``spec`` and verify."""
    total_lines = len(original.splitlines())
    anchor_loc = _coerce_anchor_loc(spec.anchor_loc, total_lines)
    if spec.class_name:
        cls = _find_class(tree, spec.class_name)
        if cls is None:
            return PatchResult(
                path=rel, ok=False, original_content=original,
                error=f"ast_insert: class '{spec.class_name}' not found in {rel}",
            )
        existing = _existing_names_in_class(cls)
        filtered, dropped = (
            _filter_dupes(parsed, existing) if spec.dedupe else (parsed, [])
        )
        if not filtered.defs and not filtered.other:
            return PatchResult(
                path=rel, ok=True, new_content=original,
                original_content=original, is_new_file=False,
                error=(
                    "ast_insert: nothing to do (already defined: "
                    + ", ".join(dropped) + ")"
                ) if dropped else "ast_insert: snippet had no insertable defs",
            )
        snippet_text = _snippet_to_text(filtered, dedented_source)
        # Line-anchored fast path: trust the retrieval-time loc when it
        # falls within the class body's line span; otherwise re-walk.
        if anchor_loc is not None:
            end_line, indent_col = anchor_loc
            cls_start = int(getattr(cls, "lineno", 0) or 0)
            cls_end = int(getattr(cls, "end_lineno", cls_start) or cls_start)
            if cls_start <= end_line <= cls_end:
                indent = indent_col if indent_col > 0 else _detect_body_indent(cls.body, default=4)
                indented = _indent_block(snippet_text, indent)
                new_content = _splice_lines(original, end_line, indented)
            else:
                indent = _detect_body_indent(cls.body, default=4)
                indented = _indent_block(snippet_text, indent)
                insert_after, _inside = _class_insert_after_line(cls, spec.anchor_symbol)
                new_content = _splice_lines(original, insert_after, indented)
        else:
            indent = _detect_body_indent(cls.body, default=4)
            indented = _indent_block(snippet_text, indent)
            insert_after, _inside = _class_insert_after_line(cls, spec.anchor_symbol)
            new_content = _splice_lines(original, insert_after, indented)
    else:
        existing = _existing_names_in_module(tree)
        filtered, dropped = (
            _filter_dupes(parsed, existing) if spec.dedupe else (parsed, [])
        )
        if not filtered.defs and not filtered.other:
            return PatchResult(
                path=rel, ok=True, new_content=original,
                original_content=original, is_new_file=False,
                error=(
                    "ast_insert: nothing to do (already defined: "
                    + ", ".join(dropped) + ")"
                ) if dropped else "ast_insert: snippet had no insertable defs",
            )
        snippet_text = _snippet_to_text(filtered, dedented_source)
        if anchor_loc is not None:
            end_line, _indent = anchor_loc
            new_content = _splice_lines(original, end_line, snippet_text)
        else:
            insert_after = _module_insert_after_line(tree, spec.anchor_symbol)
            new_content = _splice_lines(original, insert_after, snippet_text)

    # Re-parse the result to catch indentation / scoping errors before
    # any caller writes the file. This is the same safety net the rest
    # of the pipeline relies on via validate_patch_results, but we run
    # it here too so plan_ast_insertion can never produce a broken file.
    try:
        ast.parse(new_content, filename=rel)
    except SyntaxError as exc:
        return PatchResult(
            path=rel, ok=False, original_content=original,
            error=f"ast_insert: result failed reparse: {exc}",
        )
    return PatchResult(
        path=rel, ok=True, new_content=new_content,
        original_content=original, is_new_file=False,
    )


def _container_id_to_rel(
    container_id: str, container_type: str, project_root: str
) -> Tuple[Optional[str], Optional[str]]:
    """Decompose a suggest_insertion_points container_id.

    Returns ``(rel_path, class_name)`` where ``class_name`` is ``None``
    for file containers and the class' simple name for class containers.
    Returns ``(None, None)`` when the id cannot be resolved under
    ``project_root``.
    """
    if not container_id:
        return None, None
    cid = container_id
    class_name: Optional[str] = None
    if container_type == "class":
        if _CLASS_SEP not in cid:
            return None, None
        file_part, _, class_part = cid.partition(_CLASS_SEP)
        # Nested classes get joined with "." by the parser; we only
        # support top-level classes for AST insertion (consistent with
        # _find_class). Reject nested forms explicitly.
        if "." in class_part:
            return None, None
        class_name = class_part
        cid = file_part
    # cid is now an absolute or repo-relative file path.
    norm = cid.replace("\\", "/")
    if os.path.isabs(norm):
        try:
            rel = os.path.relpath(norm, start=project_root).replace("\\", "/")
        except ValueError:
            return None, None
        # Reject paths that escape project_root.
        if rel.startswith("..") or os.path.isabs(rel):
            return None, None
        return rel, class_name
    return norm.lstrip("/"), class_name


def _anchor_symbol_from_chunk_id(chunk_id: Optional[str]) -> Optional[str]:
    """Pull the leaf symbol name from a sibling-anchor chunk id."""
    if not chunk_id or not isinstance(chunk_id, str):
        return None
    for sep in (_METHOD_SEP, _FUNC_SEP, _CLASS_SEP):
        if sep in chunk_id:
            tail = chunk_id.split(sep, 1)[1]
            # method tail looks like "ClassName.func"; we want the leaf.
            return tail.split(".")[-1] or None
    return None


def plan_ast_insertion_from_suggestion(
    project_root: str,
    suggestion: dict,
    code: str,
    *,
    dedupe: bool = True,
) -> PatchResult:
    """Bridge from :func:`suggest_insertion_points` output to a PatchResult.

    ``suggestion`` is a single item from the orchestrator's anchor list:

        {
            "container_type": "file" | "class",
            "container_id": "<abs path>" | "<abs path>::class::<Name>",
            "anchors": {
                "likely_caller": <chunk_id> | None,
                "similar_signature_neighbor": <chunk_id> | None,
            },
            "score": float,
        }

    ``code`` is the Python snippet to splice in. The function picks the
    similar-signature neighbour as the preferred anchor (the same signal
    suggest_insertion_points ranks highest); falling back to the likely
    caller if absent. Either may be ``None`` — in that case the snippet
    is appended at the end of the container.
    """
    if not isinstance(suggestion, dict):
        return PatchResult(
            path="", ok=False,
            error="ast_insert: suggestion must be a dict",
        )
    container_type = str(suggestion.get("container_type") or "")
    container_id = str(suggestion.get("container_id") or "")
    rel, class_name = _container_id_to_rel(
        container_id, container_type, project_root,
    )
    if not rel:
        return PatchResult(
            path=container_id, ok=False,
            error=(
                "ast_insert: could not resolve container_id "
                f"{container_id!r} (type={container_type!r}) under "
                f"project_root={project_root!r}"
            ),
        )
    anchors = suggestion.get("anchors") or {}
    # Prefer similar-signature neighbour (the strongest ranking signal);
    # fall back to likely_caller. Pull the matching ``*_loc`` so the
    # planner can use the v3 line anchor when present.
    if anchors.get("similar_signature_neighbor"):
        anchor_chunk = anchors.get("similar_signature_neighbor")
        anchor_loc = anchors.get("similar_signature_neighbor_loc")
    elif anchors.get("likely_caller"):
        anchor_chunk = anchors.get("likely_caller")
        anchor_loc = anchors.get("likely_caller_loc")
    else:
        anchor_chunk = None
        anchor_loc = None
    anchor_symbol = _anchor_symbol_from_chunk_id(anchor_chunk)
    spec = AstInsertSpec(
        rel_path=rel,
        code=code,
        class_name=class_name,
        anchor_symbol=anchor_symbol,
        dedupe=dedupe,
        anchor_loc=anchor_loc if isinstance(anchor_loc, dict) else None,
    )
    return plan_ast_insertion(project_root, spec)


def build_unified_diff(patch: PatchResult) -> str:
    """Render a :class:`PatchResult` as a unified diff string.

    The result is shaped exactly like the diffs that
    :func:`cgx.codegen.diff_apply.parse_fenced_diffs` and
    :func:`cgx.codegen.diff_apply.apply_diffs_in_memory` consume, so a
    caller can route an AST-anchored plan back through the standard
    disk-apply / validate pipeline without any special-casing.
    """
    if patch.new_content is None:
        return ""
    a_label = "/dev/null" if patch.is_new_file else f"a/{patch.path}"
    b_label = f"b/{patch.path}"
    a_lines = (patch.original_content or "").splitlines(keepends=True) if not patch.is_new_file else []
    b_lines = patch.new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a_lines, b_lines, fromfile=a_label, tofile=b_label, n=3,
    )
    return "".join(diff)


__all__ = [
    "AstInsertSpec",
    "plan_ast_insertion",
    "plan_ast_insertion_from_suggestion",
    "build_unified_diff",
]
