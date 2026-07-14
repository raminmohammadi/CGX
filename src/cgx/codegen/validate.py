

"""Language-aware validation of post-patch file contents.

Currently this module implements:

- Python: ``ast.parse`` over the post-patch source, surfacing line/column
  information for the first ``SyntaxError`` if any.
- JavaScript / TypeScript / TSX: a tree-sitter parse that fails on the first
  ERROR / MISSING node (best-effort: skipped when the optional tree-sitter
  grammar is unavailable, mirroring the YAML validator).
- JSON: ``json.loads`` for ``*.json`` files.
- YAML: ``yaml.safe_load`` when PyYAML is importable (best-effort).

Each validator is intentionally cheap and side-effect-free so we can run them
on every iteration of an LLM generation loop. Grounding correctness in a real
parser rather than the model's self-report keeps generation quality flat
across providers -- a weak local model that emits broken JS is caught by the
same deterministic gate as a strong cloud one.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from cgx.codegen.diff_apply import PatchResult

logger = logging.getLogger(__name__)


@dataclass
class SyntaxDiagnostic:
    """Per-file validation diagnostic."""
    path: str
    ok: bool
    language: str
    error: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


# Extension -> tree-sitter grammar name for the JS/TS family. Mirrors the
# parser registrations in ``cgx.parser.js_ts_parser`` so the syntax gate
# covers exactly the files those parsers index. The grammar name doubles as
# the diagnostic ``language`` label.
_JS_TS_LANGS = {
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
}


def _detect_language(path: str) -> str:
    p = path.lower()
    dot = p.rfind(".")
    ext = p[dot:] if dot >= 0 else ""
    if ext == ".py":
        return "python"
    if ext == ".json":
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext in _JS_TS_LANGS:
        return _JS_TS_LANGS[ext]
    return "unknown"


def validate_python_source(path: str, source: str) -> SyntaxDiagnostic:
    """Parse ``source`` as Python and return a diagnostic.

    We only run ``ast.parse``; this catches grammar-level syntax errors but
    does not execute the module, so it is safe to call on untrusted output
    from an LLM.
    """
    try:
        ast.parse(source, filename=path)
        return SyntaxDiagnostic(path=path, ok=True, language="python")
    except SyntaxError as e:
        return SyntaxDiagnostic(
            path=path,
            ok=False,
            language="python",
            error=str(e),
            line=getattr(e, "lineno", None),
            column=getattr(e, "offset", None),
        )
    except Exception as e:
        return SyntaxDiagnostic(
            path=path, ok=False, language="python", error=f"{type(e).__name__}: {e}",
        )


def _validate_json(path: str, source: str) -> SyntaxDiagnostic:
    try:
        json.loads(source)
        return SyntaxDiagnostic(path=path, ok=True, language="json")
    except json.JSONDecodeError as e:
        return SyntaxDiagnostic(
            path=path, ok=False, language="json",
            error=e.msg, line=e.lineno, column=e.colno,
        )


def _validate_yaml(path: str, source: str) -> SyntaxDiagnostic:
    try:
        import yaml  # type: ignore
    except Exception:
        return SyntaxDiagnostic(path=path, ok=True, language="yaml", error="PyYAML unavailable; skipped")
    try:
        yaml.safe_load(source)
        return SyntaxDiagnostic(path=path, ok=True, language="yaml")
    except Exception as e:
        return SyntaxDiagnostic(path=path, ok=False, language="yaml", error=f"{type(e).__name__}: {e}")


def _first_error_node(root: Any) -> Optional[Any]:
    """Return the first ERROR / MISSING node in a tree-sitter tree, or ``None``.

    Iterative pre-order walk so a deeply-nested syntax error does not blow the
    Python recursion limit on a large generated file.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_missing or node.type == "ERROR":
            return node
        # Only descend where the parser flagged a problem; a clean subtree
        # cannot contain an ERROR/MISSING node.
        if node.has_error:
            stack.extend(reversed(node.children))
    return None


def validate_js_ts_source(
    path: str, source: str, language: str,
) -> SyntaxDiagnostic:
    """Parse ``source`` with the tree-sitter ``language`` grammar.

    ``language`` is a grammar name (``javascript`` / ``typescript`` / ``tsx``).
    Degrades gracefully -- when the optional ``tree_sitter_language_pack``
    dependency (or the specific grammar) is unavailable we return ``ok=True``
    with a skip note, mirroring the YAML validator, so an install without the
    ``parsers`` extra behaves exactly as before this gate existed.
    """
    from cgx.parser.treesitter_base import _get_ts_parser

    parser = _get_ts_parser(language)
    if parser is None:
        return SyntaxDiagnostic(
            path=path, ok=True, language=language,
            error="tree-sitter grammar unavailable; skipped",
        )
    try:
        tree = parser.parse(source.encode("utf-8", errors="ignore"))
    except Exception as e:  # pragma: no cover - defensive
        return SyntaxDiagnostic(
            path=path, ok=False, language=language,
            error=f"{type(e).__name__}: {e}",
        )
    if not tree.root_node.has_error:
        return SyntaxDiagnostic(path=path, ok=True, language=language)
    bad = _first_error_node(tree.root_node)
    if bad is None:
        # ``has_error`` was set but no ERROR/MISSING node surfaced; treat as OK
        # rather than emit an error with no location to feed back.
        return SyntaxDiagnostic(path=path, ok=True, language=language)
    line = bad.start_point[0] + 1
    column = bad.start_point[1] + 1
    kind = "missing token" if bad.is_missing else "syntax error"
    return SyntaxDiagnostic(
        path=path, ok=False, language=language,
        error=f"{kind} near line {line}, column {column}",
        line=line, column=column,
    )


_JS_EXTS = {".jsx", ".js", ".tsx", ".ts", ".mjs", ".cjs"}


def check_cross_file_coherence(
    results: Sequence[PatchResult],
    project_root: Optional[str] = None,
) -> List[SyntaxDiagnostic]:
    """Detect Python files that import from JS/JSX siblings in the same batch.

    Catches the common mis-generation where a Python test does
    ``from src.App import calculateResult`` but ``src/App.jsx`` is a React
    component -- not a Python module.  Checks both the in-batch file set and
    (when *project_root* is given) existing files on disk.
    """
    # Only count successfully-applied patches as "in the batch".
    # NOTE: use prefix-stripping rather than lstrip("./") so dotfiles
    # (e.g. ``.env.example``, ``.gitignore``) keep their leading dot.
    batch_paths: set = set()
    for r in results:
        if r.path and r.ok and r.new_content is not None:
            p = r.path
            while p.startswith("./"):
                p = p[2:]
            batch_paths.add(p.lstrip("/"))

    def _is_js_on_disk(rel: str) -> bool:
        """True when *rel* exists on disk under project_root and has a JS extension."""
        if not project_root:
            return False
        abs_path = os.path.join(project_root, rel)
        return os.path.isfile(abs_path)

    issues: List[SyntaxDiagnostic] = []
    for r in results:
        if not r.ok or not r.new_content:
            continue
        if not r.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(r.new_content, filename=r.path)
        except SyntaxError:
            continue  # already reported by validate_python_source
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module:
                continue
            module_rel = node.module.replace(".", "/")
            for ext in _JS_EXTS:
                candidate = module_rel + ext
                if candidate in batch_paths or _is_js_on_disk(candidate):
                    names = ", ".join(a.name for a in node.names)
                    issues.append(SyntaxDiagnostic(
                        path=r.path,
                        ok=False,
                        language="python",
                        error=(
                            f"imports '{names}' from '{node.module}' but "
                            f"'{candidate}' is a JavaScript/JSX file, not a Python module"
                        ),
                        line=node.lineno,
                    ))
                    break  # one report per import statement is enough
    return issues


def validate_patch_results(results: Sequence[PatchResult]) -> List[SyntaxDiagnostic]:
    """Run a per-language syntax check on every successfully-applied patch.

    Failed patches are surfaced as failed diagnostics so a calling loop can
    feed both classes of issues back to the LLM uniformly.
    """
    diagnostics: List[SyntaxDiagnostic] = []
    for r in results:
        lang = _detect_language(r.path)
        if not r.ok or r.new_content is None:
            diagnostics.append(SyntaxDiagnostic(
                path=r.path,
                ok=False,
                language=lang,
                error=r.error or "patch failed",
            ))
            continue
        if lang == "python":
            diagnostics.append(validate_python_source(r.path, r.new_content))
        elif lang in ("javascript", "typescript", "tsx"):
            diagnostics.append(validate_js_ts_source(r.path, r.new_content, lang))
        elif lang == "json":
            diagnostics.append(_validate_json(r.path, r.new_content))
        elif lang == "yaml":
            diagnostics.append(_validate_yaml(r.path, r.new_content))
        else:
            diagnostics.append(SyntaxDiagnostic(path=r.path, ok=True, language=lang))
    n_failed = sum(1 for d in diagnostics if not d.ok)
    if n_failed:
        logger.info("codegen.validate: %d/%d files failed syntax checks",
                    n_failed, len(diagnostics))
    return diagnostics
