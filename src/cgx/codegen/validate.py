

"""Language-aware validation of post-patch file contents.

Currently this module implements:

- Python: ``ast.parse`` over the post-patch source, surfacing line/column
  information for the first ``SyntaxError`` if any.
- JSON: ``json.loads`` for ``*.json`` files.
- YAML: ``yaml.safe_load`` when PyYAML is importable (best-effort).

Each validator is intentionally cheap and side-effect-free so we can run them
on every iteration of an LLM generation loop.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

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


def _detect_language(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith(".json"):
        return "json"
    if p.endswith((".yaml", ".yml")):
        return "yaml"
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


_JS_EXTS = {".jsx", ".js", ".tsx", ".ts", ".mjs", ".cjs"}


def check_cross_file_coherence(
    results: Sequence[PatchResult],
    project_root: Optional[str] = None,
) -> List[SyntaxDiagnostic]:
    """Detect Python files that import from JS/JSX siblings in the same batch.

    Catches the common mis-generation where a Python test does
    ``from src.App import calculateResult`` but ``src/App.jsx`` is a React
    component — not a Python module.  Checks both the in-batch file set and
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
