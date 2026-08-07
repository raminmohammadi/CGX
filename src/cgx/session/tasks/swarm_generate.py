"""The per-file generation ladder for the swarm Developer.

One file is generated at a time, mirroring the greenfield SCAFFOLD ladder so
the swarm inherits its graded degradation instead of a bespoke, brittle path:

1. **Full-file** -- :func:`generate_single_scaffold_file`, grounded on the
   *real* on-disk content of the file's ``depends_on`` (so imports name
   symbols that exist), gated on ``syntax_ok``. Re-asked once on failure.
2. **AST fallback** -- the deterministic ``header + per-symbol`` assembler
   from :mod:`cgx.session.tasks.ast_scaffold`, required symbols taken from the
   plan contracts, rejected if it degrades to an empty/symbol-less module.

The result is a small value object the Developer writes to disk and logs; the
ladder itself does no IO beyond reading dependencies for grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cgx.codegen.ast_gluer import ASTAssembler
from cgx.session.import_audit import strip_unused_imports, unused_imports
from cgx.session.tasks.swarm_ground import _safe_read, ground_dependencies
from cgx.session.tasks.swarm_log import swarm_beat


@dataclass
class GenerationOutcome:
    """What the ladder produced for one file.

    ``method`` is ``full-file`` / ``ast-fallback`` / ``failed``; ``ok`` is
    True only when ``content`` is a syntactically valid, non-empty body ready
    to write. ``error`` carries the last concrete reason on failure.
    """

    path: str
    content: str
    ok: bool
    method: str
    error: Optional[str] = None

    @property
    def bytes(self) -> int:
        return len(self.content or "")


def _dep_context(depends_on: List[str], root: str) -> List[Dict[str, str]]:
    """On-disk content of each readable dependency, as engine context."""
    ctx: List[Dict[str, str]] = []
    for dep in depends_on or []:
        src = _safe_read(dep, root)
        if src:
            ctx.append({"path": dep, "content": src})
    return ctx


def _full_file_attempt(path: str, description: str, depends_on: List[str],
                       contracts: Dict[str, Any], goal: str, root: str,
                       provider: Any, layer: str,
                       manifest_paths: Optional[List[str]]) -> Any:
    """One full-file generation. Returns ``(content, error)``."""
    from cgx.answer.engine import generate_single_scaffold_file
    context = _dep_context(depends_on, root)
    try:
        result = generate_single_scaffold_file(
            path, description, provider,
            layer=layer,
            existing_files_with_content=context,
            goal=goal,
            depends_on=list(depends_on or []),
            contracts=contracts or {},
            manifest_paths=manifest_paths,
        )
    except Exception as exc:  # pragma: no cover - defensive: provider crash
        return "", f"{type(exc).__name__}: {exc}"
    content = str(result.get("content") or "")
    if content and bool(result.get("syntax_ok")):
        return content, None
    err = (str(result.get("syntax_error") or "").strip()
           or "full-file generation failed the syntax gate")
    return "", err


def _ast_fallback(path: str, description: str, depends_on: List[str],
                  contracts: Dict[str, Any], goal: str, root: str,
                  provider: Any) -> Any:
    """Deterministic AST assembly fallback. Returns ``(content, error)``."""
    if not path.endswith(".py"):
        return "", "AST fallback only supports .py files"
    from cgx.session.tasks.ast_scaffold import (
        _assembly_rejection, _generate_header, _generate_symbol,
        _required_symbols)
    grounding = ground_dependencies(depends_on, root)
    header = _generate_header(path, provider, goal, description,
                              grounding=grounding)
    assembler = ASTAssembler(header)
    symbols = _required_symbols("", path, contracts or {})
    for name, kind in symbols:
        code = _generate_symbol(path, name, kind, provider, goal, header,
                                description, grounding=grounding)
        if code:
            assembler.add_component(code)
    try:
        content = assembler.unparse()
    except Exception as exc:  # pragma: no cover - defensive: unparse failure
        return "", f"AST assembly failed: {exc}"
    rejection = _assembly_rejection(content, symbols, assembler)
    if rejection:
        return "", rejection
    return content, None


def _sanitize_phantoms(content: str, path: str,
                       log_root: Optional[str]) -> str:
    """Guarantee no phantom import ships: strip any that survived and log it.

    Runs on every accepted body regardless of which rung produced it, so an
    unused, hallucinated import can never reach disk even when the model
    ignores the corrective re-ask.
    """
    cleaned, removed = strip_unused_imports(content, path=path)
    if removed:
        swarm_beat(log_root, "developer", "phantom_stripped", file=path,
                   removed=removed)
    return cleaned


def generate_file(*, path: str, description: str, depends_on: List[str],
                  contracts: Dict[str, Any], goal: str, root: str,
                  provider: Any, layer: str = "",
                  manifest_paths: Optional[List[str]] = None,
                  log_root: Optional[str] = None) -> GenerationOutcome:
    """Run the full-file -> AST fallback ladder for a single file.

    A phantom import (a syntactically valid body carrying an unused, invented
    dependency) is treated as a *gate failure* on the first full-file attempt
    and re-asked once, since it signals the module was hallucinated. Whatever
    the final rung yields is sanitised so a surviving phantom is stripped
    deterministically before the outcome is returned.
    """
    depends_on = list(depends_on or [])
    content = err = ""
    for attempt in range(2):
        content, err = _full_file_attempt(
            path, description, depends_on, contracts, goal, root, provider,
            layer, manifest_paths)
        if not content:
            swarm_beat(log_root, "developer", "gate", file=path, ok=False,
                       method="full-file", error=err)
            continue
        phantom = unused_imports(content, path=path)
        if phantom and attempt == 0:
            swarm_beat(log_root, "developer", "gate", file=path, ok=False,
                       method="full-file",
                       error=f"phantom imports: {', '.join(phantom)}")
            content = ""
            continue
        break
    if content:
        content = _sanitize_phantoms(content, path, log_root)
        return GenerationOutcome(path, content, True, "full-file")

    swarm_beat(log_root, "developer", "ast_fallback", file=path, error=err)
    ast_content, ast_err = _ast_fallback(
        path, description, depends_on, contracts, goal, root, provider)
    if ast_content:
        ast_content = _sanitize_phantoms(ast_content, path, log_root)
        return GenerationOutcome(path, ast_content, True, "ast-fallback")
    return GenerationOutcome(path, "", False, "failed",
                             error=ast_err or err)

