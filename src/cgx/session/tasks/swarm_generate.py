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

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cgx.codegen.ast_gluer import ASTAssembler
from cgx.session.import_audit import strip_unused_imports, unused_imports
from cgx.session.tasks.swarm_ground import _safe_read, ground_dependencies
from cgx.session.tasks.swarm_log import swarm_beat

# A pytest bootstrap that makes an un-installed project importable from its own
# root: prepending both the project root (so ``import src.pkg`` resolves) and a
# ``src/`` dir (so ``import pkg`` resolves) covers either rooting the Developer
# emitted, without hard-coding a package name. Deterministic, so a weak model
# never fabricates a broken conftest.
_CONFTEST_TEMPLATE = (
    '"""Pytest bootstrap: make the project importable without installation."""\n'
    "import os\n"
    "import sys\n\n"
    "_ROOT = os.path.dirname(os.path.abspath(__file__))\n"
    '_SRC = os.path.join(_ROOT, "src")\n'
    "for _p in (_ROOT, _SRC):\n"
    "    if os.path.isdir(_p) and _p not in sys.path:\n"
    "        sys.path.insert(0, _p)\n"
)

# Doc/config extensions the non-source path owns. Deliberately narrow: real
# source in another language (``.js``, ``.ts``, ``.go``) still runs the
# full-file rung, so only prose/config is diverted from the code ladder.
_DOC_EXT = (".md", ".rst", ".txt", ".toml", ".cfg", ".ini")


def _is_non_source(path: str) -> bool:
    """True when ``path`` is a scaffolding/doc deliverable, not code.

    ``requirements.txt`` and ``conftest.py`` are named explicitly (the latter
    is ``.py`` yet carries no plan symbols); every doc/config extension is
    diverted too. Any other extension -- including non-Python source -- stays
    on the code ladder.
    """
    base = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return (base in ("requirements.txt", "conftest.py")
            or path.lower().endswith(_DOC_EXT))


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


def _requirements_content(manifest_paths: Optional[List[str]],
                          root: str) -> str:
    """A requirements.txt built from the tree's real third-party imports.

    Deterministic and source-derived (never model-authored): scan every
    generated ``.py`` in the manifest, drop stdlib and first-party roots, map
    each surviving import root to its PyPI distribution, and always pin the
    test runner. Reuses the same import machinery the env dry-run installs
    with, so the declared manifest matches what the suite actually needs.
    """
    from cgx.codegen.env_manager import (
        _STDLIB_TOP, _import_root_to_pypi, _is_local_package, scan_imports)
    py_abs = [os.path.join(root, p) for p in (manifest_paths or [])
              if p.endswith(".py")]
    dists: set = set()
    for imp in scan_imports(py_abs):
        if imp.lower().replace("-", "_") in _STDLIB_TOP:
            continue
        if _is_local_package(imp, root):
            continue
        name = _import_root_to_pypi(imp)
        if name:
            dists.add(name)
    dists.add("pytest")
    return "\n".join(sorted(dists)) + "\n"


def _fallback_readme(goal: str, manifest_paths: Optional[List[str]]) -> str:
    """A minimal but valid README when the free-form model path is unusable."""
    title = (goal or "Project").strip().splitlines()[0][:80] or "Project"
    files = "\n".join(f"- `{p}`" for p in (manifest_paths or []))
    return (f"# {title}\n\n{goal}\n\n"
            "## Install\n\n```\npip install -r requirements.txt\n```\n\n"
            "## Testing\n\n```\npytest\n```\n"
            + (f"\n## Files\n\n{files}\n" if files else ""))


def _readme_content(description: str, goal: str, provider: Any,
                    manifest_paths: Optional[List[str]]) -> str:
    """A README authored free-form by the model, falling back deterministically.

    Prose is not source, so it bypasses the AST ladder: the model writes
    Markdown directly. Any provider failure or empty reply degrades to
    :func:`_fallback_readme` so a README always ships.
    """
    listing = "\n".join(f"- {p}" for p in (manifest_paths or []))
    system = ("You write a concise, accurate README.md for a Python project. "
              "Output ONLY Markdown -- no surrounding code fence. Include a "
              "title, a one-paragraph description, an Install section using "
              "'pip install -r requirements.txt', and a Testing section using "
              "'pytest'.")
    user = (f"Project goal:\n{goal}\n\nThis file's purpose:\n{description}\n\n"
            f"Planned files:\n{listing}\n")
    try:
        res = provider.chat(messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": user}])
        content = str((res or {}).get("content") or "").strip()
    except Exception:  # pragma: no cover - defensive: provider crash
        content = ""
    return content + "\n" if content else _fallback_readme(goal, manifest_paths)


def _generate_non_source(path: str, description: str, goal: str, root: str,
                         provider: Any,
                         manifest_paths: Optional[List[str]],
                         log_root: Optional[str]) -> GenerationOutcome:
    """Generate a planned non-source deliverable (scaffolding / docs).

    The AST ladder only understands Python symbols, so scaffolding was
    previously ungeneratable. This routes the three deliverables the planner
    now mandates: ``requirements.txt`` and ``conftest.py`` from deterministic,
    source-derived templates, and ``README.md`` (or any other doc) from a
    free-form model call with a deterministic fallback.
    """
    base = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base == "requirements.txt":
        content, method = _requirements_content(manifest_paths, root), "template"
    elif base == "conftest.py":
        content, method = _CONFTEST_TEMPLATE, "template"
    else:
        content, method = (
            _readme_content(description, goal, provider, manifest_paths),
            "freeform")
    swarm_beat(log_root, "developer", "non_source", file=path, method=method,
               bytes=len(content))
    if content.strip():
        return GenerationOutcome(path, content, True, method)
    return GenerationOutcome(path, "", False, "failed",
                             error="non-source generation produced nothing")


def generate_file(*, path: str, description: str, depends_on: List[str],
                  contracts: Dict[str, Any], goal: str, root: str,
                  provider: Any, layer: str = "",
                  manifest_paths: Optional[List[str]] = None,
                  log_root: Optional[str] = None) -> GenerationOutcome:
    """Run the full-file -> AST fallback ladder for a single file.

    A planned non-source deliverable (``requirements.txt``, ``conftest.py``,
    ``README.md`` and other doc/config files) has no Python symbols to
    assemble, so it is routed to :func:`_generate_non_source` before the
    Python ladder runs.

    A phantom import (a syntactically valid body carrying an unused, invented
    dependency) is treated as a *gate failure* on the first full-file attempt
    and re-asked once, since it signals the module was hallucinated. Whatever
    the final rung yields is sanitised so a surviving phantom is stripped
    deterministically before the outcome is returned.
    """
    if _is_non_source(path):
        return _generate_non_source(path, description, goal, root, provider,
                                    manifest_paths, log_root)
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

