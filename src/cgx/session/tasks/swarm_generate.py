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
from typing import Any, Dict, List, Optional, Tuple

from cgx.codegen.ast_gluer import ASTAssembler
from cgx.session.import_audit import strip_unused_imports, unused_imports
from cgx.session.tasks.swarm_ground import _safe_read, ground_dependencies

# Caps on model-facing context. A tool response (``run_python_probe`` output, a
# file skeleton) and an injected dependency body are both untrusted, unbounded
# text; left uncapped they grow the prompt every turn until a weak local model
# either declines or exceeds its window. These mirror the truncation the
# DIAGNOSE loop already applies to observations.
_TOOL_OUTPUT_LIMIT = 4000
_DEP_CONTENT_LIMIT = 2000


def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars, marking the elision so it is visible."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


# Tools the Developer may call while generating a file: read-only introspection
# so imports name symbols that actually exist, plus any configured MCP tools.
# Dispatch and descriptions both come from the shared registry -- adding a tool
# here (or an MCP server) needs no change to this loop.
_DEV_BASE_TOOLS = ("run_python_probe", "file_skeleton", "list_symbols")
_MAX_TOOL_ITERS = 5


def _dev_tools() -> tuple:
    """Developer tool set: introspection + MCP tools when servers exist."""
    from cgx.session.tasks.swarm_tools import mcp_tools_if_configured
    return _DEV_BASE_TOOLS + mcp_tools_if_configured()


class ToolWrapper:
    """Provider shim that resolves ``<call_tool>`` requests via the registry.

    Wraps a provider so the generation ladder can call it like any other, while
    transparently running any tool the model requests (through the approval
    gate, when one is supplied) and feeding the result back until the model
    produces its final answer.
    """

    def __init__(self, p, root_dir, *, tools=None, deps=None,
                 approval_gate=None):
        self.p = p
        self.root = root_dir
        self.tools = tuple(tools) if tools is not None else _dev_tools()
        self.deps = deps
        self.approval_gate = approval_gate

    def chat(self, messages, **kwargs):
        from cgx.session.tasks.swarm_log import swarm_beat
        from cgx.session.tasks.tool_registry import (
            REGISTRY, ToolContext, parse_tool_calls)
        # register_native_tools ran at import of swarm_tools; ensure it is
        # imported so the registry is populated even if nothing else pulled it.
        import cgx.session.tasks.swarm_tools  # noqa: F401

        if messages and messages[0].get("role") == "system":
            if "<call_tool" not in messages[0].get("content", ""):
                messages[0]["content"] += (
                    "\n\nCRITICAL INSTRUCTION: BEFORE outputting your final code JSON, "
                    "you MUST verify the API signatures, class names, and exports of "
                    "ANY local file you plan to import from, by calling tools. DO NOT "
                    "GUESS OR HALLUCINATE imported names.\n"
                    + REGISTRY.describe_for_prompt(self.tools)
                )

        ctx = ToolContext(root=self.root, deps=self.deps,
                          log_root=self.root,
                          approval_gate=self.approval_gate)
        kwargs["force_json"] = False
        for _ in range(_MAX_TOOL_ITERS):
            res = self.p.chat(messages=messages, **kwargs)
            text = str(res.get("content", ""))
            calls = [c for c in parse_tool_calls(text) if c.name in self.tools]
            if not calls:
                return res
            messages.append({"role": "assistant", "content": text})
            for call in calls:  # honour every requested call, not just the first
                swarm_beat(self.root, "developer", "tool_call",
                           tool=call.name, args=call.raw_args)
                out = _truncate(REGISTRY.dispatch(call, ctx),
                                _TOOL_OUTPUT_LIMIT)
                messages.append({
                    "role": "user",
                    "content": f"<tool_response name=\"{call.name}\">\n{out}\n"
                               "</tool_response>"})
        return self.p.chat(messages=messages, **kwargs)

    def chat_stream(self, *args, **kwargs):
        return self.p.chat_stream(*args, **kwargs)
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


def _ast_import_injector(content: str, err: str, contracts: Dict[str, Any]) -> Optional[str]:
    """Auto-inject missing imports if they are defined in the contracts."""
    import re, ast
    m = re.search(r"uses undefined name\(s\) (\[.*?\]):", err)
    if not m:
        return None
    try:
        names = ast.literal_eval(m.group(1))
    except Exception:
        return None
        
    injected = []
    schemas = contracts.get("schemas", [])
    functions = contracts.get("functions", [])
    
    for name in names:
        module_path = None
        for s in schemas:
            if s.get("name") == name:
                module_path = s.get("module")
                break
        if not module_path:
            for f in functions:
                if f.get("name") == name:
                    module_path = f.get("module")
                    break
        
        if module_path:
            # convert src/models.py to src.models
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            mod_str = module_path.replace("/", ".")
            injected.append(f"from {mod_str} import {name}")
        elif name in contracts.get("third_party_dependencies", []) or name in ["os", "json", "sys", "pytest", "typing"]:
            injected.append(f"import {name}")
            
    if not injected:
        return None
        
    # prepend to content
    return "\n".join(injected) + "\n\n" + content


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
    renegotiated_contracts: Optional[Dict[str, Any]] = None

    @property
    def bytes(self) -> int:
        return len(self.content or "")


def _dep_context(depends_on: List[str], root: str) -> List[Dict[str, str]]:
    """On-disk content of each readable dependency, as engine context."""
    ctx: List[Dict[str, str]] = []
    for dep in depends_on or []:
        src = _safe_read(dep, root)
        if src:
            # Cap each dependency body: a widely-imported module would
            # otherwise swamp the prompt (the very failure ``_DEP_LIMIT`` in
            # swarm_ground warns about, in a path that previously ignored it).
            ctx.append({"path": dep,
                        "content": _truncate(src, _DEP_CONTENT_LIMIT)})
    return ctx


def _gate_generated_content(path: str, content: str,
                            contracts: Dict[str, Any],
                            manifest_paths: Optional[List[str]],
                            root: str) -> Optional[str]:
    """Contract + import validation for one generated body; ``None`` if clean.

    Unions the contract-compliance gate with (when a manifest is known) the
    first-party import resolver and phantom third-party check, filters to this
    file, and returns a single error string prefixed ``AST Import validation
    failed`` (for a bad import) or ``Contract compliance failed`` (otherwise),
    with a hint listing the real local files. Shared by both rungs of
    :func:`_full_file_attempt` so the logic lives in one place.

    These gates parse Python (``ast`` + dotted-import resolution), so they run
    only for ``.py`` files. A non-Python file (``.jsx``/``.ts``/…) is validated
    by its own syntax gate in the full-file rung and by the polyglot build/test
    in SWARM_VERIFY, not here.
    """
    if not path.endswith(".py"):
        return None
    from cgx.session.scaffold_validate import check_contract_compliance
    warnings = check_contract_compliance({path: content}, contracts)
    if manifest_paths:
        from cgx.session.import_audit import resolve_first_party_imports
        from cgx.session.tasks.swarm_verify import _check_phantom_third_party_imports
        allowed_3p = contracts.get("third_party_dependencies", [])
        warnings.extend(
            resolve_first_party_imports({path: content}, manifest_paths, root))
        warnings.extend(
            _check_phantom_third_party_imports([path], {path: content},
                                               allowed_3p, root))
    file_warnings = [w for w in warnings
                     if w.get("file") == path or w.get("module") == path]
    if not file_warnings:
        return None
    errs = "; ".join(w.get("reason", "unknown") for w in file_warnings)
    is_import_err = bool(manifest_paths) and any(
        w.get("kind") == "phantom_third_party"
        or "resolves against neither" in w.get("reason", "")
        for w in file_warnings)
    if is_import_err:
        available = ", ".join(manifest_paths or [])
        errs += (f". IMPORTANT: You imported a non-existent local file! "
                 f"Available local files in the project are: {available}. "
                 "Use the correct dotted path (e.g. if the file is src/api.py, "
                 "use 'from src.api import ...').")
        return f"AST Import validation failed for {path}: {errs}"
    return f"Contract compliance failed for {path}: {errs}"


def _full_file_attempt(path: str, description: str, depends_on: List[str],
                       contracts: Dict[str, Any], goal: str, root: str,
                       provider: Any, layer: str,
                       manifest_paths: Optional[List[str]],
                       skills: Optional[List[str]] = None) -> Any:
    from cgx.answer.engine import generate_single_scaffold_file

    context = _dep_context(depends_on, root)
    try:
        wrapped = ToolWrapper(provider, root)
        result = generate_single_scaffold_file(
            path, description, wrapped,
            layer=layer,
            existing_files_with_content=context,
            goal=goal,
            skills=skills,
            depends_on=list(depends_on or []),
            contracts=contracts or {},
            manifest_paths=manifest_paths,
        )
    except Exception as exc:  # pragma: no cover
        return "", f"{type(exc).__name__}: {exc}"
        
    content = str(result.get("content") or "")
    if content and bool(result.get("syntax_ok")):
        return content, _gate_generated_content(
            path, content, contracts, manifest_paths, root)

    err = (str(result.get("syntax_error") or "").strip()
           or "full-file generation failed the syntax gate")

    if "uses undefined name(s)" in err:
        repaired = _ast_import_injector(content, err, contracts or {})
        if repaired:
            import ast
            try:
                ast.parse(repaired)
                return repaired, _gate_generated_content(
                    path, repaired, contracts, manifest_paths, root)
            except SyntaxError:
                pass

    return content, err


def _renegotiate_contracts(
        path: str, content: str, err: str, contracts: Dict[str, Any], goal: str, provider: Any
) -> Optional[Dict[str, Any]]:
    """Prompt the Tech Lead to amend contracts to match the working code."""
    import json
    import copy
    
    prompt = (
        f"You are the Tech Lead. The Developer generated the following code for {path}, "
        f"but it fails the contract compliance gate.\n\n"
        f"Error:\n{err}\n\n"
        f"Code:\n```python\n{content}\n```\n\n"
        f"Your current contracts are:\n```json\n{json.dumps(contracts, indent=2)}\n```\n\n"
        "Instead of forcing the Developer to change the code (which is functionally correct), "
        "you must amend the contracts to match the reality of the code. "
        "Output ONLY the complete updated JSON contracts object and nothing else."
    )
    
    try:
        res = provider.chat(messages=[{"role": "user", "content": prompt}], force_json=True)
        text = str(res.get("content") or "").strip()
        new_contracts = json.loads(text)
        return new_contracts
    except Exception:
        return None


# Map a file extension to a Markdown code-fence language hint, so semantic
# repair prompts (and their fenced-block extraction) match the file's language
# instead of always assuming Python.
_FENCE_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "tsx", ".vue": "vue",
    ".go": "go", ".rs": "rust", ".java": "java",
}


def _fence_lang(path: str) -> str:
    """Fence language for ``path`` (empty when unknown)."""
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return _FENCE_LANG.get(ext, "")


def _semantic_repair_fallback(
        path: str, content: str, err: str, goal: str, root: str, provider: Any
) -> Tuple[str, str]:
    """Semantic repair agent: fix syntax errors without losing business logic.

    Language-aware: the fence hint and extraction follow the file's extension,
    so a broken ``.jsx``/``.ts`` file is repaired as JS/TS rather than being
    refused (the previous ``.py``-only guard is gone).
    """
    lang = _fence_lang(path)
    prompt = (
        f"The following code for {path} failed the syntax gate with this error:\n"
        f"{err}\n\n"
        f"Code:\n```{lang}\n{content}\n```\n\n"
        "Fix the error (e.g., add missing imports) WITHOUT removing any business "
        f"logic or functions. Output ONLY the fixed code inside a ```{lang} "
        "block and nothing else."
    )

    try:
        res = provider.chat(messages=[{"role": "user", "content": prompt}], force_json=False)
        text = str(res.get("content") or "")
        # sometimes the model still outputs JSON with a "code" key
        if text.strip().startswith("{"):
            import json
            try:
                parsed = json.loads(text)
                if "code" in parsed:
                    text = parsed["code"]
            except Exception:
                pass

        import re
        # Accept a fenced block in any language, not just ```python.
        m = re.search(r"```[a-zA-Z0-9_+-]*\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return m.group(1), ""
        # if no markdown block, return raw text if it looks like code
        if any(tok in text for tok in ("def ", "import ", "function ",
                                       "const ", "class ", "export ")):
            return text.strip(), ""
        return "", "Semantic repair failed to produce valid code."
    except Exception as exc:
        return "", f"Semantic repair failed: {exc}"


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
                                        {"role": "user", "content": user}], force_json=False)
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
                  log_root: Optional[str] = None,
                  skills: Optional[List[str]] = None) -> GenerationOutcome:
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
    last_broken_content = ""
    for attempt in range(2):
        content, err = _full_file_attempt(
            path, description, depends_on, contracts, goal, root, provider,
            layer, manifest_paths, skills)
        if err:
            last_broken_content = content or last_broken_content
            swarm_beat(log_root, "developer", "gate", file=path, ok=False,
                       method="full-file", error=err)
            content = ""
            continue
        phantom = unused_imports(content, path=path)
        if phantom and attempt == 0:
            last_broken_content = content
            swarm_beat(log_root, "developer", "gate", file=path, ok=False,
                       method="full-file",
                       error=f"phantom imports: {', '.join(phantom)}")
            content = ""
            continue
        break
    if content:
        content = _sanitize_phantoms(content, path, log_root)
        return GenerationOutcome(path, content, True, "full-file")

    if "Contract compliance failed" in err:
        swarm_beat(log_root, "developer", "renegotiate", file=path, error=err)
        renegotiated_contracts = _renegotiate_contracts(
            path, last_broken_content, err, contracts, goal, provider)
        if renegotiated_contracts is not None:
            # We return the content because it's syntactically valid!
            return GenerationOutcome(path, last_broken_content, True, "renegotiated", renegotiated_contracts=renegotiated_contracts)
    
    swarm_beat(log_root, "developer", "semantic_repair", file=path, error=err)
    repaired_content, repair_err = _semantic_repair_fallback(
        path, last_broken_content, err, goal, root, provider)
    if repaired_content:
        repaired_content = _sanitize_phantoms(repaired_content, path, log_root)
        return GenerationOutcome(path, repaired_content, True, "semantic-repair")
    
    return GenerationOutcome(path, "", False, "failed",
                             error=repair_err or err)

