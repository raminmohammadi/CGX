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
            ctx.append({"path": dep, "content": src})
    return ctx


def _full_file_attempt(path: str, description: str, depends_on: List[str],
                       contracts: Dict[str, Any], goal: str, root: str,
                       provider: Any, layer: str,
                       manifest_paths: Optional[List[str]]) -> Any:
    """One full-file generation with interactive tool loop and pre-AST validation. Returns ``(content, error)``."""
    from cgx.answer.engine import generate_single_scaffold_file

    class ToolWrapper:
        def __init__(self, p, root_dir):
            self.p = p
            self.root = root_dir
            
        def chat(self, messages, **kwargs):
            import re, json
            from cgx.session.tasks.swarm_tools import run_python_probe, query_codebase
            from cgx.session.tasks.swarm_ground import file_skeleton, list_symbols
            
            if messages and messages[0].get("role") == "system":
                if "run_python_probe" not in messages[0].get("content", ""):
                    messages[0]["content"] += (
                        "\n\nTOOLS AVAILABLE: You may use tools before outputting your final JSON plan. "
                        "Tools: run_python_probe(code: str) to run arbitrary python code, "
                        "file_skeleton(path: str) to view file signatures, "
                        "list_symbols(path: str) to see symbols in a file. "
                        "To call a tool, output EXACTLY: <call_tool name=\"tool_name\">{\"arg\": \"val\"}</call_tool>"
                    )
            
            kwargs["force_json"] = False
            for _ in range(5):
                res = self.p.chat(messages=messages, **kwargs)
                text = str(res.get("content", ""))
                match = re.search(r'<call_tool name="(.*?)">(.*?)</call_tool>', text, re.DOTALL)
                if match:
                    t_name, t_args_str = match.group(1), match.group(2)
                    messages.append({"role": "assistant", "content": text})
                    from cgx.session.tasks.swarm_log import swarm_beat
                    swarm_beat(self.root, "developer", "tool_call", tool=t_name, args=t_args_str)
                    try:
                        args = json.loads(t_args_str)
                        if t_name == "run_python_probe":
                            out = run_python_probe(args.get("code", ""), self.root)
                        elif t_name == "file_skeleton":
                            out = file_skeleton(args.get("path", ""), self.root)
                        elif t_name == "list_symbols":
                            out = str(list_symbols(args.get("path", ""), self.root))
                        else:
                            out = f"Unknown tool: {t_name}"
                    except Exception as e:
                        out = f"Tool error: {e}"
                    messages.append({"role": "user", "content": f"<tool_response>\n{out}\n</tool_response>"})
                else:
                    return res
            return self.p.chat(messages=messages, **kwargs)
            
        def chat_stream(self, *args, **kwargs):
            return self.p.chat_stream(*args, **kwargs)

    context = _dep_context(depends_on, root)
    try:
        wrapped = ToolWrapper(provider, root)
        result = generate_single_scaffold_file(
            path, description, wrapped,
            layer=layer,
            existing_files_with_content=context,
            goal=goal,
            depends_on=list(depends_on or []),
            contracts=contracts or {},
            manifest_paths=manifest_paths,
        )
    except Exception as exc:  # pragma: no cover
        return "", f"{type(exc).__name__}: {exc}"
        
    content = str(result.get("content") or "")
    if content and bool(result.get("syntax_ok")):
        # Pre-AST Validation: ensure contracts meant for this file are fulfilled
        from cgx.session.scaffold_validate import check_contract_compliance
        warnings = check_contract_compliance({path: content}, contracts)
        # Filter warnings specifically related to this path
        file_warnings = [w for w in warnings if w.get("module") == path]
        if file_warnings:
            errs = "; ".join(w.get("reason", "unknown") for w in file_warnings)
            return content, f"Contract compliance failed for {path}: {errs}"
        return content, None
        
    err = (str(result.get("syntax_error") or "").strip()
           or "full-file generation failed the syntax gate")
           
    if "uses undefined name(s)" in err:
        repaired = _ast_import_injector(content, err, contracts or {})
        if repaired:
            import ast
            try:
                ast.parse(repaired)
                from cgx.session.scaffold_validate import check_contract_compliance
                warnings = check_contract_compliance({path: repaired}, contracts)
                file_warnings = [w for w in warnings if w.get("module") == path]
                if file_warnings:
                    errs = "; ".join(w.get("reason", "unknown") for w in file_warnings)
                    return repaired, f"Contract compliance failed for {path}: {errs}"
                return repaired, None
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


def _semantic_repair_fallback(
        path: str, content: str, err: str, goal: str, root: str, provider: Any
) -> Tuple[str, str]:
    """Semantic repair agent: fix syntax errors without losing business logic."""
    if not path.endswith(".py"):
        return "", "Semantic repair only supports .py files"
        
    prompt = (
        f"The following code for {path} failed the syntax gate with this error:\n"
        f"{err}\n\n"
        f"Code:\n```python\n{content}\n```\n\n"
        "Fix the error (e.g., add missing imports) WITHOUT removing any business logic or functions. "
        "Output ONLY the fixed Python code inside a ```python``` block and nothing else."
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
        m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            fixed = m.group(1)
            return fixed, ""
        # if no markdown block, return raw text if it looks like code
        if "def " in text or "import " in text:
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
    last_broken_content = ""
    for attempt in range(2):
        content, err = _full_file_attempt(
            path, description, depends_on, contracts, goal, root, provider,
            layer, manifest_paths)
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

