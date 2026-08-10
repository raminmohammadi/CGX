import ast
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from cgx.session.models import Artifact, ArtifactKind, TaskKind, TaskNode
from cgx.session.tasks.base import ExecutorDeps, ExecutorResult, register_executor
from cgx.session.tasks.scaffold import _emit_scaffold_progress
from cgx.codegen.ast_gluer import ASTAssembler
from cgx.trace import traced

logger = logging.getLogger(__name__)

# ``generate_project_skeleton`` returns a single unified script that separates
# per-file sections with a comment marker like ``# --- src/config.py ---``.
# This matches that marker and captures the path token.
_SKELETON_MARKER = re.compile(r"^\s*#\s*-{2,}\s*(?P<path>.+?)\s*-{2,}\s*$")


def _split_skeleton_by_path(skeleton: str) -> Dict[str, str]:
    """Split a unified skeleton script into ``{path: code}`` sections.

    ``generate_project_skeleton`` emits one string containing the signatures
    for every manifest file, delimited by ``# --- <path> ---`` comment
    markers. The AST fallback needs the section for a single file, so parse
    the markers back into a per-path map. Lines before the first marker are
    ignored; a file with no marker simply won't be found (callers degrade to
    an empty skeleton).
    """
    sections: Dict[str, List[str]] = {}
    current: str = ""
    for line in skeleton.splitlines():
        m = _SKELETON_MARKER.match(line)
        if m:
            current = m.group("path").strip()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {p: "\n".join(lines).strip() for p, lines in sections.items()}

_FENCE = re.compile(r"```(?:python|py)?\s*\n(?P<body>.*?)(?:```|\Z)",
                    re.DOTALL)


def _extract_python(text: str) -> Tuple[str, Optional[str]]:
    """The parseable Python in an LLM reply, or the reason there is none.

    Stripping fence markers and keeping the rest left any surrounding
    prose in the string: ``ASTAssembler`` then dropped the whole piece
    (or emptied itself, for a header) with no signal to the caller, and
    the file collapsed to a byte. Prefer the fenced block(s), fall back
    to the bare reply, and require the result to ``ast.parse``.
    """
    raw = str(text or "")
    blocks = [m.group("body") for m in _FENCE.finditer(raw)]
    candidate = "\n".join(b.strip() for b in blocks) if blocks else raw.strip()
    candidate = candidate.strip()
    if not candidate:
        return "", "model returned no code"
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return "", f"{type(exc).__name__}: {exc}"
    return candidate, None


def _generate_code(path: str, what: str, provider: Any, prompt: str) -> str:
    """Prompt for a Python fragment, re-prompting once if it does not parse.

    The retry quotes the parse error, which is the single most effective
    correction for the failure mode seen in ``ses_fa6f72a9d3da4217`` (a
    reply that trailed off into English). A second failure yields ``""``
    so the caller's output gate reports the file as failed.
    """
    messages = [
        {"role": "system", "content": "You are a strict code generator."},
        {"role": "user", "content": prompt},
    ]
    for attempt in (1, 2):
        try:
            res = provider.chat(messages=messages, force_json=False)
        except Exception as exc:
            logger.error("Failed to generate AST %s for %s: %s",
                         what, path, exc)
            return ""
        code, error = _extract_python(res.get("content", ""))
        if error is None:
            return code
        logger.warning("AST %s for %s did not parse (attempt %d): %s",
                       what, path, attempt, error)
        if attempt == 2:
            return ""
        messages = messages + [
            {"role": "assistant", "content": str(res.get("content", ""))},
            {"role": "user", "content":
                f"That reply is not valid Python: {error}. "
                "Return ONLY raw Python source, no prose and no commentary."},
        ]
    return ""


def _symbol_name_from_signature(signature: str) -> str:
    """The leading identifier of a ``name(args) -> ret`` contract signature."""
    m = re.match(r"\s*(?:async\s+)?(?:def\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                 str(signature or ""))
    return m.group(1) if m else ""


def _symbols_from_skeleton(skeleton_code: str) -> List[Tuple[str, str]]:
    """Top-level functions/classes declared in a file's skeleton section."""
    out: List[Tuple[str, str]] = []
    try:
        parsed = ast.parse(skeleton_code)
    except Exception:
        return out
    for node in parsed.body:
        if isinstance(node, ast.AsyncFunctionDef):
            out.append((node.name, "async function"))
        elif isinstance(node, ast.FunctionDef):
            out.append((node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, "class"))
    return out


def _symbols_from_contracts(
        path: str, contracts: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Required symbols for ``path`` taken from the plan contracts.

    The safety net for a skeleton section that will not ``ast.parse``: without
    it the assembler builds only a header and the file is rejected for
    defining none of its required symbols. Functions contracted for this
    module become symbols to generate; schemas bound to it become classes.
    """
    norm = str(path or "").replace("\\", "/")
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for fn in (contracts.get("functions") or []):
        if not isinstance(fn, dict):
            continue
        if str(fn.get("module") or "").replace("\\", "/") != norm:
            continue
        name = (str(fn.get("name") or "").strip()
                or _symbol_name_from_signature(fn.get("signature", "")))
        if name and name not in seen:
            seen.add(name)
            out.append((name, "function"))
    for sc in (contracts.get("schemas") or []):
        if not isinstance(sc, dict):
            continue
        if str(sc.get("module") or "").replace("\\", "/") != norm:
            continue
        name = str(sc.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append((name, "class"))
    return out


def _required_symbols(skeleton_code: str, path: str,
                      contracts: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Symbols the AST fallback must (re)build for ``path``.

    Prefer the skeleton (authoritative when it parses); fall back to the plan
    contracts so a malformed skeleton no longer guarantees an empty file.
    """
    symbols = _symbols_from_skeleton(skeleton_code)
    return symbols if symbols else _symbols_from_contracts(path, contracts)


def _public_signatures(code: str) -> str:
    """A compact signature view (def/class headers + first docstring line)."""
    try:
        parsed = ast.parse(code)
    except Exception:
        return ""
    lines: List[str] = []
    for node in parsed.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        if isinstance(node, ast.ClassDef):
            header = f"class {node.name}"
        else:
            kw = "async def" if isinstance(
                node, ast.AsyncFunctionDef) else "def"
            try:
                header = f"{kw} {node.name}({ast.unparse(node.args)})"
            except Exception:
                header = f"{kw} {node.name}(...)"
        lines.append(header)
        doc = ast.get_docstring(node)
        if doc:
            lines.append(f"    # {doc.splitlines()[0].strip()}")
    return "\n".join(lines)


def _dependency_paths(path: str, plan_content: Dict[str, Any]) -> List[str]:
    """The ``depends_on`` paths declared for ``path`` in the work plan."""
    for lay in (plan_content.get("layers") or []):
        for f in (lay.get("files") or []):
            if isinstance(f, dict) and str(f.get("path") or "") == path:
                return [str(d) for d in (f.get("depends_on") or []) if str(d)]
    return []


def _grounding_block(dep_paths: List[str], skeleton: Dict[str, str],
                     generated_by_path: Dict[str, str],
                     limit: int = 1600) -> str:
    """Public signatures of a file's dependencies, to prompt against.

    Prefer a dependency already (re)generated this run -- its *real*
    signatures -- and fall back to the dependency's skeleton section. Bounded
    so the prompt stays small even for a widely-depended-upon module.
    """
    parts: List[str] = []
    for dep in dep_paths:
        if dep in generated_by_path:
            sig = _public_signatures(generated_by_path[dep])
        else:
            sig = str(skeleton.get(dep, "") or "").strip()
        if sig:
            parts.append(f"# From {dep}:\n{sig}")
    block = "\n\n".join(parts).strip()
    if len(block) > limit:
        block = block[:limit] + "\n# ... (truncated)"
    return block


@traced("ast_scaffold.generate_header")
def _generate_header(path: str, provider: Any, goal: str, context: str,
                     grounding: str = "") -> str:
    """Prompt the LLM for just the imports and globals of the file."""
    ground = (f"\nSignatures available from dependencies (import from these, "
              f"do not redefine them):\n{grounding}\n" if grounding else "")
    prompt = f"""You are generating ONLY the file header (imports and global variables) for {path}.
Project Goal: {goal}
Context: {context}{ground}
Return ONLY valid Python code containing imports and module-level constants. No functions or classes."""
    return _generate_code(path, "header", provider, prompt)

@traced("ast_scaffold.generate_symbol")
def _generate_symbol(path: str, symbol_name: str, symbol_type: str, provider: Any, goal: str, header: str, context: str, grounding: str = "") -> str:
    """Prompt the LLM for a specific function or class."""
    ground = (f"\nSignatures available from dependencies (call these; do not "
              f"reimplement them):\n{grounding}\n" if grounding else "")
    prompt = f"""You are generating exactly ONE {symbol_type} named `{symbol_name}` for {path}.
Project Goal: {goal}
Context: {context}{ground}
The file currently has the following imports and globals:
```python
{header}
```

Return ONLY the code for `{symbol_name}`. Do NOT include imports. Do not wrap in markdown, output raw python code."""
    return _generate_code(path, f"symbol {symbol_name}", provider, prompt)


def _resolve_goal(task: TaskNode, deps: ExecutorDeps,
                  plan_content: Dict[str, Any]) -> str:
    """The project goal to prompt with, from the first source that has one.

    SCAFFOLD reads ``composed_goal``/``prior_goal`` off the WORK_PLAN; the
    router threads ``prior_goal`` through the regenerate inputs. Reading
    only ``composed_goal`` from ``task.inputs`` -- a key the router never
    sets -- left every fallback prompt with an empty goal, so the model
    invented an unrelated stack. Fall back to the session's objective so
    the prompt is never goal-less.
    """
    for candidate in (task.inputs.get("composed_goal"),
                      task.inputs.get("prior_goal"),
                      plan_content.get("composed_goal"),
                      plan_content.get("prior_goal")):
        text = str(candidate or "").strip()
        if text:
            return text
    if deps.store is None:
        return ""
    try:
        session = deps.store.get_session(task.session_id)
    except Exception:  # pragma: no cover - defensive: store hiccup
        return ""
    return str(getattr(session, "original_objective", "") or "").strip()


def _assembly_rejection(content: str,
                        symbols: List[Tuple[str, str]],
                        assembler: ASTAssembler) -> Optional[str]:
    """Why the assembled file must not be handed to APPLY, if it must not.

    ``ASTAssembler`` degrades to an empty module when the header does not
    parse and silently drops components that do not, so ``unparse`` can
    succeed on nothing at all. The result used to be recorded as
    ``generated`` with ``syntax_ok=True`` -- in session
    ``ses_fa6f72a9d3da4217`` a 1-byte file overwrote a real one. An empty
    assembly, or one missing a symbol the skeleton requires, is a failed
    regeneration and must be reported as such.
    """
    if assembler.base_error:
        return f"AST fallback header did not parse: {assembler.base_error}"
    if not content.strip():
        return "AST fallback produced an empty file"
    defined = {node.name for node in assembler.module.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))}
    missing = [name for name, _ in symbols if name not in defined]
    if missing:
        return ("AST fallback produced no definition for required "
                f"symbol(s): {missing}")
    return None


def _demote_import_failures(generated: List[Dict[str, Any]],
                            diffs: List[Dict[str, Any]],
                            failed: List[Dict[str, str]],
                            regenerated: set) -> int:
    """Move files importing undefined first-party symbols into ``failed``.

    The fallback fires only after SCAFFOLD's own gates rejected a file
    twice, yet it handed its output to APPLY unchecked. Run SCAFFOLD's
    cross-file import check over the assembled tree and fail any file
    this round regenerated that imports a name the target module never
    defines, so the router builds a targeted regenerate instead of
    shipping a broken import.

    Best-effort, matching the scaffold convention: any error in the
    checker abstains and leaves the tree untouched. Returns the number of
    files demoted.
    """
    contents = {e["file"]: e["content"] for e in generated
                if isinstance(e, dict) and e.get("file")
                and isinstance(e.get("content"), str)}
    if not contents:
        return 0
    try:
        from cgx.session.scaffold_validate import (
            cross_check_first_party_imports)
        warnings = cross_check_first_party_imports(contents)
    except Exception:  # pragma: no cover - defensive: checker is best-effort
        logger.exception("AST fallback: import cross-check raised; skipping")
        return 0

    reasons: Dict[str, List[str]] = {}
    for w in warnings:
        path = str(w.get("file") or "")
        if path not in regenerated:
            continue
        reasons.setdefault(path, []).append(
            f"{w.get('name')!r} is not defined in {w.get('module')!r}")
    if not reasons:
        return 0

    for path, msgs in reasons.items():
        logger.warning("AST fallback rejected %s: %s", path, "; ".join(msgs))
        failed.append({
            "file": path,
            "error": ("AST fallback imported undefined first-party "
                      f"symbol(s): {'; '.join(msgs)}"),
        })
    generated[:] = [e for e in generated
                    if not (isinstance(e, dict) and e.get("file") in reasons)]
    diffs[:] = [d for d in diffs
                if not (isinstance(d, dict) and d.get("file") in reasons)]
    return len(reasons)


@register_executor(TaskKind.AST_REGENERATE)
@traced("ast_scaffold.run")
def run_ast_scaffold(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Fallback executor that builds Python files piece-by-piece via AST."""
    if deps.provider is None:
        return ExecutorResult(failure="AST_REGENERATE requires an LLM provider")

    regen_files = task.inputs.get("regenerate_files", [])
    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "").strip()

    work_plan = None
    contracts = {}
    plan_content: Dict[str, Any] = {}
    if work_plan_id and deps.store:
        work_plan = deps.store.get_artifact(work_plan_id)
        if work_plan:
            content = work_plan.content or {}
            plan_content = content
            contracts = content.get("contracts") or {}
            if not regen_files:
                # Full-tree regeneration
                layers = content.get("layers") or []
                regen_files = []
                for lyr in layers:
                    for f in lyr.get("files", []):
                        if "path" in f:
                            regen_files.append(f["path"])
                            
    # ``project_skeleton`` is the unified string produced by
    # ``generate_project_skeleton``; split it into a per-path map. Tolerate a
    # dict (legacy / future shape) by mapping each value to its "content".
    raw_skeleton = contracts.get("project_skeleton", "")
    if isinstance(raw_skeleton, str):
        skeleton = _split_skeleton_by_path(raw_skeleton)
    elif isinstance(raw_skeleton, dict):
        skeleton = {}
        for p, v in raw_skeleton.items():
            if isinstance(v, dict):
                content = v.get("content", "")
                skeleton[p] = ("\n".join(content)
                               if isinstance(content, list) else str(content))
            elif isinstance(v, list):
                skeleton[p] = "\n".join(str(x) for x in v)
            else:
                skeleton[p] = str(v)
    else:
        skeleton = {}

    goal = _resolve_goal(task, deps, plan_content)

    diffs = []
    generated = []
    failed = []

    prior_scaffold_id = str(task.inputs.get("prior_scaffold_artifact_id") or "").strip()
    if prior_scaffold_id and deps.store:
        prior_scaffold = deps.store.get_artifact(prior_scaffold_id)
        if prior_scaffold and prior_scaffold.content:
            prior_generated = prior_scaffold.content.get("generated") or []
            prior_diffs = prior_scaffold.content.get("diffs") or []
            regen_set = set(regen_files)
            for file_entry in prior_generated:
                if isinstance(file_entry, dict) and file_entry.get("file") not in regen_set:
                    generated.append(file_entry)
            for diff_entry in prior_diffs:
                if isinstance(diff_entry, dict) and diff_entry.get("file") not in regen_set:
                    diffs.append(diff_entry)

    # Real content of everything already on the tree, keyed by path: the
    # grounding source so a file is authored against its dependencies' actual
    # signatures rather than blind (the root cause of hallucinated imports).
    generated_by_path: Dict[str, str] = {
        e["file"]: e["content"] for e in generated
        if isinstance(e, dict) and e.get("file")
        and isinstance(e.get("content"), str)}

    total_files = len(regen_files)
    progress_done = 0
    progress_failed = 0

    for path in regen_files:
        path = (path or "").strip()
        # Security: Prevent directory traversal and enforce relative paths
        if ".." in path or os.path.isabs(path):
            logger.warning("AST scaffolding rejected unsafe path: %s", path)
            failed.append({"file": path, "error": "Unsafe path detected"})
            continue

        if not path.endswith(".py"):
            # Only python is supported for AST fallback currently
            failed.append({"file": path, "error": "AST fallback only supports .py files"})
            progress_failed += 1
            _emit_scaffold_progress(
                deps, task, file=path, layer="ast_fallback",
                index=progress_done, total=total_files,
                status="failed", failed_count=progress_failed)
            continue
            
        logger.info("AST scaffolding file: %s", path)
        _emit_scaffold_progress(
            deps, task, file=path, layer="ast_fallback",
            index=progress_done + 1, total=total_files,
            status="start", failed_count=progress_failed)
        
        started = time.time()
        skeleton_code = skeleton.get(path, "")
        dep_paths = _dependency_paths(path, plan_content)
        grounding = _grounding_block(dep_paths, skeleton, generated_by_path)

        # 1. Generate Header (Imports + Globals). The file's own skeleton is
        # the context; the dependency signatures ground its imports so they
        # name symbols that actually exist instead of hallucinated ones.
        header = _generate_header(
            path, deps.provider, goal, skeleton_code, grounding=grounding)

        assembler = ASTAssembler(header)

        # Required symbols come from the skeleton when it parses, else from the
        # plan contracts -- a malformed skeleton must not silently yield an
        # empty file (every required symbol then reported missing).
        symbols = _required_symbols(skeleton_code, path, contracts)

        for sym_name, sym_type in symbols:
            symbol_code = _generate_symbol(
                path, sym_name, sym_type, deps.provider, goal, header,
                skeleton_code, grounding=grounding)
            if symbol_code:
                assembler.add_component(symbol_code)
        
        # Unparse the final assembled AST
        try:
            final_content = assembler.unparse()
            rejection = _assembly_rejection(
                final_content, symbols, assembler)
            if rejection:
                logger.warning("AST fallback rejected %s: %s", path, rejection)
                failed.append({"file": path, "error": rejection})
                progress_failed += 1
                _emit_scaffold_progress(
                    deps, task, file=path, layer="ast_fallback",
                    index=progress_done, total=total_files, status="failed",
                    elapsed_ms=int((time.time() - started) * 1000),
                    failed_count=progress_failed)
                continue
            import difflib
            patch_lines = list(difflib.unified_diff(
                [],
                final_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3
            ))
            patch = "".join(patch_lines)
            if not patch:
                patch = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +0,0 @@\n"
            diffs.append({
                "file": path,
                "patch": patch
            })
            generated.append({
                "file": path,
                "syntax_ok": True,
                "content": final_content,
                "bytes": len(final_content)
            })
            # Ground later files in this run against what was just built.
            generated_by_path[path] = final_content
            progress_done += 1
            _emit_scaffold_progress(
                deps, task, file=path, layer="ast_fallback",
                index=progress_done, total=total_files,
                status="done", bytes=len(final_content),
                elapsed_ms=int((time.time() - started) * 1000),
                failed_count=progress_failed)
        except Exception as e:
            logger.error("AST assembly failed for %s: %s", path, e)
            failed.append({"file": path, "error": f"AST assembly failed: {e}"})
            progress_failed += 1
            _emit_scaffold_progress(
                deps, task, file=path, layer="ast_fallback",
                index=progress_done, total=total_files,
                status="failed", elapsed_ms=int((time.time() - started) * 1000),
                failed_count=progress_failed)

    _demote_import_failures(
        generated, diffs, failed, {p for p in regen_files if p})

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.SCAFFOLD_PATCHES,
        content={
            "diffs": diffs,
            "generated": generated,
            "failed": failed,
            "complete": True
        }
    )
    
    return ExecutorResult(
        artifact=artifact,
        outputs={"failed": failed, "failed_count": len(failed)}
    )
