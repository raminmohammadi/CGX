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

@traced("ast_scaffold.generate_header")
def _generate_header(path: str, provider: Any, goal: str, context: str) -> str:
    """Prompt the LLM for just the imports and globals of the file."""
    prompt = f"""You are generating ONLY the file header (imports and global variables) for {path}.
Project Goal: {goal}
Context: {context}
Return ONLY valid Python code containing imports and module-level constants. No functions or classes."""
    try:
        messages = [
            {"role": "system", "content": "You are a strict code generator."},
            {"role": "user", "content": prompt}
        ]
        res = provider.chat(messages=messages, force_json=False)
        text = res.get("content", "")
        return text.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        logger.error("Failed to generate AST header for %s: %s", path, e)
        return ""

@traced("ast_scaffold.generate_symbol")
def _generate_symbol(path: str, symbol_name: str, symbol_type: str, provider: Any, goal: str, header: str, context: str) -> str:
    """Prompt the LLM for a specific function or class."""
    prompt = f"""You are generating exactly ONE {symbol_type} named `{symbol_name}` for {path}.
Project Goal: {goal}
Context: {context}

The file currently has the following imports and globals:
```python
{header}
```

Return ONLY the code for `{symbol_name}`. Do NOT include imports. Do not wrap in markdown, output raw python code."""
    try:
        messages = [
            {"role": "system", "content": "You are a strict code generator."},
            {"role": "user", "content": prompt}
        ]
        res = provider.chat(messages=messages, force_json=False)
        text = res.get("content", "")
        return text.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        logger.error("Failed to generate AST symbol %s for %s: %s", symbol_name, path, e)
        return ""


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
                        module: ast.Module) -> Optional[str]:
    """Why the assembled file must not be handed to APPLY, if it must not.

    ``ASTAssembler`` degrades to an empty module when the header does not
    parse and silently drops components that do not, so ``unparse`` can
    succeed on nothing at all. The result used to be recorded as
    ``generated`` with ``syntax_ok=True`` -- in session
    ``ses_fa6f72a9d3da4217`` a 1-byte file overwrote a real one. An empty
    assembly, or one missing a symbol the skeleton requires, is a failed
    regeneration and must be reported as such.
    """
    if not content.strip():
        return "AST fallback produced an empty file"
    defined = {node.name for node in module.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))}
    missing = [name for name, _ in symbols if name not in defined]
    if missing:
        return ("AST fallback produced no definition for required "
                f"symbol(s): {missing}")
    return None


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

        # 1. Generate Header (Imports + Globals). The file's own skeleton
        # is the context: without it the header is authored blind and its
        # imports contradict the symbols generated below.
        header = _generate_header(path, deps.provider, goal, skeleton_code)

        assembler = ASTAssembler(header)

        symbols = []
        try:
            parsed = ast.parse(skeleton_code)
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef):
                    symbols.append((node.name, "function"))
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.append((node.name, "async function"))
                elif isinstance(node, ast.ClassDef):
                    symbols.append((node.name, "class"))
        except Exception as parse_e:
            logger.warning("Failed to parse skeleton for %s: %s", path, parse_e)
            
        for sym_name, sym_type in symbols:
            symbol_code = _generate_symbol(path, sym_name, sym_type, deps.provider, goal, header, skeleton_code)
            if symbol_code:
                assembler.add_component(symbol_code)
        
        # Unparse the final assembled AST
        try:
            final_content = assembler.unparse()
            rejection = _assembly_rejection(
                final_content, symbols, assembler.module)
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
