"""High-level :func:`run_agent` entrypoint.

Wires :class:`~cgx.agents.planner.Planner`,
:class:`~cgx.agents.tracker.Tracker`, and
:class:`~cgx.agents.judge.Judge` to the existing Averix capabilities:

* ``ask``      → :func:`cgx.answer.engine.answer_with_llm`
* ``plan``     → :func:`cgx.answer.engine.generate_code_plan`
* ``scaffold`` → :func:`cgx.answer.engine.generate_project_scaffold`
* ``search``   → :func:`cgx.pipeline.auto.run_query_auto`
* ``summarize``→ inline LLM condensation of prior outputs
* ``apply``    → :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`
* ``verify``   → :func:`cgx.codegen.test_runner.run_tests_on_disk`

``scaffold`` is the only capability that does **not** require an index —
it generates a brand-new project from a plain-language idea and stores
its output as ``--- /dev/null`` new-file unified diffs so the ``apply``
capability can write them to ``project_root`` without special handling.

The capability callables are imported lazily inside :func:`run_agent`
so the agent module stays usable in environments that don't have the
embedding stack or a populated index (e.g. unit tests).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from cgx.agents.judge import Judge
from cgx.agents.planner import Planner
from cgx.agents.tracker import Tracker
from cgx.agents.types import AgentEvent, Plan

logger = logging.getLogger(__name__)


def _build_default_capabilities(
    *,
    provider: Any,
    index_dir: Optional[str],
    records_path: Optional[str],
    project_root: Optional[str],
) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Return capability callables backed by the real engine.

    Each capability tolerates ``index_dir`` / ``records_path`` being
    ``None`` by raising ``ValueError`` — the Tracker will record the
    failure and (by default) stop the plan.
    """
    def _need_index() -> None:
        if not index_dir or not records_path:
            raise ValueError("index_dir and records_path are required for this capability")

    def ask(question: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        from cgx.answer.engine import answer_with_llm
        return answer_with_llm(index_dir, records_path, question, provider, **kw)

    def plan(task_text: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        import inspect
        from cgx.answer.engine import generate_code_plan
        from cgx.codegen.symbol_map import build_symbol_context_prompt
        # Always honour project_root for self-test sandboxing if available.
        kw.setdefault("project_root", project_root)
        if project_root:
            kw.setdefault("self_test", True)
            kw.setdefault("run_tests", False)
            kw.setdefault("max_retries", 1)
        # Phase 3: inject symbol map so the SLM knows what's already defined.
        if records_path and not kw.get("symbol_context"):
            sym_ctx = build_symbol_context_prompt(records_path)
            if sym_ctx:
                kw["symbol_context"] = sym_ctx
        # plan_fix attaches ``target_files`` / ``do_not_change`` to the PLAN
        # task's inputs to scope the retry. ``generate_code_plan`` itself
        # has no parameter for either, so fold them into the task text and
        # strip them (along with any other unknown kwargs) before forwarding.
        target_files = kw.pop("target_files", None) or []
        do_not_change = kw.pop("do_not_change", None) or []
        guidance: List[str] = []
        if target_files:
            guidance.append(
                "Only modify the following file(s); do not introduce diffs "
                "for any other path:\n" + "\n".join(f"- {p}" for p in target_files)
            )
        if do_not_change:
            guidance.append(
                "Do NOT modify or re-emit these files (already verified):\n"
                + "\n".join(f"- {p}" for p in do_not_change)
            )
        if guidance:
            task_text = (task_text or "").rstrip() + "\n\n" + "\n\n".join(guidance)
        accepted = set(inspect.signature(generate_code_plan).parameters)
        safe_kw = {k: v for k, v in kw.items() if k in accepted}
        return generate_code_plan(
            index_dir, records_path, task_text, provider, **safe_kw,
        )

    def search(query: str, **kw: Any) -> Dict[str, Any]:
        _need_index()
        from cgx.pipeline.auto import run_query_auto
        return run_query_auto(index_dir, records_path, query, **kw)

    def summarize(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        # Compose a single text blob then ask the LLM to summarise.
        if provider is None:
            return {"answer_md": ""}
        body = "\n\n---\n\n".join(
            str(o.get("answer_md") or o.get("plan_md") or o.get("hits") or o)
            for o in (prior or [])
        )[:6000]
        resp = provider.chat(messages=[
            {"role": "system", "content": "Summarise the following work products in <=8 bullets."},
            {"role": "user", "content": body},
        ], temperature=0.1, max_tokens=1000, force_json=False)
        return {"answer_md": str((resp or {}).get("content") or "")}

    def scaffold(task_text: str, **kw: Any) -> Dict[str, Any]:
        # No index required — generates an entire project from scratch.
        from cgx.answer.engine import generate_project_scaffold
        kw.setdefault("project_root", project_root)
        # The tracker forwards prior task outputs to SCAFFOLD via the
        # ``_prior_outputs`` kwarg so sibling scaffold tasks share one
        # coherent file tree. Mine them for already-generated paths and
        # surface that list to the engine; don't propagate the kwarg
        # itself to the underlying capability.
        prior = kw.pop("_prior_outputs", None) or []
        existing = _scaffold_existing_files(prior)
        if existing and "existing_files" not in kw:
            kw["existing_files"] = existing
        return generate_project_scaffold(task_text, provider, **kw)

    def apply(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        if not project_root:
            raise ValueError("apply requires project_root to be set")
        from cgx.codegen.disk_apply import apply_diffs_to_disk
        diffs = _extract_prior_diffs(prior)
        if not diffs:
            return {
                "applied_files": [], "failed_files": [],
                "diffs": [], "error": "no diffs found in prior task outputs",
            }
        return apply_diffs_to_disk(project_root, diffs)

    def verify(prior: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        if not project_root:
            raise ValueError("verify requires project_root to be set")
        from cgx.codegen.test_runner import (
            discover_all_tests, run_pytest_paths, run_tests_on_disk,
        )
        changed = _changed_files_from_prior(prior)

        # Phase 2: pre-flight dependency check — scan generated Python files
        # for imports that aren't in requirements.txt and auto-install them
        # before running pytest, so ModuleNotFoundError doesn't mask real failures.
        py_files = [f for f in changed if f.endswith(".py")]
        if py_files:
            try:
                from cgx.codegen.env_manager import preflight_install, update_requirements
                abs_py = [
                    os.path.join(project_root, f) if not os.path.isabs(f) else f
                    for f in py_files
                ]
                missing, results = preflight_install(abs_py, project_root)
                if missing:
                    logger.info(
                        "verify: preflight installed %d package(s): %s",
                        sum(1 for ok in results.values() if ok),
                        [p for p, ok in results.items() if ok],
                    )
                    # Persist newly-installed packages to requirements.txt
                    # after a successful install (tests decide whether to keep them).
                    installed = [p for p, ok in results.items() if ok]
                    if installed:
                        update_requirements(project_root, installed)
            except Exception as _e:
                logger.debug("verify: preflight_install skipped: %s", _e)

        # Standalone verify (no prior APPLY) — sweep all discovered tests.
        mode = "impacted"
        if not changed:
            discovered = discover_all_tests(project_root)
            if discovered:
                outcome = run_pytest_paths(
                    project_root, discovered,
                    timeout_seconds=float(kw.get("timeout", 180.0)),
                )
                mode = "all"
            else:
                outcome = run_tests_on_disk(
                    project_root, changed,
                    timeout_seconds=float(kw.get("timeout", 180.0)),
                )
        else:
            outcome = run_tests_on_disk(
                project_root, changed,
                timeout_seconds=float(kw.get("timeout", 180.0)),
            )
        return {
            "ran": outcome.ran,
            "tests_passed": outcome.ran and outcome.returncode == 0,
            "returncode": outcome.returncode,
            "tests_selected": outcome.tests_selected,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "skipped_reason": outcome.skipped_reason,
            "mode": mode,
        }

    def scaffold_manifest(task_text: str, **kw: Any) -> Dict[str, Any]:
        from cgx.answer.engine import plan_scaffold_manifest
        kw.setdefault("project_root", project_root)
        prior = kw.pop("_prior_outputs", None) or []
        existing = _scaffold_existing_files(prior)
        if existing and "existing_files" not in kw:
            kw["existing_files"] = existing
        # Remove keys the engine doesn't accept.
        kw.pop("project_root", None)
        result = plan_scaffold_manifest(task_text, provider, **kw)
        # Convert the manifest layers into SCAFFOLD_FILE tasks that the
        # Tracker will inject immediately after this task completes.
        result["inject_tasks"] = _manifest_to_scaffold_file_tasks(
            result.get("layers") or [],
            goal=kw.get("goal", task_text),
            skills=kw.get("skills"),
        )
        return result

    def scaffold_file(task_text: str, **kw: Any) -> Dict[str, Any]:
        from cgx.answer.engine import generate_single_scaffold_file
        prior = kw.pop("_prior_outputs", None) or []
        existing_with_content = _gather_generated_files_with_content(prior)
        path = kw.pop("path", "")
        description = kw.pop("file_description", task_text)
        layer = kw.pop("layer", "")
        goal = kw.pop("goal", "")
        skills = kw.pop("skills", None)
        kw.pop("project_root", None)
        return generate_single_scaffold_file(
            path, description, provider,
            layer=layer,
            existing_files_with_content=existing_with_content,
            goal=goal,
            skills=skills,
        )

    def fill_logic(task_text: str, **kw: Any) -> Dict[str, Any]:
        """Phase 1: fill a single empty function body in an existing skeleton file.

        Expects ``kw`` to carry:
          - ``file_path``     : project-relative path to the skeleton file
          - ``function_name`` : the name of the stub to fill
          - ``skeleton``      : current file content (optional; read from disk if absent)
        """
        if not project_root:
            raise ValueError("fill_logic requires project_root to be set")
        if provider is None:
            raise ValueError("fill_logic requires a provider")

        file_path = kw.pop("file_path", "")
        function_name = kw.pop("function_name", "")
        skeleton = kw.pop("skeleton", None)

        abs_path = (
            os.path.join(project_root, file_path)
            if file_path and not os.path.isabs(file_path)
            else file_path
        )
        if skeleton is None and abs_path and os.path.exists(abs_path):
            try:
                skeleton = open(abs_path, encoding="utf-8").read()
            except Exception:
                skeleton = ""

        # Phase 3: build symbol context so the model knows what's already defined.
        sym_ctx = ""
        if records_path:
            try:
                from cgx.codegen.symbol_map import build_symbol_context_prompt
                sym_ctx = build_symbol_context_prompt(records_path)
            except Exception:
                pass

        system = (
            "You are filling in a single function body. "
            "Return ONLY the implementation code for the function body — "
            "no imports, no class definition, no other functions. "
            "The code must be indented correctly to fit inside the function. "
            "Do NOT include the `def` line itself."
        )
        if sym_ctx:
            system = sym_ctx + "\n\n" + system

        user = (
            f"Implement the body for `{function_name}` in `{file_path}`.\n\n"
            f"Current file skeleton:\n```\n{(skeleton or '')[:3000]}\n```\n\n"
            f"Return only the function body logic (no def line, correct indentation)."
        )
        if task_text and task_text not in user:
            user = task_text + "\n\n" + user

        resp = provider.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
            max_tokens=kw.get("max_tokens", 800),
            force_json=False,
        )
        body_code = str((resp or {}).get("content") or "").strip()

        # Stitch: replace `pass` / `# TODO` in the skeleton with the generated body.
        result: Dict[str, Any] = {
            "file_path": file_path,
            "function_name": function_name,
            "body_code": body_code,
        }
        if skeleton and body_code and abs_path:
            try:
                import ast as _ast
                import re as _re
                # Find the stub's pass/TODO line and replace with the body.
                stub_re = _re.compile(
                    r"(def\s+" + _re.escape(function_name) + r"\s*\([^)]*\)[^:]*:\s*\n)"
                    r"(\s+)(pass|# ?TODO[^\n]*)\n",
                    _re.MULTILINE,
                )
                indent = ""
                m = stub_re.search(skeleton)
                if m:
                    indent = m.group(2)
                    indented_body = "\n".join(
                        indent + line if line.strip() else line
                        for line in body_code.splitlines()
                    )
                    new_content = stub_re.sub(
                        m.group(1) + indented_body + "\n", skeleton, count=1
                    )
                    with open(abs_path, "w", encoding="utf-8") as fh:
                        fh.write(new_content)
                    result["applied"] = True
                    result["new_content"] = new_content
                    # Inline syntax check.
                    if abs_path.endswith(".py"):
                        try:
                            _ast.parse(new_content)
                            result["syntax_ok"] = True
                        except SyntaxError as se:
                            result["syntax_ok"] = False
                            result["syntax_error"] = str(se)
            except Exception as e:
                result["stitch_error"] = str(e)
        return result

    return {"ask": ask, "plan": plan, "scaffold": scaffold, "search": search,
            "summarize": summarize, "apply": apply, "verify": verify,
            "scaffold_manifest": scaffold_manifest, "scaffold_file": scaffold_file,
            "fill_logic": fill_logic}


def _scaffold_existing_files(prior: List[Dict[str, Any]]) -> List[str]:
    """Return the ordered list of file paths produced by prior SCAFFOLD tasks.

    Used by the ``scaffold`` capability wrapper to tell later sibling
    tasks which files have already been generated, so they extend the
    same tree instead of inventing a parallel one.
    """
    out: List[str] = []
    seen: set = set()
    for o in (prior or []):
        if not isinstance(o, dict):
            continue
        diffs = o.get("diffs")
        if not isinstance(diffs, list):
            continue
        for d in diffs:
            if not isinstance(d, dict):
                continue
            fp = str(d.get("file") or d.get("path") or "").strip()
            if fp and fp not in seen:
                seen.add(fp)
                out.append(fp)
    return out


def _gather_generated_files_with_content(
    prior: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return ``{path, content}`` pairs for every file produced by prior SCAFFOLD_FILE tasks.

    Used by the ``scaffold_file`` capability so each new file generation
    call receives the full content of already-generated files as context.
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    for o in (prior or []):
        if not isinstance(o, dict):
            continue
        fp = str(o.get("file") or "").strip()
        content = str(o.get("content") or "").strip()
        if fp and content and fp not in seen:
            seen.add(fp)
            out.append({"path": fp, "content": content})
    return out


def _manifest_to_scaffold_file_tasks(
    layers: List[Dict[str, Any]],
    *,
    goal: str = "",
    skills: Optional[List[str]] = None,
) -> List[Any]:
    """Convert a manifest's ``layers`` list into ordered ``SCAFFOLD_FILE`` Task objects.

    Files are ordered layer-by-layer so dependency-heavy files (core types,
    utilities) are generated before the files that import them.
    """
    from cgx.agents.types import Task, TaskKind
    tasks: List[Any] = []
    for layer in (layers or []):
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "project")
        for f in (layer.get("files") or []):
            if not isinstance(f, dict):
                continue
            path = str(f.get("path") or "").strip()
            description = str(f.get("description") or path)
            if not path:
                continue
            t = Task(
                description=f"Generate {path}",
                kind=TaskKind.SCAFFOLD_FILE,
                name=f"Generate {path}",
                criteria=[
                    f"File {path} has complete, non-stub content.",
                    "File passes syntax validation.",
                ],
                inputs={
                    "path": path,
                    "file_description": description,
                    "layer": layer_name,
                    "goal": goal,
                },
            )
            if skills:
                t.inputs["skills"] = list(skills)
            tasks.append(t)
    return tasks


def _extract_prior_diffs(prior: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Collect and merge diffs from ALL prior task outputs.

    Walks every output in order so that multi-scaffold plans (one task per
    layer) contribute all their files to the final APPLY step. Later entries
    win on file conflicts, allowing a follow-up scaffold task to refine a
    file produced by an earlier one.
    """
    merged: Dict[str, str] = {}
    for out in (prior or []):
        if not isinstance(out, dict):
            continue
        diffs = out.get("diffs")
        if not isinstance(diffs, list):
            continue
        for d in diffs:
            if not isinstance(d, dict):
                continue
            fp = str(d.get("file") or d.get("path") or "").strip()
            patch = str(d.get("patch") or d.get("diff") or "")
            if fp and patch:
                merged[fp] = patch
    return [{"file": fp, "patch": patch} for fp, patch in merged.items()]


def _changed_files_from_prior(prior: List[Dict[str, Any]]) -> List[str]:
    """Return the union of files written by any preceding APPLY task."""
    out: List[str] = []
    seen: set = set()
    for o in (prior or []):
        if not isinstance(o, dict):
            continue
        for f in (o.get("applied_files") or []):
            fp = str(f)
            if fp and fp not in seen:
                seen.add(fp)
                out.append(fp)
    return out


def _build_default_retriever(
    index_dir: Optional[str], records_path: Optional[str],
) -> Optional[Callable[[str], Dict[str, Any]]]:
    """Return a Planner-compatible retriever bound to the project's index.

    Returns ``None`` when an index isn't configured so the Planner falls
    back to LLM-only decomposition rather than failing on the first
    query call.
    """
    if not index_dir or not records_path:
        return None
    if not os.path.isdir(index_dir):
        return None

    def _retrieve(goal: str) -> Dict[str, Any]:
        from cgx.pipeline.auto import run_query_auto
        return run_query_auto(
            index_dir=index_dir, records_path=records_path, query=goal,
            top_k_per_view=10, neighbor_depth=1, use_lexical=True,
        )

    return _retrieve


def _extract_verify_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return outputs of VERIFY tasks that ran and failed (rc != 0)."""
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind != TaskKind.VERIFY or t.status != TaskStatus.FAILED:
            continue
        out = t.output or {}
        if not out.get("ran"):
            continue
        failures.append({
            "tests_selected": out.get("tests_selected") or [],
            "stdout_tail": str(out.get("stdout") or "")[-1200:],
            "stderr_tail": str(out.get("stderr") or "")[-600:],
            "returncode": out.get("returncode"),
            "error": t.error or "",
        })
    return failures


def _verify_failure_is_unrecoverable(
    failures: List[Dict[str, Any]],
    project_root: Optional[str],
) -> Optional[str]:
    """Return a human reason when verify failed for reasons the LLM can't fix.

    Currently detects pytest collection errors (``rc == 2``) whose
    ``ModuleNotFoundError`` names a first-party project directory — that
    is a ``sys.path`` / packaging problem in the sandbox, not a code
    issue the planner can repair by regenerating files. Returning a
    string short-circuits the re-plan loop with that message; returning
    ``None`` lets the normal retry proceed.
    """
    if not failures or not project_root:
        return None
    diagnosis = _diagnose_failure(failures)
    if diagnosis["error_type"] != "import_error":
        return None
    bad = diagnosis.get("bad_imports") or []
    if not bad:
        return None
    root = project_root
    local_hits: List[str] = []
    for mod in bad:
        head = mod.split(".")[0]
        for candidate in (
            os.path.join(root, head),
            os.path.join(root, f"{head}.py"),
            os.path.join(root, "src", head),
            os.path.join(root, "src", f"{head}.py"),
        ):
            if os.path.isdir(candidate) or os.path.isfile(candidate):
                local_hits.append(head)
                break
    if not local_hits:
        return None
    # rc==2 is pytest's "collection / usage error" — distinguishes an
    # infrastructure import failure from a real test that just happens
    # to import a missing helper.
    is_collection_error = any(int(f.get("returncode") or 0) == 2 for f in failures)
    if not is_collection_error:
        return None
    names = ", ".join(sorted(set(local_hits)))
    return (
        f"Verify failed because pytest could not import first-party module(s) "
        f"{names} from {root}. This is a sandbox sys.path / packaging issue, "
        f"not a code generation issue — re-planning would not help."
    )


def _plan_fix_index_available(index_dir: Optional[str]) -> bool:
    """Return True iff ``planner.plan_fix`` can run without crashing.

    ``plan_fix`` emits a PLAN task whose ``plan`` capability loads FAISS
    artifacts from ``<index_dir>/meta.json`` via
    :func:`cgx.io.persist.load_indices`. For a freshly-scaffolded user
    project there is no such index yet, so the PLAN task raises
    ``FileNotFoundError`` before any useful retry work happens. Callers
    check this *before* invoking ``plan_fix`` and fall back to a path
    that doesn't depend on retrieval (``_build_scaffold_retry_plan``)
    or demote the failure when no fallback applies.
    """
    if not index_dir:
        return False
    try:
        return os.path.isfile(os.path.join(index_dir, "meta.json"))
    except (OSError, TypeError):
        return False


def _diagnose_failure(failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse test/apply failure output and return a structured diagnosis.

    Returns a dict with:
      error_type  — "import_error" | "syntax_error" | "logic_error" | "unknown"
      responsible_files — project-relative paths named in tracebacks
      bad_imports — module names that failed to import
      language_mismatch — True when a Python file imports a JS/JSX module
    """
    diagnosis: Dict[str, Any] = {
        "error_type": "unknown",
        "responsible_files": [],
        "bad_imports": [],
        "language_mismatch": False,
    }
    seen_files: set = set()

    for f in failures:
        stdout = str(f.get("stdout_tail") or f.get("stdout") or "")
        stderr = str(f.get("stderr_tail") or f.get("stderr") or "")
        combined = stdout + "\n" + stderr

        # Classify error type — first match wins by priority.
        if diagnosis["error_type"] == "unknown":
            if re.search(r"ModuleNotFoundError|ImportError", combined):
                diagnosis["error_type"] = "import_error"
            elif re.search(r"SyntaxError", combined):
                diagnosis["error_type"] = "syntax_error"
            elif re.search(r"(AssertionError|TypeError|AttributeError|ValueError)", combined):
                diagnosis["error_type"] = "logic_error"

        # Collect bad module names.
        for m in re.finditer(r"No module named '([^']+)'", combined):
            mod = m.group(1)
            if mod not in diagnosis["bad_imports"]:
                diagnosis["bad_imports"].append(mod)

        # Language mismatch: Python file tries to import a JS/JSX module.
        if re.search(r"imports.*from.*is a JavaScript/JSX file", combined):
            diagnosis["language_mismatch"] = True
        # Also infer from module name — "src.App" when App.jsx is known pattern.
        for mod in diagnosis["bad_imports"]:
            mod_path = mod.replace(".", "/")
            if re.search(
                re.escape(mod_path) + r"\.(jsx|js|tsx|ts)\b", combined, re.IGNORECASE
            ):
                diagnosis["language_mismatch"] = True

        # Extract project-relative .py file paths from tracebacks and error lines.
        # Match paths that:
        #   • follow "File " or "Tests: " (pytest / traceback format)
        #   • follow "test module '" (import-collection errors)
        #   • appear at the start of a line (bare "tests/foo.py:2:" lines)
        _PATH_RE = re.compile(
            r'(?:'
            r"(?:File |Tests:\s*)[\"']?|"  # traceback / pytest prefix
            r"test module [\"']|"          # collection-error prefix
            r"(?:^|\n)\s*"                 # line-start (bare path)
            r')'
            r'([^\s\'"<>:\n]+\.py)',
            re.MULTILINE,
        )
        for m in _PATH_RE.finditer(combined):
            fp = m.group(1).strip().rstrip("'\"")
            # Reject stdlib / site-packages paths.
            if any(skip in fp for skip in ("/usr/lib", "/usr/local/lib", "site-packages",
                                           "importlib", "pluggy", "_pytest")):
                continue
            # Normalise to project-relative (strip leading absolute prefix if any).
            norm = fp
            if os.path.isabs(fp):
                parts = fp.lstrip("/").split("/")
                for i, part in enumerate(parts):
                    if part in ("tests", "src", "backend", "app"):
                        norm = "/".join(parts[i:])
                        break
            if norm and norm not in seen_files:
                seen_files.add(norm)
                diagnosis["responsible_files"].append(norm)

    return diagnosis


def _extract_error_snippet(
    project_root: str,
    responsible_files: List[str],
    combined_output: str,
) -> Optional[str]:
    """Extract ±5 lines around the first error line in the failing file.

    The 10-line buffer keeps the retry prompt tightly focused: the model
    sees exactly what broke without drowning in a 200-line traceback.
    Returns ``None`` when no line number or file can be resolved.
    """
    # Pull the first line-number reference from the traceback.
    line_re = re.compile(r'(?:File\s+"[^"]*",\s+line\s+(\d+)|\.py:(\d+))')
    m = line_re.search(combined_output)
    if not m:
        return None
    lineno = int(m.group(1) or m.group(2))

    for fp in responsible_files:
        abs_path = (
            os.path.join(project_root, fp)
            if not os.path.isabs(fp)
            else fp
        )
        if not os.path.exists(abs_path):
            continue
        try:
            file_lines = open(abs_path, encoding="utf-8", errors="ignore").readlines()
        except Exception:
            continue
        start = max(0, lineno - 6)  # 5 lines before the error (0-indexed)
        end = min(len(file_lines), lineno + 5)  # 5 lines after
        snippet_lines: List[str] = []
        for i, line in enumerate(file_lines[start:end], start=start + 1):
            marker = "  # <-- ERROR HERE" if i == lineno else ""
            snippet_lines.append(f"{i:4d}. {line.rstrip()}{marker}")
        if snippet_lines:
            return "\n".join(snippet_lines)
    return None


def _build_fix_goal(original_goal: str, failures: List[Dict[str, Any]],
                    plan: Any, project_root: str) -> str:
    """Compose a targeted fix-goal from failing test output.

    Phase 4 enhancement: instead of feeding the full pytest traceback
    (which overloads small models), we extract the exact error line plus
    5 lines of context above and below it — the "10-line buffer rule".
    """
    from cgx.agents.types import TaskKind
    applied: List[str] = []
    for t in plan.tasks:
        if t.kind == TaskKind.APPLY:
            applied.extend((t.output or {}).get("applied_files") or [])

    diagnosis = _diagnose_failure(failures)

    parts = [
        f"Fix test failures in the project at: {project_root}",
        f"Original goal: {original_goal}",
    ]
    if applied:
        preview = "\n".join(f"  - {f}" for f in applied[:20])
        parts.append(f"Files written to disk:\n{preview}")

    # Targeted file guidance: tell the LLM which files to change and which to leave alone.
    broken = diagnosis["responsible_files"]
    owned = getattr(plan, "owned_files", {}) or {}
    already_good = [fp for fp, st in owned.items() if st == "applied"
                    and fp not in broken]
    if broken:
        parts.append(
            "TARGETED FIX — only regenerate these files (they caused the failure):\n"
            + "\n".join(f"  - {f}" for f in broken[:10])
        )
    if already_good:
        parts.append(
            "DO NOT CHANGE these files (they are correct and already on disk):\n"
            + "\n".join(f"  - {f}" for f in already_good[:20])
        )

    # Augment language_mismatch detection using the file manifest.
    _JS_EXTS = (".jsx", ".js", ".tsx", ".ts", ".mjs")
    if diagnosis["error_type"] == "import_error" and not diagnosis["language_mismatch"]:
        for mod in diagnosis["bad_imports"]:
            mod_path = mod.replace(".", "/")
            for js_ext in _JS_EXTS:
                if (mod_path + js_ext) in owned:
                    diagnosis["language_mismatch"] = True
                    break

    # Error-type guidance.
    etype = diagnosis["error_type"]
    if etype == "import_error" and diagnosis["bad_imports"]:
        bad = ", ".join(f"'{m}'" for m in diagnosis["bad_imports"][:5])
        if diagnosis["language_mismatch"]:
            parts.append(
                f"ERROR TYPE: Python import failure — language mismatch detected.\n"
                f"The module(s) {bad} cannot be imported because they resolve to "
                f"JavaScript/JSX files, not Python modules.\n"
                f"You MUST fix this by doing one of:\n"
                f"  (a) Create a separate Python backend module (e.g. "
                f"backend/calculator.py) that implements the logic the test needs, "
                f"and update the test to import from that module.\n"
                f"  (b) Remove the Python test entirely and write a JavaScript test "
                f"(e.g. with Vitest or Jest) that tests the React component directly.\n"
                f"NEVER import a .jsx / .js / .ts file from a Python test file."
            )
        else:
            parts.append(
                f"ERROR TYPE: Python import failure.\n"
                f"The module(s) {bad} could not be imported.\n"
                f"Check that the module files exist, are on sys.path, and that "
                f"__init__.py files are present where required."
            )
    elif etype == "syntax_error":
        parts.append(
            "ERROR TYPE: Syntax error in generated code.\n"
            "Regenerate the failing file(s) with correct syntax — ensure all "
            "brackets, quotes, and indentation are balanced and complete."
        )

    # Phase 4: 10-line buffer — inject a tight snippet around the first error.
    # This replaces dumping the full raw traceback so small models stay focused.
    combined_for_snippet = "".join(
        str(f.get("stdout_tail") or "") + str(f.get("stderr_tail") or "")
        for f in failures[:1]
    )
    snippet = _extract_error_snippet(project_root, broken, combined_for_snippet)
    if snippet and broken:
        # Format as a micro-targeted prompt block.
        failing_file = broken[0]
        # Extract the error type/message from the first line mentioning the error.
        err_line = ""
        for line in combined_for_snippet.splitlines():
            if re.search(r"Error:|Exception:|assert", line):
                err_line = line.strip()[:160]
                break
        header = (
            f'Your code failed in `{failing_file}` with: {err_line}\n'
            f'Here is the context around the failure:\n'
            f'```python\n{snippet}\n```\n'
            f'Fix this specific issue only.'
        )
        parts.append(header)
    else:
        # Fallback: cap raw output to avoid overwhelming the model.
        for i, f in enumerate(failures[:2], 1):
            parts.append(f"Test failure {i}:")
            if f.get("tests_selected"):
                parts.append("  Tests: " + ", ".join(f["tests_selected"][:5]))
            if f.get("stdout_tail"):
                parts.append("  Output (tail):\n" + str(f["stdout_tail"])[-800:])
            if f.get("stderr_tail"):
                parts.append("  Stderr (tail):\n" + str(f["stderr_tail"])[-400:])

    parts.append(
        "Analyse the failure above, fix only the identified file(s), then re-apply and verify."
    )
    return "\n\n".join(parts)


def _extract_apply_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return APPLY tasks that failed due to syntax/patch errors in generated files.

    This catches the case where scaffold/plan tasks produce code that passes
    the in-pipeline self-test but is rejected by ``apply_diffs_to_disk``'s
    stricter smoke test (e.g. unterminated string literals, bad patches).
    """
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind != TaskKind.APPLY or t.status != TaskStatus.FAILED:
            continue
        out = t.output or {}
        failed_files: List[Dict[str, Any]] = out.get("failed_files") or []
        error = t.error or ((t.judge or {}).get("rationale") or "") if t.judge else t.error or ""
        if failed_files or error:
            failures.append({"failed_files": failed_files, "error": error})
    return failures


def _apply_broken_files(apply_failures: List[Dict[str, Any]]) -> List[str]:
    """Return the project-relative paths that failed APPLY's smoke test."""
    out: List[str] = []
    seen: set = set()
    for f in apply_failures or []:
        for ff in (f.get("failed_files") or []):
            if not isinstance(ff, dict):
                continue
            fname = str(ff.get("file") or "").strip()
            if fname and fname not in seen:
                seen.add(fname)
                out.append(fname)
    return out


def _already_good_files(plan: Any, broken_files: List[str]) -> List[str]:
    """Return files marked ``applied`` in ``plan.owned_files`` minus ``broken_files``.

    Used to tell the planner which files are already on disk and must
    NOT be regenerated by the retry pass.
    """
    owned = getattr(plan, "owned_files", {}) or {}
    broken = set(broken_files or [])
    return sorted(
        fp for fp, status in owned.items()
        if status == "applied" and fp not in broken
    )


def _build_apply_fix_goal(original_goal: str, apply_failures: List[Dict[str, Any]]) -> str:
    """Compose a retry goal when apply fails due to syntax / patch errors."""
    file_errors: List[str] = []
    for f in apply_failures:
        for ff in (f.get("failed_files") or []):
            if not isinstance(ff, dict):
                continue
            fname = str(ff.get("file") or "").strip()
            err   = str(ff.get("error") or "").strip()
            if fname and err:
                file_errors.append(f"  - {fname}: {err}")
            elif fname:
                file_errors.append(f"  - {fname}: patch or syntax error")

    parts = [original_goal]
    parts.append(
        "CRITICAL: The previous attempt generated code that failed the syntax smoke "
        "test and was NOT written to disk. You MUST regenerate the files with "
        "correct syntax. Common causes:\n"
        "  • Unterminated string literals (missing closing quote or triple-quote)\n"
        "  • Unmatched brackets, parentheses, or braces\n"
        "  • Truncated function/class bodies\n"
        "  • Invalid escape sequences inside strings\n"
        "Generate every file in full — do NOT truncate or abbreviate any section."
    )
    if file_errors:
        parts.append("Specific errors to fix:\n" + "\n".join(file_errors))
    return "\n\n".join(parts)


def _extract_core_failures(plan: Any) -> List[Dict[str, Any]]:
    """Return failed generation tasks (judge rejections or execution errors)."""
    from cgx.agents.types import TaskKind, TaskStatus
    failures = []
    for t in plan.tasks:
        if t.kind not in (TaskKind.SCAFFOLD, TaskKind.PLAN,
                          TaskKind.SCAFFOLD_MANIFEST, TaskKind.SCAFFOLD_FILE):
            continue
        if t.status != TaskStatus.FAILED:
            continue
        failures.append({
            "kind": t.kind.value,
            "name": t.name or t.description,
            "path": str((t.inputs or {}).get("path") or "").strip(),
            "error": t.error or "",
            "judge_rationale": ((t.judge or {}).get("rationale") or "") if t.judge else "",
        })
    return failures


def _demote_unrecoverable_verify(plan: Any, reason: str) -> List[Dict[str, Any]]:
    """Convert FAILED VERIFY tasks to SKIPPED when the failure is environmental.

    An ``unrecoverable`` verify failure (sandbox missing pytest, container
    timeout, etc.) doesn't mean the generated code is wrong — the files
    were applied and the project is intact. Demoting the task lets the UI
    show *complete with warnings* instead of a red "failed" badge.

    Returns a list of ``{task_id, name}`` records for the tasks that were
    demoted so the caller can emit ``task_skipped`` events.
    """
    from cgx.agents.types import TaskKind, TaskStatus
    demoted: List[Dict[str, Any]] = []
    for t in plan.tasks:
        if t.kind != TaskKind.VERIFY or t.status != TaskStatus.FAILED:
            continue
        t.status = TaskStatus.SKIPPED
        if isinstance(t.output, dict):
            t.output["skipped_reason"] = reason
            t.output["unrecoverable"] = True
        else:
            t.output = {"skipped_reason": reason, "unrecoverable": True}
        demoted.append({"task_id": t.id, "name": t.name or t.description})
    return demoted


def _scaffold_file_broken_files(failures: List[Dict[str, Any]]) -> List[str]:
    """Return the project-relative paths of failed SCAFFOLD_FILE tasks.

    Used to drive a targeted retry that regenerates only the file(s) the
    Tracker actually rejected, rather than re-running the entire manifest
    (which would discard the sibling files already written to disk).
    """
    out: List[str] = []
    seen: set = set()
    for f in failures or []:
        if f.get("kind") != "scaffold_file":
            continue
        path = str(f.get("path") or "").strip()
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _build_scaffold_retry_plan(
    original_plan: Any,
    broken_files: List[str],
    goal: str,
) -> Plan:
    """Build a [SCAFFOLD_FILE × N, APPLY, VERIFY] plan to regenerate only
    the file(s) whose original SCAFFOLD_FILE task failed.

    Unlike :meth:`Planner.plan_fix`, this path never invokes the ``plan``
    capability, so it does not require the FAISS retriever index to be
    present. ``SCAFFOLD_FILE`` calls ``generate_single_scaffold_file``
    directly against the LLM provider, which is the right tool for a
    file that came back empty or with invalid content on the first pass.

    Each new SCAFFOLD_FILE task copies its ``inputs`` from the matching
    failed task in ``original_plan`` so ``path`` / ``file_description`` /
    ``layer`` / ``goal`` / ``skills`` are preserved. The previous plan's
    ``owned_files`` manifest is carried forward so the Tracker can skip
    any stray scaffold for files that are already on disk.
    """
    from cgx.agents.types import Plan as _Plan, Task as _Task, TaskKind as _TK

    broken_set = {p for p in (broken_files or []) if p}
    orig_by_path: Dict[str, Any] = {}
    for t in (getattr(original_plan, "tasks", None) or []):
        if t.kind != _TK.SCAFFOLD_FILE:
            continue
        p = str((t.inputs or {}).get("path") or "").strip()
        if p in broken_set and p not in orig_by_path:
            orig_by_path[p] = t

    retry_hint = (
        "Previous attempt produced no usable content for this file. "
        "Regenerate the full file from scratch with valid, complete content."
    )
    retry_tasks: List[_Task] = []
    for path in (broken_files or []):
        if not path:
            continue
        original = orig_by_path.get(path)
        base_inputs: Dict[str, Any] = dict((original.inputs if original else {}) or {})
        base_inputs.setdefault("path", path)
        prev_desc = base_inputs.get("file_description")
        if prev_desc:
            base_inputs["file_description"] = f"{prev_desc}\n\n{retry_hint}"
        else:
            base_inputs["file_description"] = f"{path}: {retry_hint}"
        if goal and not base_inputs.get("goal"):
            base_inputs["goal"] = goal
        retry_tasks.append(_Task(
            description=f"Regenerate {path}",
            kind=_TK.SCAFFOLD_FILE,
            name=f"Regenerate {path}",
            criteria=[
                f"File {path} has complete, non-stub content.",
                "File passes syntax validation.",
            ],
            inputs=base_inputs,
        ))

    apply_task = _Task(
        description="Apply the regenerated file(s) to disk after a smoke test.",
        kind=_TK.APPLY,
        name="Apply regenerated files",
        criteria=["Every diff applies without rejected hunks.",
                  "Modified files parse as valid syntax."],
    )
    verify_task = _Task(
        description="Run impacted tests against the project tree.",
        kind=_TK.VERIFY,
        name="Verify with tests",
        criteria=["Impacted tests located and executed.",
                  "Test suite returns a zero exit code."],
    )

    plan_goal = goal or getattr(original_plan, "goal", "") or ""
    plan = _Plan(
        goal=plan_goal,
        tasks=[*retry_tasks, apply_task, verify_task],
        rationale=("Scaffold retry: regenerate only the file(s) that failed "
                   "the previous attempt; siblings already on disk are preserved."),
        owned_files=dict(getattr(original_plan, "owned_files", {}) or {}),
    )
    for i in range(1, len(plan.tasks)):
        plan.tasks[i].dependencies = [plan.tasks[i - 1].id]
    logger.info(
        "loop: scaffold-retry plan id=%s broken=%d carried_owned=%d",
        plan.id, len(retry_tasks), len(plan.owned_files),
    )
    return plan


# Original pattern: judge sometimes says "missing required files: X"
_MISSING_FILES_RE = re.compile(
    r"missing required files:\s*([^|\n]+)", re.IGNORECASE
)
# Judge's actual parenthetical format: "no packaging file (package.json)"
# or "no entry module (app.py, main.py, ...)"
_PARENS_FILE_RE = re.compile(
    r"no\s+(?:packaging\s+file|entry\s+module)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def _manifest_required_files_from_goal(goal: str) -> List[str]:
    """Derive the files the Judge will require from the goal text.

    Mirrors the exact tables in :mod:`cgx.agents.judge`
    (``_SKILL_REQUIRED_FILES``, ``_LANGUAGE_REQUIRED_FILES``,
    ``_LANGUAGE_BACKEND_EXTS``) so the retry prompt tells the model
    precisely what to include — without depending on the judge's
    rationale string format.

    This fixes the infinite-retry cycle where the model alternates
    between two broken manifests (one missing ``package.json``, one
    missing ``.py`` files) because the vague fallback rationale gives it
    no actionable direction.
    """
    goal_low = goal.lower()
    required: List[str] = []

    def _add(f: str) -> None:
        if f not in required:
            required.append(f)

    # ── Frontend frameworks ──────────────────────────────────────────
    if re.search(r"\bnext\.?js\b", goal_low):
        _add("package.json")
        _add("pages/index.js or app/page.js (Next.js entry)")
    elif re.search(r"\breact\b", goal_low):
        _add("package.json")
        _add("src/App.jsx or src/App.tsx (React root component)")
    elif re.search(r"\bvue\b", goal_low):
        _add("package.json")
        _add("src/App.vue (Vue root component)")
    elif re.search(r"\bsvelte\b", goal_low):
        _add("package.json")
        _add("src/App.svelte")
    elif re.search(r"\bexpress\b", goal_low):
        _add("package.json")
        _add("index.js or server.js (Express entry)")

    # ── Python backend frameworks ────────────────────────────────────
    if re.search(r"\bfastapi\b", goal_low):
        _add("requirements.txt")
        _add("main.py (FastAPI entry)")
    elif re.search(r"\bflask\b", goal_low):
        _add("requirements.txt")
        _add("app.py (Flask entry)")
    elif re.search(r"\bdjango\b", goal_low):
        _add("requirements.txt")
        _add("manage.py (Django entry)")
    elif re.search(r"\bpython\b", goal_low):
        # Generic Python — require requirements.txt and an entry module
        # when a "backend / server / api" keyword is also present.
        _add("requirements.txt")
        if any(kw in goal_low for kw in ("backend", "server", "api")):
            _add("main.py or app.py or server.py (Python backend entry)")

    return required


def _build_core_fix_goal(original_goal: str, failures: List[Dict[str, Any]]) -> str:
    """Compose a retry goal from scaffold/plan failure diagnostics.

    The shape is tailored to the failing kind: manifest failures get a
    short ``MUST INCLUDE`` directive (small models drown in verbose
    instructions and emit an empty manifest on retry), file-generation
    failures keep the full syntax/completeness guidance.

    For manifest failures the required-file list is derived in two ways:
    1. **Goal-text derivation** via :func:`_manifest_required_files_from_goal`
       — mirrors the Judge's own tables so the model receives positive
       instructions ("include X") instead of a vague rejection sentence.
    2. **Rationale parsing** — extracts filenames from the Judge's
       parenthetical format ("no packaging file (package.json)") and from
       the legacy "missing required files: X" pattern, as a supplementary
       signal.

    This combination eliminates the infinite-retry cycle that occurs when
    ``_MISSING_FILES_RE`` cannot match the judge's actual output and the
    fallback vague sentence gives the 3B model no actionable direction.
    """
    manifest_kinds = {"scaffold_manifest"}
    manifest_only = bool(failures) and all(
        f["kind"] in manifest_kinds for f in failures
    )

    rationales: List[str] = []
    required_files: List[str] = []

    for f in failures:
        reason = (f["judge_rationale"] or f["error"] or "").strip()
        if reason:
            rationales.append(reason)
        # Legacy pattern: "missing required files: X, Y"
        for m in _MISSING_FILES_RE.finditer(reason):
            for piece in m.group(1).split(","):
                p = piece.strip().rstrip(".").strip()
                if p and p not in required_files:
                    required_files.append(p)
        # Judge's actual format: "no packaging file (package.json)"
        # and "no entry module (app.py, main.py, ...)"
        for m in _PARENS_FILE_RE.finditer(reason):
            for piece in m.group(1).split(","):
                p = piece.strip().rstrip(".").strip("'")
                if p and p not in required_files:
                    required_files.append(p)

    parts: List[str] = [original_goal.strip()]
    if manifest_only:
        # Derive required files directly from the goal text — this is more
        # robust than parsing the judge rationale and ensures the model
        # always gets a positive "MUST INCLUDE X" instruction even when the
        # regex extraction finds nothing.
        for gf in _manifest_required_files_from_goal(original_goal):
            if gf not in required_files:
                required_files.append(gf)

        if required_files:
            parts.append(
                "MUST INCLUDE these files in the manifest (every one is required):\n"
                + "\n".join(f"  - {p}" for p in required_files[:20])
            )
        if rationales:
            # Include the first rationale as context but after the directive
            # so the model sees the requirement first, not the rejection.
            parts.append("Previous attempt rejected because: " + rationales[0][:200])
    else:
        if rationales:
            parts.append(
                "The previous attempt had the following issues that MUST be fixed:\n"
                + "\n".join(f"  - {r}" for r in rationales)
            )
        parts.append(
            "Regenerate the project from scratch, ensuring all files are syntactically "
            "complete, all brackets/braces closed, all imports resolve, and all logic is "
            "fully implemented. Do not truncate any file."
        )
    return "\n\n".join(parts)


def _stream_with_retry(
    plan_obj: Any,
    tracker: Tracker,
    planner: Planner,
    capabilities: Dict[str, Callable[..., Dict[str, Any]]],
    judge: Optional[Judge],
    goal: str,
    project_root: Optional[str],
    stop_on_fail: bool,
    progress_interval: float,
    max_retries: int,
    attempt: int = 1,
    index_dir: Optional[str] = None,
) -> Iterator[Any]:
    """Yield events from the initial plan, then auto-retry if tasks failed.

    Each plan (initial + retries) is streamed exactly once. Retry plans
    are surfaced to the UI by rewriting their ``plan`` event as
    ``retry_plan`` and prefixing the stream with a ``retry_start`` event.
    """
    from cgx.agents.types import AgentEvent

    current_plan = plan_obj
    current_tracker = tracker
    current_goal = goal
    attempts_left = max_retries

    while True:
        # Stream the current plan. The first attempt's plan events pass
        # through unchanged; subsequent retries rewrite ``plan`` →
        # ``retry_plan`` so the frontend appends new task rows rather
        # than replacing the existing timeline.
        if attempt == 1:
            for ev in current_tracker.stream(current_plan):
                yield ev
        else:
            for ev in current_tracker.stream(current_plan):
                if ev.type == "plan":
                    yield AgentEvent(type="retry_plan", payload=ev.payload)
                else:
                    yield ev

        # Gather all classes of failure up-front so the priority order can
        # consider them together. Fix A makes SCAFFOLD_FILE failures "soft",
        # which means APPLY/VERIFY now run even when individual file
        # generation tasks failed — so a verify failure here is often a
        # *cascade* of an unfixed file, not an independent test bug.
        verify_failures = _extract_verify_failures(current_plan)
        core_failures   = _extract_core_failures(current_plan)

        # Even when the retry budget is exhausted, an unrecoverable verify
        # failure should still be demoted so the run finishes "Complete
        # with warnings" rather than red. Without this, the final retry's
        # VERIFY task stays in FAILED status because the in-loop
        # demotion path (priority-1 / priority-2 branches below) is gated
        # on entering a new retry attempt.
        if attempts_left <= 0:
            if verify_failures and project_root:
                unrecoverable = _verify_failure_is_unrecoverable(
                    verify_failures, project_root,
                )
                if unrecoverable:
                    demoted = _demote_unrecoverable_verify(
                        current_plan, unrecoverable,
                    )
                    for entry in demoted:
                        yield AgentEvent(
                            type="task_skipped",
                            payload={"task_id": entry["task_id"],
                                     "name": entry["name"],
                                     "reason": unrecoverable},
                        )
            return
        scaffold_file_paths = _scaffold_file_broken_files(core_failures)
        scaffold_file_only_core = bool(scaffold_file_paths) and all(
            f["kind"] == "scaffold_file" for f in core_failures
        )

        # Priority 1: SCAFFOLD_FILE soft-failures. These have a targeted
        # retry path (re-run only the failed SCAFFOLD_FILE tasks, then
        # APPLY+VERIFY) and are almost always the root cause when verify
        # also failed — a missing or empty file breaks every downstream
        # test run. The retry deliberately avoids ``plan_fix`` because
        # ``plan_fix`` emits a PLAN task whose engine requires the FAISS
        # retriever index, which doesn't exist for a freshly-scaffolded
        # user project; SCAFFOLD_FILE only needs the LLM provider.
        use_scaffold_retry = False
        if scaffold_file_only_core:
            # If verify also failed and is environmental, demote it now so
            # the UI doesn't show a red badge while the file-level retry
            # is in flight. The retry itself may obviate the verify failure
            # entirely (e.g. by writing the missing package.json).
            if verify_failures and project_root:
                unrecoverable = _verify_failure_is_unrecoverable(
                    verify_failures, project_root,
                )
                if unrecoverable:
                    demoted = _demote_unrecoverable_verify(
                        current_plan, unrecoverable,
                    )
                    for entry in demoted:
                        yield AgentEvent(
                            type="task_skipped",
                            payload={"task_id": entry["task_id"],
                                     "name": entry["name"],
                                     "reason": unrecoverable},
                        )
            logger.info("run_agent: %d scaffold file failure(s) on attempt %d "
                        "— regenerating only those files",
                        len(scaffold_file_paths), attempt)
            fix_goal = _build_core_fix_goal(current_goal, core_failures)
            retry_reason = (
                f"{len(scaffold_file_paths)} file(s) failed generation — "
                "regenerating only those files"
            )
            broken_files = scaffold_file_paths
            use_fix_plan = True
            use_scaffold_retry = True
        # Priority 2: test failures (VERIFY) — build a targeted fix goal.
        elif verify_failures:
            if not project_root:
                return
            unrecoverable = _verify_failure_is_unrecoverable(
                verify_failures, project_root,
            )
            # When the FAISS index is missing (typical for a freshly-
            # scaffolded user project), ``plan_fix`` would crash on
            # ``meta.json`` before any retry work could happen, and the
            # outer fallback would then trigger a full re-plan that
            # restarts the whole scaffold from scratch. Treat that as
            # unrecoverable too: the files are already on disk, so the
            # right answer is to leave them for manual review rather
            # than scaffold-storm the project.
            if not unrecoverable and not _plan_fix_index_available(index_dir):
                unrecoverable = (
                    "Verify failed but the retry planner has no search "
                    "index for this project yet — leaving files on disk "
                    "for manual review instead of regenerating everything."
                )
            if unrecoverable:
                logger.info(
                    "run_agent: verify failure on attempt %d is unrecoverable — "
                    "skipping re-plan (%s)", attempt, unrecoverable,
                )
                # Demote each failed VERIFY to SKIPPED so the UI doesn't
                # flag the whole run as failed when the files are already
                # on disk and the only issue was the test runner itself.
                demoted = _demote_unrecoverable_verify(current_plan, unrecoverable)
                for entry in demoted:
                    yield AgentEvent(
                        type="task_skipped",
                        payload={"task_id": entry["task_id"],
                                 "name": entry["name"],
                                 "reason": unrecoverable},
                    )
                yield AgentEvent(
                    type="retry_skipped",
                    payload={"attempt": attempt, "reason": unrecoverable},
                )
                if demoted:
                    from cgx.agents.types import TaskStatus
                    completed = sum(1 for t in current_plan.tasks
                                    if t.status == TaskStatus.DONE)
                    failed = sum(1 for t in current_plan.tasks
                                 if t.status == TaskStatus.FAILED)
                    skipped = sum(1 for t in current_plan.tasks
                                  if t.status == TaskStatus.SKIPPED)
                    yield AgentEvent(
                        type="summary",
                        payload={"plan": current_plan.to_dict(),
                                 "completed": completed,
                                 "failed": failed,
                                 "skipped": skipped},
                    )
                return
            logger.info("run_agent: %d verify failure(s) on attempt %d — re-planning",
                        len(verify_failures), attempt)
            fix_goal = _build_fix_goal(current_goal, verify_failures,
                                       current_plan, project_root)
            retry_reason = f"{len(verify_failures)} test failure(s) detected — re-planning to fix"
            broken_files = list(_diagnose_failure(verify_failures)["responsible_files"])
            use_fix_plan = True
        # Priority 3: non-scaffold core failures (manifest / plan) — these
        # require a fresh full plan rather than a delta plan because the
        # upstream task that produces the file list is what's broken.
        elif core_failures:
            logger.info("run_agent: %d core failure(s) on attempt %d — re-planning",
                        len(core_failures), attempt)
            fix_goal = _build_core_fix_goal(current_goal, core_failures)
            retry_reason = f"{len(core_failures)} generation failure(s) — retrying with fixes"
            broken_files = []
            use_fix_plan = False
        else:
            # Priority 4: apply failures (syntax / patch errors in generated
            # files) — fires when scaffold/plan produced syntactically
            # invalid code that passed the in-pipeline check but was
            # rejected by apply's smoke test.
            apply_failures = _extract_apply_failures(current_plan)
            if not apply_failures:
                return
            logger.info("run_agent: %d apply failure(s) on attempt %d — regenerating",
                        len(apply_failures), attempt)
            fix_goal = _build_apply_fix_goal(current_goal, apply_failures)
            retry_reason = (
                f"{sum(len(f['failed_files']) for f in apply_failures)} file(s) had "
                "syntax / patch errors — regenerating with fixes"
            )
            broken_files = _apply_broken_files(apply_failures)
            # When the FAISS index is missing, ``plan_fix`` would crash;
            # route the retry through the scaffold-retry plan which
            # regenerates the specific files that failed without needing
            # any retrieval. ``_build_scaffold_retry_plan`` also handles
            # files that have no original SCAFFOLD_FILE task by
            # synthesising fresh inputs from the path.
            if _plan_fix_index_available(index_dir):
                use_fix_plan = True
            elif broken_files:
                use_scaffold_retry = True
            else:
                return

        # Plan the next attempt against the failure rationale, then loop
        # back to stream it. A fresh Tracker is used per attempt so its
        # internal state (if any) doesn't bleed across retries.
        if use_scaffold_retry:
            next_plan = _build_scaffold_retry_plan(
                current_plan, broken_files, fix_goal,
            )
        elif use_fix_plan:
            already_good = _already_good_files(current_plan, broken_files)
            next_plan = planner.plan_fix(
                fix_goal,
                broken_files=broken_files,
                already_good_files=already_good,
                prior_owned_files=getattr(current_plan, "owned_files", {}),
            )
        else:
            next_plan = planner.plan(fix_goal)
        next_tracker = Tracker(
            capabilities=capabilities, judge=judge,
            stop_on_fail=stop_on_fail, progress_interval=progress_interval,
        )

        yield AgentEvent(
            type="retry_start",
            payload={"attempt": attempt + 1, "reason": retry_reason},
        )

        current_plan = next_plan
        current_tracker = next_tracker
        current_goal = fix_goal
        attempts_left -= 1
        attempt += 1


def run_agent(
    goal: str,
    *,
    provider: Any = None,
    index_dir: Optional[str] = None,
    records_path: Optional[str] = None,
    project_root: Optional[str] = None,
    capabilities: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    planner: Optional[Planner] = None,
    judge: Optional[Judge] = None,
    stop_on_fail: bool = True,
    stream: bool = False,
    progress_interval: float = 2.0,
    max_retries: int = 1,
) -> Any:
    """Run a Planner → Tracker → Judge loop for ``goal``.

    Parameters
    ----------
    goal
        Natural-language user request.
    provider
        Optional :class:`~cgx.answer.providers.LLMProvider`. Required for
        the planner's LLM path and for any ``ask``/``plan`` capability;
        the deterministic fallback plan still runs without it.
    index_dir, records_path
        Paths to the indexed artifacts. Required by the default
        capabilities; ignored when ``capabilities`` is supplied.
    project_root
        Forwarded to ``generate_code_plan`` for the self-test sandbox.
    capabilities
        Override the default capability table (useful for tests).
    planner / judge
        Override the default ``Planner`` / ``Judge`` (e.g. to inject a
        deterministic stub).
    stop_on_fail
        Halt the plan after the first failed task (default True).
    stream
        If True, return a generator of :class:`AgentEvent`. If False,
        return the final :class:`Plan` after running to completion.
    """
    if planner is None:
        retriever = _build_default_retriever(index_dir, records_path)
        planner = Planner(provider=provider, retriever=retriever)
    plan_obj: Plan = planner.plan(goal)
    if capabilities is None:
        capabilities = _build_default_capabilities(
            provider=provider, index_dir=index_dir,
            records_path=records_path, project_root=project_root,
        )
    judge = judge if judge is not None else Judge(provider=provider)
    tracker = Tracker(capabilities=capabilities, judge=judge,
                      stop_on_fail=stop_on_fail,
                      progress_interval=progress_interval)
    if stream:
        return _stream_with_retry(
            plan_obj, tracker, planner, capabilities, judge,
            goal, project_root, stop_on_fail, progress_interval, max_retries,
            index_dir=index_dir,
        )
    tracker.run(plan_obj)
    if max_retries > 0:
        verify_failures = _extract_verify_failures(plan_obj)
        apply_failures  = _extract_apply_failures(plan_obj)
        core_failures   = _extract_core_failures(plan_obj)
        fix_goal: Optional[str] = None
        broken_files: List[str] = []
        use_fix_plan = False
        use_scaffold_retry = False
        # Priority order mirrors _stream_with_retry: SCAFFOLD_FILE
        # soft-failures first (root cause when verify also failed), then
        # verify, then non-scaffold core, then apply.
        scaffold_file_paths = _scaffold_file_broken_files(core_failures)
        scaffold_file_only_core = bool(scaffold_file_paths) and all(
            f["kind"] == "scaffold_file" for f in core_failures
        )
        if scaffold_file_only_core:
            if verify_failures and project_root:
                unrecoverable = _verify_failure_is_unrecoverable(
                    verify_failures, project_root,
                )
                if unrecoverable:
                    _demote_unrecoverable_verify(plan_obj, unrecoverable)
            fix_goal = _build_core_fix_goal(goal, core_failures)
            broken_files = scaffold_file_paths
            use_fix_plan = True
            use_scaffold_retry = True
        elif verify_failures and project_root:
            unrecoverable = _verify_failure_is_unrecoverable(
                verify_failures, project_root,
            )
            if not unrecoverable and not _plan_fix_index_available(index_dir):
                unrecoverable = (
                    "Verify failed but the retry planner has no search "
                    "index for this project yet — leaving files on disk "
                    "for manual review instead of regenerating everything."
                )
            if unrecoverable:
                logger.info(
                    "run_agent: verify failure is unrecoverable — skipping retry (%s)",
                    unrecoverable,
                )
                _demote_unrecoverable_verify(plan_obj, unrecoverable)
                return plan_obj
            fix_goal = _build_fix_goal(goal, verify_failures, plan_obj, project_root)
            broken_files = list(_diagnose_failure(verify_failures)["responsible_files"])
            use_fix_plan = True
        elif core_failures:
            fix_goal = _build_core_fix_goal(goal, core_failures)
        elif apply_failures:
            fix_goal = _build_apply_fix_goal(goal, apply_failures)
            broken_files = _apply_broken_files(apply_failures)
            if _plan_fix_index_available(index_dir):
                use_fix_plan = True
            elif broken_files:
                use_scaffold_retry = True
            else:
                fix_goal = None
        if fix_goal:
            if use_scaffold_retry:
                fix_plan = _build_scaffold_retry_plan(
                    plan_obj, broken_files, fix_goal,
                )
            elif use_fix_plan:
                already_good = _already_good_files(plan_obj, broken_files)
                fix_plan = planner.plan_fix(
                    fix_goal,
                    broken_files=broken_files,
                    already_good_files=already_good,
                    prior_owned_files=getattr(plan_obj, "owned_files", {}),
                )
            else:
                fix_plan = planner.plan(fix_goal)
            fix_tracker = Tracker(capabilities=capabilities, judge=judge,
                                  stop_on_fail=stop_on_fail,
                                  progress_interval=progress_interval)
            fix_tracker.run(fix_plan)
            # Mirror the streaming path: if the retry's VERIFY is also an
            # unrecoverable sandbox/env failure, demote it so the run
            # finishes "Complete with warnings" instead of red.
            fix_verify_failures = _extract_verify_failures(fix_plan)
            if fix_verify_failures and project_root:
                unrecoverable = _verify_failure_is_unrecoverable(
                    fix_verify_failures, project_root,
                )
                if unrecoverable:
                    _demote_unrecoverable_verify(fix_plan, unrecoverable)
            return fix_plan
    return plan_obj
