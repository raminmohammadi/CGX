"""The Reviser executor: apply user feedback to a finished swarm build.

The loop this closes: *user asks -> the swarm builds -> the user reviews the
running app (screenshots, live preview) -> the user says what to change ->
the agent must know exactly WHERE and HOW to apply it -> the result is
rebuilt and re-verified -> the user reviews again.*

For each piece of feedback the executor:

1. **Localizes** it to the planned files that own the change. First
   deterministically (:func:`cgx.session.revise.rank_files`: named
   components, routes/screens, quoted UI copy found in the sources, concern
   keywords mapped to file roles, the routes the last runtime smoke flagged),
   then with ONE bounded LLM call that sees the ranked candidates' exported
   symbols and the exact source lines that mention the feedback, and returns
   a per-file instruction restricted to planned paths. If the model declines
   or fails, the deterministic candidates are used with the raw feedback.
2. **Applies** each instruction as a grounded *modification* of the current
   file through the same generation ladder the Developer uses (full current
   body quoted, design brief, contracts, syntax/import gates), and checks the
   file actually changed -- a weak model that hands the file back unchanged
   gets one firmer re-ask. A theme request ("dark mode", a named theme)
   switches the deterministic tokens stylesheet with no model call.
3. **Reports** what changed where (per-file instruction, why that file, a
   capped unified diff) in a ``SWARM_REVISION`` artifact, appends the
   revision to the WORK_PLAN's history, and hands the tree to
   ``SWARM_VERIFY`` so the revised app is rebuilt, smoke-tested and
   screenshotted before the user sees it.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from cgx.session.models import Artifact, ArtifactKind, TaskKind
from cgx.session.revise import (
    Candidate, ancestor_chain, chain_with_rules, component_classes, diagnose_width_constraints,
    diff_covers, instruction_is_noop, instruction_tokens, is_layout_feedback, page_level_rules,
    rank_files, relevant_css_blocks, relevant_lines, rules_for_classes, rules_for_selector,
    split_feedback, theme_from_feedback, unified_diff)
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, TaskNode, register_executor)
from cgx.session.tasks.scaffold import _emit_scaffold_progress
from cgx.session.tasks.swarm_generate import generate_file
from cgx.session.tasks.swarm_ground import _safe_read
from cgx.session.tasks.swarm_log import swarm_beat
from cgx.session.tasks.swarm_plan import plan_specs
from cgx.session.tasks.swarm_tools import edit_file

#: How much of the current file is quoted back for a revision. A revision must
#: see the WHOLE file to change one thing and keep the rest; the Developer's
#: default cap (2 000 chars) is for dependency context, not for this.
_REVISE_EXISTING_LIMIT = 16_000
#: Most (file, instruction) changes one message may produce. Multi-part
#: feedback gets one instruction per part per file; beyond this the strongest
#: are applied and the rest reported as skipped.
_MAX_TARGETS = 6

_LOCALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "instruction": {"type": "string"},
                    "why": {"type": "string"},
                    "part": {"type": "integer"},
                },
                "required": ["path", "instruction"],
            },
        },
        "note": {"type": "string"},
    },
    "required": ["changes"],
}

_LOCALIZE_SYSTEM = (
    "You are the Tech Lead of a small team that just built the application described "
    "below. The user has reviewed the RUNNING app and sent feedback. Decide exactly WHICH "
    "planned files must change and WHAT to change in each, so a developer can apply it "
    "without asking anything.\n\n"
    "Return strict JSON only:\n"
    '{"changes": [{"path": "<a planned path>", "part": <feedback part number>, '
    '"instruction": "<the precise change for THIS file: name the CSS selector / element / '
    'function / prop to change and the exact new value, behaviour or appearance>", '
    '"why": "<one short clause: why this file>"}], '
    '"note": "<one sentence for the user: what will change, where>"}\n\n'
    "Rules:\n"
    "- Use ONLY paths from ALL PLANNED FILES. Never invent a file.\n"
    "- The feedback is split into numbered PARTS. EVERY part must be covered by at least "
    "one change. Return ONE change per (file, part): if one file needs edits for two "
    "parts, return two entries for that file, each with its own small instruction.\n"
    "- Prefer the FEWEST files per part (usually one). A visual/layout change belongs in "
    "the app stylesheet (its classes) and/or the component that renders the element; a "
    "behaviour change belongs in the component or helper that owns it; a wording change "
    "belongs in the component that renders the text.\n"
    "- For a size/width/position complaint, FIRST find the existing CSS rule that actually "
    "constrains the element -- it is often the PARENT container (a max-width, "
    "align-items:center, a fixed height), shown under 'styles' -- and name THAT selector "
    "and property in the instruction. Do not invent a new class the stylesheet does not "
    "define unless the instruction also defines it.\n"
    "- Never choose the design tokens stylesheet (CGX generates it).\n"
    "- Be concrete: 'in .app set align-items: stretch so the header spans the page' is "
    "good; 'improve the header' is not.\n"
    "- The LIKELY FILES were found by a deterministic scan with the reasons given; trust "
    "them unless the source lines show the change belongs elsewhere.\n"
    "- If the feedback is not actionable in code, return {\"changes\": []}.\n"
)


def _load_artifact(deps: ExecutorDeps, artifact_id: str) -> Any:
    if not artifact_id or deps.store is None:
        return None
    try:
        return deps.store.get_artifact(artifact_id)
    except Exception:  # pragma: no cover - store best-effort
        return None


def _collect(paths: Sequence[str], root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        src = _safe_read(p, root)
        if src is not None:
            out[p] = src
    return out


def _design_summary(design: Optional[Dict[str, Any]]) -> str:
    if not design or not design.get("applies"):
        return ""
    bits = [
        f"product: {design.get('product_name') or 'app'}",
        f"archetype: {design.get('archetype_name') or design.get('archetype') or 'generic'}",
        f"framework: {design.get('framework') or '?'} ({design.get('styling') or 'css'})",
        f"theme: {design.get('theme') or '?'}",
    ]
    screens = design.get("screens") or []
    if screens:
        bits.append("screens: " + "; ".join(str(s) for s in screens[:6]))
    paths = [f"{k.replace('_path', '')}={design.get(k)}" for k in
             ("app_path", "entry_path", "global_css_path", "api_client_path", "storage_path")
             if design.get(k)]
    if paths:
        bits.append("canonical files: " + ", ".join(paths))
    return "\n".join("- " + b for b in bits)


def _symbols_line(path: str, src: str) -> str:
    low = path.lower()
    names: List[str] = []
    if low.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs")):
        try:
            from cgx.session.js_symbols import js_exports
            ex = js_exports(src)
            names = list(ex.get("named") or [])
            if ex.get("default"):
                names.append(f"default:{ex['default']}")
        except Exception:  # pragma: no cover
            names = []
    elif low.endswith(".py"):
        names = re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", src, re.M)
    elif low.endswith((".css", ".scss")):
        names = sorted({m for m in re.findall(r"^\s*\.([a-zA-Z][\w-]*)", src, re.M)})[:30]
        return "classes: " + ", ".join(names) if names else ""
    return "exports: " + ", ".join(names[:20]) if names else ""


def _is_stylesheet(path: str) -> bool:
    return path.lower().endswith((".css", ".scss", ".sass", ".less"))


def _importers_of(path: str, contents: Dict[str, str], paths: Sequence[str]) -> List[str]:
    """Planned files that import ``path`` (the components that render it)."""
    try:
        from cgx.session.js_symbols import js_imports, resolve_relative
    except Exception:  # pragma: no cover
        return []
    out: List[str] = []
    for other, src in contents.items():
        if other == path or not other.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
            continue
        for stmt in js_imports(src or ""):
            if resolve_relative(stmt.spec, other, paths) == path:
                out.append(other)
                break
    return out


def _context_classes(candidates: List[Candidate], contents: Dict[str, str],
                     paths: Sequence[str]) -> List[str]:
    """Classes rendered by the candidate components AND by the files that render them.

    A header is narrow because its parent is ``display:flex; align-items:center``
    -- the constraint lives in the container's rule, so the container's classes
    are part of the picture.
    """
    out: List[str] = []
    for c in candidates:
        if _is_stylesheet(c.path):
            continue
        files = [c.path] + _importers_of(c.path, contents, paths)
        for f in files:
            for cls in component_classes(contents.get(f) or ""):
                if cls not in out:
                    out.append(cls)
    return out


def _styles_for(classes: Sequence[str], contents: Dict[str, str], paths: Sequence[str],
                tokens_path: str, *, limit: int = 10, page_level: bool = False) -> str:
    """``file:line selector { body }`` for the rules that style ``classes`` (all stylesheets).

    With ``page_level`` the ``html`` / ``body`` / ``#root`` rules come first:
    they are ancestors of every component, and a centring ``body`` or a
    ``#root`` max-width is a classic reason an element "does not span the page".
    """
    lines: List[str] = []
    stylesheets = [p for p in paths if _is_stylesheet(p)]

    def _fmt(p: str, rule: Dict[str, Any]) -> str:
        tag = (" (design tokens -- do not edit; override in the app stylesheet)"
               if p == tokens_path else "")
        return f"{p}:{rule['line']} {rule['selector']} {{ {rule['body'][:200]} }}{tag}"

    if page_level:
        for p in stylesheets:
            for rule in page_level_rules(contents.get(p) or ""):
                lines.append(_fmt(p, rule) + "  <- page-level ancestor")
    for p in stylesheets:
        for rule in rules_for_classes(contents.get(p) or "", classes, limit=limit):
            lines.append(_fmt(p, rule))
            if len(lines) >= limit + 4:
                return "\n".join(lines)
    return "\n".join(lines)


def _ancestry_block(chain: List[Dict[str, str]], contents: Dict[str, str], paths: Sequence[str],
                    tokens_path: str) -> str:
    lines: List[str] = []
    stylesheets = [p for p in paths if _is_stylesheet(p)]
    for i, ring in enumerate(chain):
        sel = ring["selector"]
        found: List[str] = []
        for p in stylesheets:
            for rule in rules_for_selector(contents.get(p) or "", sel):
                tag = " [tokens]" if p == tokens_path else ""
                found.append(f"{p}:{rule['line']}{tag} {{ {rule['body'][:160]} }}")
        indent = "  " * i
        if found:
            lines.append(f"{indent}{sel}  ({ring['source']})")
            lines.extend(f"{indent}    {f}" for f in found[:3])
        else:
            lines.append(f"{indent}{sel}  ({ring['source']}) -- NO rule in any stylesheet (content-sized)")
    return "\n".join(lines)


def _candidate_block(c: Candidate, specs: Dict[str, Any], contents: Dict[str, str],
                     feedback: str, *, paths: Sequence[str] = (),
                     context_classes: Sequence[str] = (), tokens_path: str = "") -> str:
    src = contents.get(c.path) or ""
    desc = str((specs.get(c.path) or {}).get("description") or "").strip()
    head = f"### {c.path}  (why: {'; '.join(c.reasons) or 'planned file'})"
    parts = [head]
    if desc:
        parts.append("purpose: " + desc[:240])
    sym = _symbols_line(c.path, src)
    if sym:
        parts.append(sym)
    if _is_stylesheet(c.path):
        # whole rules, not bare lines: a "max-width" means nothing without its selector
        blocks = relevant_css_blocks(src, feedback, context_classes)
        if blocks:
            parts.append("relevant rules:\n" + blocks)
    else:
        lines = relevant_lines(src, feedback)
        if lines:
            parts.append("relevant lines:\n" + lines)
        classes = component_classes(src)
        if classes:
            importers = _importers_of(c.path, contents, paths)
            container = [cls for f in importers for cls in component_classes(contents.get(f) or "")]
            styles = _styles_for(classes + container, contents, paths, tokens_path,
                                 page_level=is_layout_feedback(feedback))
            if importers:
                parts.append("rendered by: " + ", ".join(importers)
                             + (f" (container classes: {', '.join(container[:6])})" if container else ""))
            if is_layout_feedback(feedback):
                # The DOM ancestry with each ring's rules: a width/position
                # constraint on ANY ring (a centring body, an unstyled mount
                # root) narrows everything inside it.
                chain = ancestor_chain(c.path, contents, paths, importers_of=_importers_of)
                parts.append("DOM ANCESTRY (outermost first) with each ring's rules -- a constraint on any "
                             "ring narrows everything inside it; fix the ring that constrains:\n"
                             + _ancestry_block(chain, contents, paths, tokens_path))
                stylesheets = {p: contents.get(p) or "" for p in paths if _is_stylesheet(p)}
                diagnosis = diagnose_width_constraints(chain_with_rules(chain, stylesheets), feedback)
                if diagnosis:
                    parts.append("LAYOUT DIAGNOSIS (deterministic CSS analysis of the ancestry -- the "
                                 "instruction MUST fix the FIRST item at its own ring, in the app "
                                 "stylesheet; do not edit rings below it):\n"
                                 + "\n".join(f"- {d['ring']}: {d['problem']}. Fix: `{d['fix']}`" for d in diagnosis))
            if styles:
                parts.append("styles for this component and its container:\n" + styles)
    if not src:
        parts.append("(file is missing on disk)")
    return "\n".join(parts)


def refine_with_model(provider: Any, *, feedback: str, goal: str,
                      design: Optional[Dict[str, Any]], candidates: List[Candidate],
                      specs: Dict[str, Any], contents: Dict[str, str],
                      paths: Sequence[str], previous: Sequence[Dict[str, Any]],
                      log_root: Optional[str] = None,
                      parts: Optional[Sequence[str]] = None,
                      part_candidates: Optional[Dict[int, List[Candidate]]] = None
                      ) -> Optional[Dict[str, Any]]:
    """One bounded LLM call: ranked candidates -> ``{changes: [...], note}``.

    Returns ``None`` when the provider fails or the reply is unusable, so the
    caller falls back to the deterministic candidates. Paths the plan does not
    own are dropped; the tokens stylesheet is dropped; at most ``_MAX_TARGETS``
    survive, deterministic candidates first. ``parts`` are the feedback's
    separately actionable clauses (see :func:`split_feedback`); each returned
    change carries the ``part`` it answers so the caller can see which parts
    the model left uncovered.
    """
    if provider is None:
        return None
    tokens_path = str((design or {}).get("tokens_path") or "")
    parts = list(parts or [feedback])
    prev_lines = [f"- #{p.get('index')}: \"{str(p.get('feedback') or '')[:160]}\" -> "
                  + ", ".join(str(x) for x in (p.get("changed") or [])[:5])
                  for p in list(previous)[-4:]]
    ctx_classes = _context_classes(candidates[:6], contents, paths)
    blocks = [_candidate_block(c, specs, contents, feedback, paths=paths,
                               context_classes=ctx_classes, tokens_path=tokens_path)
              for c in candidates[:6]]
    part_lines = []
    for i, part in enumerate(parts, 1):
        likely = ", ".join(c.path for c in (part_candidates or {}).get(i, [])[:3])
        part_lines.append(f"{i}) {part}" + (f"   [likely: {likely}]" if likely else ""))
    user = (
        f"PROJECT GOAL:\n{goal}\n\n"
        + (f"PRODUCT:\n{_design_summary(design)}\n\n" if design else "")
        + (f"PREVIOUS REVISIONS (already applied):\n" + "\n".join(prev_lines) + "\n\n" if prev_lines else "")
        + f"USER FEEDBACK:\n\"{feedback.strip()}\"\n\n"
        + "FEEDBACK PARTS (cover every one):\n" + "\n".join(part_lines) + "\n\n"
        + ("LIKELY FILES (ranked by a deterministic scan):\n\n" + "\n\n".join(blocks) + "\n\n"
           if blocks else "LIKELY FILES: none found by the deterministic scan.\n\n")
        + "ALL PLANNED FILES:\n" + "\n".join(f"- {p}" + (f": {str((specs.get(p) or {}).get('description') or '')[:100]}"
                                                       if (specs.get(p) or {}).get("description") else "")
                                             for p in paths)
    )
    try:
        from cgx.answer.engine import _extract_json_object
        from cgx.answer.model_caps import get_summary_budget
        budget = get_summary_budget(provider)
        raw = provider.chat(
            messages=[{"role": "system", "content": _LOCALIZE_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.1, max_tokens=min(1200, int(budget.get("output_tokens") or 1200)),
            force_json=True, json_schema=_LOCALIZE_SCHEMA,
        ).get("content", "")
        parsed = _extract_json_object(raw or "")
    except Exception as e:  # pragma: no cover - provider hiccup -> deterministic fallback
        swarm_beat(log_root, "reviser", "localize_error", error=repr(e))
        return None
    entries = parsed.get("changes") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return None
    known = set(paths)
    cand_order = {c.path: i for i, c in enumerate(candidates)}
    out: List[Dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        path = str(e.get("path") or "").strip().replace("\\", "/").lstrip("./")
        instr = str(e.get("instruction") or "").strip()
        if not path or not instr or path not in known or path == tokens_path:
            continue
        if any(o["path"] == path and o["instruction"] == instr for o in out):
            continue
        try:
            part = int(e.get("part") or 0)
        except (TypeError, ValueError):
            part = 0
        if not (1 <= part <= len(parts)):
            part = _guess_part(instr, parts) if len(parts) > 1 else 1
        out.append({"path": path, "instruction": instr, "part": part,
                    "why": str(e.get("why") or "").strip()
                    or "; ".join(next((c.reasons for c in candidates if c.path == path), [])),
                    "source": "model" if path not in cand_order else "model+scan"})
    out.sort(key=lambda o: (o["part"], cand_order.get(o["path"], 10_000)))
    return {"changes": out[:_MAX_TARGETS], "note": str(parsed.get("note") or "").strip()
            if isinstance(parsed, dict) else ""}


def _guess_part(instruction: str, parts: Sequence[str]) -> int:
    """The feedback part an instruction most plausibly answers (keyword overlap), 1-based."""
    from cgx.session.revise import _keywords
    iw = set(_keywords(instruction))
    best, best_n = 1, -1
    for i, part in enumerate(parts, 1):
        n = len(iw & set(_keywords(part)))
        if n > best_n:
            best, best_n = i, n
    return best


def ensure_part_coverage(changes: List[Dict[str, Any]], parts: Sequence[str],
                         part_candidates: Dict[int, List[Candidate]],
                         feedback: str) -> List[Dict[str, Any]]:
    """Add a deterministic change for every feedback part the model left uncovered.

    A part is covered when a change claims it. An uncovered part whose own
    deterministic scan found a candidate becomes one more change on that file
    with the part's text as the instruction -- so a two-part message never
    silently loses its second half.
    """
    out = list(changes)
    for i, part in enumerate(parts, 1):
        if any(int(c.get("part") or 0) == i for c in out):
            continue
        cands = part_candidates.get(i) or []
        if not cands:
            continue
        top = cands[0]
        if any(c["path"] == top.path and c["instruction"] == part for c in out):
            continue
        out.append({"path": top.path, "instruction": part, "part": i,
                    "why": "; ".join(top.reasons) + " (part not covered by the model's plan)",
                    "source": "scan:part"})
    return out[:_MAX_TARGETS]


def _fallback_changes(feedback: str, candidates: List[Candidate],
                      parts: Optional[Sequence[str]] = None,
                      part_candidates: Optional[Dict[int, List[Candidate]]] = None
                      ) -> List[Dict[str, Any]]:
    """Deterministic plan: per part, its top candidates with the part text as the ask."""
    parts = list(parts or [feedback])
    out: List[Dict[str, Any]] = []
    if len(parts) > 1 and part_candidates:
        for i, part in enumerate(parts, 1):
            for c in (part_candidates.get(i) or [])[:2]:
                out.append({"path": c.path, "instruction": part, "part": i,
                            "why": "; ".join(c.reasons), "source": "scan"})
        if out:
            return out[:_MAX_TARGETS]
    return [{"path": c.path, "instruction": feedback.strip(), "part": 1,
             "why": "; ".join(c.reasons), "source": "scan"}
            for c in candidates[:_MAX_TARGETS]]


def grounding_notes(instruction: str, current: str, path: str) -> List[str]:
    """Facts about the current file the instruction's author may have assumed wrongly.

    A localizer that says "in .app set align-items: stretch" while the
    stylesheet has no ``.app`` rule leaves a weak model stuck: told to edit in
    place, it finds nothing to edit and echoes the file back. So every selector
    the instruction names that has no rule in a stylesheet (or identifier that
    does not occur in a source file) is called out explicitly as something to
    ADD -- deterministic, checkable, and exactly the hint the model needs.
    """
    notes: List[str] = []
    tokens = instruction_tokens(instruction)
    if _is_stylesheet(path):
        for tok in tokens:
            if not tok.startswith((".", "#")):
                continue
            name = tok[1:]
            if not re.search(r"(?<![\w-])" + re.escape(tok) + r"(?![\w-])", current):
                notes.append(
                    f"`{tok}` has NO rule in this file yet: ADD a new `{tok} {{ ... }}` rule block "
                    f"(the class/id `{name}` is used by the markup) -- do not just echo the file.")
    else:
        for tok in tokens:
            if tok.startswith((".", "#")) or not re.match(r"^[A-Za-z_][\w$]*$", tok):
                continue
            if tok not in current:
                notes.append(f"`{tok}` does not occur in this file yet: ADD it as the change requires.")
    return notes


def _revision_description(base: str, feedback: str, instruction: str, firmer: bool,
                          missing: Optional[Sequence[str]] = None,
                          current: str = "", path: str = "") -> str:
    text = (
        (base.strip() + "\n\n" if base.strip() else "")
        + "USER FEEDBACK ON THE RUNNING APP: " + feedback.strip()
        + "\nREQUIRED CHANGE IN THIS FILE: " + instruction.strip()
        + "\nApply exactly this change. Keep every other behaviour, export, import, class "
        "name and the design-system usage unchanged. Edit the existing rule / element in "
        "place when it exists; when it does not exist yet, ADD it. Never return the file "
        "unchanged. Output the complete updated file.")
    for note in grounding_notes(instruction, current, path) if current else []:
        text += "\nNOTE: " + note
    if firmer and missing:
        text += ("\n\nYour previous rewrite did NOT touch " + ", ".join(missing[:6])
                 + " -- the change above names them and is REQUIRED (the user asked for it "
                 "after reviewing the app). Edit exactly those so the request is satisfied.")
    elif firmer:
        text += ("\n\nYour previous answer returned the file UNCHANGED. The change above is "
                 "REQUIRED -- the user asked for it after reviewing the app. Modify the file "
                 "so it visibly satisfies the request.")
    return text


def apply_change(change: Dict[str, Any], *, feedback: str, specs: Dict[str, Any],
                 contracts: Dict[str, Any], goal: str, root: str, provider: Any,
                 paths: Sequence[str], skills: List[str], design: Optional[Dict[str, Any]],
                 deps: Any = None) -> Dict[str, Any]:
    """Apply one per-file instruction; return the change row with its outcome.

    Outcome is ``changed`` (written, body differs), ``unchanged`` (the model
    returned the same body twice -- reported, never silently counted as done),
    or ``failed`` (the generation ladder produced nothing acceptable). When the
    instruction names concrete things (a selector, a property, a prop), the
    diff must touch at least one of them: a rewrite that edited something else
    gets one firmer re-ask naming what it missed, and is flagged ``partial``
    if it still misses.
    """
    path = change["path"]
    spec = specs.get(path) or {}
    before = _safe_read(path, root) or ""
    row = dict(change)
    tokens = instruction_tokens(change["instruction"])
    row.update({"bytes_before": len(before), "bytes_after": len(before), "outcome": "failed",
                "diff": "", "method": "", "tokens": tokens, "covered": [], "partial": False})
    last_err = ""
    missing: List[str] = []
    current = before
    for attempt in range(2):
        desc = _revision_description(str(spec.get("description") or ""), feedback,
                                     change["instruction"], firmer=attempt == 1,
                                     missing=missing, current=current, path=path)
        outcome = generate_file(
            path=path, description=desc, depends_on=list(spec.get("depends_on") or []),
            contracts=contracts, goal=goal, root=root, provider=provider, layer=path,
            manifest_paths=list(paths), log_root=root, skills=skills, deps=deps,
            modify_existing=True, design=design, existing_limit=_REVISE_EXISTING_LIMIT)
        if not outcome.ok:
            last_err = str(getattr(outcome, "error", "") or "generation failed")
            swarm_beat(root, "reviser", "apply", file=path, ok=False, attempt=attempt + 1,
                       error=last_err[:200])
            continue
        after = outcome.content or ""
        if after.strip() == current.strip():
            swarm_beat(root, "reviser", "apply", file=path, ok=True, attempt=attempt + 1,
                       unchanged=True)
            if row["outcome"] != "changed":
                row["outcome"] = "unchanged"
            row["method"] = outcome.method
            continue  # one firmer re-ask
        edit_file(path, after, root)
        current = after
        diff = unified_diff(path, before, after)
        covered = diff_covers(diff, tokens)
        row.update({"outcome": "changed", "bytes_after": len(after), "method": outcome.method,
                    "diff": diff, "covered": covered})
        missing = [t for t in tokens if t not in covered]
        applied = not tokens or bool(covered)
        swarm_beat(root, "reviser", "apply", file=path, ok=True, attempt=attempt + 1,
                   method=outcome.method, bytes=len(after), covered=covered,
                   missing=missing if not applied else [])
        if applied:
            row["partial"] = False
            return row
        row["partial"] = True
    if row["outcome"] == "failed":
        row["error"] = last_err
    return row


def _app_stylesheet(design: Optional[Dict[str, Any]], paths: Sequence[str]) -> Optional[str]:
    """The app stylesheet revisions may edit (never the deterministic tokens file)."""
    tokens_path = str((design or {}).get("tokens_path") or "")
    preferred = str((design or {}).get("global_css_path") or "")
    if preferred and preferred in paths and preferred != tokens_path:
        return preferred
    for p in paths:
        if _is_stylesheet(p) and p != tokens_path:
            return p
    return None


def layout_diagnosis(candidates: List[Candidate], contents: Dict[str, str], paths: Sequence[str],
                     feedback: str) -> List[Dict[str, str]]:
    """The deterministic width diagnosis for the component the feedback names (top-ranked)."""
    if not is_layout_feedback(feedback):
        return []
    comp = next((c.path for c in candidates
                 if not _is_stylesheet(c.path) and c.path.lower().endswith((".jsx", ".tsx", ".vue", ".svelte"))),
                None)
    if not comp:
        return []
    chain = ancestor_chain(comp, contents, paths, importers_of=_importers_of)
    stylesheets = {p: contents.get(p) or "" for p in paths if _is_stylesheet(p)}
    return diagnose_width_constraints(chain_with_rules(chain, stylesheets), feedback)


def _apply_layout_knowledge(changes: List[Dict[str, Any]], candidates: List[Candidate],
                            contents: Dict[str, str], paths: Sequence[str],
                            design: Optional[Dict[str, Any]], feedback: str
                            ) -> "tuple[List[Dict[str, Any]], List[str]]":
    """Drop no-op CSS instructions; ensure the diagnosed constraining ring gets fixed."""
    stylesheets = {p: contents.get(p) or "" for p in paths if _is_stylesheet(p)}
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for ch in changes:
        if _is_stylesheet(ch["path"]):
            why = instruction_is_noop(ch["instruction"], stylesheets)
            if why:
                dropped.append(f"{ch['path']}: {ch['instruction']} ({why})")
                continue
        kept.append(ch)
    diagnosis = layout_diagnosis(candidates, contents, paths, feedback)
    css_path = _app_stylesheet(design, paths)
    if diagnosis and css_path:
        first = diagnosis[0]
        ring = first["ring"]
        targets_ring = any(
            _is_stylesheet(ch["path"]) and re.search(r"(?<![\w.#-])" + re.escape(ring) + r"(?![\w-])", ch["instruction"])
            for ch in kept)
        if not targets_ring:
            kept.insert(0, {
                "path": css_path,
                "instruction": (f"Add or update the rule `{first['fix']}` -- {first['problem']}. "
                                f"Make this change at {ring} itself; leave the inner elements alone."),
                "why": f"deterministic layout diagnosis of the DOM ancestry ({ring} constrains the element)",
                "part": 1, "source": "diagnosis"})
    return kept[:_MAX_TARGETS], dropped


def _snapshot_screenshots(root: str, revision_index: int) -> Dict[str, str]:
    """Copy the current route screenshots aside; return ``{slug: copied path}``."""
    import shutil
    src_dir = os.path.join(root, ".cgx", "screenshots")
    if not os.path.isdir(src_dir):
        return {}
    dst_dir = os.path.join(src_dir, f"before-rev{revision_index}")
    out: Dict[str, str] = {}
    try:
        os.makedirs(dst_dir, exist_ok=True)
        for name in sorted(os.listdir(src_dir)):
            if not name.lower().endswith(".png"):
                continue
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, name)
            shutil.copyfile(src, dst)
            out[name[:-4]] = dst
    except OSError as e:  # pragma: no cover - best-effort
        swarm_beat(root, "reviser", "snapshot_error", error=repr(e))
    return out


def _switch_theme(design: Dict[str, Any], new_key: str, paths: List[str], root: str) -> Optional[Dict[str, Any]]:
    """Rewrite the deterministic tokens stylesheet for ``new_key``; no model call."""
    tokens_path = str(design.get("tokens_path") or "")
    if not tokens_path:
        return None
    old = str(design.get("theme") or "")
    try:
        from cgx.design import tokens_content
        design["theme"] = new_key
        before = _safe_read(tokens_path, root) or ""
        content = tokens_content(design)
        edit_file(tokens_path, content, root)
        if tokens_path not in paths:
            paths.append(tokens_path)
    except Exception as e:  # pragma: no cover - best-effort
        design["theme"] = old
        swarm_beat(root, "reviser", "theme_error", error=repr(e))
        return None
    swarm_beat(root, "reviser", "theme", file=tokens_path, old=old, new=new_key)
    return {"from": old, "to": new_key, "tokens_path": tokens_path,
            "diff": unified_diff(tokens_path, before, content, max_lines=40)}


@register_executor(TaskKind.SWARM_REVISE)
def swarm_revise(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Localize the feedback, apply it file by file, report what changed where."""
    if deps.provider is None:
        return ExecutorResult(
            failure="No provider configured for Swarm mode.", retryable=False)

    feedback = str(task.inputs.get("feedback") or "").strip()
    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "")
    plan_art = _load_artifact(deps, work_plan_id)
    plan: Dict[str, Any] = dict(plan_art.content) if plan_art is not None and plan_art.content else {}
    paths: List[str] = list(plan.get("paths") or [])
    specs = plan_specs({"layers": plan.get("layers") or []})
    contracts = plan.get("contracts") or {}
    skills = list(plan.get("skills") or [])
    design = plan.get("design") if isinstance(plan.get("design"), dict) else None
    if design is not None and not design.get("applies"):
        design = None
    goal = str(task.inputs.get("goal") or plan.get("goal") or "")
    project_root = os.path.abspath(str(task.inputs.get("project_root") or plan.get("project_root")
                                       or deps.project_root or "."))
    prior_status = str(task.inputs.get("prior_status") or "completed")
    revision_index = int(task.inputs.get("revision_index") or (len(plan.get("revisions") or []) + 1))
    base_outputs = {"work_plan_artifact_id": work_plan_id, "goal": goal,
                    "project_root": project_root, "feedback": feedback,
                    "prior_status": prior_status, "revision_index": revision_index}

    def _give_up(reason: str, candidates: List[Candidate]) -> ExecutorResult:
        art = Artifact.new(
            session_id=task.session_id, produced_by_task_id=task.task_id,
            kind=ArtifactKind.SWARM_REVISION,
            content={**base_outputs, "status": "unlocalized", "reason": reason, "note": reason,
                     "method": "none", "candidates": [c.as_dict() for c in candidates],
                     "changes": [], "changed": [], "failed": [], "unchanged": [], "theme": None})
        swarm_beat(project_root, "reviser", "report", ok=False, reason=reason[:200])
        return ExecutorResult(artifact=art, outputs={
            **base_outputs, "changed": [], "failed_paths": [], "reason": reason, "note": reason,
            "revision_artifact_id": art.artifact_id})

    if not feedback:
        return _give_up("No feedback text was provided.", [])
    if not paths:
        return _give_up("The build's plan could not be loaded, so there is nothing to revise.", [])

    swarm_beat(project_root, "reviser", "localize", feedback=feedback[:200], files=len(paths),
               revision=revision_index)
    contents = _collect(paths, project_root)
    # Keep the pre-revision screenshots: the verifier compares BEFORE/AFTER to
    # judge whether the feedback was visibly addressed (the smoke overwrites
    # <root>/.cgx/screenshots/<route>.png in place).
    before_shots = _snapshot_screenshots(project_root, revision_index)

    # Routes the last runtime smoke flagged: a problem report ("the settings
    # page is blank") is localized to the files behind those routes.
    failing_routes: List[str] = []
    report = _load_artifact(deps, str(task.inputs.get("verify_report_artifact_id") or ""))
    if report is not None and isinstance(report.content, dict):
        runtime = report.content.get("runtime") or {}
        for r in (runtime.get("routes") or []) if isinstance(runtime, dict) else []:
            if not (r.get("mounted") and not r.get("errors")):
                failing_routes.append(str(r.get("route") or "/"))

    # Deterministic theme switch (no model call): "dark mode", a named theme.
    theme_change = None
    new_theme = theme_from_feedback(feedback, design)
    if new_theme and design:
        theme_change = _switch_theme(design, new_theme, paths, project_root)
        if theme_change:
            plan["design"] = design

    candidates = rank_files(feedback, paths, specs, design, contents,
                            failing_routes=failing_routes, limit=_MAX_TARGETS)
    # Multi-part feedback: localize each part on its own so no part is lost.
    parts = split_feedback(feedback)
    part_candidates: Dict[int, List[Candidate]] = {}
    if len(parts) > 1:
        for i, part in enumerate(parts, 1):
            part_candidates[i] = rank_files(part, paths, specs, design, contents,
                                            failing_routes=failing_routes, limit=3)
        # a part's strong candidate joins the overall ranking so the model sees it
        seen = {c.path for c in candidates}
        for cl in part_candidates.values():
            for c in cl:
                if c.path not in seen and c.score >= 6:
                    candidates.append(c)
                    seen.add(c.path)
    else:
        part_candidates[1] = list(candidates)
    swarm_beat(project_root, "reviser", "candidates",
               files=[(c.path, round(c.score, 1)) for c in candidates], parts=len(parts))

    previous = list(plan.get("revisions") or [])
    refined = refine_with_model(
        deps.provider, feedback=feedback, goal=goal, design=design, candidates=candidates,
        specs=specs, contents=contents, paths=paths, previous=previous, log_root=project_root,
        parts=parts, part_candidates=part_candidates)
    method = "model"
    if refined is None:
        method = "scan"
        changes = _fallback_changes(feedback, candidates, parts, part_candidates)
        note = ""
    else:
        changes = list(refined.get("changes") or [])
        note = str(refined.get("note") or "")
        if not changes and candidates and not theme_change:
            # The model declined; the deterministic scan still has concrete
            # evidence -- apply its strongest candidate rather than drop the ask.
            method = "scan"
            changes = _fallback_changes(feedback, candidates[:2], parts, part_candidates)
        elif changes and len(parts) > 1:
            before_n = len(changes)
            changes = ensure_part_coverage(changes, parts, part_candidates, feedback)
            if len(changes) > before_n:
                method = "model+scan"
    # A pure theme request needs no component change unless the model named one.
    if theme_change and not refined:
        changes = []
    # Deterministic layout knowledge beats a weak model's guess: drop a CSS
    # instruction that would change nothing (the declaration already holds that
    # value -- the model mis-located the constraint), and when the ancestry
    # diagnosis names the ring that actually constrains the element, make sure
    # a change fixes THAT ring in the app stylesheet.
    changes, dropped = _apply_layout_knowledge(changes, candidates, contents, paths, design,
                                               feedback)
    if dropped:
        swarm_beat(project_root, "reviser", "noop_dropped", dropped=dropped)
        if method == "model":
            method = "model+diagnosis"
    swarm_beat(project_root, "reviser", "plan", method=method,
               targets=[(c["path"], c.get("part")) for c in changes], note=note[:200])

    if not changes and not theme_change:
        return _give_up(
            "Could not map the feedback to any file of this build. Name the screen, "
            "component or text you mean (e.g. \"the Send button on the chat page\").",
            candidates)

    rows: List[Dict[str, Any]] = []
    total = len(changes)
    for i, change in enumerate(changes):
        _emit_scaffold_progress(deps, task, file=change["path"], layer="revise",
                                index=i + 1, total=total, status="start")
        started = time.time()
        row = apply_change(change, feedback=feedback, specs=specs, contracts=contracts,
                           goal=goal, root=project_root, provider=deps.provider, paths=paths,
                           skills=skills, design=design, deps=deps)
        rows.append(row)
        _emit_scaffold_progress(
            deps, task, file=change["path"], layer="revise", index=i + 1, total=total,
            status="done" if row["outcome"] == "changed" else "failed",
            bytes=row.get("bytes_after"), elapsed_ms=int((time.time() - started) * 1000),
            failed_count=sum(1 for r in rows if r["outcome"] == "failed"))

    def _uniq(items: List[str]) -> List[str]:
        seen: List[str] = []
        for it in items:
            if it not in seen:
                seen.append(it)
        return seen

    changed = _uniq([r["path"] for r in rows if r["outcome"] == "changed"])
    unchanged = _uniq([r["path"] for r in rows if r["outcome"] == "unchanged"
                       and r["path"] not in changed])
    failed = _uniq([r["path"] for r in rows if r["outcome"] == "failed"
                    and r["path"] not in changed])
    partial = [r for r in rows if r.get("partial")]
    if theme_change:
        changed.insert(0, theme_change["tokens_path"])

    if changed:
        # Certain mechanical defects a rewrite can introduce (a hook used without
        # its import, a dangling stylesheet import) are fixed without a model
        # call before the verifier sees the tree.
        try:
            from cgx.session.tasks.swarm_verify import _apply_deterministic_repairs
            _apply_deterministic_repairs(paths, project_root, skills or
                                         ([str(design.get("framework"))] if design else []))
        except Exception:  # pragma: no cover - best-effort
            pass

    if not note:
        if changed:
            note = "Applied the feedback to " + ", ".join(changed[:4]) + (
                f" (+{len(changed) - 4} more)" if len(changed) > 4 else "")
        elif unchanged:
            note = ("The model returned " + ", ".join(unchanged[:3])
                    + " unchanged; the feedback could not be applied. Try naming the exact element.")
        else:
            note = "No file could be regenerated for this feedback."
    if partial:
        note += (" Caution: " + ", ".join(
            f"{r['path']} did not touch " + ", ".join(t for t in r["tokens"] if t not in r["covered"])[:80]
            for r in partial[:2]) + " -- check the result and rephrase if needed.")
    status = "applied" if changed else "nothing_changed"

    content = {
        **base_outputs,
        "verify_report_artifact_id": task.inputs.get("verify_report_artifact_id"),
        "status": status, "method": method, "note": note,
        "parts": parts,
        "before_screenshots": before_shots,
        "auto_round": int(task.inputs.get("auto_round") or 0),
        "candidates": [c.as_dict() for c in candidates],
        "changes": rows, "changed": changed, "unchanged": unchanged, "failed": failed,
        "partial": [r["path"] for r in partial],
        "theme": theme_change,
        "reason": "" if changed else note,
    }
    artifact = Artifact.new(
        session_id=task.session_id, produced_by_task_id=task.task_id,
        kind=ArtifactKind.SWARM_REVISION, content=content)

    # Revision history on the plan: the next round knows what was already asked
    # and done (threaded into the localizer's prompt), and the UI lists it.
    if plan_art is not None and deps.store is not None:
        try:
            history = list(plan.get("revisions") or [])
            instructions: Dict[str, List[str]] = {}
            for r in rows:
                if r["outcome"] == "changed":
                    instructions.setdefault(r["path"], []).append(r["instruction"])
            history.append({"index": revision_index, "feedback": feedback, "changed": changed,
                            "unchanged": unchanged, "failed": failed, "note": note,
                            # per-file asks, so a later verifier regeneration of one
                            # of these files preserves what the user asked for
                            "instructions": instructions,
                            "revision_artifact_id": artifact.artifact_id, "at": time.time()})
            plan_art.content["revisions"] = history
            if theme_change:
                plan_art.content["design"] = design
            if theme_change and theme_change["tokens_path"] not in (plan_art.content.get("paths") or []):
                plan_art.content["paths"] = paths
            deps.store.save_artifact(plan_art)
        except Exception as e:  # pragma: no cover - history is best-effort
            swarm_beat(project_root, "reviser", "history_error", error=repr(e))

    swarm_beat(project_root, "reviser", "report", ok=bool(changed), changed=changed,
               unchanged=unchanged, failed=failed, method=method)
    return ExecutorResult(artifact=artifact, outputs={
        **base_outputs, "changed": changed, "unchanged": unchanged, "failed_paths": failed,
        "note": note, "reason": "" if changed else note,
        "revision_artifact_id": artifact.artifact_id,
        "auto_round": int(task.inputs.get("auto_round") or 0),
        "auto_rounds_max": int(task.inputs.get("auto_rounds_max") or 0)})
