"""Parse and apply unified diffs in memory.

We deliberately do NOT touch the user's filesystem here. Callers receive the
post-patch contents and decide what to do (preview, write to a sandbox, or
prompt the user to confirm a real write).

The parser accepts two shapes:

1. Fenced blocks with a ``path=`` header::

       ```diff path=src/module.py
       --- a/src/module.py
       +++ b/src/module.py
       @@ -1,3 +1,4 @@
        line a
       +new line
        line b
       ```

2. A raw unified diff containing ``--- a/<path>`` / ``+++ b/<path>`` headers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


_FENCE_RE = re.compile(
    r"```(?:diff|patch)\s+path=([^\s`]+)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
_HUNK_HEADER_RE = re.compile(
    r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@"
)


@dataclass
class PatchTarget:
    """A single file's worth of diff text plus its declared relative path."""
    path: str
    diff_text: str


@dataclass
class PatchResult:
    """Outcome of applying a single :class:`PatchTarget`."""
    path: str
    ok: bool
    new_content: Optional[str] = None
    original_content: Optional[str] = None
    is_new_file: bool = False
    error: Optional[str] = None
    rejected_hunks: List[str] = field(default_factory=list)


def parse_fenced_diffs(text: str) -> List[PatchTarget]:
    """Extract ``diff path=...`` fenced blocks from arbitrary LLM output."""
    out: List[PatchTarget] = []
    for m in _FENCE_RE.finditer(text or ""):
        out.append(PatchTarget(path=m.group(1).strip(), diff_text=m.group(2)))
    if out:
        return out
    # Fallback: a raw unified diff with --- a/PATH headers, no fence.
    blocks: List[Tuple[str, List[str]]] = []
    cur_path: Optional[str] = None
    cur_lines: List[str] = []
    for line in (text or "").splitlines():
        if line.startswith("+++ b/"):
            cur_path = line[len("+++ b/"):].strip()
            cur_lines.append(line)
        elif line.startswith("--- ") and cur_path is None:
            cur_lines.append(line)
        elif line.startswith("--- ") and cur_path is not None:
            blocks.append((cur_path, cur_lines))
            cur_path = None
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_path is not None:
        blocks.append((cur_path, cur_lines))
    for path, lines in blocks:
        out.append(PatchTarget(path=path, diff_text="\n".join(lines)))
    return out


def _read_file(project_root: str, rel_path: str) -> Optional[str]:
    abs_p = os.path.normpath(os.path.join(project_root, rel_path))
    if not os.path.isfile(abs_p):
        return None
    try:
        with open(abs_p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _apply_hunks(original: str, hunks: List[List[str]]) -> Tuple[Optional[str], List[str]]:
    """Apply hunk bodies (lists of diff lines starting at the line after @@).

    Returns ``(new_text, rejected_hunk_strings)``. If any hunk fails to match,
    that hunk is skipped and reported in the rejected list; remaining hunks
    are still attempted on the original text so the caller gets best-effort
    output. For strict apply, callers should treat any rejected hunk as a
    failure.
    """
    src_lines = original.splitlines(keepends=False)
    out: List[str] = list(src_lines)
    rejected: List[str] = []
    # We apply sequentially with an offset that tracks net insertions/deletions.
    offset = 0
    for hunk in hunks:
        header = hunk[0]
        m = _HUNK_HEADER_RE.match(header)
        if not m:
            rejected.append("\n".join(hunk))
            continue
        old_start = int(m.group(1))
        body = hunk[1:]
        context_before: List[str] = []
        for ln in body:
            if ln.startswith(" "):
                context_before.append(ln[1:])
            elif ln.startswith("-"):
                context_before.append(ln[1:])
            elif ln.startswith("+"):
                break
            else:
                # Unexpected line; treat as context for resilience.
                context_before.append(ln)
        anchor = old_start - 1 + offset
        if anchor < 0 or anchor > len(out):
            rejected.append("\n".join(hunk))
            continue
        new_block: List[str] = []
        consumed = 0
        for ln in body:
            if ln.startswith(" "):
                new_block.append(ln[1:]); consumed += 1
            elif ln.startswith("-"):
                consumed += 1
            elif ln.startswith("+"):
                new_block.append(ln[1:])
            # ignore other markers (\ No newline at end of file)
        out[anchor:anchor + consumed] = new_block
        offset += len(new_block) - consumed
    return "\n".join(out) + ("\n" if original.endswith("\n") else ""), rejected


def _split_hunks(diff_text: str) -> List[List[str]]:
    """Split a unified-diff body into a list of hunk line-arrays."""
    hunks: List[List[str]] = []
    cur: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if cur:
                hunks.append(cur)
            cur = [line]
        elif line.startswith("--- ") or line.startswith("+++ "):
            continue
        elif line.startswith("diff --git"):
            continue
        elif cur:
            cur.append(line)
    if cur:
        hunks.append(cur)
    return hunks


def apply_diffs_in_memory(
    project_root: str,
    targets: List[PatchTarget],
    *,
    allow_new_files: bool = True,
) -> List[PatchResult]:
    """Dry-run apply each target's diff against ``project_root`` content.

    Files are read from disk but never modified. New-file diffs (where the
    target doesn't exist yet) are recognized and their additive lines are
    materialized when ``allow_new_files`` is True.
    """
    results: List[PatchResult] = []
    for tgt in targets:
        rel = tgt.path
        original = _read_file(project_root, rel)
        is_new = original is None
        hunks = _split_hunks(tgt.diff_text)
        if not hunks:
            # New-file diffs from small local models often omit the @@ header.
            # Reconstruct the file from '+' lines (skipping '+++ b/...' headers)
            # so a missing hunk header doesn't kill the whole plan.
            if is_new and allow_new_files:
                synth: List[str] = []
                for ln in (tgt.diff_text or "").splitlines():
                    if ln.startswith("+++ ") or ln.startswith("--- "):
                        continue
                    if ln.startswith("diff --git"):
                        continue
                    if ln.startswith("+"):
                        synth.append(ln[1:])
                new_content = ("\n".join(synth) + "\n") if synth else ""
                results.append(PatchResult(
                    path=rel, ok=True, new_content=new_content,
                    original_content=None, is_new_file=True,
                ))
                continue
            results.append(PatchResult(
                path=rel, ok=False, error="No @@ hunks found in diff",
                original_content=original, is_new_file=is_new,
            ))
            continue
        if is_new:
            if not allow_new_files:
                results.append(PatchResult(
                    path=rel, ok=False, error="File does not exist and new files disallowed",
                    is_new_file=True,
                ))
                continue
            # Materialize a new file from the '+' lines only.
            new_lines: List[str] = []
            for hunk in hunks:
                for ln in hunk[1:]:
                    if ln.startswith("+"):
                        new_lines.append(ln[1:])
            results.append(PatchResult(
                path=rel, ok=True, new_content="\n".join(new_lines) + "\n",
                original_content=None, is_new_file=True,
            ))
            continue
        try:
            new_text, rejected = _apply_hunks(original, hunks)
            results.append(PatchResult(
                path=rel,
                ok=not rejected and new_text is not None,
                new_content=new_text,
                original_content=original,
                is_new_file=False,
                rejected_hunks=rejected,
                error=("partial apply" if rejected else None),
            ))
        except Exception as e:
            results.append(PatchResult(
                path=rel, ok=False, error=f"{type(e).__name__}: {e}",
                original_content=original, is_new_file=False,
            ))
    return results

