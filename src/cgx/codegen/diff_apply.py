

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

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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


def _build_hunk_images(body: List[str]) -> Tuple[List[str], List[str]]:
    """Split a hunk body into the pre-image (lines expected in the source)
    and the post-image (lines the source should be replaced with).

    ``" "`` lines contribute to both; ``"-"`` lines appear only in the
    pre-image; ``"+"`` lines appear only in the post-image. ``"\\ No newline
    at end of file"`` markers and other non-diff lines are ignored.
    """
    pre: List[str] = []
    post: List[str] = []
    for ln in body:
        if not ln:
            # A blank line inside a hunk body is a context line for an empty
            # source line -- unified diff requires a leading space, but some
            # models drop it. Treat as context on both sides.
            pre.append("")
            post.append("")
            continue
        tag, rest = ln[0], ln[1:]
        if tag == " ":
            pre.append(rest)
            post.append(rest)
        elif tag == "-":
            pre.append(rest)
        elif tag == "+":
            post.append(rest)
        # any other leading char (e.g. "\") is metadata; skip.
    return pre, post


def _locate_pre_image(
    buf: List[str], pre: List[str], hint: int, *, window: int = 50
) -> Optional[int]:
    """Find the index in ``buf`` at which ``pre`` matches exactly.

    Tries, in order:

    1. The hinted index supplied by the @@ header (adjusted by prior offsets).
    2. A sliding-window scan ±``window`` lines around the hint.
    3. A whole-buffer scan, but only accepted when the match is unique.

    Returns the start index or ``None`` if nothing matches uniquely.
    An empty ``pre`` (pure insertion hunk) returns ``hint`` clamped to
    ``[0, len(buf)]``.
    """
    n_pre = len(pre)
    if n_pre == 0:
        return max(0, min(hint, len(buf)))
    if n_pre > len(buf):
        return None

    def _matches_at(i: int) -> bool:
        return 0 <= i <= len(buf) - n_pre and buf[i:i + n_pre] == pre

    if _matches_at(hint):
        return hint
    for delta in range(1, window + 1):
        for cand in (hint - delta, hint + delta):
            if _matches_at(cand):
                return cand
    # Global unique-match fallback.
    matches: List[int] = []
    for i in range(0, len(buf) - n_pre + 1):
        if buf[i:i + n_pre] == pre:
            matches.append(i)
            if len(matches) > 1:
                return None
    return matches[0] if len(matches) == 1 else None


def _apply_hunks(original: str, hunks: List[List[str]]) -> Tuple[Optional[str], List[str]]:
    """Apply hunk bodies (lists of diff lines starting at the line after @@).

    Each hunk is located by matching its pre-image (context + deletion lines)
    against the working buffer. The @@ line numbers are only used as a hint;
    if they drift, a windowed scan and then a unique-match global scan are
    tried before giving up. Hunks whose pre-image cannot be located, or whose
    location is ambiguous, are reported in the rejected list and left
    unapplied -- this prevents the silent structural overwrite that would
    otherwise corrupt the file when a model emits wrong line numbers or
    hallucinated context.

    Returns ``(new_text, rejected_hunk_strings)``.
    """
    src_lines = original.splitlines(keepends=False)
    out: List[str] = list(src_lines)
    rejected: List[str] = []
    offset = 0
    for hunk in hunks:
        header = hunk[0]
        m = _HUNK_HEADER_RE.match(header)
        if not m:
            rejected.append("\n".join(hunk))
            continue
        old_start = int(m.group(1))
        body = hunk[1:]
        pre, post = _build_hunk_images(body)
        hint = max(0, old_start - 1 + offset)
        loc = _locate_pre_image(out, pre, hint)
        if loc is None:
            logger.info(
                "codegen.diff_apply: rejecting hunk -- pre-image not found "
                "(hint=%d, pre_lines=%d)", hint, len(pre),
            )
            rejected.append("\n".join(hunk))
            continue
        out[loc:loc + len(pre)] = post
        offset += len(post) - len(pre)

    no_newline = False
    has_newline_marker = False
    if hunks:
        last_hunk = hunks[-1]
        for idx in range(len(last_hunk) - 1, 0, -1):
            ln = last_hunk[idx]
            if ln.startswith("\\"):
                has_newline_marker = True
                prev_line = last_hunk[idx - 1]
                if prev_line.startswith("+") or prev_line.startswith(" "):
                    no_newline = True
                break
        if not has_newline_marker and last_hunk[-1].startswith("+"):
            endswith_newline = True
        else:
            endswith_newline = original.endswith("\n") if not no_newline else False
    else:
        endswith_newline = original.endswith("\n")

    return "\n".join(out) + ("\n" if endswith_newline else ""), rejected


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
    logger.info("codegen.diff_apply: applying %d target(s) root=%s allow_new=%s",
                len(targets), project_root, allow_new_files)
    results: List[PatchResult] = []
    for tgt in targets:
        rel = tgt.path
        original = _read_file(project_root, rel)
        is_new = (original is None) or ("--- /dev/null" in tgt.diff_text)
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
                    original_content=original, is_new_file=True,
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
                original_content=original, is_new_file=True,
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
            logger.warning("codegen.diff_apply: exception applying %s: %s: %s",
                           rel, type(e).__name__, e)
            results.append(PatchResult(
                path=rel, ok=False, error=f"{type(e).__name__}: {e}",
                original_content=original, is_new_file=False,
            ))
    n_ok = sum(1 for r in results if r.ok)
    logger.info("codegen.diff_apply: done ok=%d failed=%d",
                n_ok, len(results) - n_ok)
    return results

