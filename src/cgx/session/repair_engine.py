"""General, language-agnostic failure localization + repair-context selection.

The verifier used to localize a red build by scanning the runner output for
Python/pytest-specific patterns and to *classify* it (assertion vs import) to
pick a repair strategy. That was brittle, and one branch asked the weak model
to *recall* "the standard import for X" -- pure hallucination bait for a small
model.

This module replaces error-text *classification* with two general primitives
that every toolchain's output supports:

* :func:`localize_paths` -- which planned files a failure implicates, from the
  fact that a compiler/test-runner/bundler (pytest, tsc, vite, cargo, go build,
  eslint, ...) prints the offending FILE PATH in its diagnostics. Longest-first
  and word-boundary-guarded so ``a.py`` never matches inside ``data.py`` and a
  planned ``test_app.py`` still matches when the runner prints
  ``tests/test_app.py``. Overlapping shorter matches are suppressed so the most
  specific planned path wins.
* :func:`python_module_targets` -- one data-driven *resolver* (Python) mapping a
  ``ModuleNotFoundError`` / ``ImportError`` module name back to the planned file
  meant to provide it. A new ecosystem is a new resolver, never a new branch in
  the loop; an ecosystem with no resolver simply contributes nothing.
* :func:`neighbor_context` -- the focused source set to hand a repairer: the
  localized files, their direct ``depends_on``, and their reverse dependencies,
  source-filtered and localized-first, so a weak model gets a small,
  defect-centred prompt (it declines on the whole tree) yet still sees both
  sides of a test<->impl disagreement.

Nothing here scans an error for words like ``AssertionError`` to choose a
strategy: the repair prompt itself carries the "fix the wrong side, never
weaken a correct test" guardrail, so one failure-driven repair path serves every
red signal.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cgx.session.scaffold_validate import _module_name_for_path

_WORD_RE = re.compile(r"\w")

# ModuleNotFoundError / ImportError name the missing dotted module in quotes.
_MODULE_ERR_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):[^\n]*?['\"]([\w.]+)['\"]")


def _is_word(ch: str) -> bool:
    return bool(_WORD_RE.match(ch))


def _overlaps(start: int, end: int, claimed: List[Tuple[int, int]]) -> bool:
    return any(start < ce and cs < end for cs, ce in claimed)


def localize_paths(text: str, paths: Sequence[str]) -> List[str]:
    """Planned paths that appear as file paths in a runner/compiler diagnostic.

    A path matches when it occurs in ``text`` bounded by non-word characters on
    both sides -- so ``/`` and ``:`` around ``tests/test_app.py:2`` are fine but
    ``app.py`` inside ``test_app.py`` (preceded by ``_``) is rejected. Paths are
    tried longest-first and each match claims its span, so a specific planned
    path (``sub/app.py``) suppresses the generic suffix (``app.py``) when the
    diagnostic named the specific one. Returned in longest-first order.
    """
    text = text or ""
    claimed: List[Tuple[int, int]] = []
    found: List[str] = []
    for p in sorted({str(x).replace("\\", "/") for x in paths if x},
                    key=len, reverse=True):
        matched = False
        start = 0
        while True:
            i = text.find(p, start)
            if i < 0:
                break
            j = i + len(p)
            before_ok = i == 0 or not _is_word(text[i - 1])
            after_ok = j >= len(text) or not _is_word(text[j])
            if before_ok and after_ok and not _overlaps(i, j, claimed):
                claimed.append((i, j))
                matched = True
            start = i + 1
        if matched:
            found.append(p)
    return found


def python_module_targets(text: str, paths: Sequence[str]) -> List[str]:
    """Planned ``.py`` files that would provide a missing imported module.

    Maps each ``ModuleNotFoundError`` / ``ImportError`` module name in ``text``
    to the planned file whose dotted module name (or its ``src.``-stripped
    variant) equals it -- the file expected to *define* the module that failed
    to import. Empty when the failure names no missing module.
    """
    wanted = set(_MODULE_ERR_RE.findall(text or ""))
    if not wanted:
        return []
    out: List[str] = []
    for p in paths:
        mod = _module_name_for_path(p)
        if not mod:
            continue
        variants = {mod}
        if mod.startswith("src."):
            variants.add(mod[len("src."):])
        if (variants & wanted) and p not in out:
            out.append(p)
    return out


def neighbor_context(localized: Sequence[str], paths: Sequence[str],
                     specs: Optional[Dict[str, Any]],
                     exts: Tuple[str, ...]) -> List[str]:
    """Localized files + their ``depends_on`` + reverse deps, source-filtered.

    When ``localized`` is empty there is no better hint, so every source file
    (by ``exts``) is offered. Otherwise the context is centred on the localized
    files plus the siblings a test<->impl disagreement needs: what each localized
    file imports (``depends_on``) and what imports it (reverse deps). Order is
    localized-first so a downstream ``max_files`` cap never drops a target.
    """
    src = [p for p in paths if str(p).endswith(exts)]
    if not localized:
        return src
    specs = specs or {}
    revdeps: Dict[str, List[str]] = {}
    for owner, spec in specs.items():
        for dep in (spec or {}).get("depends_on") or []:
            revdeps.setdefault(str(dep), []).append(owner)
    keep: List[str] = []

    def _add(p: str) -> None:
        if p in src and p not in keep:
            keep.append(p)

    for t in localized:
        _add(t)
        for dep in (specs.get(t, {}) or {}).get("depends_on") or []:
            _add(str(dep))
        for r in revdeps.get(t, []):
            _add(r)
    return keep or src


__all__ = ["localize_paths", "python_module_targets", "neighbor_context"]
