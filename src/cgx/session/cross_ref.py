"""Language-agnostic cross-file module resolution.

A generated file that imports a *sibling* module which was never generated is
an integration break the per-file gates miss: each file parses in isolation,
yet the tree cannot build or run because an import points at a file that does
not exist (the ``sendMessage is not exported by App.jsx`` failure class -- here
the coarser, un-gameable half: the imported file is simply absent). Python
already has this check (``resolve_first_party_imports`` /
``cross_check_first_party_imports`` in :mod:`cgx.session.scaffold_validate`);
this module fills the identical gap for the JS/TS family and any future
ecosystem, WITHOUT a model.

Design (honours the harness's data-driven-registry rule -- a new language is a
new row, never a bespoke branch, and a stack with no row ABSTAINS rather than
getting a misapplied check):

* :data:`_SPEC_TABLE` maps a set of file extensions to a *relative-specifier
  extractor*. A file whose extension has no row is never checked here. Python
  is deliberately absent -- its richer resolver already runs elsewhere, and
  double-reporting would only cause churn.
* Only the highest-precision, un-gameable signal is returned: a **relative**
  import specifier (``./x`` / ``../y`` / ``/abs``) that resolves to no file in
  the generated manifest. This is pure path arithmetic -- no grammar, no
  symbol table -- so it cannot false-positive on export shapes or truncated
  bodies, and a stub cannot fake a sibling file that does not exist. Bare
  specifiers (``react``, ``@scope/pkg``, the ``@/x`` alias) are third-party or
  build-tool-aliased and are never gated.
* Specifiers pointing at non-code assets (``./styles.css``, ``./logo.svg``,
  data ``.json``) are skipped: whether an asset was "planned" is ambiguous, and
  the abstain-on-uncertainty rule says never turn an ambiguous case into a
  file-dropping gate. A genuinely missing asset still fails the real build.
* Never raises; abstains on anything it cannot resolve confidently.
"""

from __future__ import annotations

import posixpath
import re
from typing import Callable, Dict, FrozenSet, List, Tuple

# Code extensions a relative specifier may resolve to (extensionless imports are
# completed with these, mirroring a bundler's module-resolution order).
_JS_CODE_EXTS: Tuple[str, ...] = (
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue")

# Asset/data extensions that must NOT gate (ambiguous whether planned; a real
# miss is caught by the build). Kept broad on purpose -- abstain over false-red.
_ASSET_EXTS: FrozenSet[str] = frozenset({
    ".css", ".scss", ".sass", ".less", ".styl", ".json", ".svg", ".png",
    ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".avif", ".woff",
    ".woff2", ".ttf", ".eot", ".otf", ".mp3", ".mp4", ".webm", ".wav",
    ".ogg", ".md", ".txt", ".yml", ".yaml", ".wasm", ".graphql", ".gql",
})

# Extract every quoted import/require/export specifier from JS/TS-family source.
# Four forms cover ES modules and CommonJS; comments are not stripped, which can
# only *over*-collect a commented-out import -- harmless, because a commented
# specifier that happens to resolve is fine and one that doesn't is abstained
# on below unless it is a real relative code import (a rare, low-risk edge).
_JS_SPEC_RES: Tuple[re.Pattern, ...] = (
    re.compile(r"""(?:import|export)\b[^;\n]*?\bfrom\s*['"]([^'"]+)['"]"""),
    re.compile(r"""\brequire\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""", re.MULTILINE),
)


def _js_relative_specifiers(content: str) -> List[str]:
    """Relative import specifiers (``.`` / ``..`` / ``/`` prefixed) in JS/TS."""
    out: List[str] = []
    seen = set()
    for rx in _JS_SPEC_RES:
        for m in rx.finditer(content or ""):
            spec = (m.group(1) or "").strip()
            if not spec or spec in seen:
                continue
            if spec[0] == "." or spec[0] == "/":
                seen.add(spec)
                out.append(spec)
    return out


# ext-set -> relative-specifier extractor. A file's extension selects the row;
# no matching row means abstain (this module never checks that file).
_SPEC_TABLE: List[Tuple[FrozenSet[str], Callable[[str], List[str]]]] = [
    (frozenset(_JS_CODE_EXTS), _js_relative_specifiers),
]


def _ext(path: str) -> str:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def _extractor_for(path: str):
    ext = _ext(path)
    for exts, fn in _SPEC_TABLE:
        if ext in exts:
            return fn
    return None


def _resolves(spec: str, importer: str, manifest: FrozenSet[str]) -> bool:
    """Whether a relative ``spec`` imported from ``importer`` names a manifest file.

    Tries the specifier verbatim, with each code extension appended, and as a
    directory ``index.*`` -- the resolution order a JS bundler uses. A target
    that normalises to outside the project root abstains (returns ``True`` so it
    is never flagged: it may reference a real file we did not generate).
    """
    base = posixpath.dirname(importer.replace("\\", "/"))
    target = posixpath.normpath(posixpath.join(base, spec))
    if target.startswith(".."):
        return True  # escapes the project root -- out of our knowledge, abstain
    target = target.lstrip("./") or target
    # Verbatim (the spec already carried an extension) or with a code ext, plus
    # the directory-index forms.
    candidates = {target}
    for e in _JS_CODE_EXTS:
        candidates.add(target + e)
        candidates.add(posixpath.join(target, "index" + e))
    return bool(candidates & manifest)


def unresolved_references(
    file_contents: Dict[str, str], paths: List[str], root: str = "",
) -> List[Dict[str, str]]:
    """Relative imports that resolve to no generated file (gating warnings).

    ``file_contents`` maps generated path -> source; ``paths`` is the full
    planned/generated manifest. Returns ``{file, module, reason, kind}`` dicts,
    one per unresolved relative code import, for files whose language has a spec
    row. Best-effort and defensive: any per-file error is swallowed so a bad
    parse can never fail the build via an exception.
    """
    manifest = frozenset(
        p.replace("\\", "/").lstrip("./") for p in (paths or []) if p)
    warnings: List[Dict[str, str]] = []
    for path, content in (file_contents or {}).items():
        if not isinstance(content, str) or not content:
            continue
        extractor = _extractor_for(path)
        if extractor is None:
            continue  # no spec row for this language -> abstain
        importer = path.replace("\\", "/").lstrip("./")
        try:
            specs = extractor(content)
        except Exception:  # pragma: no cover - extractor is best-effort
            continue
        for spec in specs:
            # Skip specifiers that clearly name a non-code asset/data file:
            # whether it was planned is ambiguous, so never gate on it.
            if _ext(spec) in _ASSET_EXTS:
                continue
            try:
                if _resolves(spec, importer, manifest):
                    continue
            except Exception:  # pragma: no cover - path math is best-effort
                continue
            warnings.append({
                "file": path,
                "module": spec,
                "kind": "unresolved_module",
                "reason": (f"relative import {spec!r} resolves to no file in "
                           "the project manifest (the imported module was "
                           "never generated)"),
            })
    return warnings


__all__ = ["unresolved_references"]
