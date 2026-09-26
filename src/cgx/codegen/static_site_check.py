"""Stdlib-only static-site link/asset resolver used by ``StaticSiteRunner``.

Walks a built static site and returns every *local* ``href``/``src``/
``link``/``url(...)`` reference that does not resolve to a file on disk.
No third-party dependency, no network, no browser: rendering is the user's
job in the sandboxed preview; this is the deterministic "does every link
point at something real" gate that a build-less site otherwise lacks.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Set, Tuple

__all__ = ["check_static_site"]

# Attributes whose value is a URL we should resolve.
_URL_ATTRS = {"href", "src", "poster", "data-src"}
_CSS_URL_RE = re.compile(r"""url\(\s*['"]?([^'")]+)['"]?\s*\)""", re.IGNORECASE)
_IGNORE_DIRS = {".git", ".cgx", ".cgx-backups", "node_modules", "__pycache__"}


def _is_external(ref: str) -> bool:
    r = (ref or "").strip()
    if not r:
        return True
    low = r.lower()
    return low.startswith((
        "http://", "https://", "//", "data:", "mailto:", "tel:",
        "javascript:", "#", "blob:", "about:",
    ))


class _RefParser(HTMLParser):
    """Collect local URL references from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        for name, value in attrs:
            if name and name.lower() in _URL_ATTRS and value:
                self.refs.append(value)


def _iter_site_files(root: Path, max_files: int):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fn in filenames:
            if count >= max_files:
                return
            yield Path(dirpath) / fn
            count += 1


def check_static_site(
    project_root: str, *, max_files: int = 400, max_bytes: int = 2 * 1024 * 1024,
) -> Tuple[List[Tuple[str, str]], int]:
    """Return ``(broken, checked)``.

    ``broken`` is a de-duplicated list of ``(source_file_relpath, reference)``
    for local references that do not resolve to a file inside the site.
    ``checked`` is the number of HTML/CSS files scanned. A reference is only
    flagged when it stays inside the project (a ``../`` escape or an absolute
    path pointing outside the site is reported too, since it can't be
    served). Query strings and fragments are stripped before resolving.
    """
    root = Path(project_root)
    root_real = os.path.realpath(str(root))
    broken: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    checked = 0

    for path in _iter_site_files(root, max_files):
        ext = path.suffix.lower()
        if ext not in (".html", ".htm", ".css"):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        checked += 1
        rel_src = os.path.relpath(str(path), root_real)

        refs: List[str] = []
        if ext in (".html", ".htm"):
            p = _RefParser()
            try:
                p.feed(text)
            except Exception:  # noqa: BLE001 - malformed markup: skip its refs
                pass
            refs.extend(p.refs)
            refs.extend(_CSS_URL_RE.findall(text))  # inline <style>/style=
        else:  # .css
            refs.extend(_CSS_URL_RE.findall(text))

        base = path.parent
        for ref in refs:
            if _is_external(ref):
                continue
            # Strip query/fragment; leading "/" means site-root-relative.
            clean = ref.split("#", 1)[0].split("?", 1)[0].strip()
            if not clean:
                continue
            if clean.startswith("/"):
                target = os.path.realpath(os.path.join(root_real, clean.lstrip("/")))
            else:
                target = os.path.realpath(os.path.join(str(base), clean))
            inside = target == root_real or target.startswith(root_real + os.sep)
            ok = inside and (os.path.isfile(target) or os.path.isdir(target))
            if not ok:
                key = (rel_src, ref)
                if key not in seen:
                    seen.add(key)
                    broken.append(key)

    return broken, checked
