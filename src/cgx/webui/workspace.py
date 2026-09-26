"""Managed workspaces for generated sites.

For the "build me a website" flow we don't want to make users invent an
absolute path. Instead a named site gets a folder under
``$CGX_CONFIG_DIR/sites/<slug>`` (default ``~/.cgx/sites/<slug>``). The
slug is a strict identifier so it can never traverse out of the sites root.
Users who want a specific location can still pass an explicit project_root.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
SITES_DIR = CONFIG_DIR / "sites"

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 48


def slugify(name: str) -> str:
    """Turn a human name into a safe, lower-kebab directory slug."""
    s = _SLUG_STRIP_RE.sub("-", (name or "").strip().lower()).strip("-")
    s = s[:_MAX_SLUG_LEN].strip("-")
    return s or "site"


def _sites_root_real() -> str:
    SITES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(SITES_DIR, 0o700)
    except OSError:
        pass
    return os.path.realpath(str(SITES_DIR))


def site_path(slug: str) -> Path:
    """Resolve ``<sites>/<slug>`` with a containment guard (raises ValueError)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", slug or ""):
        raise ValueError(f"invalid site slug: {slug!r}")
    base = _sites_root_real()
    candidate = os.path.realpath(os.path.join(base, slug))
    if candidate != base and not candidate.startswith(base + os.sep):
        raise ValueError(f"invalid site slug: {slug!r}")
    return Path(candidate)


def create_site(name: str) -> Dict[str, str]:
    """Create (idempotently) a workspace for ``name`` and return its info."""
    slug = slugify(name)
    path = site_path(slug)
    path.mkdir(parents=True, exist_ok=True)
    return {"name": name or slug, "slug": slug, "project_root": str(path)}


def list_sites() -> List[Dict[str, object]]:
    """List managed site workspaces, newest first."""
    root = Path(_sites_root_real())
    out: List[Dict[str, object]] = []
    for child in sorted(root.iterdir() if root.exists() else [],
                        key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        out.append({
            "slug": child.name,
            "project_root": str(child),
            "has_index": (child / "index.html").is_file(),
            "modified": child.stat().st_mtime,
        })
    return out
