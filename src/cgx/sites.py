"""Managed workspaces for generated sites (engine-level).

For the "build me a website" flow we don't want to make users invent an
absolute path. Instead a named site gets a folder under
``$CGX_CONFIG_DIR/sites/<slug>`` (default ``~/.cgx/sites/<slug>``). The
slug is a strict identifier so it can never traverse out of the sites root.
Users who want a specific location can still pass an explicit project_root.

This module lives in the engine (``cgx.sites``) rather than under
``cgx.webui`` so both surfaces -- the web UI *and* the CLI -- can manage
site workspaces without pulling in the optional ``ui``/FastAPI extra.
``cgx.webui.workspace`` re-exports these names for backward compatibility.
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


def _validate_slug(slug: str) -> str:
    """Validate a slug used as a single directory name component."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", slug or ""):
        raise ValueError(f"invalid site slug: {slug!r}")
    return slug


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
    """Resolve ``<sites>/<slug>`` with a containment guard (raises ValueError).

    ``safe_slug`` already matches ``^[a-z0-9][a-z0-9-]{0,63}$`` (a single path
    component), and ``os.path.basename`` strips any residual separator so the
    join can never escape ``base`` -- the primary sanitizer. The realpath +
    containment check is defense in depth.
    """
    safe_slug = os.path.basename(_validate_slug(slug))
    base = _sites_root_real()
    joined = os.path.join(base, safe_slug)
    real = os.path.realpath(joined)
    if real != base and not real.startswith(base + os.sep):
        raise ValueError(f"invalid site slug: {safe_slug!r}")
    return Path(joined)


def create_site(name: str) -> Dict[str, str]:
    """Create (idempotently) a workspace for ``name`` and return its info."""
    slug = _validate_slug(slugify(name))
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
