"""Backward-compatible shim for managed site workspaces.

The implementation moved to the engine-level module :mod:`cgx.sites` so the
CLI can manage site workspaces without importing ``cgx.webui`` (and thus
without requiring the optional ``ui``/FastAPI extra). This module re-exports
the same names so existing importers -- ``cgx.webui.routes.sites`` and the
webui tests -- keep working unchanged.
"""

from __future__ import annotations

from cgx.sites import (  # noqa: F401
    CONFIG_DIR,
    SITES_DIR,
    _sites_root_real,
    _validate_slug,
    create_site,
    list_sites,
    site_path,
    slugify,
)

__all__ = [
    "CONFIG_DIR",
    "SITES_DIR",
    "create_site",
    "list_sites",
    "site_path",
    "slugify",
]
