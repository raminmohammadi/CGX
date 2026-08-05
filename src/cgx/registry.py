

"""Model / prompt version registry and run + index lineage.

Three provenance primitives so any record (an index manifest, an
``LLM_CALL`` fact, a future feedback row) can be joined back to *exactly*
what produced it:

- **prompt versions** -- :func:`fingerprint` hashes a prompt template to a
  short, stable content id; :class:`PromptRegistry` maps stable names to
  their current fingerprint and exposes a manifest for the registry API.
- **run id** -- :func:`new_run_id` mints a per-execution join key that the
  trace context propagates and the LLM tracer stamps onto every call.
- **index lineage** -- :func:`build_index_lineage` captures the CGX version,
  the indexed repo's git revision, and the embedder identity so a stale or
  foreign index is detectable after the fact.

Stdlib-only; git/version probes are best-effort and never raise.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from typing import Any, Dict, Optional


def cgx_version() -> str:
    """Best-effort installed package version (``0+unknown`` if undetermined)."""
    try:
        from importlib.metadata import version
        return version("cgx")
    except Exception:  # pragma: no cover - editable/uninstalled tree
        return "0+unknown"


def fingerprint(text: str, *, length: int = 12) -> str:
    """Stable short content id for a prompt template (sha256 prefix)."""
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return digest[:length]


def new_run_id() -> str:
    """Mint a per-execution provenance join key."""
    return "run_" + uuid.uuid4().hex[:16]


def git_revision(project_root: Optional[str]) -> Optional[str]:
    """Return the short git SHA of ``project_root``'s HEAD, or None.

    Best-effort: returns None when git is absent, the path is not a repo,
    or the command errors. Never raises.
    """
    if not project_root:
        return None
    try:
        root = os.path.realpath(os.fspath(project_root))
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        rev = (out.stdout or "").strip()
        return rev or None
    except Exception:
        return None


class PromptRegistry:
    """Name -> current prompt template + content fingerprint.

    Registration is idempotent and last-writer-wins; a name re-registered
    with different text simply takes the new fingerprint (prompts are
    versioned by content, not by an incrementing counter).
    """

    def __init__(self) -> None:
        self._prompts: Dict[str, str] = {}

    def register(self, name: str, template: str) -> str:
        self._prompts[name] = template or ""
        return fingerprint(self._prompts[name])

    def version_of(self, name: str) -> Optional[str]:
        tmpl = self._prompts.get(name)
        return fingerprint(tmpl) if tmpl is not None else None

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {"version": fingerprint(tmpl), "chars": len(tmpl)}
            for name, tmpl in sorted(self._prompts.items())
        }


_REGISTRY = PromptRegistry()
_KNOWN_LOADED = False


def registry() -> PromptRegistry:
    return _REGISTRY


def register_known_prompts() -> PromptRegistry:
    """Populate the default registry from the engine's prompt constants.

    Imported lazily and defensively so a partial install (or an import
    cycle during module load) degrades to an empty manifest rather than
    breaking callers.
    """
    global _KNOWN_LOADED
    if _KNOWN_LOADED:
        return _REGISTRY
    try:
        from cgx.answer import engine as _engine
        for mode, tmpl in getattr(_engine, "SYSTEM_PROMPTS", {}).items():
            _REGISTRY.register(f"ask:{mode}", tmpl)
        for mode, tmpl in getattr(_engine, "SYSTEM_PROMPTS_STREAM", {}).items():
            _REGISTRY.register(f"ask_stream:{mode}", tmpl)
        for const in ("SYSTEM", "SYSTEM_STREAM", "SYSTEM2"):
            tmpl = getattr(_engine, const, None)
            if isinstance(tmpl, str):
                _REGISTRY.register(f"engine:{const}", tmpl)
    except Exception:  # pragma: no cover - defensive
        pass
    _KNOWN_LOADED = True
    return _REGISTRY


def build_index_lineage(
    *, project_root: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_dim: Optional[int] = None,
    index_type: Optional[str] = None,
    metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Provenance block embedded into an index ``meta.json``."""
    return {
        "index_id": "idx_" + uuid.uuid4().hex[:16],
        "cgx_version": cgx_version(),
        "git_revision": git_revision(project_root),
        "embed_model": embed_model,
        "embed_dim": embed_dim,
        "index_type": index_type,
        "metric": metric,
    }
