"""Automatic external-dependency grounding for the Tech Lead.

A weak model rarely takes the initiative to research a third-party package, so
it implements against a *remembered* (often hallucinated) API. This module
removes that from the model's discretion: for every dependency the plan
DECLARES, the harness fetches the package's real **registry metadata** (PyPI /
npm JSON -- structured, cheap, and it carries the summary + README) and injects
it into the plan so the Developer implements against real text instead of
guessing.

Honesty about scope (the design rule here is provenance, not optimism):

* Ecosystem is chosen by an **extension-intersection table** -- each row owns
  its own source extensions and its registry endpoint. Present ecosystems are
  those whose extensions actually appear in the plan's files. A new ecosystem
  is one row; a project whose files match no row is simply not grounded (never
  a misapplied fetch). This deliberately does NOT use a python/node-only
  ``_langs_present`` helper that cannot even report a Rust/Go project.
* Registry metadata is fetched BEFORE anything is installed, so it is
  **UNVERIFIED** third-party text (a summary/README, not introspected
  signatures). It is labelled as such at render time (see
  ``engine._render_contracts_for_prompt``): it narrows the model toward the
  real API and names that the package exists, but the model must still confirm
  exact signatures with tools. Only introspection of an *installed* artifact
  earns a "verified" label, which is a later (verify-time) upgrade.
* Every fetched blob is screened through the prompt-injection guardrail and
  truncated. All network egress goes through the shared SSRF + allowlist
  policy. Nothing here ever raises -- grounding is best-effort and degrades to
  "unavailable" so an offline or private-registry project is never blocked.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; CGX/1.0; +https://github.com/raminmohammadi/CGX)"
_DETAIL_LIMIT = 2500
_MAX_DEPS = 16
# Fetched metadata is stable within a run; avoid re-fetching the same package
# across debate drafts / re-ask rounds. Keyed by (ecosystem, name).
_CACHE: Dict[str, Dict[str, Any]] = {}


def _extract_pypi(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    info = data.get("info") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return None
    summary = str(info.get("summary") or "").strip()
    detail = str(info.get("description") or "").strip()
    if not summary and not detail:
        return None
    return {"summary": summary, "detail": detail}


def _extract_npm(data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(data, dict):
        return None
    summary = str(data.get("description") or "").strip()
    detail = str(data.get("readme") or "").strip()
    if not summary and not detail:
        return None
    return {"summary": summary, "detail": detail}


# Each row OWNS its source extensions + registry endpoint + extractor. Add a
# language by appending a row; a file whose extension matches no row is never
# grounded here.
ECOSYSTEM_REGISTRIES: List[Dict[str, Any]] = [
    {
        "name": "python",
        "exts": frozenset({".py"}),
        "registry_json": "https://pypi.org/pypi/{name}/json",
        "extract": _extract_pypi,
    },
    {
        "name": "node",
        "exts": frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                           ".vue"}),
        "registry_json": "https://registry.npmjs.org/{name}",
        "extract": _extract_npm,
    },
]


def _ext(path: str) -> str:
    base = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def present_ecosystems(paths: List[str]) -> List[Dict[str, Any]]:
    """Ecosystem rows whose source extensions appear in the plan's files."""
    exts = {_ext(p) for p in (paths or []) if p}
    return [row for row in ECOSYSTEM_REGISTRIES if row["exts"] & exts]


def _default_fetch_json(url: str) -> Optional[Dict[str, Any]]:
    """SSRF/allowlist-safe HTTP GET returning parsed JSON, or ``None``.

    Never raises. Refuses non-allowlisted / private hosts via the shared egress
    guardrail before making any request, and caps the response size.
    """
    try:
        from cgx.guardrails.net import url_refusal
        if url_refusal(url):
            return None
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            raw = resp.read(600_000)
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:  # pragma: no cover - network is best-effort
        return None


def _dep_names(contracts: Dict[str, Any]) -> List[str]:
    """Declared third-party dependency names (tolerant of str or {name} form)."""
    out: List[str] = []
    for d in (contracts.get("third_party_dependencies") or []):
        name = (d if isinstance(d, str) else str((d or {}).get("name") or "")
                ).strip()
        if name and name not in out:
            out.append(name)
    return out


def ground_dependency(
    name: str, ecosystems: List[Dict[str, Any]], *,
    fetch_json: Callable[[str], Optional[Dict[str, Any]]],
    screen: Callable[[str, str], str],
) -> Optional[Dict[str, Any]]:
    """Fetch + screen registry metadata for one dependency, or ``None``.

    Tries each present ecosystem's registry until one resolves the package, so
    a polyglot repo grounds a PyPI dep and an npm dep correctly without knowing
    in advance which side owns it. Result is cached per (ecosystem, name).
    """
    import urllib.parse
    for row in ecosystems:
        cache_key = f"{row['name']}:{name}"
        if cache_key in _CACHE:
            cached = _CACHE[cache_key]
            return cached or None
        url = row["registry_json"].format(name=urllib.parse.quote(name, safe=""))
        data = fetch_json(url)
        extracted = row["extract"](data) if data else None
        if not extracted:
            _CACHE[cache_key] = {}
            continue
        summary = screen(extracted.get("summary") or "", f"{name} registry summary")
        detail = extracted.get("detail") or ""
        if len(detail) > _DETAIL_LIMIT:
            detail = detail[:_DETAIL_LIMIT] + "\n... [truncated]"
        detail = screen(detail, f"{name} registry readme")
        result = {
            "ecosystem": row["name"],
            "verified": False,           # registry text, not introspection
            "summary": summary.strip(),
            "detail": detail.strip(),
        }
        _CACHE[cache_key] = result
        return result
    return None


def ground_external_dependencies(
    contracts: Dict[str, Any], paths: List[str], *,
    fetch_json: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    screen: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Registry-grounded reference for each declared third-party dependency.

    Returns ``{name: {ecosystem, verified, summary, detail}}`` for every
    dependency a present ecosystem's registry could resolve. Empty when no
    ecosystem is present, nothing is declared, or fetching yields nothing
    (offline / private registry) -- the caller then emits a "grounding
    unavailable" beat and proceeds; the plan is never blocked on this.
    """
    ecos = present_ecosystems(paths)
    names = _dep_names(contracts)
    if not ecos or not names:
        return {}
    if screen is None:
        from cgx.guardrails.injection import screen_untrusted as screen
    if fetch_json is None:
        fetch_json = _default_fetch_json
    grounded: Dict[str, Any] = {}
    for name in names[:_MAX_DEPS]:
        ref = ground_dependency(name, ecos, fetch_json=fetch_json, screen=screen)
        if ref:
            grounded[name] = ref
    return grounded


__all__ = [
    "ECOSYSTEM_REGISTRIES", "present_ecosystems",
    "ground_dependency", "ground_external_dependencies",
]
