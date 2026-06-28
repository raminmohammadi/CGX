


"""Thin PyPI JSON client with an on-disk cache.

Used by the dependency-aware repair proposer (and, eventually, the
SCAFFOLD-time pin validator) to look up package metadata without making
the repair loop dependent on a live network. Two surfaces:

* :meth:`PyPIClient.get_package` -- ``/pypi/{name}/json``: lists every
  released version with upload timestamps. Used to find the highest
  peer version released contemporaneously with a given consumer.
* :meth:`PyPIClient.get_release` -- ``/pypi/{name}/{version}/json``:
  detailed metadata for one release, including ``info.requires_dist``
  which lists the declared peer constraints. Used to detect explicit
  upper bounds (``Werkzeug<3``) when the consumer's authors knew about
  the incompatibility.

Both calls cache to ``~/.cgx/pypi-cache/<name>/<key>.json`` (key is
``_root`` for the package call, the version string for the release
call). The cache is read-through with a per-entry TTL of 7 days for the
package roll-up (so newly published peer versions get picked up
eventually) and never-expire for per-release records (immutable on
PyPI).

The ``fetcher`` constructor argument exists so tests can stub the
network entirely: pass a callable that returns the bytes for a given
URL. The default uses :mod:`urllib.request` with a short timeout.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


Fetcher = Callable[[str], bytes]


_DEFAULT_TIMEOUT_SECONDS = 8.0
_PACKAGE_TTL_SECONDS = 7 * 24 * 60 * 60
_USER_AGENT = "cgx-repair (+https://github.com/cgx)"


def _default_fetcher(url: str) -> bytes:
    """Fetch ``url`` with a short timeout and a polite User-Agent."""
    req = Request(url, headers={"User-Agent": _USER_AGENT,
                                "Accept": "application/json"})
    with urlopen(req, timeout=_DEFAULT_TIMEOUT_SECONDS) as resp:
        return resp.read()


class PyPIClient:
    """Read-through PyPI JSON client with disk caching.

    The cache layout mirrors PyPI's URL shape: one directory per package
    under ``cache_dir``, with ``_root.json`` for the package roll-up and
    ``{version}.json`` for each release. Cache misses fall through to
    ``fetcher``; non-200 / network failures return ``None`` so callers
    can degrade gracefully rather than raise.
    """

    def __init__(self, *, cache_dir: Optional[Path] = None,
                 fetcher: Optional[Fetcher] = None) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir is not None
            else Path.home() / ".cgx" / "pypi-cache"
        )
        self._fetcher: Fetcher = fetcher or _default_fetcher

    # --------------------- public api ---------------------

    def get_package(self, name: str) -> Optional[Dict[str, Any]]:
        """Return ``/pypi/{name}/json`` payload, or ``None`` on error.

        Cached on disk with a 7-day TTL so newly-published peer versions
        are eventually visible without burning a network round-trip on
        every repair attempt.
        """
        key = self._safe_name(name)
        if not key:
            return None
        cached = self._read_cache(key, "_root.json", ttl=_PACKAGE_TTL_SECONDS)
        if cached is not None:
            return cached
        url = f"https://pypi.org/pypi/{key}/json"
        data = self._fetch_json(url)
        if data is not None:
            self._write_cache(key, "_root.json", data)
        return data

    def get_release(self, name: str,
                    version: str) -> Optional[Dict[str, Any]]:
        """Return ``/pypi/{name}/{version}/json``, or ``None`` on error.

        Per-release metadata is immutable on PyPI, so the cache never
        expires. The ``version`` argument is used verbatim (PyPI accepts
        canonical forms like ``2.1.2``).
        """
        key = self._safe_name(name)
        ver = self._safe_version(version)
        if not key or not ver:
            return None
        cached = self._read_cache(key, f"{ver}.json", ttl=None)
        if cached is not None:
            return cached
        url = f"https://pypi.org/pypi/{key}/{ver}/json"
        data = self._fetch_json(url)
        if data is not None:
            self._write_cache(key, f"{ver}.json", data)
        return data

    # --------------------- helpers ---------------------

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = (name or "").strip().lower().replace("_", "-")
        if not cleaned or "/" in cleaned or ".." in cleaned:
            return ""
        return cleaned

    @staticmethod
    def _safe_version(version: str) -> str:
        v = (version or "").strip()
        if not v or "/" in v or ".." in v:
            return ""
        return v

    def _read_cache(self, pkg_key: str, filename: str,
                    *, ttl: Optional[float]) -> Optional[Dict[str, Any]]:
        path = self._cache_dir / pkg_key / filename
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if ttl is not None and (time.time() - stat.st_mtime) > ttl:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, pkg_key: str, filename: str,
                     payload: Dict[str, Any]) -> None:
        path = self._cache_dir / pkg_key / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.debug("PyPIClient: cache write failed for %s: %s",
                         path, exc)

    def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._fetcher(url)
        except (URLError, OSError, ValueError) as exc:
            logger.debug("PyPIClient: fetch failed %s: %s", url, exc)
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.debug("PyPIClient: bad json from %s: %s", url, exc)
            return None
