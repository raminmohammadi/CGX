"""Network egress guardrail: where the agent is allowed to fetch from.

Shared by BOTH web-access paths so they enforce one policy:

* the built-in ``fetch_url`` / ``search_web`` tools, and
* the MCP path (``mcp_call`` arguments that carry a URL).

Two layers:

* **SSRF hard-block** (``host_is_blocked``) -- loopback/private/link-local/
  reserved hosts are always refused so a model-chosen URL can't reach internal
  services or cloud metadata (``169.254.169.254``). Not configurable.
* **Domain allowlist** (``host_allowed``) -- only trusted documentation /
  package / code-host / provider domains by default; extend with
  ``CGX_FETCH_ALLOWLIST`` (CSV) or open fully with ``CGX_FETCH_ALLOW_ANY=1``
  (still SSRF-blocked).
"""

from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import urlparse

# Trusted hosts to read from BY DEFAULT: official language/package docs,
# package registries, code hosts, and the major cloud/LLM provider docs -- i.e.
# where real API documentation lives. ``example.com``/``example.org`` are IANA
# doc-reserved and safe to allow.
DEFAULT_FETCH_ALLOWLIST = frozenset({
    "example.com", "example.org",
    # language / package docs + registries
    "python.org", "readthedocs.io", "readthedocs.org", "pypi.org",
    "npmjs.com", "nodejs.org", "developer.mozilla.org", "pkg.go.dev",
    "go.dev", "docs.rs", "crates.io", "rubygems.org", "packagist.org",
    # code hosts / Q&A
    "github.com", "githubusercontent.com", "gitlab.com", "stackoverflow.com",
    "stackexchange.com",
    # major provider / cloud / framework docs
    "ai.google.dev", "developers.google.com", "cloud.google.com",
    "platform.openai.com", "docs.anthropic.com", "learn.microsoft.com",
    "docs.aws.amazon.com", "developer.apple.com", "huggingface.co",
    "palletsprojects.com", "fastapi.tiangolo.com", "djangoproject.com",
    "react.dev", "vuejs.org", "angular.io", "vitejs.dev", "expressjs.com",
    "tailwindcss.com", "stripe.com", "twilio.com",
})


def fetch_allowlist() -> frozenset:
    """Default allowlist plus any domains from ``CGX_FETCH_ALLOWLIST`` (CSV)."""
    extra = os.environ.get("CGX_FETCH_ALLOWLIST", "")
    if not extra.strip():
        return DEFAULT_FETCH_ALLOWLIST
    return DEFAULT_FETCH_ALLOWLIST | {
        d.strip().lower().lstrip(".") for d in extra.split(",") if d.strip()}


def host_is_blocked(host: str) -> bool:
    """True for loopback/private/link-local/reserved hosts (SSRF hard-block).

    Literal private IPs are rejected outright; hostnames are resolved and every
    resolved address is checked. Unresolvable hosts return False (left for the
    caller's fetch to fail naturally, so DNS failures don't look like a policy
    block).
    """
    import ipaddress
    import socket
    h = (host or "").strip().lower()
    if not h or h == "localhost" or h.endswith(".local") or h.endswith(".internal"):
        return True
    try:
        addrs = [info[4][0] for info in socket.getaddrinfo(h, None)]
    except Exception:
        return False
    for ip in addrs:
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (a.is_private or a.is_loopback or a.is_link_local
                or a.is_reserved or a.is_multicast or a.is_unspecified):
            return True
    return False


def host_allowed(host: str) -> bool:
    """True if ``host`` is (a subdomain of) an allowlisted domain.

    Bypassed when ``CGX_FETCH_ALLOW_ANY`` is truthy -- any public host (still
    SSRF-blocked).
    """
    if os.environ.get("CGX_FETCH_ALLOW_ANY", "").strip().lower() in (
            "1", "true", "yes", "on"):
        return True
    h = (host or "").strip().lower().rstrip(".")
    return any(h == d or h.endswith("." + d) for d in fetch_allowlist())


def host_refusal(host: str) -> Optional[str]:
    """Reason to refuse ``host`` (SSRF or not-allowlisted), else ``None``."""
    if host_is_blocked(host):
        return f"private/loopback host ({host})"
    if not host_allowed(host):
        return (f"host {host!r} is not in the fetch allowlist -- add it via the "
                "CGX_FETCH_ALLOWLIST env var (comma-separated domains) or set "
                "CGX_FETCH_ALLOW_ANY=1 to permit any public host")
    return None


def url_refusal(url: str) -> Optional[str]:
    """Reason to refuse a full ``url`` (scheme + host policy), else ``None``.

    Non-http(s) schemes and disallowed hosts are refused. Used to screen a URL
    an agent (or an MCP tool call) is about to fetch.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        return f"non-http(s) URL ({parsed.scheme or 'no'} scheme)"
    return host_refusal(parsed.hostname or "")


def find_urls(value) -> List[str]:
    """Extract http(s) URLs from an arbitrary MCP argument value (nested).

    MCP tool arguments are free-form; recurse dict/list values and collect any
    string that looks like an http(s) URL so the egress policy can screen a
    fetch-type call regardless of which parameter carries the URL.
    """
    out: List[str] = []

    def _walk(v):
        if isinstance(v, str):
            s = v.strip()
            if s[:7].lower() == "http://" or s[:8].lower() == "https://":
                out.append(s)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)

    _walk(value)
    return out


__all__ = [
    "DEFAULT_FETCH_ALLOWLIST", "fetch_allowlist", "host_is_blocked",
    "host_allowed", "host_refusal", "url_refusal", "find_urls",
]
