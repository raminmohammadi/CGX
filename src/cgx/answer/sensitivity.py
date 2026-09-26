"""One shared file-sensitivity scorer (the JEV "trust" axis).

The paper adds a third routing axis beyond difficulty and cost -- TRUST -- and
uses the same signal for content-inspecting permissions: estimate which kinds
of files an action touches, then attach policy to file *types*. This module is
the SINGLE source of truth for that estimate, consumed by
:mod:`cgx.guardrails.command` (permissions, now) and, in a later phase, model
routing (which providers may see a file). Building it once keeps the two from
drifting apart.

Deterministic path/glob rules -- no model call:

* ``RESTRICTED`` -- secrets, credentials, keys, and infra state (``.env*``,
  ``~/.ssh``, ``*.pem``/``*.key``, ``.aws``/``.npmrc``/``.netrc``,
  ``*.tfstate``, ``terraform.tfvars`` ...). Locally these never need to leave
  the machine; at a cloud opt-in they gate to first-party providers only.
* ``PUBLIC`` -- docs and vendored/third-party dependencies (``*.md``,
  ``docs/``, ``node_modules/``, ``site-packages/``, ``vendor/``, ``.venv/``).
* ``STANDARD`` -- ordinary application code (the default).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable


class Sensitivity(str, Enum):
    PUBLIC = "public"
    STANDARD = "standard"
    RESTRICTED = "restricted"


# Ordinal for "most restrictive wins" aggregation.
_ORDER = {Sensitivity.PUBLIC: 0, Sensitivity.STANDARD: 1, Sensitivity.RESTRICTED: 2}

_RESTRICTED = [
    re.compile(r"(^|/)\.env(\.[^/]*)?$"),          # .env, .env.local, .env.prod
    re.compile(r"(^|/)\.ssh(/|$)"),
    re.compile(r"(^|/)id_rsa"),
    re.compile(r"\.(pem|key|p12|pfx|keystore|jks)$"),
    re.compile(r"(^|/)\.(aws|gnupg)(/|$)"),
    re.compile(r"(^|/)\.(npmrc|netrc|git-credentials|pypirc)$"),
    re.compile(r"(^|/)secrets?(/|\.[^/]*$)"),       # secrets/ dir or secret.* file
    re.compile(r"(^|/)credentials?($|[./])"),
    re.compile(r"\.tfstate(\.backup)?$"),
    re.compile(r"(^|/)terraform\.tfvars$"),
]

_PUBLIC = [
    re.compile(r"\.(md|rst)$"),
    re.compile(r"(^|/)docs?/"),
    re.compile(r"(^|/)(node_modules|site-packages|vendor|\.venv|venv|dist-info|\.tox)/"),
    re.compile(r"(^|/)(license|readme)"),
]


def score_path(path: object) -> Sensitivity:
    """Classify a single path. RESTRICTED wins over PUBLIC on overlap."""
    p = str(path or "").strip().replace("\\", "/").lower()
    if not p:
        return Sensitivity.STANDARD
    if any(pat.search(p) for pat in _RESTRICTED):
        return Sensitivity.RESTRICTED
    if any(pat.search(p) for pat in _PUBLIC):
        return Sensitivity.PUBLIC
    return Sensitivity.STANDARD


def score_paths(paths: Iterable[object]) -> Sensitivity:
    """Most-restrictive sensitivity across ``paths``; STANDARD when empty."""
    worst = Sensitivity.PUBLIC
    seen = False
    for p in paths or []:
        seen = True
        s = score_path(p)
        if _ORDER[s] > _ORDER[worst]:
            worst = s
    return worst if seen else Sensitivity.STANDARD


__all__ = ["Sensitivity", "score_path", "score_paths"]
