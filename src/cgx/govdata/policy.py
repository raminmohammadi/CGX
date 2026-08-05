"""Data-governance policy (Subsystem M).

One env-driven :class:`GovernanceConfig` that the write paths and the
retention driver read: how long observation rows live (``retention_days``),
whether stores keep full user text or only a capped preview
(``store_full_text``), and whether PII is scrubbed before persistence
(``scrub_pii``). Zero-config defaults keep today's behaviour (retain
forever, store full text, no PII scrubbing) so the policy is opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_PREVIEW_CAP = 500


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GovernanceConfig:
    """Resolved data-lifecycle policy for the observation stores."""

    retention_days: int = 0            # 0 => keep forever (no TTL purge)
    store_full_text: bool = True       # False => persist a capped preview only
    scrub_pii: bool = False            # True => strip PII before persistence
    preview_cap: int = _PREVIEW_CAP

    @classmethod
    def from_env(cls) -> "GovernanceConfig":
        return cls(
            retention_days=max(0, _env_int("CGX_RETENTION_DAYS", 0)),
            store_full_text=_env_bool("CGX_STORE_FULL_TEXT", True),
            scrub_pii=_env_bool("CGX_SCRUB_PII", False),
            preview_cap=max(0, _env_int("CGX_PREVIEW_CAP", _PREVIEW_CAP)),
        )

    def apply_text_policy(self, text: str) -> str:
        """Scrub PII (if enabled) then cap to a preview (if full text is off)."""
        if not text:
            return text
        out = text
        if self.scrub_pii:
            from cgx.govdata.pii import scrub_pii as _scrub
            out = _scrub(out)
        if not self.store_full_text and len(out) > self.preview_cap:
            out = out[: self.preview_cap]
        return out
