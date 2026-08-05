"""Data governance: retention, right-to-erasure, and PII/DLP (Subsystem M).

Layers a data-lifecycle policy over the observation stores: a
:class:`GovernanceConfig` (env-driven TTL, full-vs-preview text, PII toggle),
a :mod:`~cgx.govdata.pii` scan/scrub pass that complements the credential
redaction in :mod:`cgx.redact`, and a :mod:`~cgx.govdata.retention` driver
that sweeps expired rows and honours per-run / per-owner deletion requests.
"""

from __future__ import annotations

from cgx.govdata.pii import has_pii, scan_pii, scrub_mapping, scrub_pii
from cgx.govdata.policy import GovernanceConfig
from cgx.govdata.retention import erase_owner, erase_run, purge_expired

__all__ = [
    "GovernanceConfig",
    "scan_pii",
    "scrub_pii",
    "scrub_mapping",
    "has_pii",
    "purge_expired",
    "erase_run",
    "erase_owner",
]
