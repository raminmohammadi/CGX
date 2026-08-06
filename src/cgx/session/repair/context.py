"""``FailureContext`` -- one normalized input for DIAGNOSE (Workstream D1).

``VERIFY_REPORT``, ``SMOKE_REPORT``, ``API_CHECK_REPORT``, and
``RUNTIME_REPORT`` each carry a bespoke content shape. The DIAGNOSE
executor should not learn four schemas, so this module folds whichever
report drove the repair into a single frozen dataclass, reusing the
existing :mod:`cgx.session.repair.classify` plumbing rather than adding
new parsing. It is pure (no I/O), so it is trivially unit-testable and
traceable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from cgx.session.repair import classify

# Gate tokens -- which upstream report drove the repair.
GATE_VERIFY = "verify"
GATE_SMOKE = "smoke"
GATE_API_CHECK = "api_check"
GATE_RUNTIME = "runtime"
GATES: Tuple[str, ...] = (GATE_VERIFY, GATE_SMOKE, GATE_API_CHECK, GATE_RUNTIME)

# Keep the failure blob bounded so a small local model's context window is
# not blown by a multi-thousand-line pytest dump.
FAILURE_TEXT_LIMIT = 4000


@dataclass(frozen=True)
class FailureContext:
    """Normalized, provider-ready view of a single gate failure."""

    gate: str
    classification: str
    failure_signature: str
    failure_text: str
    traceback_files: Tuple[str, ...]
    installed_packages: Tuple[str, ...]
    goal: str
    manifest_files: Tuple[str, ...]

    @classmethod
    def from_report(
        cls,
        gate: str,
        content: Dict[str, Any],
        *,
        goal: str = "",
        manifest_files: Sequence[str] = (),
        installed_packages: Sequence[str] = (),
        classification: Optional[str] = None,
    ) -> "FailureContext":
        """Fold a report ``content`` dict into a :class:`FailureContext`.

        ``classification`` may be passed by a caller that already computed
        it (the router / executor); when omitted it is derived from the
        gate via the matching ``classify.py`` entry point. ``failure_text``
        is normalized per gate and truncated to :data:`FAILURE_TEXT_LIMIT`.
        """
        content = content or {}
        gate = str(gate or "").strip() or GATE_VERIFY
        text = _gate_failure_text(gate, content)[:FAILURE_TEXT_LIMIT]
        sig = str(content.get("failure_signature") or "").strip()
        if not sig:
            sig = classify.failure_signature(content)
        cls_token = (classification
                     if classification is not None
                     else _classify(gate, content))
        # Reuse classify's frame scanner over the *normalized* blob so
        # every gate localizes to the ``.py`` files the failure flowed
        # through without a second parser.
        tb = classify.traceback_source_files({"stdout": text})
        return cls(
            gate=gate,
            classification=cls_token,
            failure_signature=sig,
            failure_text=text,
            traceback_files=tb,
            installed_packages=tuple(installed_packages),
            goal=str(goal or ""),
            manifest_files=tuple(manifest_files),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for tracing / persistence (lists, not tuples)."""
        d = asdict(self)
        d["traceback_files"] = list(self.traceback_files)
        d["installed_packages"] = list(self.installed_packages)
        d["manifest_files"] = list(self.manifest_files)
        return d


# --------------------- helpers (pure) ---------------------

def _classify(gate: str, content: Dict[str, Any]) -> str:
    """Derive the ``classify.py`` token for ``gate`` from ``content``."""
    if gate == GATE_VERIFY:
        return classify.classify_verify_report(content)
    if gate == GATE_RUNTIME:
        return classify.classify_runtime_report(content)
    # SMOKE / API_CHECK have no dedicated classifier: their failures are
    # ambiguous authoring/env work that DIAGNOSE reasons over, so they map
    # to the "unknown" needs-reasoning token (a caller with a more specific
    # verdict passes ``classification=`` to override).
    return "unknown"


def _gate_failure_text(gate: str, content: Dict[str, Any]) -> str:
    """Return the human-readable failure blob for ``gate``."""
    if gate == GATE_VERIFY:
        return classify.failure_text(content)
    if gate == GATE_RUNTIME:
        return classify.runtime_failure_text(content)
    if gate == GATE_SMOKE:
        return _smoke_text(content)
    if gate == GATE_API_CHECK:
        return _api_check_text(content)
    return classify.failure_text(content)


def _smoke_text(content: Dict[str, Any]) -> str:
    """Concatenate each failing module's stderr tail plus the build smoke."""
    parts = []
    for mod in content.get("modules") or []:
        if isinstance(mod, dict) and not mod.get("ok"):
            label = str(mod.get("module") or mod.get("label") or "").strip()
            tail = str(mod.get("stderr_tail") or "").strip()
            parts.append(f"{label}:\n{tail}".rstrip())
    build = content.get("build_smoke")
    if isinstance(build, dict) and not build.get("ok"):
        label = str(build.get("label") or "build").strip()
        tail = str(build.get("stderr_tail") or "").strip()
        parts.append(f"{label}:\n{tail}".rstrip())
    return "\n\n".join(p for p in parts if p)


def _api_check_text(content: Dict[str, Any]) -> str:
    """Render the failing API references as ``module.name: error`` lines."""
    parts = []
    for ref in content.get("failed_references") or []:
        if isinstance(ref, dict):
            mod = str(ref.get("module") or "").strip()
            name = str(ref.get("name") or "").strip()
            err = str(ref.get("error") or "").strip()
            parts.append(f"{mod}.{name}: {err}".strip())
    probe_error = str(content.get("probe_error") or "").strip()
    if probe_error:
        parts.append(probe_error)
    return "\n".join(p for p in parts if p)
