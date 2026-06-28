


"""PyPI-aware ``requirements.txt`` pin validator for SCAFFOLD.

Phase 4.1 prevention gate: before SCAFFOLD persists its diff bundle,
inspect any ``requirements.txt`` the generator emitted and tighten
upper bounds on known-fragile peers using PyPI metadata. The curated
:data:`FRAGILE_PEERS` table maps a consumer package (e.g. ``flask``)
to peers whose unbounded releases have historically broken the
consumer (``werkzeug``, ``jinja2``, ``itsdangerous``, ``click``).

For each pinned consumer in the file, we fetch ``info.requires_dist``
from PyPI and reuse any declared constraint on a fragile peer verbatim;
unpinned consumers or PyPI fetch failures degrade to no-op so SCAFFOLD
never blocks on transient network errors. Reuses
:class:`cgx.session.repair.pypi_client.PyPIClient` (cache + DI hook)
from Phase 3.2.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cgx.session.repair.pypi_client import PyPIClient

logger = logging.getLogger(__name__)


# Consumer (key) -> peers (normalised lower-case) we will enforce the
# consumer's declared ``requires_dist`` constraint on. Kept small and
# conservative; expand only when a real failure case justifies adding
# a peer (each addition costs one PyPI round-trip per scaffold).
FRAGILE_PEERS: Dict[str, List[str]] = {
    "flask": ["werkzeug", "jinja2", "itsdangerous", "click"],
    "alembic": ["sqlalchemy"],
    "scipy": ["numpy"],
    "pydantic": ["pydantic-core"],
}


_REQ_BASENAMES = {"requirements.txt", "requirements-dev.txt"}

_PIN_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_.\-]*)\s*([=~<>!].*)?\s*$"
)
_EXACT_RE = re.compile(r"^==\s*([0-9][A-Za-z0-9_.\-]*)\s*$")


def is_requirements_path(path: str) -> bool:
    """Return True when ``path`` looks like a pip requirements file."""
    p = (path or "").strip().lower()
    if not p:
        return False
    base = Path(p).name
    if base in _REQ_BASENAMES:
        return True
    parts = Path(p).parts
    return len(parts) >= 2 and parts[0] == "requirements" and base.endswith(".txt")


def _normalise_name(raw: str) -> str:
    return (raw or "").strip().lower().replace("_", "-")


def _parse_pin_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(raw_name, spec, normalised_name)`` or ``None``."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    # Skip ``-r foo.txt`` / ``-c constraints.txt`` includes verbatim.
    if stripped.startswith("-"):
        return None
    m = _PIN_RE.match(stripped)
    if not m:
        return None
    name = m.group(1)
    spec = (m.group(2) or "").strip()
    return name, spec, _normalise_name(name)


def _exact_version(spec: str) -> Optional[str]:
    """Extract ``X.Y.Z`` from ``==X.Y.Z`` (and nothing else)."""
    m = _EXACT_RE.match(spec or "")
    return m.group(1) if m else None


def _peer_constraint_from_requires_dist(
        info: Dict[str, Any], peer_key: str) -> Optional[str]:
    """Pull the consumer's declared constraint for ``peer_key`` out of
    ``info.requires_dist``. Returns ``"<canonical_name><spec>"`` (no
    surrounding spaces) or ``None`` when no constraint is declared.
    """
    reqs = info.get("requires_dist") or []
    if not isinstance(reqs, list):
        return None
    for raw in reqs:
        if not isinstance(raw, str):
            continue
        head = raw.split(";", 1)[0].strip()
        if not head:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9_.\-]*)\s*(.*)$", head)
        if not m:
            continue
        if _normalise_name(m.group(1)) != peer_key:
            continue
        rest = m.group(2).strip()
        if not rest:
            continue
        return f"{m.group(1)}{rest}"
    return None


def _replace_or_append_pin(lines: List[str], peer_key: str,
                           constraint: str) -> List[str]:
    """Replace the existing peer line in-place or append ``constraint``."""
    out: List[str] = []
    replaced = False
    for raw in lines:
        parsed = _parse_pin_line(raw)
        if parsed and parsed[2] == peer_key:
            ending = "\n" if raw.endswith("\n") else ""
            out.append(constraint + ending)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(constraint + "\n")
    return out


def _content_to_new_file_patch(path: str, content: str) -> str:
    """Render ``content`` as a new-file unified diff (mirrors engine.py).

    Kept in lockstep with ``cgx.answer.engine._content_to_new_file_patch``
    so the swapped diff round-trips through ``apply_diffs_to_disk`` the
    same way the generator's original diff did.
    """
    lines = content.splitlines()
    n = len(lines)
    header = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{n} @@\n"
    body = "\n".join(f"+{line}" for line in lines)
    return header + (body if body else "+")


def validate_requirements_text(
        text: str, *,
        pypi_client: PyPIClient,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Tighten fragile peer pins in a ``requirements.txt`` body.

    Returns ``(new_text, adjustments)`` where ``adjustments`` is a list
    of ``{consumer, consumer_version, peer, before, after, source}``
    records describing each rewrite. When no consumer in
    :data:`FRAGILE_PEERS` is pinned exactly, or PyPI lookups fail, the
    returned ``new_text`` is identical to ``text`` and ``adjustments``
    is empty.
    """
    if not text:
        return text, []
    lines: List[str] = text.splitlines(keepends=True)

    def _index() -> Dict[str, Tuple[str, str]]:
        out: Dict[str, Tuple[str, str]] = {}
        for raw in lines:
            parsed = _parse_pin_line(raw)
            if not parsed:
                continue
            _, spec, key = parsed
            out.setdefault(key, (raw.rstrip("\n"), spec))
        return out

    adjustments: List[Dict[str, Any]] = []
    pins = _index()
    for consumer_key, peers in FRAGILE_PEERS.items():
        consumer_pin = pins.get(consumer_key)
        if not consumer_pin:
            continue
        version = _exact_version(consumer_pin[1])
        if not version:
            continue
        release = pypi_client.get_release(consumer_key, version)
        if release is None:
            logger.debug(
                "scaffold pin validator: PyPI lookup failed for %s==%s",
                consumer_key, version)
            continue
        info = (release or {}).get("info") or {}
        for peer_key in peers:
            constraint = _peer_constraint_from_requires_dist(info, peer_key)
            if not constraint:
                continue
            existing = pins.get(peer_key)
            existing_spec = existing[1] if existing else None
            # When the existing line already carries the consumer's
            # declared constraint verbatim, skip -- no churn.
            if existing_spec and constraint.strip().lower() == (
                    existing[0].strip().lower()):
                continue
            lines = _replace_or_append_pin(lines, peer_key, constraint)
            adjustments.append({
                "consumer": consumer_key,
                "consumer_version": version,
                "peer": peer_key,
                "before": existing[0].strip() if existing else None,
                "after": constraint,
                "source": "requires_dist",
            })
            pins = _index()
    if not adjustments:
        return text, []
    return "".join(lines), adjustments


def validate_scaffold_diffs(
        diffs: List[Dict[str, str]],
        file_contents: Dict[str, str], *,
        pypi_client: PyPIClient,
) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict[str, Any]]]:
    """Run :func:`validate_requirements_text` over each requirements file.

    ``file_contents`` is a ``{path: content}`` map (e.g. derived from
    the scaffold loop's ``existing_files_with_content``). For every
    diff whose ``file`` is a requirements path AND for which we have
    the source content, we rewrite the patch in place. Returns the
    possibly-rewritten ``(diffs, file_contents, adjustments)``.
    """
    if not diffs:
        return diffs, file_contents, []
    new_diffs = list(diffs)
    new_contents = dict(file_contents)
    all_adjustments: List[Dict[str, Any]] = []
    for idx, entry in enumerate(new_diffs):
        path = str(entry.get("file") or "").strip()
        if not path or not is_requirements_path(path):
            continue
        original = new_contents.get(path)
        if not isinstance(original, str) or not original:
            continue
        rewritten, adjustments = validate_requirements_text(
            original, pypi_client=pypi_client)
        if not adjustments or rewritten == original:
            continue
        new_diffs[idx] = {
            "file": path,
            "patch": _content_to_new_file_patch(path, rewritten),
        }
        new_contents[path] = rewritten
        for adj in adjustments:
            adj["file"] = path
            all_adjustments.append(adj)
    return new_diffs, new_contents, all_adjustments
