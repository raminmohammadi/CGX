

"""Output policy checks on LLM-generated changes (Subsystem K).

Two concerns before a plan's diffs reach the user (and, later, the disk):

* **Secret-shaped literals.** A model can hallucinate a real-looking
  credential into generated code. We match only high-specificity provider key
  shapes (``sk-``/``AIza``/``gh?_``/``AKIA``/PEM headers) so ordinary code
  survives; a hit is ``critical`` because a committed key is a live incident.
* **Path containment.** :mod:`cgx.codegen.disk_apply` already *refuses* to
  write outside ``project_root`` at apply time; this surfaces the same
  violation earlier (advisory) so a traversal target is visible in the plan
  meta / admin alerts rather than only failing silently at write.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from cgx.guardrails.policy import Finding

# High-specificity credential shapes only (mirrors cgx.redact's provider
# prefixes) -- deliberately not the generic ``key=value`` rule, which would
# flag legitimate ``api_key=os.getenv(...)`` code.
_SECRET_LITERALS = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("google_key", re.compile(r"AIza[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"gh[posru]_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def scan_secret_literals(code: Any) -> List[Finding]:
    """Return ``critical`` findings for secret-shaped literals in ``code``."""
    if not isinstance(code, str) or not code:
        return []
    seen: set = set()
    out: List[Finding] = []
    for code_name, pattern in _SECRET_LITERALS:
        if pattern.search(code) and code_name not in seen:
            seen.add(code_name)
            out.append(Finding(
                code="secret_output", severity="critical",
                message=f"generated code contains a {code_name}-shaped secret",
                detail=code_name))
    return out


def _escapes_root(path: str, root_real: Optional[str]) -> bool:
    if os.path.isabs(path):
        # Only absolute paths *inside* the root are acceptable.
        if not root_real:
            return True
        return not os.path.realpath(path).startswith(root_real + os.sep)
    if root_real:
        target = os.path.realpath(os.path.join(root_real, path))
        return not target.startswith(root_real + os.sep)
    # No root to resolve against: flag traversal segments defensively.
    return ".." in path.replace("\\", "/").split("/")


def check_diffs(diffs: List[Dict[str, Any]], *,
                project_root: Optional[str] = None) -> List[Finding]:
    """Scan a plan's structured diffs for secret literals + path escapes.

    ``diffs`` rows may be raw plan diffs (``{path, new_content|diff}``) or the
    normalised ``diffs_payload`` rows (``{file, patch}``); both shapes are
    handled. Returns the combined findings; the caller records them and may
    block on a ``critical`` when ``CGX_GUARDRAIL_BLOCK_SECRETS`` is set.
    """
    if not isinstance(diffs, list):
        return []
    root_real = os.path.realpath(project_root) if project_root else None
    out: List[Finding] = []
    for d in diffs:
        if not isinstance(d, dict):
            continue
        path = str(d.get("path") or d.get("file") or "")
        if path and path != "(unknown)" and _escapes_root(path, root_real):
            out.append(Finding(
                code="path_escape", severity="critical",
                message=f"diff target escapes project_root: {path}",
                detail=path))
        body = (d.get("new_content") or d.get("content")
                or d.get("patch") or d.get("diff") or "")
        out.extend(scan_secret_literals(body))
    return out


__all__ = ["scan_secret_literals", "check_diffs"]
