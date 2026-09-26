"""Config / CI / infra file parser implementing the :class:`BaseParser` seam.

CI/CD and configuration files -- ``bitbucket-pipelines.yml``,
``.github/workflows/*.yml``, ``.gitlab-ci.yml``, ``Dockerfile``,
``docker-compose.yml``, ``*.toml``, ``package.json``, Terraform, deploy
scripts -- carry first-class facts users ask about ("is there a CI
pipeline?", "how is this deployed?", "what's the build config?"). The code
and Markdown parsers never ingested them, so they were invisible to
retrieval and questions about them came back "the sources don't contain
that" even when the file sat in the repo root.

This parser makes them searchable: one ``doc``-kind chunk per file whose
``code`` is the (bounded) file text, mirroring the Markdown parser so it
flows through record-building, the lexical index (which tokenizes ``code``
and the file name), and the embeddings unchanged. There is no call graph,
so ``call_relations`` is always empty.

Lockfiles (huge, low-signal) and files above the walker's size cap are
skipped to keep the index lean. Secret-bearing ``.env`` files are NOT in
the handled set, so they are never ingested/embedded.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from cgx.parser.base import BaseParser

# Cap the indexed text per config file so a large generated manifest can't
# bloat the index; the head of the file carries the identifying content.
_MAX_CONTENT_CHARS = 12_000

_LOCK_BASENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "composer.lock", "cargo.lock", "gemfile.lock",
}


def _is_lockfile(basename: str) -> bool:
    b = basename.lower()
    return b in _LOCK_BASENAMES or b.endswith(".lock") or b.endswith("-lock.json")


def _summary(basename: str, content: str) -> str:
    """A short one-line gist: the first few non-blank, non-comment lines."""
    out: List[str] = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s[:1] in ("#", ";") or s.startswith("//"):
            continue
        out.append(s)
        if len(out) >= 3:
            break
    head = "; ".join(out)
    return f"{basename}: {head}" if head else basename


def _config_kind(filepath: str, basename: str) -> str:
    """Coarse label (informational; surfaced in meta) for the config file."""
    p, b = filepath.replace("\\", "/").lower(), basename.lower()
    if ("/.github/workflows/" in p or "/.circleci/" in p
            or "pipeline" in b or b.startswith(".gitlab-ci")
            or b.startswith("azure-pipelines") or b == "jenkinsfile"):
        return "ci"
    if b.startswith("dockerfile") or "docker-compose" in b or "compose" in b:
        return "docker"
    if b.endswith((".tf", ".hcl")):
        return "infra"
    if b.endswith((".sh", ".bash")):
        return "script"
    return "config"


class ConfigParser(BaseParser):
    """Parser for config / CI / infra files; one ``doc`` chunk per file."""

    #: Matched by file extension (the common case: bitbucket-pipelines.yml,
    #: .github/workflows/*.yml, package.json, pyproject-style .toml, ...).
    extensions: Tuple[str, ...] = (
        ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".properties",
        ".json", ".sh", ".bash", ".tf", ".hcl", ".gradle",
    )
    #: Matched by exact basename (extensionless infra files the walker's
    #: ``splitext`` can't key on).
    filenames: Tuple[str, ...] = (
        "Dockerfile", "Makefile", "Procfile", "Jenkinsfile", "Vagrantfile",
    )

    def parse_file(
        self,
        filepath: str,
        source_code: str,
        project_root: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not source_code or not source_code.strip():
            return [], []
        basename = os.path.basename(filepath)
        if _is_lockfile(basename):
            return [], []

        lines = source_code.splitlines()
        content = source_code
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n... (truncated)"

        chunk = {
            "id": f"{filepath}::doc::config",
            "type": "doc",
            "name": basename,
            "file": filepath,
            "module_path": None,
            "code": content,
            "start_line": 1,
            "end_line": max(1, len(lines)),
            "col_offset": 0,
            "meta": {
                "docstring": _summary(basename, content),
                "source_kind": "doc",
                "config_kind": _config_kind(filepath, basename),
                "metrics": {"n_loc": len(lines)},
            },
        }
        return [chunk], []


__all__ = ["ConfigParser"]
