

"""Deterministic scope estimation for greenfield planning (P1.1).

The manifest planner (:func:`cgx.answer.engine.plan_scaffold_manifest`)
otherwise treats every goal alike -- its prompt even says "prefer
completeness over brevity" -- so "a calculator" pulls in a database,
migrations, auth, a React frontend, and Selenium E2E tests. Those extra
tiers are the true root cause of the recovery churn we saw: the more a
plan over-reaches, the more the coherence gate, scaffold synthesis, and
repair ladder have to claw back.

This module derives -- with NO LLM call -- a coarse complexity tier and
an explicit "minimal viable stack" constraint straight from the goal
text. DECOMPOSE threads the constraint into the planner as a hard scope
ceiling and CLARIFY records the tier on the requirements sheet.

Deterministic-first is a design constraint: the estimate is a pure
function of the text, so the same goal always yields the same ceiling
and the signal is cheap enough to compute on every plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Heavy capability -> substrings that signal the user actually asked for
# it. Matched case-insensitively against the goal padded with spaces, so
# short tokens like " db " never fire on "double" or "adb".
_FEATURE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "database": ("database", "postgres", "postgresql", "mysql", "sqlite",
                 "sqlalchemy", "mongodb", "mongo", "orm", " db ",
                 "persistence", "persist ", "migration"),
    "auth": ("auth", "login", "signup", "sign up", "oauth", "jwt",
             "authentication", "authorization", "user account",
             "sign-in", "signin"),
    "frontend": ("react", "vue", "angular", "svelte", "next.js", "nextjs",
                 "frontend", "front-end", "single-page", " spa ", "jsx",
                 "tailwind", "web ui", "web interface"),
    "browser_e2e": ("selenium", "playwright", "cypress", "e2e",
                    "end-to-end", "end to end", "browser test"),
    "realtime": ("websocket", "socket.io", "realtime", "real-time",
                 "streaming", "pub/sub"),
    "async_infra": ("celery", "redis", "rabbitmq", "kafka", "task queue",
                    "background job", "worker queue", "scheduler"),
    "api_server": ("fastapi", "flask", "django", "express", "rest api",
                   "http api", "web api", "web service", "web server",
                   "microservice", "backend api"),
    "containerization": ("docker", "kubernetes", "k8s", "docker-compose",
                         "docker compose", "helm"),
}

# Human-readable label for each capability, used to spell out what the
# planner must NOT add unless the goal explicitly asked for it.
_FEATURE_LABELS: Dict[str, str] = {
    "database": "a database, ORM, or migrations",
    "auth": "authentication or user accounts",
    "frontend": "a frontend framework (React/Vue/Angular)",
    "browser_e2e": "browser/E2E tests (Selenium/Playwright/Cypress)",
    "realtime": "websockets or realtime streaming",
    "async_infra": "task queues or background workers (Celery/Redis)",
    "api_server": "an HTTP API server or web framework",
    "containerization": "Docker/Kubernetes packaging",
}


@dataclass(frozen=True)
class ScopeProfile:
    """A deterministic read of how much project a goal actually asks for.

    ``complexity`` is one of ``trivial``/``small``/``standard``/``complex``.
    ``max_files`` is the file-count ceiling handed to the planner.
    ``requested_features`` are the heavy capabilities the goal explicitly
    named (so the planner is allowed to build them). ``constraint`` is the
    ready-to-inject "minimal viable stack" directive.
    """

    complexity: str
    max_files: int
    requested_features: Tuple[str, ...]
    constraint: str

    def as_dict(self) -> Dict[str, object]:
        """JSON-friendly view for persistence + trace records."""
        return {
            "complexity": self.complexity,
            "max_files": self.max_files,
            "requested_features": list(self.requested_features),
        }


# Capabilities that cannot run in the unattended agent sandbox no matter
# who asked for them: browser/E2E suites need a real display the headless
# environment does not provide. These are the only capabilities eligible
# for deterministic de-scoping (P1.4), and even then only when the goal did
# NOT explicitly request them -- an explicit request is honoured, so the
# de-scope only ever trims speculative, unrunnable architecture.
SANDBOX_UNRUNNABLE_FEATURES: Tuple[str, ...] = ("browser_e2e",)


def unrunnable_descope_needles(
        requested_features: Tuple[str, ...]) -> Tuple[str, ...]:
    """Keyword needles for sandbox-unrunnable capabilities the goal skipped.

    Returns the union of :data:`_FEATURE_KEYWORDS` needles for every
    :data:`SANDBOX_UNRUNNABLE_FEATURES` capability *not* in
    ``requested_features`` -- the substrings DECOMPOSE matches against a
    manifest file's path/description to flag it for de-scoping. Empty when
    every unrunnable capability was explicitly requested.
    """
    needles: List[str] = []
    for feature in SANDBOX_UNRUNNABLE_FEATURES:
        if feature not in requested_features:
            needles.extend(_FEATURE_KEYWORDS[feature])
    return tuple(needles)


# Filename suffixes that can only belong to a given capability, used as
# evidence that it survived the plan critique. Checked alongside the
# capability's own keyword needles.
_FEATURE_EVIDENCE_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "frontend": (".jsx", ".tsx", ".vue", ".svelte", ".js", ".ts", ".html",
                 ".css", "package.json"),
    "containerization": ("dockerfile", "docker-compose.yml",
                         "docker-compose.yaml"),
}


def unmet_requested_features(
        requested_features: Tuple[str, ...],
        entries: List[Tuple[str, str]]) -> Tuple[str, ...]:
    """Requested capabilities that no surviving manifest file evidences.

    ``entries`` are ``(path, description)`` pairs from the post-critique
    manifest. A capability counts as covered when some file's path ends
    in one of its :data:`_FEATURE_EVIDENCE_SUFFIXES` or when its own
    keyword needles appear in a path/description. Advisory only: the
    match is a heuristic, so callers warn rather than block.
    """
    unmet: List[str] = []
    haystacks = [(str(path or "").lower(),
                  f" {str(path or '')} {str(text or '')} ".lower())
                 for path, text in entries]
    for feature in requested_features:
        suffixes = _FEATURE_EVIDENCE_SUFFIXES.get(feature, ())
        needles = _FEATURE_KEYWORDS.get(feature, ())
        covered = any(
            any(path.endswith(sfx) for sfx in suffixes)
            or any(needle in blob for needle in needles)
            for path, blob in haystacks)
        if not covered:
            unmet.append(feature)
    return tuple(unmet)


def feature_label(feature: str) -> str:
    """Human-readable name for a capability, for log/warning text."""
    return _FEATURE_LABELS.get(feature, feature)


def _detect_features(goal: str) -> Tuple[str, ...]:
    padded = f" {(goal or '').lower()} "
    found = [
        feature
        for feature, needles in _FEATURE_KEYWORDS.items()
        if any(needle in padded for needle in needles)
    ]
    return tuple(found)


def estimate_scope(goal: str) -> ScopeProfile:
    """Derive a :class:`ScopeProfile` from raw goal text -- no LLM, pure.

    The tier is driven by how many *distinct heavy capabilities* the goal
    explicitly requests; a plain "calculator" names none and lands at
    ``trivial`` with a tight file budget, while a goal that spells out an
    API + database + auth + frontend climbs to ``complex``.
    """
    requested = _detect_features(goal)
    n = len(requested)
    if n == 0:
        complexity, max_files = "trivial", 5
    elif n == 1:
        complexity, max_files = "small", 8
    elif n <= 3:
        complexity, max_files = "standard", 12
    else:
        complexity, max_files = "complex", 20

    avoided = [
        _FEATURE_LABELS[f] for f in _FEATURE_KEYWORDS if f not in requested
    ]
    lines = [
        f"SCOPE CEILING (complexity: {complexity}). Build the MINIMAL "
        "viable stack that satisfies the goal -- nothing more.",
        f"- Target at most {max_files} files total.",
        "- Prefer a single well-factored module plus its unit tests over a "
        "multi-tier architecture.",
        "- Every file and dependency must trace to an explicit requirement "
        "in the goal; do not add speculative layers.",
    ]
    if avoided:
        lines.insert(2, "- Do NOT introduce " + "; ".join(avoided)
                     + " unless the goal explicitly requires it.")
    return ScopeProfile(
        complexity=complexity,
        max_files=max_files,
        requested_features=requested,
        constraint="\n".join(lines),
    )
