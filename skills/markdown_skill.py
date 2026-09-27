"""Markdown-authored skills -- the friendly, code-free skill format.

A :class:`MarkdownSkill` adapts an Anthropic-style ``SKILL.md`` document
onto the existing :class:`skills.base.Skill` contract, so a plain-markdown
file participates in detection, prompt composition and pinning exactly
like a built-in Python skill -- without the user writing (or CGX
executing) any Python.

Why markdown, and why deterministic detection
---------------------------------------------
CGX targets small, local, often non-tool-calling models (the default is a
~3B Ollama coder). Those models cannot be trusted to *select* a skill the
way a frontier model does with progressive disclosure. So a markdown
skill activates **deterministically**:

* ``always_on: true``           -> active on every turn (like a scoped CGX.md);
* ``triggers`` / ``trigger_regex`` matched against the goal text -> active;
* otherwise                     -> only when the user explicitly *pins* it.

The body is injected as plain system-prompt text (no tools required),
gated by ``surfaces`` so a "how we write code here" skill can target
``plan``/``scaffold`` while a "domain knowledge" skill targets ``chat``.
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple

from skills.base import ROLES, Skill, SkillVerdict, file_paths
from skills.frontmatter import parse_frontmatter

__all__ = [
    "MarkdownSkill",
    "VALID_SURFACES",
    "DEFAULT_SURFACES",
    "MAX_SKILL_CHARS",
    "from_source",
    "check_source",
]

#: Where a markdown skill's body may be injected.
#:  * ``chat``     -> conversational surfaces: the Ask chatbot and the
#:                    read-only agent tasks (investigate/recommend/clarify).
#:  * ``scaffold`` -> greenfield project generation.
#:  * ``plan``     -> code-change planning.
#:  * ``all``      -> every surface above.
VALID_SURFACES: Tuple[str, ...] = ("chat", "scaffold", "plan", "all")

#: Knowledge skills default to conversational surfaces only, so authoring a
#: skill never silently perturbs the JSON-strict codegen prompts.
DEFAULT_SURFACES: Tuple[str, ...] = ("chat",)

#: Hard ceiling on a single skill document. Injected text is un-budgeted in
#: the system message, so an unbounded skill could push retrieved SOURCES
#: past a small model's ``num_ctx`` and silently truncate citations. The
#: per-call injection layer trims further; this is the authoring-time cap.
MAX_SKILL_CHARS: int = 20_000

#: Fixed detection confidence for a deterministic trigger/always-on match.
_MATCH_CONFIDENCE: float = 0.9

# Same conservative identifier allowlist the Python-skill path uses.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _as_str_list(value: Any) -> List[str]:
    """Coerce a frontmatter value into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    out: List[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return out


def _build_trigger_matcher(triggers: List[str],
                           trigger_regex: str) -> Optional[re.Pattern]:
    """Compile triggers + optional regex into one case-insensitive matcher.

    Plain keyword triggers are word-boundaried so a short token like ``eob``
    matches "EOB required" but not "neobank". Returns ``None`` when there is
    nothing to match (the skill is then pin-only unless ``always_on``).
    """
    parts: List[str] = []
    for t in triggers:
        t = t.strip()
        if not t:
            continue
        esc = re.escape(t)
        # Word-boundary only where the edge char is word-like; keeps phrases
        # and symbol-y tokens (e.g. "c++") matchable as substrings.
        left = r"\b" if t[:1].isalnum() else ""
        right = r"\b" if t[-1:].isalnum() else ""
        parts.append(f"{left}{esc}{right}")
    if trigger_regex.strip():
        parts.append(f"(?:{trigger_regex.strip()})")
    if not parts:
        return None
    try:
        return re.compile("|".join(parts), re.IGNORECASE)
    except re.error:
        return None


def _compile_all(patterns: List[str]) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns, silently dropping any that don't compile.

    ``check_source`` validates these at save time; this stays lenient so a
    slightly-malformed skill never throws during detection/validation.
    """
    out: List[Tuple[re.Pattern, str]] = []
    for p in patterns or []:
        try:
            out.append((re.compile(p), p))
        except re.error:
            continue
    return out


def _diff_body(d: Dict[str, Any]) -> str:
    return str(d.get("patch") or d.get("diff") or d.get("content")
               or d.get("new_content") or "")


def _matches_glob(path: str, pattern: str) -> bool:
    p = path.replace("\\", "/")
    return fnmatch(p, pattern) or fnmatch(p.rsplit("/", 1)[-1], pattern)


class MarkdownSkill(Skill):
    """A :class:`Skill` backed by a ``SKILL.md`` document.

    Instances are built by :func:`from_source`; do not construct directly.
    """

    def __init__(self, *, name: str, description: str, body: str,
                 aliases: Tuple[str, ...] = (), role: str = "infra",
                 triggers: Optional[List[str]] = None, trigger_regex: str = "",
                 surfaces: Tuple[str, ...] = DEFAULT_SURFACES,
                 always_on: bool = False, priority: int = 0,
                 require_files: Optional[List[str]] = None,
                 forbid_files: Optional[List[str]] = None,
                 require_patch_regex: Optional[List[str]] = None,
                 forbid_patch_regex: Optional[List[str]] = None,
                 validate_surfaces: Tuple[str, ...] = (),
                 context_globs: Optional[List[str]] = None,
                 context_exts: Optional[List[str]] = None,
                 scope: str = "global", source_path: str = "") -> None:
        self.name = name
        self.description = description
        self.aliases = tuple(aliases)
        self.role = role
        self.body = body
        self.triggers = list(triggers or [])
        self.trigger_regex = trigger_regex
        self.surfaces = tuple(surfaces)
        self.always_on = bool(always_on)
        self.priority = int(priority)
        #: "global" (~/.cgx/skills), "repo" (<root>/.cgx/skills) or "builtin".
        self.scope = scope
        self.source_path = source_path
        # Declarative, lightweight validation (the only way a markdown skill can
        # CATCH a bad diff rather than only steer via its prompt body).
        self.require_files = list(require_files or [])
        self.forbid_files = list(forbid_files or [])
        self.validate_surfaces = tuple(validate_surfaces or ())
        self._require_pat = _compile_all(require_patch_regex or [])
        self._forbid_pat = _compile_all(forbid_patch_regex or [])
        # Condition-bound activation: files the current task touches (globs) or
        # their extensions (JEV conditional instructions -- detect_context).
        self.context_globs = list(context_globs or [])
        self.context_exts = tuple(
            "." + e.lstrip(".*").lower() for e in (context_exts or []) if e.strip())
        self._matcher = _build_trigger_matcher(self.triggers, self.trigger_regex)

    # ---- detection ---------------------------------------------------
    def detect(self, goal: str) -> float:
        if self.always_on:
            return 1.0
        if not goal or self._matcher is None:
            return 0.0
        return _MATCH_CONFIDENCE if self._matcher.search(goal) else 0.0

    # ---- prompt composition (gated by surfaces) ----------------------
    def _wants(self, surface: str) -> bool:
        return "all" in self.surfaces or surface in self.surfaces

    def ask_system_prompt(self) -> str:
        return self.body if self._wants("chat") else ""

    def scaffold_system_prompt(self) -> str:
        return self.body if self._wants("scaffold") else ""

    def plan_system_prompt(self) -> str:
        return self.body if self._wants("plan") else ""

    # ---- conditional activation by working context ------------------
    def detect_context(self, ctx: Dict[str, Any]) -> float:
        """Activate when the current task touches a matching file (JEV
        conditional instructions): a glob in ``context_globs`` or an extension
        in ``context_exts``. ``ctx`` carries ``files`` / ``files_touched``.
        """
        if not (self.context_globs or self.context_exts):
            return 0.0
        files = []
        if isinstance(ctx, dict):
            files = ctx.get("files") or ctx.get("files_touched") or []
        for f in files or []:
            fp = str(f).replace("\\", "/")
            if self.context_exts and fp.lower().endswith(self.context_exts):
                return _MATCH_CONFIDENCE
            if any(_matches_glob(fp, g) for g in self.context_globs):
                return _MATCH_CONFIDENCE
        return 0.0

    # ---- declarative validation (gated by validate_surfaces) --------
    def _validates(self, surface: str) -> bool:
        # Validate on the codegen surfaces this skill targets (never chat --
        # there are no diffs there). A chat-only skill has no non-chat surface
        # and so validates nothing.
        surfs = self.validate_surfaces or tuple(
            s for s in self.surfaces if s != "chat")
        return bool(surfs) and ("all" in surfs or surface in surfs)

    def _forbidden_file(self, diffs: List[Dict[str, Any]]) -> Optional[SkillVerdict]:
        paths = [str(p) for p in file_paths(diffs)]
        for g in self.forbid_files:
            for p in paths:
                if _matches_glob(p, g):
                    return SkillVerdict(
                        passed=False, confidence=0.8, skill=self.name,
                        rationale=(f"{self.name} skill: '{p}' matches the "
                                   f"forbidden file pattern '{g}'."))
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        # Paths-only at plan time -> only the safe direction (forbid_files);
        # require_files would false-positive on a small incremental edit.
        if not self.forbid_files or not self._validates("plan"):
            return None
        return self._forbidden_file(diffs)

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        if not self._validates("scaffold"):
            return None
        forb = self._forbidden_file(diffs)
        if forb is not None:
            return forb
        paths = [str(p) for p in file_paths(diffs)]
        for g in self.require_files:
            if not any(_matches_glob(p, g) for p in paths):
                return SkillVerdict(
                    passed=False, confidence=0.8, skill=self.name,
                    rationale=(f"{self.name} skill: no file matches the required "
                               f"pattern '{g}'."))
        bodies = [_diff_body(d) for d in (diffs or []) if isinstance(d, dict)]
        for rx, src in self._forbid_pat:
            if any(rx.search(b) for b in bodies):
                return SkillVerdict(
                    passed=False, confidence=0.8, skill=self.name,
                    rationale=(f"{self.name} skill: generated content matches the "
                               f"forbidden pattern /{src}/."))
        for rx, src in self._require_pat:
            if not any(rx.search(b) for b in bodies):
                return SkillVerdict(
                    passed=False, confidence=0.8, skill=self.name,
                    rationale=(f"{self.name} skill: no generated content matches "
                               f"the required pattern /{src}/."))
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (f"<MarkdownSkill {self.name!r} scope={self.scope} "
                f"surfaces={self.surfaces} always_on={self.always_on}>")


def _normalize_meta(meta: Dict[str, Any], body: str,
                    name_hint: str) -> Dict[str, Any]:
    """Turn raw frontmatter into a validated, defaulted spec dict."""
    name = str(meta.get("name") or name_hint or "").strip()
    surfaces = _as_str_list(meta.get("surfaces")) or list(DEFAULT_SURFACES)
    roles = _as_str_list(meta.get("roles") or meta.get("role"))
    role = roles[0] if roles else "infra"
    triggers = _as_str_list(meta.get("triggers") or meta.get("keywords"))
    priority = meta.get("priority")
    try:
        priority = int(priority) if priority is not None else 0
    except (TypeError, ValueError):
        priority = 0
    return {
        "name": name,
        "description": str(meta.get("description") or "").strip(),
        "aliases": tuple(_as_str_list(meta.get("aliases"))),
        "role": role if role in ROLES else "infra",
        "surfaces": tuple(s for s in surfaces if s in VALID_SURFACES)
        or DEFAULT_SURFACES,
        "triggers": triggers,
        "trigger_regex": str(meta.get("trigger_regex") or "").strip(),
        "always_on": bool(meta.get("always_on") or False),
        "priority": priority,
        "require_files": _as_str_list(meta.get("require_files")),
        "forbid_files": _as_str_list(meta.get("forbid_files")),
        "require_patch_regex": _as_str_list(meta.get("require_patch_regex")),
        "forbid_patch_regex": _as_str_list(meta.get("forbid_patch_regex")),
        "validate_surfaces": tuple(
            s for s in _as_str_list(meta.get("validate_surfaces"))
            if s in VALID_SURFACES),
        "context_globs": _as_str_list(meta.get("context_globs")),
        "context_exts": _as_str_list(meta.get("context_exts")),
        "body": body,
    }


def from_source(content: str, *, name_hint: str = "", scope: str = "global",
                source_path: str = "") -> MarkdownSkill:
    """Build a :class:`MarkdownSkill` from raw ``SKILL.md`` text.

    Lenient by design -- missing/odd fields fall back to defaults so the
    loader never throws on a slightly malformed skill. Use :func:`check_source`
    for the strict, user-facing validation that gates a *save*.
    """
    meta, body = parse_frontmatter(content or "")
    spec = _normalize_meta(meta, body, name_hint)
    return MarkdownSkill(scope=scope, source_path=source_path, **spec)


def check_source(content: str, *, name_hint: str = "",
                 known_names: Optional[set] = None
                 ) -> Tuple[bool, str, str, Dict[str, Any]]:
    """Strictly validate a candidate skill document before it is persisted.

    Returns ``(ok, error_kind, error_detail, meta)``. ``error_kind`` is one
    of ``invalid_name`` / ``empty_body`` / ``too_large`` / ``invalid_field``
    / ``name_collision`` (mirroring the Python path's kinds so the same UI
    error panel renders both). No code is executed -- markdown skills are
    strictly safer than Python skills.
    """
    if content is None or len(content) > MAX_SKILL_CHARS:
        return (False, "too_large",
                f"skill document exceeds {MAX_SKILL_CHARS} characters", {})
    meta, body = parse_frontmatter(content)
    spec = _normalize_meta(meta, body, name_hint)

    name = spec["name"]
    if not name:
        return (False, "invalid_name",
                "skill needs a `name:` in its frontmatter (or a name)", {})
    if not _NAME_RE.match(name):
        return (False, "invalid_name",
                f"'{name}' must be 1-64 chars of letters/digits/_/- "
                "and start alphanumeric", {})
    if not spec["body"].strip():
        return (False, "empty_body",
                "skill body is empty -- add the instructions below the "
                "frontmatter", {})

    # Surfaces / roles must be recognised (a typo should fail loudly, not
    # silently drop the skill from a surface the author expected).
    declared_surfaces = _as_str_list(meta.get("surfaces"))
    bad = [s for s in declared_surfaces if s not in VALID_SURFACES]
    if bad:
        return (False, "invalid_field",
                f"unknown surface(s) {bad}; valid: {list(VALID_SURFACES)}", {})
    declared_roles = _as_str_list(meta.get("roles") or meta.get("role"))
    bad_roles = [r for r in declared_roles if r not in ROLES]
    if bad_roles:
        return (False, "invalid_field",
                f"unknown role(s) {bad_roles}; valid: {list(ROLES)}", {})
    if spec["trigger_regex"]:
        try:
            re.compile(spec["trigger_regex"])
        except re.error as e:
            return (False, "invalid_field",
                    f"trigger_regex does not compile: {e}", {})
    for key in ("require_patch_regex", "forbid_patch_regex"):
        for pat in _as_str_list(meta.get(key)):
            try:
                re.compile(pat)
            except re.error as e:
                return (False, "invalid_field",
                        f"{key} entry {pat!r} does not compile: {e}", {})

    if known_names:
        low = name.lower()
        aliases_low = [a.lower() for a in spec["aliases"]]
        if low in known_names or any(a in known_names for a in aliases_low):
            return (False, "name_collision",
                    f"'{name}' collides with an existing skill's name or alias",
                    {})

    return (True, "", "", {k: v for k, v in spec.items() if k != "body"})
