"""The feedback loop's pure core: where does a piece of user feedback belong?

After a swarm build the user looks at the running app and says things like
"make the top bar span the full width", "the Send button should sit level
with the textarea", "remember the API key after a reload" or "why is the
settings page blank?". For the loop *user asks -> agent builds -> user
reviews -> user gives feedback -> agent revises -> user reviews again* to
work with a weak local model, the agent must first decide **exactly which
planned files** the feedback lands in -- and it must do so from evidence,
not from a guess.

This module is IO-free so the router can use its classifiers without
importing the executor stack. It provides:

* :func:`is_change_request` -- does a message on a built session ask for a
  change (revise) or ask a question (explore)?
* :func:`latest_build` -- the build a session's task tree currently holds
  (plan artifact, last verify report, revision count, in-flight state).
* :func:`rank_files` -- deterministic localization: score every planned file
  against the feedback using named files/components/symbols, routes and
  screens, quoted UI copy found in the sources, concern keywords mapped to
  file roles (stylesheet / app shell / API client / storage / README ...),
  plus a weak description-overlap signal. Each candidate carries the
  *reasons* it was picked so the revision report can show "why this file".
* :func:`theme_from_feedback` -- "dark mode" / a named theme switches the
  deterministic tokens stylesheet without a model call.
* :func:`unified_diff` -- what actually changed, for the report.

The executor (:mod:`cgx.session.tasks.swarm_revise`) refines the ranked
candidates with one bounded LLM call that returns a per-file instruction,
applies each as a grounded modification of the current file, and hands the
tree back to ``SWARM_VERIFY`` so the result is rebuilt, smoke-tested and
screenshotted again.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cgx.session.models import SessionStatus, TaskKind, TaskNode, TaskNodeStatus

# --------------------------------------------------------------------------
# message classification
# --------------------------------------------------------------------------

# Words that mark a message as *asking for a change* even when it is phrased
# as a question ("can you make the header bigger?") or as a problem report
# ("the settings page is blank"). On a built swarm session such a message is
# a revision, never a fresh plan or a read-only investigation.
_CHANGE_RE = re.compile(
    r"\b(make|change|move|add|remove|delete|drop|fix|use|increase|reduce|decrease|"
    r"align|replace|rename|put|place|show|hide|turn|set|switch|swap|resize|enlarge|"
    r"shrink|should|shouldn'?t|needs?|want|wants|please|instead|rather|bigger|smaller|"
    r"larger|wider|narrower|taller|shorter|full[- ]?width|center|centre|darker|lighter|"
    r"bolder|doesn'?t|does ?not|isn'?t|is ?not|aren'?t|won'?t|can'?t|not working|broken|"
    r"wrong|missing|can you|could you|would be|better|improve|update|adjust|tweak|"
    r"style|restyle|redesign|rework|refactor|support|allow|enable|disable|also|revert|"
    r"undo|blank|crash|crashes|error|errors|fails?|bug|persist|remember|save|keep)\b",
    re.I)


def is_change_request(message: Optional[str]) -> bool:
    """True when ``message`` asks for a change to a built app (not a question).

    Deterministic: anything that is not question-shaped is a change request,
    and a question-shaped message still counts when it names a change or a
    problem ("can you make the header bigger?", "why is the page blank?").
    A pure question ("how does the login work?") stays a question.
    """
    text = (message or "").strip()
    if not text:
        return False
    from cgx.session.mode import is_question
    if not is_question(text):
        return True
    return bool(_CHANGE_RE.search(text))


# --------------------------------------------------------------------------
# what build does this session hold?
# --------------------------------------------------------------------------

_SWARM_BUILD_KINDS = (TaskKind.SWARM_TECH_LEAD, TaskKind.SWARM_DEVELOPER,
                      TaskKind.SWARM_VERIFY, TaskKind.SWARM_REVISE)
_IN_FLIGHT = (TaskNodeStatus.READY, TaskNodeStatus.IN_PROGRESS,
              TaskNodeStatus.PENDING, TaskNodeStatus.BLOCKED)


@dataclass
class BuildRef:
    """The swarm build a session's task tree currently holds."""
    work_plan_artifact_id: str
    goal: str = ""
    project_root: Optional[str] = None
    verify_report_artifact_id: Optional[str] = None
    last_verify_ok: Optional[bool] = None
    revisions: int = 0
    in_flight: bool = False


def latest_build(tasks: Sequence[TaskNode]) -> Optional[BuildRef]:
    """The build this task tree holds, or ``None`` when nothing was built.

    A build exists once a Tech Lead plan was persisted AND at least one
    Developer / Verify / Revise task finished (so files were written). The
    newest ``work_plan_artifact_id`` wins; the newest finished SWARM_VERIFY
    supplies the report id and verdict. ``in_flight`` is set while any
    non-ASK swarm task is still queued or running, so a caller can refuse to
    fork a build that is mid-chain.
    """
    ordered = sorted(tasks, key=lambda t: t.created_at)
    plan_id = ""
    goal = ""
    root: Optional[str] = None
    report_id: Optional[str] = None
    verify_ok: Optional[bool] = None
    written = False
    revisions = 0
    in_flight = False
    for t in ordered:
        if t.kind not in _SWARM_BUILD_KINDS:
            continue
        if t.status in _IN_FLIGHT:
            in_flight = True
        outputs = t.outputs or {}
        inputs = t.inputs or {}
        pid = str(outputs.get("work_plan_artifact_id")
                  or inputs.get("work_plan_artifact_id") or "")
        if pid:
            plan_id = pid
        goal = str(outputs.get("goal") or inputs.get("goal") or goal or "")
        root = outputs.get("project_root") or inputs.get("project_root") or root
        if t.status is TaskNodeStatus.DONE and t.kind in (
                TaskKind.SWARM_DEVELOPER, TaskKind.SWARM_VERIFY, TaskKind.SWARM_REVISE):
            written = True
        if t.kind is TaskKind.SWARM_VERIFY and t.status is TaskNodeStatus.DONE:
            rid = outputs.get("verify_report_artifact_id")
            if rid:
                report_id = str(rid)
            verify_ok = bool(outputs.get("verify_ok"))
        if t.kind is TaskKind.SWARM_REVISE:
            revisions += 1
    if not plan_id or not written:
        return None
    return BuildRef(work_plan_artifact_id=plan_id, goal=goal, project_root=root,
                    verify_report_artifact_id=report_id, last_verify_ok=verify_ok,
                    revisions=revisions, in_flight=in_flight)


def prior_status_value(session_status: SessionStatus) -> str:
    """The status a no-op / crashed revision restores (terminal ones only)."""
    if session_status in (SessionStatus.COMPLETED, SessionStatus.FAILED):
        return session_status.value
    return SessionStatus.COMPLETED.value


# --------------------------------------------------------------------------
# deterministic localization
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    path: str
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def add(self, points: float, reason: str) -> None:
        self.score += points
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def as_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "score": round(self.score, 1), "reasons": list(self.reasons)}


_STOP = frozenset("""
a an the and or but if then than so to of in on at for with from by as is are was were be been
being it its this that these those there here i you we they he she them us our your my me
please make change should would could can will just also very really like want need get
have has had do does did not no yes into onto out up down over under about after before
when while where why how what which who whom whose all any some more most much many few
app page screen button thing stuff bit little big use using used
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
_QUOTED_RE = re.compile(r"[\"“”'‘’`]([^\"“”'‘’`]{2,60})[\"“”'‘’`]")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ROUTE_RE = re.compile(r"""<Route\b[^>]*\bpath\s*=\s*['"]([^'"]+)['"][^>]*\belement\s*=\s*\{\s*<([A-Z]\w*)""", re.S)
_ROUTE_PATH_RE = re.compile(r"(?<![\w/])/[a-z][\w\-/]*")

# Concern keywords -> the file ROLE (``cgx.design.prompts.file_kind``) that owns
# them. A visual complaint lands in the stylesheet (+ the component that renders
# the element, found by name/copy); a connection problem lands in the API
# client; "remember after reload" lands in the storage helper; and so on.
_CONCERNS: Dict[str, Tuple[re.Pattern, Dict[str, float], str]] = {
    "style": (re.compile(
        r"\b(colou?rs?|palette|fonts?|typography|typeface|spacing|padding|margin|gaps?|width|"
        r"height|full[- ]?width|align(ed|ment)?|cent(er|re)(ed)?|borders?|shadows?|radius|"
        r"rounded|background|layout|responsive|mobile|bigger|smaller|larger|size|look|looks|"
        r"style|styling|styled|ugly|contrast|overflow|wrap|wrapping|scroll(ing|s)?|sticky|"
        r"fixed|position|hover|spacing|cramped|crowded|tiny|huge|bold|italic|underline|"
        r"dark|light|theme|gradient|opacity|animation|transition|icon|icons)\b", re.I),
        {"stylesheet": 5.0, "app": 1.0}, "visual/layout feedback"),
    "shell": (re.compile(
        r"\b(top ?bar|header|nav(bar|igation)?|menu|logo|brand(ing)?|title ?bar|sidebar|footer|"
        r"shell|tabs?|links? to|navigate)\b", re.I),
        {"app": 5.0, "stylesheet": 2.0}, "navigation/shell feedback"),
    "api": (re.compile(
        r"\b(api|endpoint|requests?|fetch(ing)?|responses?|api ?key|token|auth(orization)?|"
        r"bearer|base ?url|url|address|connect(ion|ing|s)?|timeout|retry|retries|stream(ing)?|"
        r"model|server|cors|status ?code|404|401|500)\b", re.I),
        {"api": 6.0, "storage": 1.0}, "API/connection feedback"),
    "storage": (re.compile(
        r"\b(save[ds]?|saving|persist(s|ed|ence)?|remember(s|ed)?|storage|local ?storage|"
        r"stored?|reload(s|ed|ing)?|refresh(es|ed|ing)?|forget(s)?|forgot|reset|clear(s|ed)?|"
        r"history|lost|lose|keep)\b", re.I),
        {"storage": 5.0}, "persistence feedback"),
    "sample": (re.compile(
        r"\b(sample|seed(ed)?|mock|demo data|placeholder data|fake data|dummy|fixtures?)\b", re.I),
        {"sample": 6.0}, "sample-data feedback"),
    "html": (re.compile(
        r"\b(html|<title>|page title|browser tab|favicon|meta tags?|viewport|lang attribute)\b", re.I),
        {"html": 6.0}, "document/HTML feedback"),
    "package": (re.compile(
        r"\b(dependenc(y|ies)|packages?|npm|yarn|pnpm|install(ing|ed|s)?|versions?|package\.json|"
        r"scripts?|vite|build fails?|lint(ing)?)\b", re.I),
        {"package": 5.0, "config": 3.0}, "dependency/build feedback"),
    "readme": (re.compile(
        r"\b(readme|documentation|docs|instructions|how to run|setup guide|getting started)\b", re.I),
        {"other": 0.0}, "documentation feedback"),
    "tests": (re.compile(r"\b(tests?|spec|coverage|vitest|jest|pytest)\b", re.I),
              {"test": 5.0}, "test feedback"),
    "runtime": (re.compile(
        r"\b(crash(es|ed|ing)?|error|errors|blank|white page|empty page|nothing happens|doesn'?t "
        r"work|does not work|not working|broken|undefined|console|exception|stuck|hangs?|freez(e|es|ing))\b",
        re.I),
        {}, "problem report"),
}

# UI element words -> a source pattern the owning file must contain. Structural
# elements (a header, the composer, a sidebar) are strong evidence of WHICH
# component owns the change; generic ones (a button, an input) only break ties.
_STRUCTURAL_ELEMENTS = frozenset({
    "composer", "sidebar", "footer", "header", "top bar", "topbar", "nav", "navigation",
    "menu", "logo", "bubble", "modal", "dialog", "toast", "table", "chart", "kpi",
    "empty state", "spinner", "loading", "form", "textarea",
})
_ELEMENTS: Dict[str, re.Pattern] = {
    "button": re.compile(r"<button\b|\bButton\b"),
    "textarea": re.compile(r"<textarea\b"),
    "input": re.compile(r"<input\b"),
    "form": re.compile(r"<form\b"),
    "select": re.compile(r"<select\b"),
    "dropdown": re.compile(r"<select\b|dropdown|Dropdown"),
    "checkbox": re.compile(r"type=\"checkbox\"|type='checkbox'"),
    "toggle": re.compile(r"toggle|Toggle|switch|Switch"),
    "modal": re.compile(r"modal|Modal|dialog|Dialog"),
    "dialog": re.compile(r"modal|Modal|dialog|Dialog"),
    "toast": re.compile(r"toast|Toast"),
    "badge": re.compile(r"badge|Badge"),
    "card": re.compile(r"card|Card"),
    "table": re.compile(r"<table\b|<Table\b"),
    "list": re.compile(r"<ul\b|<ol\b|\.map\("),
    "chip": re.compile(r"chip|Chip"),
    "avatar": re.compile(r"avatar|Avatar"),
    "placeholder": re.compile(r"placeholder="),
    "label": re.compile(r"<label\b"),
    "spinner": re.compile(r"spinner|Spinner|loading|Loading"),
    "loading": re.compile(r"spinner|Spinner|loading|Loading"),
    "empty state": re.compile(r"empty|Empty"),
    "bubble": re.compile(r"bubble|Bubble"),
    "message": re.compile(r"message|Message"),
    "composer": re.compile(r"composer|Composer"),
    "sidebar": re.compile(r"sidebar|Sidebar"),
    "footer": re.compile(r"<footer\b|footer|Footer"),
    "header": re.compile(r"<header\b|header|Header|TopBar|topbar"),
    "top bar": re.compile(r"<header\b|TopBar|topbar|top-bar"),
    "topbar": re.compile(r"<header\b|TopBar|topbar|top-bar"),
    "nav": re.compile(r"<nav\b|NavLink|<Link\b"),
    "navigation": re.compile(r"<nav\b|NavLink|<Link\b"),
    "menu": re.compile(r"<nav\b|menu|Menu"),
    "link": re.compile(r"<a\b|<Link\b|NavLink"),
    "title": re.compile(r"<h1\b|<title>"),
    "heading": re.compile(r"<h[1-3]\b"),
    "logo": re.compile(r"logo|Logo|brand|Brand"),
    "icon": re.compile(r"icon|Icon|<svg\b"),
    "image": re.compile(r"<img\b|background-image"),
    "chart": re.compile(r"chart|Chart|<svg\b"),
    "kpi": re.compile(r"kpi|Kpi|KPI|stat|Stat"),
}

_SHELL_STEM_RE = re.compile(r"(shell|layout|topbar|top-?bar|header|navbar|nav\b|sidebar|footer)", re.I)

_HOME_RE = re.compile(r"\b(main page|home ?page|home|landing|front page|first page|start page|index page|the main screen)\b", re.I)


def _words(text: str) -> List[str]:
    return [w.lower().strip("'") for w in _WORD_RE.findall(text or "")]


def _keywords(text: str) -> List[str]:
    """Content words of the feedback (lower-cased, stopwords dropped, >= 3 chars)."""
    out: List[str] = []
    for w in _words(text):
        if len(w) < 3 or w in _STOP:
            continue
        if w not in out:
            out.append(w)
    return out


def split_identifier(name: str) -> List[str]:
    """``SettingsForm`` -> ``["settings", "form"]``; ``app.css`` -> ``["app"]``."""
    stem = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.split(".", 1)[0]
    parts = re.split(r"[_\-\s]+", _CAMEL_RE.sub(" ", stem))
    return [p.lower() for p in parts if p]


def _phrase_in(words: List[str], text_low: str) -> bool:
    """All of ``words`` appear in ``text_low`` as whole words (any order), none trivial."""
    if not words:
        return False
    for w in words:
        if len(w) < 3 and w not in ("ui", "db"):
            return False
        if not re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?:s|es)?(?![a-z0-9])", text_low):
            return False
    return True


def _route_component_files(contents: Dict[str, str],
                           paths: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    """``{route: {"files": [routed component], "imports": [its first-party imports]}}``.

    Read from the ``<Route path element>`` tables in the sources; the routed
    component is the primary suspect for anything said about that screen, its
    direct first-party imports (the form, the list, the helper it renders) the
    second ring.
    """
    try:
        from cgx.session.js_symbols import js_imports, resolve_relative
    except Exception:  # pragma: no cover - defensive
        return {}
    out: Dict[str, Dict[str, List[str]]] = {}
    for path, src in contents.items():
        for m in _ROUTE_RE.finditer(src or ""):
            route, comp = m.group(1), m.group(2)
            entry = out.setdefault(route, {"files": [], "imports": []})
            files = entry["files"]
            for stmt in js_imports(src):
                if stmt.default == comp or any(local == comp for _, local in stmt.named):
                    hit = resolve_relative(stmt.spec, path, paths)
                    if hit and hit not in files:
                        files.append(hit)
            if not files:
                files.append(path)  # component defined inline in the router file
            for f in list(files):
                for stmt in js_imports(contents.get(f) or ""):
                    hit = resolve_relative(stmt.spec, f, paths)
                    if (hit and hit not in files and hit not in entry["imports"]
                            and hit.lower().endswith((".js", ".jsx", ".ts", ".tsx"))):
                        entry["imports"].append(hit)
    return out


def _file_role(design: Optional[Dict[str, Any]], path: str) -> str:
    """The design-layer role of ``path`` (``stylesheet`` / ``app`` / ``api`` ...), or a path guess."""
    if design:
        try:
            from cgx.design import file_kind
            return file_kind(design, path)
        except Exception:  # pragma: no cover - design layer must not break localization
            pass
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    if low.endswith((".css", ".scss", ".sass", ".less")):
        return "stylesheet"
    if base == "package.json":
        return "package"
    if base.endswith(".html"):
        return "html"
    if re.search(r"(^|/)(test_|tests?/|.*\.(test|spec)\.)", low):
        return "test"
    if re.search(r"(api|client|http|request)", base):
        return "api"
    if re.search(r"(storage|settings|persist|store)", base):
        return "storage"
    if base.startswith("readme"):
        return "other"
    return "source"


def _source_symbols(path: str, src: str) -> List[str]:
    """Exported/defined names of a source file (JS exports, Python defs/classes)."""
    low = path.lower()
    names: List[str] = []
    if low.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs")):
        try:
            from cgx.session.js_symbols import js_exports
            ex = js_exports(src)
            names.extend(ex.get("named") or [])
            if ex.get("default"):
                names.append(str(ex["default"]))
        except Exception:  # pragma: no cover
            pass
    elif low.endswith(".py"):
        names.extend(re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)", src, re.M))
    return [n for n in names if n]


def rank_files(feedback: str, paths: Sequence[str], specs: Dict[str, Any],
               design: Optional[Dict[str, Any]], contents: Dict[str, str], *,
               failing_routes: Optional[Sequence[str]] = None,
               limit: int = 4) -> List[Candidate]:
    """Score every planned file against ``feedback``; return the likely targets.

    Signals (each attaches a human-readable reason):

    * the feedback names the file, its stem words ("top bar" -> ``TopBar.jsx``,
      "settings form" -> ``SettingsForm.jsx``) or one of its exported symbols;
    * the feedback names a route or screen ("settings page", "/settings", "main
      page") -> the files that implement that route;
    * a quoted string or a Capitalised word in the feedback appears as UI copy
      in the file ("the 'Send' button", "Not Connected");
    * a concern keyword maps to a file role: visual words -> the stylesheet,
      navigation words -> the app shell, connection words -> the API client,
      "remember after reload" -> the storage helper, "readme" -> README.md;
    * a generic element word ("textarea", "button") -> files that render it;
    * a problem report ("blank", "crash") -> the files behind the routes the
      last runtime smoke flagged;
    * weak: overlap between the feedback and the file's planned description.

    The deterministic tokens stylesheet is never a target (CGX owns it; see
    :func:`theme_from_feedback`). Strong candidates (score >= 6) are returned
    first, capped at ``limit``; with no strong candidate the top two weak ones
    are returned so the model refinement still has something concrete to look
    at; an empty list means "could not localize".
    """
    text = feedback or ""
    low = text.lower()
    keywords = _keywords(text)
    cands: Dict[str, Candidate] = {p: Candidate(p) for p in paths}
    tokens_path = str((design or {}).get("tokens_path") or "")
    roles = {p: _file_role(design, p) for p in paths}
    routes = _route_component_files(contents, paths)

    # 1. the feedback names the file / its stem words / an exported symbol
    for p in paths:
        base = p.replace("\\", "/").rsplit("/", 1)[-1]
        stem = base.split(".", 1)[0]
        if base.lower() in low or (p.lower() in low):
            cands[p].add(10.0, f"named in the feedback ({base})")
            continue
        words = split_identifier(stem)
        if words and _phrase_in(words, low) and (len(words) > 1 or len(words[0]) >= 4):
            cands[p].add(8.0, f"the feedback mentions \"{' '.join(words)}\" ({base})")
        for sym in _source_symbols(p, contents.get(p, "")):
            sw = split_identifier(sym)
            if sw and sym.lower() != stem.lower() and _phrase_in(sw, low) and (
                    len(sw) > 1 or len(sw[0]) >= 5):
                cands[p].add(6.0, f"mentions \"{' '.join(sw)}\", exported by {base}")

    # 2. routes / screens
    wants_home = bool(_HOME_RE.search(text))
    mentioned_routes = {m.group(0).rstrip("/") or "/" for m in _ROUTE_PATH_RE.finditer(low)}
    for route, entry in routes.items():
        files = entry["files"]
        route_words = [w for seg in route.strip("/").split("/") for w in split_identifier(seg) if len(w) >= 4]
        hit = route.rstrip("/") in mentioned_routes and route != "/"
        if route == "/" and wants_home:
            hit = True
        if not hit and route_words and any(
                re.search(r"(?<![a-z0-9])" + re.escape(w) + r"s?(?![a-z0-9])", low) for w in route_words):
            hit = True
        if hit:
            for f in files:
                if f in cands:
                    cands[f].add(6.0, f"implements the {route} route the feedback refers to")
            for f in entry["imports"]:
                if f in cands:
                    cands[f].add(2.0, f"rendered by the {route} route the feedback refers to")

    # 3. quoted UI copy / Capitalised words found in the sources
    literals = [q.strip() for q in _QUOTED_RE.findall(text) if q.strip()]
    literals += [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
                 if w.lower() not in _STOP and not text.startswith(w)]
    for lit in literals:
        rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(lit) + r"(?![a-z0-9_])")
        for p in paths:
            src = contents.get(p) or ""
            if not src or roles.get(p) in ("package", "config", "tokens"):
                continue
            # UI copy: between tags, in a string literal or an attribute value.
            if re.search(r"(>|[\"'`]|=\s*)\s*[^<>\"'`]*" + re.escape(lit), src) and rx.search(src):
                pts = 7.0 if len(literals) and lit in [q.strip() for q in _QUOTED_RE.findall(text)] else 3.0
                cands[p].add(pts, f"contains the text \"{lit}\"")

    # 4. concern keywords -> file roles
    named_component = any(c.score >= 6 for c in cands.values())
    for key, (rx, weights, why) in _CONCERNS.items():
        if not rx.search(text):
            continue
        if key == "readme":
            for p in paths:
                if p.lower().rsplit("/", 1)[-1].startswith("readme"):
                    cands[p].add(8.0, why)
            continue
        if key == "runtime":
            for route in (failing_routes or []):
                entry = routes.get(route) or {"files": [], "imports": []}
                for f in entry["files"]:
                    if f in cands:
                        cands[f].add(5.0, f"the {route} route failed the last runtime check")
                for f in entry["imports"]:
                    if f in cands:
                        # a crash in a child component blanks the whole page
                        cands[f].add(4.0, f"rendered by the {route} route that failed the last runtime check")
            continue
        if key == "shell":
            for p in paths:
                if _SHELL_STEM_RE.search(p.rsplit("/", 1)[-1]) and roles.get(p) in ("component", "app"):
                    cands[p].add(5.0, "a shell/navigation component by name")
        for role, pts in weights.items():
            for p in paths:
                if roles.get(p) == role and p != tokens_path:
                    bonus = pts
                    if key == "style" and role == "stylesheet" and not named_component:
                        bonus += 2.0  # global visual feedback -> the app stylesheet
                    cands[p].add(bonus, why)

    # 5. generic element words -> files that render that element
    for word, rx in _ELEMENTS.items():
        if not re.search(r"(?<![a-z])" + re.escape(word) + r"s?(?![a-z])", low):
            continue
        pts = 6.0 if word in _STRUCTURAL_ELEMENTS else 3.0
        for p in paths:
            if roles.get(p) in ("component", "app", "html") and rx.search(contents.get(p) or ""):
                cands[p].add(pts, f"renders the {word}" if pts >= 6 else f"renders a {word}")

    # 6. weak: description overlap
    for p in paths:
        desc = str((specs.get(p) or {}).get("description") or "")
        if not desc:
            continue
        dw = set(_keywords(desc))
        shared = [k for k in keywords if k in dw]
        if len(shared) >= 2:
            cands[p].add(min(4.0, 1.0 * len(shared)), "its planned purpose mentions " + ", ".join(shared[:4]))

    # the tokens stylesheet is generated deterministically; never a model target
    ranked = sorted((c for c in cands.values() if c.score > 0 and c.path != tokens_path),
                    key=lambda c: (-c.score, c.path))
    strong = [c for c in ranked if c.score >= 6.0][:limit]
    if strong:
        return strong
    return [c for c in ranked if c.score >= 3.0][:2]


# --------------------------------------------------------------------------
# deterministic theme switch
# --------------------------------------------------------------------------

_DARK_RE = re.compile(r"\b(dark(er)? (mode|theme|look|background|colou?rs?)|make it dark(er)?|go dark|night mode)\b", re.I)
_LIGHT_RE = re.compile(r"\b(light(er)? (mode|theme|look|background)|make it light(er)?|white background|day mode)\b", re.I)


def theme_from_feedback(feedback: str, design: Optional[Dict[str, Any]]) -> Optional[str]:
    """A new theme key the feedback asks for, or ``None``.

    Recognises a theme by name ("use the Midnight Console theme", "warm
    artisan") and the generic "dark mode" / "light mode" requests, which pick
    the archetype's natural dark/light theme. Returns ``None`` when the
    feedback does not ask for a theme change or already matches the current
    theme, so the caller never rewrites the tokens file for nothing.
    """
    if not design or not design.get("applies"):
        return None
    try:
        from cgx.design import THEMES, theme_by_key
    except Exception:  # pragma: no cover
        return None
    low = (feedback or "").lower()
    current = str(design.get("theme") or "")
    for t in THEMES:
        names = {t.key.replace("-", " "), t.name.lower()}
        if any(n and n in low for n in names):
            return None if t.key == current else t.key
    cur = theme_by_key(current)
    archetype = str(design.get("archetype") or "")
    if _DARK_RE.search(low) and not (cur and cur.dark):
        for t in THEMES:
            if t.dark and archetype in t.default_for:
                return t.key
        return "midnight-console" if archetype in ("developer-tool", "analytics-dashboard",
                                                   "admin-dashboard", "chat-assistant") else "bold-dark"
    if _LIGHT_RE.search(low) and cur and cur.dark:
        for t in THEMES:
            if not t.dark and archetype in t.default_for:
                return t.key
        return "clean-tech"
    return None


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def unified_diff(path: str, before: str, after: str, max_lines: int = 120) -> str:
    """A capped unified diff of one file, for the revision report."""
    lines = list(difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=2))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


def relevant_lines(src: str, feedback: str, *, max_lines: int = 24, context: int = 1) -> str:
    """Numbered source lines that mention a feedback keyword (plus context).

    Gives the localizing model the exact region a change belongs to instead of
    a whole file, which keeps the prompt small enough for a local model while
    still being evidence rather than a summary. Falls back to the file head.
    """
    lines = (src or "").splitlines()
    if not lines:
        return ""
    kws = [k for k in _keywords(feedback) if len(k) >= 3]
    hits: List[int] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in kws):
            hits.append(i)
    if not hits:
        chosen = list(range(min(len(lines), max_lines)))
    else:
        chosen_set = set()
        for i in hits:
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                chosen_set.add(j)
        chosen = sorted(chosen_set)[:max_lines]
    out: List[str] = []
    prev = None
    for i in chosen:
        if prev is not None and i != prev + 1:
            out.append("      ...")
        out.append(f"{i + 1:5d}: {lines[i][:160]}")
        prev = i
    return "\n".join(out)


__all__ = [
    "BuildRef", "Candidate", "is_change_request", "latest_build", "prior_status_value",
    "rank_files", "relevant_lines", "split_identifier", "theme_from_feedback", "unified_diff",
]


# --------------------------------------------------------------------------
# multi-part feedback, CSS awareness and apply-coverage checks
# --------------------------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\n+|;\s+|\.\s+(?=[A-Z])|,\s+and\s+(?:also\s+|then\s+)?|\s+and\s+(?:also\s+|then\s+)"
    r"|\s+and\s+(?=(?:make|put|move|add|remove|delete|change|align|use|show|hide|fix|set|place|"
    r"keep|turn|increase|reduce|replace|rename|give|let|have|center|centre|resize|switch|drop)\b)"
    r"|\s+-\s+|\s*•\s*|(?:^|\s)\d+[.)]\s+)", re.I)

_ACTION_VERB_RE = re.compile(
    r"^(make|put|move|add|remove|delete|change|align|use|show|hide|fix|set|place|keep|turn|"
    r"increase|reduce|replace|rename|give|let|have|center|centre|resize|switch|drop|the|a|an|"
    r"it|its|there|please|also|then|and|but|so|i|we|you|why|what|when)\b", re.I)


def split_feedback(feedback: str) -> List[str]:
    """Split one message into its separately actionable parts.

    "make the top bar full width, and put the Send button on the same row as
    the textarea" is two asks that usually land in different files; a weak
    model given both in one instruction tends to apply one. Parts are split on
    sentence ends, semicolons, bullets / numbering and an ``and`` that starts a
    new imperative. A one-word fragment is folded into its
    predecessor. A message with one part returns ``[feedback]``.
    """
    text = (feedback or "").strip()
    if not text:
        return []
    raw = [p.strip(" .;,-\t") for p in _CLAUSE_SPLIT_RE.split(text)]
    parts: List[str] = []
    for p in raw:
        if not p:
            continue
        if len(p.split()) < 2 and parts:
            parts[-1] = parts[-1] + ", " + p
        else:
            parts.append(p)
    return parts if len(parts) > 1 else [text]


_SELECTOR_TOKEN_RE = re.compile(r"(?<![\w@])[.#][A-Za-z_][\w-]*")
_HYPHEN_TOKEN_RE = re.compile(r"\b[a-z]+(?:-[a-z]+)+\b")
_CAMEL_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")
_VALUE_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?(?:px|rem|em|%|vh|vw|fr)(?![A-Za-z0-9])")
_PROP_WORDS = frozenset("""
width height display padding margin color background position top bottom left right flex grid
gap border opacity overflow float order
""".split())


def instruction_tokens(instruction: str) -> List[str]:
    """The concrete code-level things an instruction names.

    CSS selectors (``.topbar``, ``#root``), hyphenated CSS-ish words
    (``max-width``, ``align-items``, ``flex-end``), quoted strings, Camel/camel
    identifiers (``TopBar``, ``onSend``) and sized values (``100%``,
    ``16px``). These are what the resulting diff must touch for the change
    to count as applied -- a rule that is checkable without a model.
    """
    text = instruction or ""
    out: List[str] = []

    def _add(tok: str) -> None:
        tok = tok.strip()
        if tok and tok not in out:
            out.append(tok)

    for m in _SELECTOR_TOKEN_RE.finditer(text):
        _add(m.group(0))
    for q in _QUOTED_RE.findall(text):
        if 1 < len(q.strip()) <= 40:
            _add(q.strip())
    selectors = {t[1:] for t in out if t.startswith((".", "#"))}
    for m in _HYPHEN_TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok in selectors:
            continue  # already covered by its ``.class`` / ``#id`` form
        if tok.lower() not in ("full-width", "re-ask", "top-level", "so-called", "built-in", "re-run"):
            _add(tok)
    for m in _CAMEL_TOKEN_RE.finditer(text):
        _add(m.group(0))
    for m in _VALUE_TOKEN_RE.finditer(text):
        _add(m.group(0))
    return out


def diff_covers(diff: str, tokens: Sequence[str]) -> List[str]:
    """The ``tokens`` that appear in the diff's changed (``+``/``-``) lines."""
    changed = [ln[1:] for ln in (diff or "").splitlines()
               if (ln.startswith("+") or ln.startswith("-"))
               and not ln.startswith("+++") and not ln.startswith("---")]
    blob = "\n".join(changed).lower()
    return [t for t in tokens if t.lower() in blob]


def css_rules(src: str) -> List[Dict[str, Any]]:
    """Flat ``[{selector, body, line, media}]`` for a stylesheet (nested @media unrolled)."""
    out: List[Dict[str, Any]] = []
    text = src or ""
    # strip comments but keep line numbers
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    i = 0
    n = len(text)
    stack: List[str] = []  # enclosing @-rule prelude(s)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            break
        prelude = text[i:j].strip()
        # prelude may contain a trailing '}' from the previous block
        prelude = prelude.rsplit("}", 1)[-1].strip() if "}" in prelude else prelude
        line = text.count("\n", 0, j) + 1
        if prelude.startswith("@") and not prelude.startswith(("@font-face", "@page")):
            stack.append(prelude)
            i = j + 1
            continue
        # find the matching close brace (rule bodies have no nested braces)
        k = text.find("}", j + 1)
        if k == -1:
            break
        body = " ".join(text[j + 1:k].split())
        if prelude:
            out.append({"selector": " ".join(prelude.split()), "body": body, "line": line,
                        "media": " ".join(stack)})
        i = k + 1
        # closing braces of enclosing @-rules
        while stack:
            m = re.match(r"\s*}", text[i:])
            if not m:
                break
            stack.pop()
            i += m.end()
    return out


def component_classes(src: str) -> List[str]:
    """CSS class names a component renders (``className="a b"`` / ``class="a"``)."""
    out: List[str] = []
    for m in re.finditer(r"class(?:Name)?\s*=\s*[\"'`]([^\"'`]+)[\"'`]", src or ""):
        for c in m.group(1).split():
            if re.match(r"^[A-Za-z_][\w-]*$", c) and c not in out:
                out.append(c)
    # template literals / cn(...) with quoted class fragments
    for m in re.finditer(r"class(?:Name)?\s*=\s*\{[^}]*?[\"'`]([^\"'`{}$]+)[\"'`]", src or ""):
        for c in m.group(1).split():
            if re.match(r"^[A-Za-z_][\w-]*$", c) and c not in out:
                out.append(c)
    return out


def rules_for_classes(css_src: str, classes: Sequence[str], *, limit: int = 12) -> List[Dict[str, Any]]:
    """Rules whose selector targets any of ``classes`` (``.topbar``, ``.topbar .nav`` ...)."""
    wanted = {c for c in classes if c}
    if not wanted:
        return []
    out: List[Dict[str, Any]] = []
    for rule in css_rules(css_src):
        sel = rule["selector"]
        if any(re.search(r"\." + re.escape(c) + r"(?![\w-])", sel) for c in wanted):
            out.append(rule)
            if len(out) >= limit:
                break
    return out


#: Selectors that constrain EVERY component's layout from above: a ``body``
#: that centres its flex children, an ``#root`` with a max-width. Shown for any
#: layout feedback -- the constraint that narrows a header is often here.
_PAGE_SELECTOR_RE = re.compile(r"^(html|body|#root|#app|\.root|\.app|#__next|html,\s*body)(\s*,\s*(html|body))?$", re.I)
_LAYOUT_RE = re.compile(
    r"\b(width|height|wide|narrow|full|stretch|span|edge|cent(er|re)(ed)?|align(ed|ment)?|"
    r"layout|position|overflow|row|column|side|left|right|top|bottom|gap|spacing|padding|margin|"
    r"viewport|page|screen|fit|overlap|cut ?off|squeezed|cramped|responsive|mobile)\b", re.I)


def is_layout_feedback(feedback: str) -> bool:
    """Does the feedback concern size/position/layout (so page-level rules matter)?"""
    return bool(_LAYOUT_RE.search(feedback or ""))


def page_level_rules(css_src: str, *, limit: int = 6) -> List[Dict[str, Any]]:
    """Rules on ``html`` / ``body`` / ``#root``-style selectors (ancestors of everything)."""
    out: List[Dict[str, Any]] = []
    for rule in css_rules(css_src):
        if _PAGE_SELECTOR_RE.match(rule["selector"].strip()):
            out.append(rule)
            if len(out) >= limit:
                break
    return out


def relevant_css_blocks(src: str, feedback: str, extra_classes: Sequence[str] = (),
                        *, limit: int = 10) -> str:
    """Whole CSS rules that bear on the feedback, as ``line: selector { body }``.

    A stylesheet line that mentions a keyword is useless without its selector:
    "max-width: 820px" only means something as part of ``.app { ... }``. So the
    localizer shows complete rules whose selector or body mentions a feedback
    keyword or one of the classes the implicated components render -- that is
    how the model learns that a header is narrow because its *parent* has a
    max-width, not the header itself. For layout feedback the page-level rules
    (``html``, ``body``, ``#root``) are always included: a ``body`` that centres
    its flex children shrinks every top-level block to its content width.
    """
    kws = [k for k in _keywords(feedback) if len(k) >= 3]
    classes = [c.lower() for c in extra_classes]
    layout = is_layout_feedback(feedback)
    lines: List[str] = []
    for rule in css_rules(src):
        sel_low = rule["selector"].lower()
        body_low = rule["body"].lower()
        hit = any(k in sel_low or k in body_low for k in kws) or any(
            re.search(r"\." + re.escape(c) + r"(?![\w-])", sel_low) for c in classes)
        if not hit and layout and _PAGE_SELECTOR_RE.match(rule["selector"].strip()):
            hit = True
        if not hit:
            continue
        media = f" (in {rule['media']})" if rule["media"] else ""
        lines.append(f"{rule['line']:5d}: {rule['selector']} {{ {rule['body'][:220]} }}{media}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


_MOUNT_ID_RE = re.compile(r"""getElementById\(\s*['"]([\w-]+)['"]\s*\)|querySelector\(\s*['"]#([\w-]+)['"]\s*\)""")
_HTML_ID_RE = re.compile(r"""<(div|main|section)\b[^>]*\bid=["']([\w-]+)["']""", re.I)


def mount_selector(contents: Dict[str, str]) -> str:
    """The id selector the SPA mounts into (``#root``), read from the entry or index.html."""
    for path, src in contents.items():
        if path.lower().endswith((".jsx", ".tsx", ".js", ".ts")):
            m = _MOUNT_ID_RE.search(src or "")
            if m:
                return "#" + (m.group(1) or m.group(2))
    for path, src in contents.items():
        if path.lower().endswith(".html"):
            m = _HTML_ID_RE.search(src or "")
            if m:
                return "#" + m.group(2)
    return "#root"


def ancestor_chain(component: str, contents: Dict[str, str], paths: Sequence[str],
                   *, importers_of: Any) -> List[Dict[str, str]]:
    """The DOM ancestry of a component, outermost first: ``html > body > #root > ...``.

    Each ring is ``{"selector": ..., "source": ...}``. The page ancestors come
    from the document (``html``, ``body``, the mount id from the entry); then,
    walking the import graph outwards from the component, the wrapper element
    classes of every component that renders it (its importer, its importer's
    importer ...), read from the outermost ``className`` in each file. A
    layout constraint on any ring narrows everything inside it -- this is the
    chain the localizer shows so the model can name the ring that actually
    constrains the element instead of the element itself.
    """
    chain: List[Dict[str, str]] = []
    seen = {component}
    current = component
    rings: List[Tuple[str, str]] = []
    for _ in range(6):
        importers = [i for i in importers_of(current, contents, paths) if i not in seen]
        if not importers:
            break
        parent = importers[0]
        seen.add(parent)
        classes = component_classes(contents.get(parent) or "")
        if classes:  # a wrapper that renders no element (the entry file) adds no CSS ring
            rings.append((f".{classes[0]}", parent))
        current = parent
    chain.append({"selector": "html", "source": "document"})
    chain.append({"selector": "body", "source": "document"})
    chain.append({"selector": mount_selector(contents), "source": "mount point (entry / index.html)"})
    for wrapper, source in reversed(rings):
        chain.append({"selector": wrapper, "source": source})
    own = component_classes(contents.get(component) or "")
    chain.append({"selector": f".{own[0]}" if own else component.rsplit("/", 1)[-1], "source": component})
    return chain


def rules_for_selector(css_src: str, selector: str, *, limit: int = 4) -> List[Dict[str, Any]]:
    """Rules whose selector list targets exactly ``selector`` (``body``, ``#root``, ``.app``)."""
    sel = selector.strip()
    out: List[Dict[str, Any]] = []
    for rule in css_rules(css_src):
        parts = [p.strip() for p in rule["selector"].split(",")]
        if sel in parts or any(re.match(re.escape(sel) + r"(?![\w-])", p) and p == sel for p in parts):
            out.append(rule)
            if len(out) >= limit:
                break
    return out


_WIDTH_COMPLAINT_RE = re.compile(
    r"\b(full[- ]?width|span|stretch|edge to edge|wider|narrow|too small|not wide|whole (page|width|viewport|screen)|"
    r"across the (page|screen|viewport)|fill the|fills? the|100%)\b", re.I)


def _declarations(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for decl in (body or "").split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out


def diagnose_width_constraints(chain: Sequence[Dict[str, Any]], feedback: str) -> List[Dict[str, str]]:
    """Deterministic CSS diagnosis for "X should span the full width" complaints.

    ``chain`` is the DOM ancestry (outermost first) where each ring carries
    ``selector`` and ``rules`` (the merged declarations of every rule that
    targets it). Walks outward-in and names the ring whose declarations make
    everything inside it content-sized or capped. Each finding is
    ``{"ring", "problem", "fix"}`` where ``fix`` is a CSS snippet ready to be
    written into the app stylesheet:

    * a ``display:flex`` ring in row direction with ``justify-content``
      centring -> its children shrink to their content width;
    * a column flex ring with ``align-items`` other than stretch -> children
      shrink-wrap;
    * ``max-width`` / ``width: fit-content|max-content|min-content`` /
      ``display: inline(-block|-flex)``;
    * a ring with NO rule at all under a centring parent (the classic unstyled
      ``#root``).

    A weak model told "make the header full width" otherwise edits the header
    (or a ring that is already fine); the diagnosis names the ring and the
    exact declaration, and the executor can apply it without the model.
    """
    if not _WIDTH_COMPLAINT_RE.search(feedback or ""):
        return []
    out: List[Dict[str, str]] = []
    parent_centres = False
    for ring in chain:
        sel = ring.get("selector") or "?"
        decls = ring.get("rules") or {}
        display = decls.get("display", "")
        direction = decls.get("flex-direction", "row")
        if not decls:
            if parent_centres:
                out.append({"ring": sel,
                            "problem": (f"{sel} has NO width rule and its parent centres its children, so "
                                        f"{sel} is content-sized -> nothing inside can be wider than its content"),
                            "fix": f"{sel} {{ width: 100%; }}"})
            continue
        if "max-width" in decls and decls["max-width"] not in ("none", "100%", "100vw"):
            out.append({"ring": sel,
                        "problem": f"{sel} caps its width with max-width: {decls['max-width']} -> nothing inside can be wider",
                        "fix": f"{sel} {{ max-width: none; }}"})
        width = decls.get("width", "")
        if width in ("fit-content", "max-content", "min-content") or display.startswith("inline"):
            out.append({"ring": sel,
                        "problem": f"{sel} is content-sized (width: {width or display}) -> children cannot be wider",
                        "fix": f"{sel} {{ width: 100%; display: block; }}"})
        centres_here = False
        if display in ("flex", "inline-flex"):
            if direction.startswith("row") and decls.get("justify-content", "") in ("center", "space-around", "space-evenly"):
                centres_here = True
                out.append({"ring": sel,
                            "problem": (f"{sel} is a ROW flex container with justify-content: {decls['justify-content']} "
                                        f"-> its children shrink to their content width"),
                            "fix": f"{sel} {{ display: block; }}"})
            if direction.startswith("column") and decls.get("align-items", "") in ("center", "flex-start", "flex-end", "start", "end"):
                centres_here = True
                out.append({"ring": sel,
                            "problem": (f"{sel} is a COLUMN flex container with align-items: {decls['align-items']} "
                                        f"-> its children shrink to their content width"),
                            "fix": f"{sel} {{ align-items: stretch; }}"})
        if display == "grid" and decls.get("justify-items", "") in ("center", "start", "end"):
            centres_here = True
            out.append({"ring": sel,
                        "problem": f"{sel} is a grid with justify-items: {decls['justify-items']} -> children shrink-wrap",
                        "fix": f"{sel} {{ justify-items: stretch; }}"})
        parent_centres = centres_here or (parent_centres and width in ("", "auto") and "max-width" not in decls)
    return out


_SET_DECL_RE = re.compile(
    r"(?:in|on|for|to)\s+`?([.#]?[A-Za-z_][\w-]*)`?\s*(?:,|:)?\s*(?:set|add|make|change|use|give|apply)\s+`?([a-z-]+)`?\s*(?::|to|=)\s*`?([^`;,.]+?)`?(?:\s*;|\s*$|\s+so\b|\s+to\b|\s*,|\s*\.|\s+\()",
    re.I)
_DECL_FIRST_RE = re.compile(
    r"(?:set|add|make|change|use)\s+`?([a-z-]+)`?\s*(?::|to|=)\s*`?([^`;,.]+?)`?\s+(?:on|in|to|for)\s+(?:the\s+)?`?([.#]?[A-Za-z_][\w-]*)`?",
    re.I)


def instruction_is_noop(instruction: str, stylesheets: Dict[str, str]) -> Optional[str]:
    """Why a CSS instruction would change nothing, or ``None``.

    Recognises "in .app set align-items: stretch" / "add width: 100% to #root"
    phrasing and checks the selector's merged base declarations across the
    stylesheets: when the property already holds that value the instruction is
    a no-op -- the model has mis-located the constraint, and asking a weak
    model to "apply" it only makes it echo the file back.
    """
    text = instruction or ""
    m = _SET_DECL_RE.search(text)
    if m:
        sel, prop, value = m.group(1), m.group(2), m.group(3)
    else:
        m = _DECL_FIRST_RE.search(text)
        if not m:
            return None
        prop, value, sel = m.group(1), m.group(2), m.group(3)
    sel = sel.strip()
    prop = prop.strip().lower()
    value = " ".join(value.strip().lower().rstrip(";").split())
    merged: Dict[str, str] = {}
    for src in stylesheets.values():
        for rule in rules_for_selector(src, sel):
            if not rule.get("media"):
                merged.update(_declarations(rule["body"]))
    if not merged:
        return None
    have = merged.get(prop)
    if have is not None and have.rstrip(";") == value:
        return f"{sel} already has {prop}: {value} -- the instruction changes nothing"
    return None


def chain_with_rules(chain: Sequence[Dict[str, str]], stylesheets: Dict[str, str]) -> List[Dict[str, Any]]:
    """Attach the merged declarations of every rule targeting each ring (later files win)."""
    out: List[Dict[str, Any]] = []
    for ring in chain:
        merged: Dict[str, str] = {}
        for _path, src in stylesheets.items():
            for rule in rules_for_selector(src, ring["selector"]):
                if rule.get("media"):
                    continue  # base rules only; media overrides are conditional
                merged.update(_declarations(rule["body"]))
        out.append({**ring, "rules": merged})
    return out


__all__ += [  # noqa: F405 - extend the public surface defined above
    "chain_with_rules", "diagnose_width_constraints", "instruction_is_noop",
    "ancestor_chain", "component_classes", "css_rules", "diff_covers", "instruction_tokens",
    "is_layout_feedback", "mount_selector", "page_level_rules", "relevant_css_blocks",
    "rules_for_classes", "rules_for_selector", "split_feedback",
]
