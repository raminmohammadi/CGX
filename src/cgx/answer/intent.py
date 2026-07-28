

"""
Intent detection module for developer-style queries.

This module classifies a user question into one of several
retrieval/answering "modes". These modes influence how
`answer_with_llm` in engine.py constructs prompts and
which metadata or graph traversal logic to emphasize.

Supported modes
---------------
- "overview"        : High-level summary of the repo.
- "qa"              : General/conceptual question grounded in SOURCES, without
                      a forced section template (default for natural-language
                      questions that aren't symbol-, howto-, or change-shaped).
- "change_plan"     : Requests about adding/refactoring/extending.
- "howto"           : "How to" usage or workflow questions.
- "symbol_explain"  : Explain a specific function/class in depth.
- "symbol_location" : Identify file(s)/chunk(s) containing a symbol.
- "line_number"     : Identify line spans where to edit.
- "callers_list"    : List all functions/classes that call a target symbol.
- "callees_list"    : List all functions/classes that a target symbol calls.
- "enumerate"       : Count/list API endpoints or routes deterministically.
"""

import re
from typing import Literal

Intent = Literal[
    "overview",
    "qa",
    "change_plan",
    "howto",
    "symbol_explain",
    "symbol_location",
    "line_number",
    "callers_list",
    "callees_list",
    "enumerate",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_IDENT_RE = re.compile(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]")
_DOTTED_REF_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]+")
_MIXED_CASE_RE = re.compile(r"[a-z][A-Z]")


def _has_symbol_token(q: str) -> bool:
    r"""Return True if ``q`` contains a *structurally* code-like identifier.

    The check is intentionally strict: plain English words are rejected so
    conceptual questions ("how does the world model encode images?") are not
    misclassified as ``symbol_explain``. A token counts as a symbol when it
    is quoted (``\`foo\``` / ``"foo"``), uses ``snake_case``, contains a
    lower-to-upper transition (``CamelCase`` / ``camelCase`` -- note that a
    sentence-initial capital like ``How`` is intentionally **not** a match
    because it has no internal lower\u2192upper boundary), is a dotted
    reference (``mod.func``), or is a short all-uppercase acronym (``VAE``,
    ``RNN``) of 2-6 characters.
    """
    if _QUOTED_IDENT_RE.search(q):
        return True
    if _DOTTED_REF_RE.search(q):
        return True
    for tok in _IDENTIFIER_RE.findall(q):
        if "_" in tok:
            return True
        if _MIXED_CASE_RE.search(tok):
            return True
        if 2 <= len(tok) <= 6 and tok.isupper():
            return True
    return False


def detect_intent(question: str) -> Intent:
    """
    Detect the intent of a developer's natural language question.

    Parameters
    ----------
    question : str
        User's natural language or code-related query.

    Returns
    -------
    Intent
        One of the supported modes. Empty input returns ``"overview"``; any
        natural-language question that doesn't match a more specific rule
        falls through to ``"qa"`` so the engine answers it directly instead
        of forcing a templated repo overview.

    Notes
    -----
    The detection is rule-based (keyword + regex).
    Rules are ordered most-specific first; broad keywords like "change" or
    "add" only route to `change_plan` when no clear symbol-targeted phrasing
    is present.
    """
    q = (question or "").strip()
    if not q:
        return "overview"
    ql = q.lower()
    has_sym = _has_symbol_token(q)

    # High-level repo summaries (most specific phrases first)
    if any(k in ql for k in ["repo overview", "what does this repo", "high level overview", "high-level overview", "summary of the repo"]):
        return "overview"

    # Deterministic endpoint enumeration: counting/listing API routes.
    # Requires BOTH an enumeration cue (how many / list / count / ...) AND an
    # api/endpoint/route keyword so ordinary questions ("how does the API
    # work?") are not hijacked into the enumeration path. Placed high so a
    # phrasing like "list all endpoints" wins before broader branches.
    if (
        any(k in ql for k in [
            "how many", "how much", "number of", "count of", "count the",
            "list all", "list the", "list every", "list of", "enumerate",
            "what are the", "which ", "all the", "show me all", "show all",
        ])
        and re.search(r"\b(api|apis|endpoint|endpoints|route|routes)\b", ql)
    ):
        return "enumerate"

    # Callers / callees via graph (require an explicit verb AND a symbol)
    if has_sym and any(k in ql for k in ["who calls", "functions that call", "callers of", "what calls", "invokes ", "invoked by"]):
        return "callers_list"
    if has_sym and any(k in ql for k in ["functions called by", "callees of", "calls to ", "what does this function call", "what functions does"]):
        return "callees_list"

    # Symbol location (explicit "where" phrasing)
    if any(k in ql for k in ["where is", "location of", "which file contains", "which file has", "find the file", "in which file"]):
        return "symbol_location"

    # Line number queries
    if any(k in ql for k in ["which line", "line number", "line should i change", "what line"]):
        return "line_number"

    # Overview-shaped phrasings about a project/repo/codebase as a whole.
    # Must come BEFORE the symbol_explain branch so questions like
    # "what is the CGX project about?" -- which contain both the
    # "what is the " trigger AND a short ALLCAPS token -- still route
    # to the high-level summary path rather than being treated as a
    # request to explain a specific symbol.
    if any(k in ql for k in [
        "project about", "repo about", "codebase about",
        "about this project", "about the project", "about this repo",
        "about the repo", "about this codebase", "about the codebase",
        "what is this project", "what is the project", "what is this repo",
        "what is the repo", "what is this codebase", "what is the codebase",
        "what's this project", "what's the project", "what's this repo",
        "tell me about this", "tell me about the project",
        "tell me about the repo", "tell me about the codebase",
        "summarize the project", "summarize this project", "summarize the repo",
        "project overview", "codebase overview", "repo summary",
    ]):
        return "overview"

    # Also catch phrasings like "tell me about the CGX project" or
    # "summarize the Foo codebase" where a project token sits between the
    # verb phrase and the noun. Treat any "(tell me about|what is|summarize)
    # ... (project|repo|codebase)" shape as an overview request -- the
    # in-between token is the project's name, not a symbol to dissect.
    if re.search(
        r"\b(?:tell\s+me\s+about|what\s+is|what's|summari[sz]e)\b[^?.!]{0,40}\b(?:project|repo|codebase)\b",
        ql,
    ):
        return "overview"

    # Symbol explanation: explicit verbs + a symbol token
    if has_sym and any(k in ql for k in ["what does", "explain", "describe", "purpose of", "what is the ", "how does"]):
        return "symbol_explain"

    # Optimization / recommendation requests: the caller wants actionable
    # advice -- which parameters to pick, what to tune, how to make something
    # faster or lighter -- not a description of an existing symbol. This must
    # beat the symbol_explain fallback below: an anchored goal like "best
    # FAISS parameters for build_faiss_index" carries a symbol token but no
    # explain-verb, so without this branch it fell through to symbol_explain
    # and produced a Purpose/Signature write-up instead of recommendations.
    # ``change_plan`` framing (Goal / Affected files / Step-by-step edits /
    # Tests / Risks) is the closest fit for a grounded recommendation.
    if any(k in ql for k in [
        "optimize", "optimise", "optimization", "optimisation",
        "optimal", "recommend", "improve", "speed up", "faster",
        "reduce memory", "tune ", "tuning", "best ",
        "which parameters", "what parameters",
    ]):
        return "change_plan"

    # Usage / workflow questions (no concrete symbol target)
    if any(k in ql for k in ["how do i", "how to ", "where to ", "how can i"]):
        return "howto"

    # Code modification requests (broad; only after symbol-targeted branches)
    if any(k in ql for k in ["add ", "implement", "feature", "refactor", "plan ", "change ", "extend ", "modify", "introduce", "create a "]):
        return "change_plan"

    # Bare "what is X" / "what is X about" / "tell me about X" where X is a
    # short ALLCAPS acronym is almost always asking about the project as a
    # whole rather than a specific symbol.
    m = re.match(
        r"^\s*(?:what\s+is|what's|tell\s+me\s+about)\s+([A-Za-z][A-Za-z0-9_]*)"
        r"(?:\s+(?:about|project|repo|codebase))?\s*\??\s*$",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        tok = m.group(1)
        if tok.isupper() and 2 <= len(tok) <= 6:
            return "overview"

    # Fallback: prefer symbol_explain if a symbol is present, else qa.
    # ``qa`` (not ``overview``) is the default so conceptual questions like
    # "what indexing is being used?" get a focused answer instead of a
    # templated Purpose/Components/Entry-Points sandwich.
    if has_sym:
        return "symbol_explain"
    return "qa"
