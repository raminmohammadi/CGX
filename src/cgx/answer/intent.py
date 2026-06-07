

"""
Intent detection module for developer-style queries.

This module classifies a user question into one of several
retrieval/answering "modes". These modes influence how
`answer_with_llm` in engine.py constructs prompts and
which metadata or graph traversal logic to emphasize.

Supported modes
---------------
- "overview"        : High-level summary of the repo.
- "change_plan"     : Requests about adding/refactoring/extending.
- "howto"           : "How to" usage or workflow questions.
- "symbol_explain"  : Explain a specific function/class in depth.
- "symbol_location" : Identify file(s)/chunk(s) containing a symbol.
- "line_number"     : Identify line spans where to edit.
- "callers_list"    : List all functions/classes that call a target symbol.
- "callees_list"    : List all functions/classes that a target symbol calls.
"""

import re
from typing import Literal

Intent = Literal[
    "overview",
    "change_plan",
    "howto",
    "symbol_explain",
    "symbol_location",
    "line_number",
    "callers_list",
    "callees_list",
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
        One of the supported modes (default: "overview").

    Notes
    -----
    The detection is rule-based (keyword + regex).
    Rules are ordered most-specific first; broad keywords like "change" or
    "add" only route to `change_plan` when no clear symbol-targeted phrasing
    is present.
    """
    q = (question or "").strip()
    ql = q.lower()
    has_sym = _has_symbol_token(q)

    # High-level repo summaries (most specific phrases first)
    if any(k in ql for k in ["repo overview", "what does this repo", "high level overview", "high-level overview", "summary of the repo"]):
        return "overview"

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

    # Symbol explanation: explicit verbs + a symbol token
    if has_sym and any(k in ql for k in ["what does", "explain", "describe", "purpose of", "what is the ", "how does"]):
        return "symbol_explain"

    # Usage / workflow questions (no concrete symbol target)
    if any(k in ql for k in ["how do i", "how to ", "where to ", "how can i"]):
        return "howto"

    # Code modification requests (broad; only after symbol-targeted branches)
    if any(k in ql for k in ["add ", "implement", "feature", "refactor", "plan ", "change ", "extend ", "modify", "introduce", "create a "]):
        return "change_plan"

    # Overview-shaped phrasings about a project/repo/codebase as a whole.
    # Must come before the fallback so short all-uppercase tokens (often the
    # project's own name, e.g. "CGX") don't force symbol_explain.
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

    # Fallback: prefer symbol_explain if a symbol is present, else overview
    if has_sym:
        return "symbol_explain"
    return "overview"
