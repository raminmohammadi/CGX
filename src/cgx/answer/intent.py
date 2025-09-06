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
    For production, could be swapped with a classifier model.
    """
    q = (question or "").lower()

    # High-level repo summaries
    if any(k in q for k in ["overview", "what does this repo", "summary", "high level"]):
        return "overview"

    # Code modification requests
    if any(k in q for k in ["add", "implement", "feature", "refactor", "plan", "change", "extend"]):
        return "change_plan"

    # Usage / workflow questions
    if any(k in q for k in ["how do i", "how to", "where to"]):
        return "howto"

    # Symbol locations
    if any(k in q for k in ["where is", "location of", "which file contains"]):
        return "symbol_location"

    # Line number queries
    if any(k in q for k in ["which line", "line number", "line should i change"]):
        return "line_number"

    # Callers / callees via graph
    if any(k in q for k in ["who calls", "functions that call", "callers of", "invokes"]):
        return "callers_list"
    if any(k in q for k in ["what does", "functions called by", "callees of", "calls to"]):
        return "callees_list"

    # Symbol explanation
    if re.search(r"\b(what does|explain|describe)\b.*\b([A-Za-z_][A-Za-z0-9_]*)\b", q):
        return "symbol_explain"

    return "overview"
