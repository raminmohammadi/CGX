"""Intent detection tests."""

from cgx.answer.intent import detect_intent
from cgx.answer.scope import (
    apply_scope_penalty,
    detect_scope,
    resolve_scope_for_intent,
)


def test_overview_default():
    # Empty input keeps the high-level summary default.
    assert detect_intent("") == "overview"
    assert detect_intent("   ") == "overview"
    # Explicit overview phrasings still route to overview.
    assert detect_intent("repo overview please") == "overview"


# Conceptual / factual questions without a symbol token used to fall back to
# "overview", which forced the engine to produce a templated Purpose / Major
# Components / Entry-Points sandwich regardless of what was asked. The new
# default is "qa" so the engine answers the actual question directly.
def test_qa_is_default_for_conceptual_questions():
    assert detect_intent("what specific indexing is being used?") == "qa"
    assert detect_intent("what database does this use?") == "qa"
    assert detect_intent("what models can it run with?") == "qa"
    assert detect_intent("how is data flowing through the pipeline?") == "qa"
    assert detect_intent("what embeddings are produced?") == "qa"


def test_qa_does_not_override_explicit_overview_phrasings():
    # Explicit project-wide phrasing must still beat the qa fallback.
    assert detect_intent("what is this project?") == "overview"
    assert detect_intent("summarize the codebase") == "overview"
    assert detect_intent("repo overview") == "overview"
    assert detect_intent("tell me about this project") == "overview"


def test_symbol_explain_routes_correctly():
    assert detect_intent("what does parse_codebase do?") == "symbol_explain"
    assert detect_intent("explain `HybridRetriever`") == "symbol_explain"


def test_callers_callees():
    assert detect_intent("who calls parse_codebase") == "callers_list"
    assert detect_intent("functions called by HybridRetriever") == "callees_list"


def test_change_plan_requires_keywords():
    assert detect_intent("add CSV export") == "change_plan"
    assert detect_intent("refactor the indexer") == "change_plan"


def test_howto_without_symbol():
    assert detect_intent("how do i run tests?") == "howto"


def test_symbol_location():
    assert detect_intent("where is parse_codebase defined?") == "symbol_location"


# Regression: plain English words must not be treated as symbols so that
# conceptual questions don't get routed to the symbol-explain code path.
def test_conceptual_how_does_does_not_route_to_symbol_explain():
    assert detect_intent("how does world model encode images?") != "symbol_explain"
    assert detect_intent("how does the encoder work conceptually?") != "symbol_explain"


def test_short_acronym_still_classified_as_symbol():
    # Acronyms like VAE/RNN/MLP are real class names in many code bases and
    # should keep their symbol_explain routing.
    assert detect_intent("explain VAE") == "symbol_explain"
    assert detect_intent("what does RNN do?") == "symbol_explain"


def test_dotted_reference_classified_as_symbol():
    assert detect_intent("explain module.func") == "symbol_explain"


# Regression: a short ALLCAPS token that is the project's own name (e.g. "CGX")
# must not force overview-shaped questions into the symbol_explain branch.
# Bare "what is <lowercase>?" no longer auto-routes to overview -- it falls
# through to ``qa`` because the token may equally be a function/module name
# that the engine should answer about based on retrieval.
def test_project_overview_phrasings_route_to_overview():
    assert detect_intent("What is CGX project about?") == "overview"
    assert detect_intent("what is CGX?") == "overview"
    assert detect_intent("tell me about CGX") == "overview"
    assert detect_intent("tell me about this project") == "overview"
    assert detect_intent("project overview please") == "overview"


# Regression for the platform "What is the CGX project about?" query:
# the question contains BOTH the symbol_explain trigger ("what is the ")
# AND a short ALLCAPS token ("CGX") AND the overview phrase "project about".
# The classifier used to fire symbol_explain first, sending the LLM down the
# heavier per-symbol prompt path. Overview phrasings must win.
def test_overview_phrasings_outrank_symbol_explain_trigger():
    assert detect_intent("What is the CGX project about?") == "overview"
    assert detect_intent("what is this CGX project about?") == "overview"
    assert detect_intent("Tell me about the CGX project") == "overview"
    assert detect_intent("summarize this project") == "overview"


# --- scope detection -------------------------------------------------------

def test_detect_scope_defaults_to_src():
    assert detect_scope("") == "src"
    assert detect_scope("what is the structure of this repo?") == "src"
    assert detect_scope("explain HybridRetriever") == "src"
    assert detect_scope("what specific indexing is being used?") == "src"


def test_detect_scope_picks_tests_for_test_questions():
    assert detect_scope("how do the parser tests work?") == "tests"
    assert detect_scope("explain the pytest fixtures") == "tests"
    assert detect_scope("what does the test suite cover?") == "tests"
    assert detect_scope("how is HybridRetriever tested?") == "tests"
    assert detect_scope("show me the conftest setup") == "tests"


def test_detect_scope_any_for_explicit_inclusive_phrasing():
    assert detect_scope("explain the parser including tests") == "any"
    assert detect_scope("everything related to indexing") == "any"
    assert detect_scope("across the codebase, where is BM25 used?") == "any"


def test_resolve_scope_change_plan_always_any():
    assert resolve_scope_for_intent("add CSV export", "change_plan") == "any"
    assert resolve_scope_for_intent("refactor parser tests", "change_plan") == "any"


def test_resolve_scope_defers_to_question_otherwise():
    assert resolve_scope_for_intent("explain HybridRetriever", "symbol_explain") == "src"
    assert resolve_scope_for_intent("how is it tested?", "howto") == "tests"


# --- scope penalty ---------------------------------------------------------

def _hit(path, score):
    return {"chunk_id": f"{path}::func::x", "score": float(score)}


def test_apply_scope_penalty_src_demotes_test_paths():
    hits = [
        _hit("/repo/tests/test_x.py", 1.0),
        _hit("/repo/src/cgx/parser/parse_codebase.py", 0.8),
        _hit("/repo/tests/test_y.py", 0.9),
        _hit("/repo/src/cgx/answer/engine.py", 0.5),
    ]
    out = apply_scope_penalty(hits, "src", penalty=0.3)
    # source files should now top the list; tests should be demoted.
    paths_in_order = [h["chunk_id"].split("::")[0] for h in out]
    assert paths_in_order[0].endswith("parse_codebase.py")
    assert paths_in_order[1].endswith("engine.py")
    # Demoted hits retain a marker.
    demoted = [h for h in out if h.get("scope_demoted")]
    assert len(demoted) == 2
    assert all("/tests/" in h["chunk_id"] for h in demoted)


def test_apply_scope_penalty_tests_demotes_source_paths():
    hits = [
        _hit("/repo/src/cgx/parser/parse_codebase.py", 1.0),
        _hit("/repo/tests/test_x.py", 0.5),
    ]
    out = apply_scope_penalty(hits, "tests", penalty=0.3)
    assert out[0]["chunk_id"].split("::")[0].endswith("test_x.py")
    assert out[1].get("scope_demoted") is True


def test_apply_scope_penalty_any_is_noop():
    hits = [
        _hit("/repo/tests/test_x.py", 1.0),
        _hit("/repo/src/cgx/answer/engine.py", 0.5),
    ]
    out = apply_scope_penalty(hits, "any")
    assert [h["score"] for h in out] == [1.0, 0.5]
    assert not any(h.get("scope_demoted") for h in out)


def test_apply_scope_penalty_handles_root_test_filename_without_tests_dir():
    # File named test_foo.py living outside a tests/ directory.
    hits = [
        _hit("/repo/pkg/test_foo.py", 1.0),
        _hit("/repo/pkg/foo.py", 0.6),
    ]
    out = apply_scope_penalty(hits, "src", penalty=0.3)
    assert out[0]["chunk_id"].split("::")[0].endswith("foo.py")
    assert out[1].get("scope_demoted") is True
