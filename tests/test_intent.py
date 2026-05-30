"""Intent detection tests."""

from cgx.answer.intent import detect_intent


def test_overview_default():
    assert detect_intent("") == "overview"
    assert detect_intent("repo overview please") == "overview"


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
