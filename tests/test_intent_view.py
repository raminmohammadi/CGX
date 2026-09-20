"""Deterministic assertions for the natural-language intent view.

The intent view is the primary semantic-search input. These tests pin the
properties the Phase-1 rewrite guarantees without needing an embedding model:
class context is never dropped for methods, the absolute path and the old
``key: value`` scaffold never leak into the embedded text, and HTTP routes /
docstrings surface as matchable prose.
"""

from __future__ import annotations

from cgx.embeddings.views import build_intent_view


def _method_chunk():
    return {
        "type": "function",  # Python methods are emitted as 'function' + ::method:: id
        "id": "/home/u/repo/store/repositories.py::method::UserRepository.save",
        "name": "save",
        "file": "/home/u/repo/store/repositories.py",
        "module_path": "store.repositories",
        "code": "def save(self, user):\n    \"\"\"Insert or update a user row.\"\"\"\n    return True",
        "meta": {
            "class_name": "UserRepository",
            "signature": "(self, user)",
            "docstring": "Insert or update a user account row in the accounts table.",
        },
    }


def test_method_intent_view_keeps_class_and_drops_path_and_scaffold():
    view = build_intent_view(_method_chunk())
    # Class context is present (qualified name), so method retrieval can
    # disambiguate homonyms across classes.
    assert "UserRepository.save" in view
    # Clean dotted location, never the absolute path or machine dirs.
    assert "store.repositories" in view
    assert "/home/u/repo" not in view
    # None of the old rigid scaffold labels leak into the vector text.
    for scaffold in ("type:", "symbol:", "file:", "class:", "called_by_count:", "metrics:"):
        assert scaffold not in view
    # The full docstring is embedded (not just discarded).
    assert "accounts table" in view


def test_route_function_surfaces_http_endpoint_prose():
    chunk = {
        "type": "function",
        "id": "/x/api/routes.py::function::refund_invoice",
        "name": "refund_invoice",
        "file": "/x/api/routes.py",
        "module_path": "api.routes",
        "code": "def refund_invoice(invoice_id):\n    return {}",
        "meta": {
            "signature": "(invoice_id)",
            "docstring": "Issue a refund against a paid invoice.",
            "route": {"methods": ["POST"], "path": "/invoices/<invoice_id>/refund"},
        },
    }
    view = build_intent_view(chunk)
    assert "HTTP endpoint" in view
    assert "POST" in view and "/invoices/<invoice_id>/refund" in view


def test_undocumented_symbol_is_nonempty_and_scaffold_free():
    chunk = {
        "type": "function",
        "id": "/x/net/client.py::function::_request",
        "name": "_request",
        "file": "/x/net/client.py",
        "module_path": "net.client",
        "code": "def _request(url):\n    return 200",
        "meta": {"signature": "(url)"},
    }
    view = build_intent_view(chunk).strip()
    # No docstring, yet the card is a meaningful NL line (name + sig + location),
    # never an all-empty scaffold like the old "summary: \nclass: \n...".
    assert view
    assert "_request" in view and "net.client" in view
    assert "summary:" not in view
