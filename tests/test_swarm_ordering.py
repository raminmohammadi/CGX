"""Phase R: global manifest toposort + context-aware AST fallback helpers.

The old ordering bucketed by filename keyword *before* sorting, so a
``depends_on`` edge crossing pipeline ranks was silently lost. The AST fallback
helpers were context-starved (never saw sibling signatures) and produced an
empty file whenever the skeleton would not parse. These tests pin the fixes.
"""

from cgx.session.tasks.decompose import (
    toposort_manifest_files, _order_manifest_layers, _bucket_rank)
from cgx.session.tasks.ast_scaffold import (
    _symbol_name_from_signature, _symbols_from_skeleton,
    _symbols_from_contracts, _required_symbols, _public_signatures,
    _dependency_paths, _grounding_block)


# --------------------------- global toposort ---------------------------

def test_cross_bucket_back_edge_is_honoured():
    # A models-ranked file depends on an api-ranked file: the old per-bucket
    # sort inverted this; the global sort must place the dependency first.
    files = [
        {"path": "src/config.py", "depends_on": ["src/server.py"]},
        {"path": "src/server.py", "depends_on": []},
    ]
    order = [f["path"] for f in toposort_manifest_files(files)]
    assert order.index("src/server.py") < order.index("src/config.py")


def test_aligned_edge_orders_dependency_first():
    files = [
        {"path": "src/app.py", "depends_on": ["src/models.py"]},
        {"path": "src/models.py", "depends_on": []},
    ]
    order = [f["path"] for f in toposort_manifest_files(files)]
    assert order.index("src/models.py") < order.index("src/app.py")


def test_residual_cycle_keeps_every_node():
    files = [
        {"path": "a.py", "depends_on": ["b.py"]},
        {"path": "b.py", "depends_on": ["a.py"]},
        {"path": "c.py", "depends_on": []},
    ]
    order = [f["path"] for f in toposort_manifest_files(files)]
    assert set(order) == {"a.py", "b.py", "c.py"}


def test_independent_peers_keep_declared_order():
    files = [{"path": f"src/x{i}.py", "depends_on": []} for i in range(3)]
    order = [f["path"] for f in toposort_manifest_files(files)]
    assert order == ["src/x0.py", "src/x1.py", "src/x2.py"]


def test_bucket_rank_partitions_by_keyword():
    assert _bucket_rank("tests/test_x.py") == 4
    assert _bucket_rank("src/models.py") == 1
    assert _bucket_rank("src/main.py") == 3
    assert _bucket_rank("src/service.py") == 2


def test_order_manifest_layers_preserves_bucket_partition():
    # Dependency-free files in distinct keyword buckets stay in distinct
    # layers (the greenfield layer_count contract).
    layers = _order_manifest_layers([{"name": "x", "files": [
        {"path": "app.py"}, {"path": "README.md"}]}])
    names = [lay["name"] for lay in layers]
    assert len(layers) == 2
    assert names == ["core_logic_auth", "api_routes_main"]


# ----------------------- AST fallback helpers -----------------------

def test_symbol_name_from_signature_variants():
    assert _symbol_name_from_signature("evaluate(expr: str) -> float") == "evaluate"
    assert _symbol_name_from_signature("def compute(x)") == "compute"
    assert _symbol_name_from_signature("async def run() -> None") == "run"


def test_symbols_from_skeleton():
    sk = "def add(a, b):\n    ...\nclass Calc:\n    ...\nasync def go():\n    ..."
    assert _symbols_from_skeleton(sk) == [
        ("add", "function"), ("Calc", "class"), ("go", "async function")]
    assert _symbols_from_skeleton("def (bad") == []


CONTRACTS = {
    "functions": [
        {"signature": "evaluate(expr: str) -> float", "module": "src/calc.py"},
        {"name": "helper", "module": "src/other.py"},
    ],
    "schemas": [{"name": "Expr", "module": "src/calc.py"}],
}


def test_symbols_from_contracts_for_module():
    assert _symbols_from_contracts("src/calc.py", CONTRACTS) == [
        ("evaluate", "function"), ("Expr", "class")]


def test_required_symbols_prefers_skeleton_else_contracts():
    sk = "def add(a, b):\n    ..."
    assert _required_symbols(sk, "src/calc.py", CONTRACTS) == [("add", "function")]
    assert _required_symbols("def (broken", "src/calc.py", CONTRACTS) == [
        ("evaluate", "function"), ("Expr", "class")]


def test_public_signatures_renders_headers_and_docstrings():
    code = ('def add(a, b):\n    """Sum two numbers."""\n    return a + b\n'
            'class P:\n    """A point."""\n    pass')
    sig = _public_signatures(code)
    assert "def add(a, b)" in sig
    assert "# Sum two numbers." in sig
    assert "class P" in sig


def test_dependency_paths_reads_depends_on():
    plan = {"layers": [{"files": [
        {"path": "src/main.py", "depends_on": ["src/models.py", "src/util.py"]},
        {"path": "src/models.py"}]}]}
    assert _dependency_paths("src/main.py", plan) == ["src/models.py", "src/util.py"]
    assert _dependency_paths("src/models.py", plan) == []


def test_grounding_prefers_generated_else_skeleton_and_is_bounded():
    code = 'def add(a, b):\n    """Sum."""\n    return a + b'
    gen = {"src/models.py": code}
    skel = {"src/util.py": "def helper(x):\n    ..."}
    block = _grounding_block(["src/models.py", "src/util.py"], skel, gen)
    assert "def add(a, b)" in block and "# From src/models.py" in block
    assert "def helper(x)" in block and "# From src/util.py" in block
    big = _grounding_block(["m.py"], {}, {"m.py": "x=1\n" * 5000}, limit=100)
    assert len(big) <= 130
