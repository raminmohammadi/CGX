"""Schema tests for the swarm plan (mirrors the greenfield WORK_PLAN).

Covers extraction of a plan object from an imperfect LLM reply, normalization
(dedupe by path, prune dangling ``depends_on``) and the dependency-first
flatten the Developer executes -- reusing the shared manifest toposort.
"""

from cgx.session.tasks.swarm_plan import (
    parse_plan_reply, normalize_plan, iter_plan_files, plan_is_buildable,
    verify_plan, ensure_scaffolding, ensure_test_coverage)

FENCE = "```"


def _sample_plan():
    reply = (
        "Here is the plan:\n" + FENCE + "json\n"
        '{"goal":"build api","layers":['
        '{"name":"api","files":['
        '{"path":"src/main.py","description":"app",'
        '"depends_on":["src/models.py","ghost.py"]},'
        '{"path":"src/main.py","description":"dup"}]},'
        '{"name":"models","files":['
        '{"path":"src/models.py","description":"schemas"}]},'
        '{"name":"tests","files":['
        '{"path":"tests/test_main.py","description":"t",'
        '"depends_on":["src/main.py"]}]},'
        '{"name":"meta","files":['
        '{"path":"README.md","description":"overview"},'
        '{"path":"requirements.txt","description":"deps"},'
        '{"path":"conftest.py","description":"pytest path setup"}]}'
        ']}\n' + FENCE)
    return normalize_plan(parse_plan_reply(reply))


def test_parse_fenced_json():
    assert parse_plan_reply(FENCE + 'json\n{"goal":"x"}\n' + FENCE) == {"goal": "x"}


def test_parse_bare_brace_fallback():
    raw = parse_plan_reply('prose {"goal":"x","layers":[]} tail')
    assert raw is not None and raw["goal"] == "x"


def test_parse_non_json_returns_none():
    assert parse_plan_reply("no json here") is None


def test_normalize_dedupes_by_path():
    files = iter_plan_files(_sample_plan())
    paths = [f["path"] for f in files]
    assert paths.count("src/main.py") == 1
    # 3 code files (deduped) + 3 scaffolding files (README, requirements,
    # conftest) the complete-project sample now carries.
    assert len(files) == 6


def test_normalize_prunes_dangling_dependency():
    files = iter_plan_files(_sample_plan())
    main = next(f for f in files if f["path"] == "src/main.py")
    assert main["depends_on"] == ["src/models.py"]


def test_iter_plan_files_is_dependency_first():
    # models -> main -> test_main, regardless of declared layer order.
    order = [f["path"] for f in iter_plan_files(_sample_plan())]
    assert order.index("src/models.py") < order.index("src/main.py")
    assert order.index("src/main.py") < order.index("tests/test_main.py")


def test_plan_is_buildable_true_with_source():
    assert plan_is_buildable(_sample_plan()) is True


def test_plan_is_buildable_false_when_only_tests_and_docs():
    plan = normalize_plan({"layers": [{"name": "meta", "files": [
        {"path": "tests/test_x.py"}, {"path": "README.md"}]}]})
    assert plan_is_buildable(plan) is False


def test_normalize_drops_pathless_entries():
    plan = normalize_plan({"layers": [{"name": "x", "files": [
        {"description": "no path"}, {"path": "src/a.py"}]}]})
    files = iter_plan_files(plan)
    assert [f["path"] for f in files] == ["src/a.py"]


# --------------------- verify_plan (pre-flight gate) ---------------------

def test_verify_plan_accepts_a_coherent_plan():
    assert verify_plan(_sample_plan()) == []


def test_verify_plan_rejects_mixed_rooting():
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "b.py", "description": "b"}]}]})
    problems = verify_plan(plan)
    assert any("inconsistent layout" in p for p in problems)


def test_verify_plan_rejects_unsafe_path():
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "../escape.py", "description": "x"}]}]})
    assert any("escapes the project root" in p for p in verify_plan(plan))


def test_verify_plan_rejects_dependency_cycle():
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a", "depends_on": ["src/b.py"]},
        {"path": "src/b.py", "description": "b", "depends_on": ["src/a.py"]}]}]})
    assert any("cycle" in p for p in verify_plan(plan))


def test_verify_plan_rejects_orphan_test():
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "tests/test_a.py", "description": "t"}]}]})
    assert any("no depends_on" in p for p in verify_plan(plan))


def test_verify_plan_rejects_unbuildable_plan():
    plan = normalize_plan({"layers": [{"name": "meta", "files": [
        {"path": "tests/test_x.py", "depends_on": ["README.md"]},
        {"path": "README.md"}]}]})
    assert any("runnable non-test source" in p for p in verify_plan(plan))


def test_verify_plan_rejects_missing_scaffolding():
    # A src/ layout with source + test but none of the scaffolding files ->
    # a concrete, re-askable problem for each missing deliverable.
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "tests/test_a.py", "description": "t",
         "depends_on": ["src/a.py"]}]}]})
    problems = verify_plan(plan)
    assert any("README.md" in p for p in problems)
    assert any("requirements.txt" in p for p in problems)
    assert any("conftest.py" in p for p in problems)


def test_verify_plan_conftest_only_required_for_src_layout():
    # A top-level layout needs README + a manifest, but not a root conftest.py
    # (no package to add to sys.path), so a complete top-level plan verifies.
    plan = normalize_plan({"layers": [{"name": "all", "files": [
        {"path": "a.py", "description": "a"},
        {"path": "test_a.py", "description": "t", "depends_on": ["a.py"]},
        {"path": "README.md", "description": "overview"},
        {"path": "requirements.txt", "description": "deps"}]}]})
    assert verify_plan(plan) == []


def test_verify_plan_conftest_is_not_an_orphan_test():
    # conftest.py carries fixtures/path setup and legitimately declares no
    # depends_on; despite the "test" substring it must not be flagged as an
    # orphan test module.
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "tests/test_a.py", "description": "t",
         "depends_on": ["src/a.py"]},
        {"path": "README.md", "description": "overview"},
        {"path": "requirements.txt", "description": "deps"},
        {"path": "conftest.py", "description": "pytest path setup"}]}]})
    problems = verify_plan(plan)
    assert not any("conftest.py" in p and "depends_on" in p for p in problems)
    assert problems == []


# ----------------- ensure_scaffolding (deterministic inject) --------------

def test_ensure_scaffolding_injects_missing_src_layout():
    # A src/ layout with source + test but no scaffolding: injection adds all
    # three deliverables and the augmented plan then clears verify_plan.
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "tests/test_a.py", "description": "t",
         "depends_on": ["src/a.py"]}]}]})
    assert verify_plan(plan)  # missing scaffolding before injection
    plan = ensure_scaffolding(plan)
    paths = [f["path"] for f in iter_plan_files(plan)]
    assert {"README.md", "requirements.txt", "conftest.py"} <= set(paths)
    assert verify_plan(plan) == []
    # README + requirements are generated after the sources they describe/scan.
    assert paths.index("src/a.py") < paths.index("requirements.txt")
    assert paths.index("src/a.py") < paths.index("README.md")


def test_ensure_scaffolding_is_idempotent_and_skips_present():
    # A complete plan is untouched (no duplicate scaffolding), and a top-level
    # layout does not get a needless conftest.py.
    complete = _sample_plan()
    before = [f["path"] for f in iter_plan_files(complete)]
    after = [f["path"] for f in iter_plan_files(ensure_scaffolding(complete))]
    assert sorted(after) == sorted(before)
    top = normalize_plan({"layers": [{"name": "all", "files": [
        {"path": "a.py", "description": "a"},
        {"path": "test_a.py", "description": "t", "depends_on": ["a.py"]}]}]})
    top = ensure_scaffolding(top)
    paths = [f["path"] for f in iter_plan_files(top)]
    assert "conftest.py" not in paths
    assert {"README.md", "requirements.txt"} <= set(paths)


# --------------- ensure_test_coverage (deterministic inject) --------------

def test_ensure_test_coverage_injects_for_uncovered_module():
    # Two source modules, a test only for the first: the second gets a
    # pytest module injected that depends on it (so it is generated after,
    # and grounded against, the code it exercises).
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/a.py", "description": "a"},
        {"path": "src/b.py", "description": "b"},
        {"path": "tests/test_a.py", "description": "t",
         "depends_on": ["src/a.py"]}]}]})
    plan = ensure_test_coverage(plan)
    files = {f["path"]: f for f in iter_plan_files(plan)}
    assert "tests/test_b.py" in files
    assert files["tests/test_b.py"]["depends_on"] == ["src/b.py"]
    # The already-covered module is not given a second test.
    assert sum(1 for p in files if p.endswith("test_a.py")) == 1


def test_ensure_test_coverage_skips_init_and_is_idempotent():
    # Package markers are not worth a test; a second pass injects nothing.
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/pkg/__init__.py", "description": "marker"},
        {"path": "src/pkg/core.py", "description": "core"}]}]})
    once = ensure_test_coverage(plan)
    paths = [f["path"] for f in iter_plan_files(once)]
    assert "tests/test_core.py" in paths
    assert not any("__init__" in p for p in paths if p.startswith("tests/"))
    twice = [f["path"] for f in iter_plan_files(ensure_test_coverage(once))]
    assert sorted(twice) == sorted(paths)


def test_ensure_test_coverage_disambiguates_basename_collision():
    # Two modules share a basename in different dirs; the second injected
    # test is qualified by its parent dir so the paths do not collide.
    plan = normalize_plan({"layers": [{"name": "core", "files": [
        {"path": "src/api/handler.py", "description": "a"},
        {"path": "src/worker/handler.py", "description": "b"}]}]})
    paths = [f["path"] for f in iter_plan_files(ensure_test_coverage(plan))]
    injected = [p for p in paths if p.startswith("tests/")]
    assert len(injected) == 2 and len(set(injected)) == 2
