"""Schema tests for the swarm plan (mirrors the greenfield WORK_PLAN).

Covers extraction of a plan object from an imperfect LLM reply, normalization
(dedupe by path, prune dangling ``depends_on``) and the dependency-first
flatten the Developer executes -- reusing the shared manifest toposort.
"""

from cgx.session.tasks.swarm_plan import (
    parse_plan_reply, normalize_plan, iter_plan_files, plan_is_buildable,
    verify_plan)

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
        '"depends_on":["src/main.py"]}]}'
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
    assert len(files) == 3


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
