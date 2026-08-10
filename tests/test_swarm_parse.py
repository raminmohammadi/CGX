"""Tolerance tests for the swarm action parser.

Small local models routinely drop the ``<tool_call>`` envelope, wrap content
in a Markdown fence, or truncate mid-file. The first swarm cut parsed with a
strict ``text.find("<tool_call>")`` and saw no action for any of these, draining
the loop budget. These tests pin the tolerant behaviour of
:func:`parse_swarm_action` against exactly those failure modes.
"""

from cgx.session.tasks.swarm_parse import parse_swarm_action, strip_fence

DEV = ("bash_repl", "edit_file", "patch_file")
FENCE = "```"


def test_wrapped_tool_call_is_parsed():
    a = parse_swarm_action(
        "<tool_call>\n<name>edit_file</name>\n<path>src/a.py</path>\n"
        "<content>print(1)</content>\n</tool_call>",
        allowed_tools=DEV, allow_report=True)
    assert a.kind == "tool_call"
    assert a.name == "edit_file"
    assert a.fields.get("path") == "src/a.py"
    assert a.fields.get("content") == "print(1)"


def test_bare_name_without_envelope_is_honoured():
    a = parse_swarm_action(
        "<name>edit_file</name>\n<path>src/b.py</path>\n<content>x=1</content>",
        allowed_tools=DEV)
    assert a.kind == "tool_call"
    assert a.name == "edit_file"
    assert a.fields.get("path") == "src/b.py"


def test_fenced_content_is_stripped():
    a = parse_swarm_action(
        "<name>edit_file</name><path>c.py</path>"
        f"<content>{FENCE}python\nprint(2)\n{FENCE}</content>",
        allowed_tools=DEV)
    assert a.fields.get("content") == "print(2)"


def test_unclosed_content_is_captured():
    a = parse_swarm_action(
        "<name>edit_file</name><path>d.py</path>"
        "<content>def f():\n    return 1",
        allowed_tools=DEV)
    assert "return 1" in a.fields.get("content", "")


def test_delegate_then_finish_finish_wins():
    a = parse_swarm_action(
        "<delegate>build the thing</delegate>\nAll done.\n"
        "<finish>complete</finish>",
        allowed_tools=(), allow_delegate=True, allow_finish=True)
    assert a.kind == "finish"
    assert a.text == "complete"


def test_report_block_is_parsed():
    a = parse_swarm_action("<report>wrote 3 files</report>",
                           allowed_tools=DEV, allow_report=True)
    assert a.kind == "report"
    assert a.text == "wrote 3 files"


def test_disallowed_tool_flagged_as_error():
    a = parse_swarm_action("<name>rm_rf</name>", allowed_tools=DEV)
    assert a.kind == "none"
    assert a.error is not None


def test_no_action_returns_none():
    a = parse_swarm_action("Let me think about this first.",
                           allowed_tools=DEV, allow_report=True)
    assert a.kind == "none"


def test_delegate_disabled_is_not_an_action():
    # A stray <delegate> in the Developer role (delegate disabled) must not
    # be honoured as a terminal action.
    a = parse_swarm_action("<delegate>do it</delegate>", allowed_tools=DEV)
    assert a.kind == "none"


def test_strip_fence_leaves_unfenced_text():
    assert strip_fence("print(1)") == "print(1)"
    assert strip_fence(f"{FENCE}py\nprint(1)\n{FENCE}") == "print(1)"
