"""``cgx context`` -- show / edit / rm the project's CGX.md context file.

Mirrors the web-UI context-file routes
(``src/cgx/webui/routes/context_file.py``) exactly: the same
:mod:`cgx.context_files` engine calls, the same fixed-filename write target
(so the filename itself can't be used for path traversal), the same 64k
write ceiling, and the same not-found / too-large cases -- surfaced here as
``die()`` exits instead of HTTP errors. The context file is *always on*: the
chatbot and agents inject it on every turn for the project it lives in.
"""

from __future__ import annotations

import argparse
import os
import sys

from cgx.cli import _render

# Same generous ceiling the route/loader applies; keeps a pathological paste
# out of the repo. The per-surface injection layer trims further to budget.
_MAX_WRITE_CHARS = 64_000


def register(sub) -> None:
    p = sub.add_parser(
        "context",
        help="Manage the project's CGX.md context file (show/edit/rm).")
    verbs = p.add_subparsers(dest="context_cmd", required=True)

    p_show = verbs.add_parser("show", help="Print the project's context file (raw).")
    _add_root(p_show)
    p_show.set_defaults(func=_show)

    p_edit = verbs.add_parser(
        "edit", help="Create or replace the context file from a file or stdin.")
    _add_root(p_edit)
    p_edit.add_argument("--file", "-f", default=None,
                        help="Source file (default: read stdin).")
    p_edit.set_defaults(func=_edit)

    p_rm = verbs.add_parser("rm", help="Delete the project's context file, if present.")
    _add_root(p_rm)
    p_rm.set_defaults(func=_rm)


def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project-root", default=None,
                   help="Project directory (default: current dir).")


def _root(args: argparse.Namespace) -> str:
    """Resolve --project-root to cwd when omitted; reject an explicit blank."""
    raw = getattr(args, "project_root", None)
    root = (raw if raw is not None else os.getcwd()).strip()
    if not root:  # mirrors the route's 400 when project_root is empty
        _render.die("project_root is required", code=3)
    return root


def _show(args: argparse.Namespace) -> None:
    from cgx import context_files as _cf
    root = _root(args)
    found = _cf.find_context_file(root)
    if found is None:
        # Not an error: mirror the GET route, which returns exists=False
        # rather than 404. Note goes to stderr so stdout stays pipeable.
        default = _cf.default_context_path(root)
        print(f"no context file under {root!r} "
              f"(create one at {default} with `cgx context edit`)",
              file=sys.stderr)
        return
    try:
        content = found.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _render.die(f"could not read context file: {e}")
    print(content, end="")  # raw content, not JSON -- pipeable to a file/editor


def _edit(args: argparse.Namespace) -> None:
    from cgx import context_files as _cf
    root = _root(args)
    content = _render.read_text_input(args.file)
    if len(content) > _MAX_WRITE_CHARS:
        _render.die(
            f"too_large: context file exceeds {_MAX_WRITE_CHARS} characters",
            code=3)
    # Write the existing file if one is present, else the default CGX.md --
    # the filename is fixed, so it can't be used for path traversal.
    target = _cf.find_context_file(root) or _cf.default_context_path(root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        _render.die(f"could not write context file: {e}")
    _cf.clear_cache()
    _render.print_json({"path": str(target), "filename": target.name,
                        "bytes": len(content.encode("utf-8"))})


def _rm(args: argparse.Namespace) -> None:
    from cgx import context_files as _cf
    root = _root(args)
    found = _cf.find_context_file(root)
    if found is None:
        _render.die("no context file to delete", code=2)
    try:
        os.remove(found)
    except OSError as e:
        _render.die(f"could not delete context file: {e}")
    _cf.clear_cache()
    _render.print_json({"deleted": str(found)})

