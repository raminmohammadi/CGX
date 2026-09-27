"""``cgx skills`` -- list / show / create / edit / delete skills.

Mirrors the web-UI skill routes (``src/cgx/webui/routes/skills.py``) exactly:
same engine calls, same built-in guard, same validation gate before any
write. Custom skills live under ``~/.cgx/skills/``; built-ins are read-only.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

_LIST_COLUMNS = [
    ("NAME", "name"),
    ("ROLE", "role"),
    ("FORMAT", "format"),
    ("CUSTOM", "is_custom"),
    ("DESCRIPTION", "description"),
]


def register(sub) -> None:
    p = sub.add_parser("skills", help="Manage skills (list/show/create/edit/delete).")
    verbs = p.add_subparsers(dest="skills_cmd", required=True)

    p_list = verbs.add_parser("list", help="List built-in + custom skills.")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_show = verbs.add_parser("show", help="Print a skill's source.")
    p_show.add_argument("name")
    p_show.set_defaults(func=_show)

    p_create = verbs.add_parser("create",
                                help="Create a custom skill from a file or stdin.")
    p_create.add_argument("--file", "-f", default=None,
                          help="Source file (default: read stdin).")
    p_create.add_argument("--markdown", action="store_true",
                          help="Treat the source as a SKILL.md markdown skill.")
    p_create.add_argument("--name", default="",
                          help="Name hint for markdown skills (optional).")
    p_create.set_defaults(func=_create)

    p_edit = verbs.add_parser("edit", help="Update an existing custom skill.")
    p_edit.add_argument("name")
    p_edit.add_argument("--file", "-f", default=None,
                        help="Source file (default: read stdin).")
    p_edit.add_argument("--markdown", action="store_true",
                        help="Treat the source as a SKILL.md markdown skill.")
    p_edit.set_defaults(func=_edit)

    p_del = verbs.add_parser("delete", help="Delete a custom skill (built-ins protected).")
    p_del.add_argument("name")
    p_del.set_defaults(func=_delete)


def _list(args: argparse.Namespace) -> None:
    import skills as _skills
    _render.emit(_skills.describe_skills(),
                 table=getattr(args, "table", False), columns=_LIST_COLUMNS)


def _show(args: argparse.Namespace) -> None:
    import skills as _skills
    src = _skills.read_skill_source(args.name)
    if src is None:
        _render.die(f"skill {args.name!r} not found", code=2)
    print(src)  # raw source, not JSON -- pipeable into an editor


def _create(args: argparse.Namespace) -> None:
    import skills as _skills
    from skills.loader import (
        save_custom_skill,
        save_markdown_skill,
        validate_markdown_skill_source,
        validate_skill_source,
    )
    source = _render.read_text_input(args.file)
    if args.markdown:
        result = validate_markdown_skill_source(
            source, known_names=_skills.known_skill_names(), name_hint=args.name)
    else:
        result = validate_skill_source(source, known_names=_skills.known_skill_names())
    if not result.ok:
        _render.die(f"{result.error_kind}: {result.error_detail}", code=3)
    meta = result.meta or {}
    name = meta["name"]
    (save_markdown_skill if args.markdown else save_custom_skill)(name, source)
    _render.print_json(
        {"created": name, "format": "markdown" if args.markdown else "python"})


def _edit(args: argparse.Namespace) -> None:
    import skills as _skills
    from skills.loader import (
        read_custom_skill_source,
        read_markdown_skill_source,
        save_custom_skill,
        save_markdown_skill,
        validate_markdown_skill_source,
        validate_skill_source,
    )
    name = args.name
    if name.lower() in _builtin_names(_skills):
        _render.die("cannot edit a built-in skill", code=3)
    read = read_markdown_skill_source if args.markdown else read_custom_skill_source
    if read(name) is None:
        kind = "markdown" if args.markdown else "custom"
        _render.die(f"{kind} skill {name!r} not found", code=2)
    source = _render.read_text_input(args.file)
    known = _skills.known_skill_names(exclude=name)
    if args.markdown:
        result = validate_markdown_skill_source(source, known_names=known, name_hint=name)
    else:
        result = validate_skill_source(source, known_names=known)
    if not result.ok:
        _render.die(f"{result.error_kind}: {result.error_detail}", code=3)
    meta = result.meta or {}
    if meta.get("name") != name:
        _render.die(
            f"source defines name {meta.get('name')!r}, expected {name!r} "
            "-- rename via delete + create instead", code=3)
    (save_markdown_skill if args.markdown else save_custom_skill)(name, source)
    _render.print_json(
        {"updated": name, "format": "markdown" if args.markdown else "python"})


def _delete(args: argparse.Namespace) -> None:
    import skills as _skills
    from skills.loader import delete_custom_skill, delete_markdown_skill
    if args.name.lower() in _builtin_names(_skills):
        _render.die("cannot delete a built-in skill", code=3)
    if delete_custom_skill(args.name) or delete_markdown_skill(args.name):
        _render.print_json({"deleted": args.name})
    else:
        _render.die(f"custom skill {args.name!r} not found", code=2)


def _builtin_names(_skills) -> set:
    return {s.name.lower() for s in _skills.SKILLS}
