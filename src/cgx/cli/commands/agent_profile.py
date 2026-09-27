"""``cgx agent-profile`` -- list / show / add / rm saved agent-launch bundles.

Mirrors the web-UI agent-profile routes
(``src/cgx/webui/routes/agent_profiles.py``) exactly: same engine calls, same
validation gate (name + objective required) before any write, same 404 on a
missing delete target. An agent profile is a saved
``{objective, project_root, mode, skills}`` template an agent session can be
launched from -- unrelated to the LLM connection presets in
``cgx.answer.profiles`` (they merely share the word "profile"). Profiles live
in ``~/.cgx/agent_profiles.json``; there are no secrets and nothing built-in
to protect, so ``rm`` deletes any named profile outright.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

_LIST_COLUMNS = [
    ("NAME", "name"),
    ("MODE", "mode"),
    ("PROJECT ROOT", "project_root"),
    ("SKILLS", "skills"),
    ("OBJECTIVE", "objective"),
]


def register(sub) -> None:
    p = sub.add_parser("agent-profile",
                       help="Manage agent-launch profiles (list/show/add/rm).")
    verbs = p.add_subparsers(dest="agent_profile_cmd", required=True)

    p_list = verbs.add_parser("list", help="List saved agent profiles.")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_show = verbs.add_parser("show", help="Show one agent profile as JSON.")
    p_show.add_argument("name")
    p_show.set_defaults(func=_show)

    # ``add`` mirrors ``PUT /agent-profiles/{name}`` -- upsert semantics, so it
    # creates or overwrites the named profile. Body fields map 1:1 to flags.
    p_add = verbs.add_parser("add",
                             help="Create or overwrite an agent profile (upsert).")
    p_add.add_argument("name")
    p_add.add_argument("--objective", default="",
                       help="Task/objective the profile launches (required).")
    p_add.add_argument("--project-root", default="",
                       help="Project directory the agent runs in.")
    p_add.add_argument("--mode", default="",
                       help="Session mode: '' (auto), 'explore', or 'greenfield'.")
    p_add.add_argument("--skill", dest="skills", action="append", default=None,
                       metavar="NAME", help="Skill to attach (repeatable).")
    p_add.set_defaults(func=_add)

    p_rm = verbs.add_parser("rm", help="Delete an agent profile.")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=_rm)


def _list(args: argparse.Namespace) -> None:
    from cgx.answer.agent_profiles import list_agent_profiles
    rows = [p.to_public_dict() for p in list_agent_profiles()]
    _render.emit(rows, table=getattr(args, "table", False), columns=_LIST_COLUMNS)


def _show(args: argparse.Namespace) -> None:
    from cgx.answer.agent_profiles import get_agent_profile
    profile = get_agent_profile(args.name)
    if profile is None:
        _render.die(f"agent profile {args.name!r} not found", code=2)
    _render.print_json(profile.to_public_dict())


def _add(args: argparse.Namespace) -> None:
    from cgx.answer.agent_profiles import AgentProfile, save_agent_profile
    # Same guards the route raises as HTTP 400 -- surface them as validation
    # exits (code 3) before touching the store.
    if not args.name.strip():
        _render.die("agent profile name is required", code=3)
    if not args.objective.strip():
        _render.die("objective is required", code=3)
    profile = AgentProfile(
        name=args.name.strip(),
        objective=args.objective.strip(),
        project_root=args.project_root.strip(),
        mode=args.mode,
        skills=list(args.skills or []),
    )
    save_agent_profile(profile)
    _render.print_json(profile.to_public_dict())


def _rm(args: argparse.Namespace) -> None:
    from cgx.answer.agent_profiles import delete_agent_profile
    if delete_agent_profile(args.name):
        _render.print_json({"deleted": args.name})
    else:
        _render.die(f"agent profile {args.name!r} not found", code=2)

