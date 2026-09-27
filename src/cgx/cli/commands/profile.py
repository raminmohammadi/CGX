"""``cgx profile`` -- list / show / add / rm saved LLM connection presets.

Mirrors the web-UI profile routes (``src/cgx/webui/routes/profiles.py``)
exactly: the same engine calls in :mod:`cgx.answer.profiles`, the same
upsert body fields, and the same error cases (name-required / not-found /
save-failure). Profiles persist under ``~/.cgx/profiles.json``; API keys go
to the OS keyring (or a 0600 file fallback) and are *never* echoed back --
only the ``has_api_key`` flag is surfaced, just like the route.

API keys are read from ``$CGX_API_KEY`` by default (preferred, keeps the
secret out of shell history); ``--api-key`` overrides it when given.
"""

from __future__ import annotations

import argparse
import os

from cgx.cli import _render

#: Env var consulted for the upsert secret when ``--api-key`` is omitted.
_API_KEY_ENV = "CGX_API_KEY"

#: Provider kinds documented on the ``Profile`` dataclass / upsert body.
_KINDS = ("ollama", "openai-compat", "gemini", "huggingface", "custom")

_LIST_COLUMNS = [
    ("NAME", "name"),
    ("KIND", "kind"),
    ("MODEL", "model"),
    ("BASE_URL", "base_url"),
    ("HAS_KEY", "has_api_key"),
    ("TEMP", "temperature"),
    ("MAX_TOK", "num_predict"),
]


def register(sub) -> None:
    p = sub.add_parser("profile",
                       help="Manage LLM connection presets (list/show/add/rm).")
    verbs = p.add_subparsers(dest="profile_cmd", required=True)

    p_list = verbs.add_parser("list", help="List saved provider profiles.")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_show = verbs.add_parser("show", help="Show one profile (no secret material).")
    p_show.add_argument("name")
    p_show.set_defaults(func=_show)

    p_add = verbs.add_parser(
        "add", help="Create or update a profile (mirrors the PUT upsert body).")
    p_add.add_argument("name")
    p_add.add_argument("--kind", default="ollama", choices=_KINDS,
                       help="Provider kind (default: ollama).")
    p_add.add_argument("--model", default="qwen2.5-coder:3b",
                       help="Model name (default: qwen2.5-coder:3b).")
    p_add.add_argument("--base-url", default="http://localhost:11434",
                       help="Provider base URL (default: http://localhost:11434).")
    p_add.add_argument("--api-key", default=None,
                       help=f"API key; defaults to ${_API_KEY_ENV}. Never echoed back.")
    p_add.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature (default: 0.2).")
    p_add.add_argument("--num-predict", type=int, default=1024,
                       help="Max tokens to generate (default: 1024).")
    p_add.add_argument("--num-ctx", type=int, default=None,
                       help="Optional Ollama KV-cache size (>0); omit for auto.")
    p_add.add_argument("--endpoint-path", default="/v1/chat/completions",
                       help="Custom endpoint suffix (default: /v1/chat/completions).")
    p_add.add_argument("--allow-no-auth", action="store_true",
                       help="Skip Bearer auth (private-subnet servers).")
    p_add.set_defaults(func=_add)

    p_rm = verbs.add_parser("rm", help="Delete a saved profile.")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=_rm)


def _summary(p) -> dict:
    """Secret-free public view of a profile -- mirrors ``ProfileSummary``."""
    return {
        "name": p.name,
        "kind": p.kind,
        "model": p.model,
        "base_url": p.base_url,
        "has_api_key": bool(p.has_api_key),
        "temperature": p.temperature,
        "num_predict": p.num_predict,
        "num_ctx": getattr(p, "num_ctx", None),
        "endpoint_path": getattr(p, "endpoint_path", "/v1/chat/completions"),
        "allow_no_auth": bool(getattr(p, "allow_no_auth", False)),
    }


def _list(args: argparse.Namespace) -> None:
    from cgx.answer import profiles as _profiles
    rows = [_summary(p) for p in _profiles.list_profiles()]
    _render.emit(rows, table=getattr(args, "table", False), columns=_LIST_COLUMNS)


def _show(args: argparse.Namespace) -> None:
    from cgx.answer import profiles as _profiles
    prof = _profiles.get_profile(args.name)
    if prof is None:
        _render.die(f"profile {args.name!r} not found", code=2)
    _render.print_json(_summary(prof))


def _add(args: argparse.Namespace) -> None:
    from cgx.answer import profiles as _profiles
    # Mirror the route's name guard (HTTP 400 -> validation exit code 3).
    name = args.name.strip()
    if not name:
        _render.die("profile name is required", code=3)
    # Prefer the secret from the environment; --api-key overrides. Sending no
    # key preserves any key already attached to this profile name (see
    # save_profile), matching the route's edit round-trip.
    api_key = args.api_key or os.environ.get(_API_KEY_ENV) or None
    nc = args.num_ctx
    try:
        prof = _profiles.Profile(
            name=name,
            kind=args.kind,
            model=args.model.strip(),
            base_url=args.base_url.strip(),
            temperature=float(args.temperature),
            num_predict=int(args.num_predict),
            num_ctx=(int(nc) if isinstance(nc, (int, float)) and nc > 0 else None),
            endpoint_path=(args.endpoint_path or "/v1/chat/completions"),
            allow_no_auth=bool(args.allow_no_auth),
        )
        _profiles.save_profile(prof, api_key=api_key)
    except Exception as e:  # mirror the route's 400 on any save failure
        _render.die(f"{type(e).__name__}: {e}", code=3)
    # Re-read to reflect the persisted has_api_key bit (route does the same).
    for live in _profiles.list_profiles():
        if live.name == prof.name:
            _render.print_json(_summary(live))
            return
    _render.print_json(_summary(prof))


def _rm(args: argparse.Namespace) -> None:
    from cgx.answer import profiles as _profiles
    if _profiles.delete_profile(args.name):
        _render.print_json({"deleted": args.name})
    else:
        _render.die(f"profile {args.name!r} not found", code=2)

