"""``cgx site`` -- Site Studio: managed workspaces + one-shot generate/revise.

Mirrors the web-UI Site Studio routes (``src/cgx/webui/routes/sites.py``):
managed site workspaces under ``~/.cgx/sites/<slug>`` (:mod:`cgx.sites`), a
focused single-shot generator that emits one self-contained ``index.html``
(:func:`cgx.answer.static_site_gen.generate_site` + one refine pass), and
in-place revision from feedback. ``path`` prints the workspace directory --
open ``<path>/index.html`` in a browser to view it (the CLI has no server).
"""

from __future__ import annotations

import argparse
import os

from cgx.cli import _render

# A full page is a few thousand tokens; the default budget truncates it into a
# thin stub, so site generation forces a large budget (matches the route).
_SITE_NUM_PREDICT = 8000

_LIST_COLUMNS = [
    ("SLUG", "slug"),
    ("HAS_INDEX", "has_index"),
    ("PROJECT_ROOT", "project_root"),
]

_PROVIDER_KINDS = ("ollama", "openai", "openai-compat", "gemini", "huggingface", "custom")


def register(sub) -> None:
    p = sub.add_parser("site", help="Site Studio (list/new/generate/revise/path/themes).")
    verbs = p.add_subparsers(dest="site_cmd", required=True)

    p_list = verbs.add_parser("list", help="List managed site workspaces.")
    p_list.add_argument("--table", action="store_true", help="Human table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_themes = verbs.add_parser("themes", help="List curated design themes.")
    p_themes.set_defaults(func=_themes)

    p_new = verbs.add_parser("new", help="Create an empty named site workspace.")
    p_new.add_argument("name")
    p_new.set_defaults(func=_new)

    p_path = verbs.add_parser("path", help="Print a site's workspace directory.")
    p_path.add_argument("slug")
    p_path.set_defaults(func=_path)

    p_gen = verbs.add_parser("generate", help="Generate a site from a brief (one-shot).")
    p_gen.add_argument("name", help="Site name (becomes the workspace slug).")
    p_gen.add_argument("--brief", "-b", required=True, help="What the site is for.")
    p_gen.add_argument("--flavor", default="modern", help="Design flavor (default: modern).")
    p_gen.add_argument("--theme", default=None, help="Explicit theme key (see `site themes`).")
    _add_provider_flags(p_gen)
    p_gen.set_defaults(func=_generate)

    p_rev = verbs.add_parser("revise", help="Revise an existing site in place from feedback.")
    p_rev.add_argument("slug")
    p_rev.add_argument("--feedback", "-m", required=True, help="What to change.")
    p_rev.add_argument("--flavor", default="modern", help="Design flavor (default: modern).")
    p_rev.add_argument("--theme", default=None, help="Explicit theme key.")
    _add_provider_flags(p_rev)
    p_rev.set_defaults(func=_revise)


def _add_provider_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default=None,
                   help="Saved provider profile (overrides --provider/--model).")
    p.add_argument("--provider", default="ollama", choices=_PROVIDER_KINDS,
                   help="Provider kind when no --profile is given (default: ollama).")
    p.add_argument("--model", default=None, help="LLM model name.")
    p.add_argument("--base-url", default="http://localhost:11434", help="Provider base URL.")
    p.add_argument("--api-key", default=None, help="API key; defaults to $CGX_API_KEY.")


def _provider_from_args(args: argparse.Namespace):
    """Build a provider object with the large site token budget."""
    from cgx.answer.provider_factory import build_provider, provider_from_profile_name
    if getattr(args, "profile", None):
        return provider_from_profile_name(args.profile)
    model = args.model
    if not model and args.provider == "ollama":
        try:
            from cgx.answer import ollama_discovery
            model = ollama_discovery.recommend_default_model()
        except Exception:
            model = "qwen2.5-coder:3b"
    return build_provider(
        kind=args.provider, model=model or "", base_url=args.base_url,
        api_key=(args.api_key or os.environ.get("CGX_API_KEY")),
        temperature=0.4, num_predict=_SITE_NUM_PREDICT)


def _write_index(project_root: str, html: str) -> int:
    """Write index.html into the site workspace (containment-guarded)."""
    from cgx import sites
    base = sites._sites_root_real()
    root = os.path.realpath(project_root)
    if root != base and not root.startswith(base + os.sep):
        _render.die(f"invalid site root: {project_root!r}", code=3)
    path = os.path.join(root, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return len(html.encode("utf-8"))


def _list(args: argparse.Namespace) -> None:
    from cgx import sites
    _render.emit(sites.list_sites(), table=getattr(args, "table", False), columns=_LIST_COLUMNS)


def _themes(args: argparse.Namespace) -> None:
    from cgx.answer.web_themes import theme_summaries
    _render.print_json({"themes": theme_summaries()})


def _new(args: argparse.Namespace) -> None:
    from cgx import sites
    name = (args.name or "").strip()
    if not name:
        _render.die("name is required", code=3)
    try:
        _render.print_json(sites.create_site(name))
    except ValueError as exc:
        _render.die(str(exc), code=3)


def _path(args: argparse.Namespace) -> None:
    from cgx import sites
    try:
        print(sites.site_path(args.slug))
    except ValueError as exc:
        _render.die(str(exc), code=3)


def _generate(args: argparse.Namespace) -> None:
    from cgx import sites
    from cgx.answer.static_site_gen import generate_site, refine_site
    brief = (args.brief or "").strip()
    name = (args.name or "").strip()
    if not name or not brief:
        _render.die("name and brief are required", code=3)
    try:
        info = sites.create_site(name)
    except ValueError as exc:
        _render.die(str(exc), code=3)
    theme_key = (args.theme or "").strip() or None
    try:
        prov = _provider_from_args(args)
        html = generate_site(brief, prov, flavor=args.flavor, theme_key=theme_key)
        html = refine_site(html, prov, flavor=args.flavor, theme_key=theme_key, rounds=1)
    except Exception as exc:  # noqa: BLE001 -- surface any generation failure
        _render.die(f"generation failed: {type(exc).__name__}: {exc}", code=1)
    nbytes = _write_index(info["project_root"], html)
    _render.print_json({"slug": info["slug"], "project_root": info["project_root"],
                        "entry": "index.html", "bytes": nbytes})


def _revise(args: argparse.Namespace) -> None:
    from cgx import sites
    from cgx.answer.static_site_gen import generate_site
    feedback = (args.feedback or "").strip()
    if not feedback:
        _render.die("feedback is required", code=3)
    try:
        root = sites.site_path(args.slug)
    except ValueError as exc:
        _render.die(str(exc), code=3)
    index = root / "index.html"
    if not index.is_file():
        _render.die("site has no index.html to revise", code=2)
    current = index.read_text(encoding="utf-8", errors="replace")
    theme_key = (args.theme or "").strip() or None
    try:
        html = generate_site("", _provider_from_args(args), flavor=args.flavor,
                             theme_key=theme_key, current_html=current, feedback=feedback)
    except Exception as exc:  # noqa: BLE001
        _render.die(f"revision failed: {type(exc).__name__}: {exc}", code=1)
    nbytes = _write_index(str(root), html)
    _render.print_json({"slug": args.slug, "project_root": str(root),
                        "entry": "index.html", "bytes": nbytes})
