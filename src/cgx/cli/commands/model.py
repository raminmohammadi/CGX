"""``cgx model`` -- discover, size, and pull LLM + embedding models.

Mirrors three web-UI routes (the CLI needs no FastAPI, so every leaf calls
the same engine functions the routes call, never ``cgx.webui``):

* ``src/cgx/webui/routes/setup.py`` -- ``GET /setup/models`` (``list``) and
  ``POST /ollama/pull`` (``pull``), over :mod:`cgx.answer.ollama_discovery`
  and :data:`cgx.answer.hardware_matrix.LOCAL_MODEL_CATALOG`.
* ``src/cgx/webui/routes/hardware.py`` -- ``GET /hardware/matrix`` (``matrix``)
  and ``GET /hardware/hf_fit`` (``fit``), over :mod:`cgx.answer.hardware_matrix`.
* ``src/cgx/webui/routes/embed.py`` -- ``GET /embed/models`` (``embed-list``)
  and ``POST /embed/pull`` (``embed-pull``), over :mod:`cgx.embeddings.catalog`.

The two pull verbs stream human progress to stderr and print a final JSON
summary to stdout. Neither pull route wraps a reusable engine generator --
each inlines the download loop -- so ``pull``/``embed-pull`` call the same
lower-level functions the handlers call (``validate_base_url`` + ``requests``;
``huggingface_hub`` + ``find_by_name``) and replicate the few route-local
helper constants that have no engine home (noted where they appear).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from cgx.cli import _render

# Flag default kept as a literal (not imported from ollama_discovery) so
# ``register`` stays free of any engine import -- matches ``site.py``. Same
# value as :data:`cgx.answer.ollama_discovery.DEFAULT_BASE_URL`.
_DEFAULT_BASE_URL = "http://localhost:11434"

# --- table columns (list verbs render these with --table) -------------------
_LIST_COLUMNS = [
    ("MODEL", "model"),
    ("INSTALLED", "installed"),
    ("RECOMMENDED", "recommended"),
]
_MATRIX_COLUMNS = [
    ("MODEL", "model"),
    ("PARAMS_B", "params_b"),
    ("MIN_RAM_GB", "min_ram_gb"),
    ("REC_VRAM_GB", "rec_vram_gb"),
    ("CTX", "ctx_window"),
    ("FAMILY", "family"),
    ("FIT", "fit"),
    ("INSTALLED", "installed"),
]
_EMBED_COLUMNS = [
    ("NAME", "name"),
    ("LABEL", "label"),
    ("KIND", "kind"),
    ("DIM", "dim"),
    ("MAX_TOKENS", "max_tokens"),
    ("SIZE_GB", "size_gb"),
    ("CACHED", "cached"),
]

# Replicated from the routes because no engine module exports them. The Hub
# host is a compile-time constant and the repo id is validated to a single
# ``owner/name`` path segment, so nothing user-supplied can redirect the
# request off-host (SSRF barrier) -- same guard as ``routes/hardware.py``.
_HF_HUB_BASE = "https://huggingface.co"
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# A valid Ollama tag is ``name[:tag]`` -- a path segment plus an optional tag.
# Restricting ``--local-name`` to this shape stops it smuggling a registry
# host/namespace into the /api/copy destination (mirrors ``routes/setup.py``).
_OLLAMA_NAME_RE = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")

# HF siblings the embed route never fetches (duplicate weights / runtime
# variants transformers won't load); replicated from ``routes/embed.py``.
_EMBED_SKIP_EXTS = (".onnx", ".gguf", ".mlpackage", ".msgpack", ".h5", ".tflite")


def register(sub) -> None:
    p = sub.add_parser("model", help="Discover / size / pull LLM + embedding models.")
    verbs = p.add_subparsers(dest="model_cmd", required=True)

    p_list = verbs.add_parser("list", help="Installed + available Ollama models.")
    p_list.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="Ollama base URL.")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_matrix = verbs.add_parser("matrix", help="Hardware fit matrix for local models.")
    p_matrix.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="Ollama base URL.")
    p_matrix.add_argument("--table", action="store_true",
                          help="Human-readable table instead of JSON.")
    p_matrix.set_defaults(func=_matrix)

    p_fit = verbs.add_parser("fit", help="Score a Hugging Face repo against local hardware.")
    p_fit.add_argument("repo", help="HF repo id (owner/name).")
    p_fit.set_defaults(func=_fit)

    p_pull = verbs.add_parser("pull", help="Pull an Ollama model (progress on stderr).")
    p_pull.add_argument("model", help="Ollama tag, e.g. qwen2.5-coder:3b or hf.co/<repo>.")
    p_pull.add_argument("--base-url", default=_DEFAULT_BASE_URL, help="Ollama base URL.")
    p_pull.add_argument("--local-name", default=None,
                        help="Re-alias to this short name:tag once pulled (best-effort).")
    p_pull.set_defaults(func=_pull)

    p_el = verbs.add_parser("embed-list", help="Curated embedding models + cache status.")
    p_el.add_argument("--table", action="store_true",
                      help="Human-readable table instead of JSON.")
    p_el.set_defaults(func=_embed_list)

    p_ep = verbs.add_parser("embed-pull", help="Download an embedding model from HF (progress on stderr).")
    p_ep.add_argument("model", help="HF repo id of the embedding model.")
    p_ep.set_defaults(func=_embed_pull)


# --------------------------------------------------------------------------- #
# discovery verbs
# --------------------------------------------------------------------------- #

def _list(args: argparse.Namespace) -> None:
    """Twin of ``GET /setup/models``: union of installed tags + catalogue."""
    from cgx.answer import ollama_discovery
    from cgx.answer.hardware_matrix import LOCAL_MODEL_CATALOG

    base_url = args.base_url or _DEFAULT_BASE_URL
    ollama_reachable = False
    try:
        installed = [m["name"] for m in ollama_discovery.list_installed_models(base_url)]
        # list_installed_models returns [] both when unreachable and when empty;
        # the health check distinguishes the two (mirrors the route).
        ollama_reachable = bool(ollama_discovery.health_check(base_url).get("ok"))
        choices = ollama_discovery.model_choices(base_url)
    except Exception:
        installed = []
        choices = [tag for tag, *_ in ollama_discovery.RECOMMENDED_LADDER]

    # Merge the full hardware catalogue so every known local model appears.
    seen = set(choices)
    for entry in LOCAL_MODEL_CATALOG:
        if entry["name"] not in seen:
            choices.append(entry["name"])
            seen.add(entry["name"])

    # Cluster by family/version/size using exact catalogue params as the
    # size tiebreaker (installed-only tags fall back to the regex hint).
    params_lookup = {e["name"]: float(e["params_b"]) for e in LOCAL_MODEL_CATALOG}
    choices = ollama_discovery.sort_model_choices_by_family(choices, params_lookup)

    try:
        default = ollama_discovery.recommend_default_model(base_url=base_url)
    except Exception:
        default = choices[0] if choices else "qwen2.5-coder:3b"

    if getattr(args, "table", False):
        inst = set(installed)
        rows = [{"model": c, "installed": c in inst, "recommended": c == default}
                for c in choices]
        _render.print_table(rows, _LIST_COLUMNS)
        return
    _render.print_json({
        "choices": choices,
        "recommended_default": default,
        "installed": installed,
        "ollama_reachable": ollama_reachable,
    })


def _matrix(args: argparse.Namespace) -> None:
    """Twin of ``GET /hardware/matrix``: catalogue fit + installed-only rows."""
    from cgx.answer import ollama_discovery
    from cgx.answer.hardware_matrix import (
        compute_local_fit,
        make_fit_row,
        params_from_name,
        parse_parameter_size,
        tradeoffs_rows,
    )

    base_url = args.base_url or _DEFAULT_BASE_URL
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    rows = compute_local_fit(hw)

    # Flag catalogue rows the user has pulled and append installed-only tags
    # that aren't in the static catalogue so the table mirrors disk.
    try:
        installed = ollama_discovery.list_installed_models(base_url)
    except Exception:
        installed = []
    installed_by_name = {m["name"]: m for m in installed if m.get("name")}
    catalog_names = {r["model"] for r in rows}
    for r in rows:
        r["installed"] = r["model"] in installed_by_name
    for name, m in installed_by_name.items():
        if name in catalog_names:
            continue
        params = parse_parameter_size(m.get("parameter_size")) or params_from_name(name)
        rows.append(make_fit_row(
            name, params, hw,
            family=(m.get("family") or "installed"),
            notes="installed locally", installed=True))

    if getattr(args, "table", False):
        _render.print_table(rows, _MATRIX_COLUMNS)
        return
    _render.print_json({"hardware": hw, "rows": rows, "tradeoffs": tradeoffs_rows()})


def _fit(args: argparse.Namespace) -> None:
    """Twin of ``GET /hardware/hf_fit``: score a HF repo against the budget."""
    from cgx.answer import ollama_discovery
    from cgx.answer.hardware_matrix import make_fit_row, params_from_name

    repo = (args.repo or "").strip()
    # The route degrades to a 200 with a "reason"; a CLI should exit non-zero
    # on bad input, so this is a validation error (code 3).
    if not _HF_REPO_RE.match(repo):
        _render.die("invalid repo id (expected owner/name)", code=3)

    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}

    params_b = 0.0
    params_source = "unknown"
    pipeline_tag = None
    gated = False
    try:
        spec = _hf_model_spec(repo)
        st = spec.get("safetensors")
        total = st.get("total") if isinstance(st, dict) else None
        if isinstance(total, (int, float)) and total > 0:
            params_b = round(float(total) / 1e9, 2)
            params_source = "safetensors"
        pipeline_tag = spec.get("pipeline_tag") or None
        gated = bool(spec.get("gated"))
    except Exception:
        # Degrade to the size hint in the repo id on any upstream failure,
        # exactly like the route (no raise).
        pass

    if params_b <= 0:
        params_b = params_from_name(repo)
        if params_b > 0:
            params_source = "name"

    row = make_fit_row(repo, params_b, hw, family="huggingface")
    _render.print_json({
        "repo": repo,
        "params_b": params_b,
        "params_source": params_source,
        "min_ram_gb": row["min_ram_gb"],
        "rec_vram_gb": row["rec_vram_gb"],
        "ctx_window": row["ctx_window"],
        "fit": row["fit"],
        "reason": row["reason"],
        "pipeline_tag": pipeline_tag,
        "gated": gated,
        "hardware": hw,
    })


def _hf_model_spec(repo: str) -> dict:
    """Fetch a repo's public Hub metadata (params/gating/pipeline).

    Replicated from ``routes/hardware.py`` (no engine equivalent). ``repo`` is
    pre-validated to ``owner/name`` and the host is a constant, so it can only
    ever be a single path segment (SSRF barrier). Raises on upstream failure.
    """
    import requests

    r = requests.get(f"{_HF_HUB_BASE}/api/models/{repo}", timeout=15)
    r.raise_for_status()
    data = r.json() if r.content else {}
    return data if isinstance(data, dict) else {}


def _embed_list(args: argparse.Namespace) -> None:
    """Twin of ``GET /embed/models``: curated embedders + per-model cache flag."""
    from cgx.embeddings.catalog import EMBED_MODEL_CATALOG, is_cached

    choices = [{
        "name": m.name, "label": m.label, "kind": m.kind, "dim": m.dim,
        "max_tokens": m.max_tokens, "size_gb": m.size_gb,
        "description": m.description, "cached": is_cached(m.name),
    } for m in EMBED_MODEL_CATALOG]

    if getattr(args, "table", False):
        _render.print_table(choices, _EMBED_COLUMNS)
        return
    # recommended_default is a fixed string in the route.
    _render.print_json({
        "choices": choices,
        "recommended_default": "jinaai/jina-embeddings-v2-base-code",
    })


# --------------------------------------------------------------------------- #
# pull verbs (progress -> stderr, final JSON summary -> stdout)
# --------------------------------------------------------------------------- #

def _pull(args: argparse.Namespace) -> None:
    """Twin of ``POST /ollama/pull`` consumed to the terminal.

    The route has no reusable engine generator -- it inlines the streaming
    ``POST {base}/api/pull`` loop -- so this calls the same lower-level pieces:
    the engine's :func:`ollama_discovery.validate_base_url` (SSRF guard) plus a
    streaming ``requests`` call, then the optional /api/copy + /api/delete
    re-alias. Progress NDJSON is rendered to stderr; a JSON summary to stdout.
    """
    import requests
    from cgx.answer import ollama_discovery

    model = (args.model or "").strip()
    if not model:
        _render.die("model is required", code=3)

    # Validate the user-supplied base URL before fetching it (SSRF guard),
    # matching the route: strip a trailing /v1 first so an OpenAI-style URL
    # still resolves to the native Ollama API root.
    raw = (args.base_url or _DEFAULT_BASE_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    try:
        base = ollama_discovery.validate_base_url(raw)
    except ValueError as exc:
        _render.die(f"invalid base_url: {exc}", code=3)

    dest = _sanitize_local_name(args.local_name or "")
    saw_success = False
    saw_error = None
    line_count = 0
    last = None
    try:
        with requests.post(f"{base}/api/pull",
                           json={"model": model, "stream": True},
                           stream=True, timeout=600) as r:
            if r.status_code >= 400:
                body = ""
                try:
                    body = r.text[:300]
                except Exception:
                    pass
                # 404 -> tag not found (code 2); 412/etc -> other error (code 1).
                code = 2 if r.status_code == 404 else 1
                _render.die(
                    f"ollama /api/pull returned HTTP {r.status_code} for "
                    f"model={model!r}" + (f": {body}" if body else ""), code=code)
            for line in r.iter_lines():
                if not line:
                    continue
                line_count += 1
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("status") == "success":
                    saw_success = True
                err_field = parsed.get("error")
                if err_field and saw_error is None:
                    saw_error = str(err_field)[:300]
                last = _emit_progress(_fmt_pull_line(parsed), last)
    except requests.RequestException as exc:
        _render.die(f"ollama pull failed: {type(exc).__name__}: {exc}", code=1)
    finally:
        # Close out the in-place progress line on a TTY.
        if last is not None and sys.stderr.isatty():
            print("", file=sys.stderr)

    if saw_error:
        _render.die(f"ollama pull failed: {saw_error}", code=1)
    if not saw_success:
        _render.die("ollama pull did not report success (incomplete stream)", code=1)

    renamed = None
    if dest and dest != model:
        renamed = _reassign_ollama_tag(requests, base, model, dest)

    _render.print_json({
        "model": renamed or model,
        "pulled": True,
        "lines": line_count,
        "renamed_to": renamed,
    })


def _reassign_ollama_tag(requests, base: str, source: str, dest: str):
    """Best-effort re-alias ``source`` -> ``dest`` via /api/copy + /api/delete.

    Mirrors the route's post-pull rename so an ``hf.co/<repo>`` download no
    longer shows up as a full web address. Any failure leaves the original
    tag in place and returns ``None`` (never turns a good pull into an error).
    """
    try:
        cr = requests.post(f"{base}/api/copy",
                           json={"source": source, "destination": dest}, timeout=60)
        if cr.status_code < 400:
            requests.delete(f"{base}/api/delete", json={"model": source}, timeout=60)
            print(f"re-aliased {source} -> {dest}", file=sys.stderr)
            return dest
        print(f"warning: copy to {dest!r} returned HTTP {cr.status_code}; "
              "keeping original tag", file=sys.stderr)
    except Exception as exc:
        print(f"warning: re-alias to {dest!r} failed "
              f"({type(exc).__name__}); keeping original tag", file=sys.stderr)
    return None


def _embed_pull(args: argparse.Namespace) -> None:
    """Twin of ``POST /embed/pull`` consumed to the terminal.

    The route inlines the HF download loop (no engine generator), so this calls
    the same lower-level pieces: :func:`catalog.find_by_name` plus
    ``huggingface_hub``'s ``HfApi.model_info`` + ``hf_hub_download``. It runs on
    the main thread, so the route's worker-thread SSL warm-up / HF_HUB_OFFLINE
    toggling (both living in ``cgx.webui``) are not needed here.
    """
    from cgx.embeddings.catalog import find_by_name

    model = (args.model or "").strip()
    if not model:
        _render.die("model is required", code=3)
    if not find_by_name(model):
        print(f"note: {model!r} is not in the embedding catalog -- pulling anyway",
              file=sys.stderr)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        _render.die(f"huggingface_hub is not installed: {exc}", code=1)

    api = HfApi()
    print("resolving files...", file=sys.stderr)
    try:
        info = api.model_info(model, files_metadata=True)
    except Exception as exc:
        # A missing/gated repo surfaces as an HTTP error here -> not-found (2).
        _render.die(f"could not resolve {model!r}: {type(exc).__name__}: {exc}", code=2)

    siblings = list(info.siblings or [])
    has_st = any((s.rfilename or "").endswith(".safetensors") for s in siblings)
    files = []
    for s in siblings:
        fname = s.rfilename or ""
        if not fname or fname.endswith(_EMBED_SKIP_EXTS):
            continue
        # Prefer .safetensors over legacy .bin / .pt duplicates.
        if has_st and fname.endswith((".bin", ".pt")):
            continue
        files.append((fname, int(s.size or 0)))

    total_bytes = sum(sz for _, sz in files)
    completed = 0
    print(f"downloading {len(files)} files ({_human_bytes(total_bytes)})...",
          file=sys.stderr)
    for fname, sz in files:
        print(f"pulling {fname}", file=sys.stderr)
        try:
            hf_hub_download(repo_id=model, filename=fname)
        except Exception as exc:
            _render.die(f"download failed for {fname}: "
                        f"{type(exc).__name__}: {exc}", code=1)
        completed += sz
        pct = (100.0 * completed / total_bytes) if total_bytes else 100.0
        print(f"pulled {fname}  {pct:5.1f}%", file=sys.stderr)

    _render.print_json({
        "model": model,
        "files": len(files),
        "bytes": total_bytes,
        "cached": True,
    })


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _sanitize_local_name(raw: str):
    """Coerce ``raw`` into a safe ``name[:tag]`` Ollama name, or ``None``.

    Replicated from ``routes/setup.py`` (no engine equivalent).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    name, _, tag = raw.partition(":")
    if not _OLLAMA_NAME_RE.fullmatch(name):
        return None
    tag = tag or "latest"
    if not _OLLAMA_NAME_RE.fullmatch(tag):
        return None
    return f"{name}:{tag}"


def _fmt_pull_line(d: dict) -> str:
    """Render one Ollama pull NDJSON dict as a human progress line."""
    status = str(d.get("status") or d.get("error") or "").strip() or "..."
    total = d.get("total")
    completed = d.get("completed")
    if (isinstance(total, (int, float)) and total > 0
            and isinstance(completed, (int, float))):
        pct = 100.0 * float(completed) / float(total)
        return (f"{status}  {pct:5.1f}%  "
                f"({_human_bytes(completed)}/{_human_bytes(total)})")
    return status


def _emit_progress(text: str, last):
    """Print ``text`` to stderr; overwrite-in-place on a TTY, de-dup otherwise.

    Returns the last-emitted text so the caller can suppress repeats when
    piped and close the transient line on a TTY.
    """
    if sys.stderr.isatty():
        print("\r\033[2K" + text, end="", file=sys.stderr, flush=True)
    elif text != last:
        print(text, file=sys.stderr, flush=True)
    return text


def _human_bytes(n) -> str:
    """Format a byte count as a compact human-readable string."""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

