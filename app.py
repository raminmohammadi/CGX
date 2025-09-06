#!/usr/bin/env python3
"""
Gradio Frontend for Codebase RAG
================================

What this app lets you do
-------------------------
- **Index a codebase** into a two-view (intent/impl) FAISS index
- **Ask questions** about the codebase in natural language
- **See likely files/classes** to modify and **suggested insertion points**
- Optionally include lexical and graph expansion signals
- Optionally run a single-view semantic helper for debugging

How to run
----------
1) Install deps (ensure your project deps + gradio are installed)
   pip install -r requirements.txt
   pip install gradio

2) Make the src/ layout importable
   export PYTHONPATH="$PWD/src:$PYTHONPATH"

3) Start the app
   python app_gradio.py

Bring-Your-Own Embedder
-----------------------
Pass an import spec "module:attr" that yields an object with a method:
    encode(list[str]) -> numpy.ndarray
The app will import it and instantiate/call as needed.
"""

import os
import sys
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

# Ensure this repo's src/ is importable for cgx.*
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.cgx.pipeline.auto import run_index_auto, run_query_auto


def _load_embedder(spec: str) -> Any:
    """
    Load an object or factory from "module:attr".
    - If it's a class: instantiate with no args.
    - If it's a callable (factory) without .encode: call it to get the object.
    - If it's an object: return as-is.
    The object must expose `.encode(list[str]) -> ndarray` when used.
    """
    import importlib
    import inspect

    if not spec or ":" not in spec:
        raise ValueError('Embedder spec must be "module:attr" (got %r)' % spec)
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if inspect.isclass(obj):
        return obj()
    if callable(obj) and not hasattr(obj, "encode"):
        return obj()  # factory
    return obj


def _maybe_extract_zip(upload) -> Optional[str]:
    """
    If a zip file was uploaded, extract it and return the extracted root path.
    Otherwise return None.
    """
    if not upload:
        return None
    try:
        tmpdir = tempfile.mkdtemp(prefix="cgx_zip_")
        with zipfile.ZipFile(upload.name, "r") as zf:
            zf.extractall(tmpdir)
        # Heuristic: if there's a single top-level directory, use that
        entries = [p for p in Path(tmpdir).iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            return str(entries[0])
        return tmpdir
    except Exception as e:
        raise RuntimeError(f"Failed to extract zip: {type(e).__name__}: {e}")


def do_index(project_root: str,
             embedder_spec: str,
             out_dir: str,
             metric: str,
             index_type: str,
             code_zip) -> tuple[str, dict]:
    """
    Index handler for Gradio.
    - If code_zip is provided, it takes precedence and project_root is ignored.
    - Returns a status string and the summary JSON.
    """
    try:
        if code_zip is not None:
            project_root = _maybe_extract_zip(code_zip)
        if not project_root or not os.path.exists(project_root):
            raise ValueError(f"project_root not found: {project_root!r}")
        os.makedirs(out_dir, exist_ok=True)

        embedder = _load_embedder(embedder_spec)
        summary = run_index_auto(
            project_root=project_root,
            embedder=embedder,
            out_dir=out_dir,
            metric=metric,
            index_type=index_type,
        )
        status = f"Index OK. Root={project_root}  -> out_dir={out_dir}"
        return status, summary
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}", {}


def do_query(index_dir: str,
             records_path: str,
             embedder_spec: str,
             query: str,
             top_k: int,
             depth: int,
             no_lexical: bool,
             single_view: str | None,
             chunks_jsonl,
             graph_json) -> tuple[str, dict, list, list, list]:
    """
    Query handler for Gradio.
    - Returns a status string, raw result JSON, and tables for top_files/classes/anchors.
    """
    try:
        embedder = _load_embedder(embedder_spec)
        chunks_path = chunks_jsonl.name if chunks_jsonl else None
        graph_path = graph_json.name if graph_json else None

        res = run_query_auto(
            index_dir=index_dir,
            records_path=records_path,
            embedder=embedder,
            query=query,
            chunks_path=chunks_path,
            graph_path=graph_path,
            top_k_per_view=int(top_k),
            neighbor_depth=int(depth),
            use_lexical=(not no_lexical),
            single_view=(single_view or None),
        )

        # Normalize tabular outputs if present
        def _to_rows(obj):
            if isinstance(obj, list):
                if all(isinstance(x, dict) for x in obj):
                    # return headers + rows for Gradio Dataframe
                    if not obj:
                        return []
                    keys = sorted({k for d in obj for k in d.keys()})
                    rows = [[d.get(k, "") for k in keys] for d in obj]
                    return [keys] + rows
                else:
                    return [["value"]] + [[str(x)] for x in obj]
            return []

        files_tbl = _to_rows(res.get("top_files", []))
        classes_tbl = _to_rows(res.get("top_classes", []))
        anchors_tbl = _to_rows(res.get("anchors", []))

        return "Query OK.", res, files_tbl, classes_tbl, anchors_tbl
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}", {}, [], [], []


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Codebase RAG", theme="default") as demo:
        gr.Markdown("# 🧭 Codebase RAG — Index • Ask • Insert")
        gr.Markdown(
            "Index your repository, ask NL questions, and get grounded suggestions "
            "for where to add new code — with insertion anchors."
        )

        with gr.Tab("Index"):
            with gr.Row():
                code_zip = gr.File(label="(Optional) Upload codebase .zip", file_types=[".zip"])
            with gr.Row():
                project_root = gr.Textbox(label="Project root (ignored if zip uploaded)", placeholder="/path/to/your/codebase")
                out_dir = gr.Textbox(label="Output directory", value="/tmp/cgx_index")
            with gr.Row():
                embedder = gr.Textbox(label='Embedder import spec "module:attr"', placeholder='myproj.embed:make_model')
            with gr.Row():
                metric = gr.Dropdown(label="Metric", choices=["cosine", "l2", "ip"], value="cosine")
                index_type = gr.Dropdown(label="Index type", choices=["flat", "ivf", "hnsw"], value="flat")
            run_index_btn = gr.Button("🚀 Run Index", variant="primary")
            index_status = gr.Textbox(label="Status", interactive=False)
            index_json = gr.JSON(label="Index summary")

        with gr.Tab("Query"):
            with gr.Row():
                index_dir = gr.Textbox(label="Index dir (from index step)", value="/tmp/cgx_index/indices")
                records = gr.Textbox(label="Records path (from index step)", value="/tmp/cgx_index/records.jsonl")
            with gr.Row():
                embedder_q = gr.Textbox(label='Embedder import spec "module:attr"', placeholder='myproj.embed:make_model')
                query = gr.Textbox(label="Your question", placeholder="Where do I add CSV export?")
            with gr.Row():
                top_k = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Top-K per view")
                depth = gr.Slider(minimum=0, maximum=3, value=1, step=1, label="Graph neighbor depth")
                no_lexical = gr.Checkbox(label="Disable lexical", value=False)
                single_view = gr.Dropdown(label="Also run single-view helper", choices=[None, "intent", "impl"], value=None)
            with gr.Row():
                chunks_jsonl = gr.File(label="(Optional) chunks.jsonl for lexical", file_types=[".jsonl", ".json"])
                graph_json = gr.File(label="(Optional) graph.json for graph expansion", file_types=[".json"])
            run_query_btn = gr.Button("🔎 Search", variant="primary")
            query_status = gr.Textbox(label="Status", interactive=False)
            res_json = gr.JSON(label="Raw results (JSON)")

            gr.Markdown("### Aggregations & Anchors")
            top_files_tbl = gr.Dataframe(label="Top Files (aggregated)", interactive=False)
            top_classes_tbl = gr.Dataframe(label="Top Classes (aggregated)", interactive=False)
            anchors_tbl = gr.Dataframe(label="Suggested Insertion Anchors", interactive=False)

        # Events
        run_index_btn.click(
            fn=do_index,
            inputs=[project_root, embedder, out_dir, metric, index_type, code_zip],
            outputs=[index_status, index_json],
        )

        run_query_btn.click(
            fn=do_query,
            inputs=[index_dir, records, embedder_q, query, top_k, depth, no_lexical, single_view, chunks_jsonl, graph_json],
            outputs=[query_status, res_json, top_files_tbl, top_classes_tbl, anchors_tbl],
        )

        gr.Markdown(
            "> Pro tip: Use the same embedder for **query** as used for **index**. "
            "Set `PYTHONPATH` to include your repo's `src/` to import cgx.* internally."
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # You can pass share=True for quick remote testing in notebooks.
    ui.launch()