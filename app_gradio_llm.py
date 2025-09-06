#!/usr/bin/env python3
"""
Gradio app with local LLM synthesis (Ollama or OpenAI-compatible endpoint).
Now exposes full retrieval debug (hits with fusion/provenance) and the exact SOURCES/prompt.
"""
import os, sys, json
from pathlib import Path
import gradio as gr

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.answer.providers import OllamaProvider, OpenAICompatProvider
from cgx.answer.engine import answer_with_llm, generate_code_plan
from cgx.answer.engine import _split_chunk_id as _split_chunk_id_engine  # reuse
from cgx.answer.engine import _read_readme as _read_readme_engine        # reuse
from cgx.answer.engine import _guess_root as _guess_root_engine          # reuse
from cgx.answer.engine import SYSTEM as ENGINE_SYSTEM                    # reuse system text

def ui_build():
    with gr.Blocks(title="Codebase RAG + LLM") as demo:
        gr.Markdown("# 🧠 Codebase RAG + LLM (Local or Remote) — with full retrieval debug")

        # ---------------- Index Tab ----------------
        with gr.Tab("Index"):
            project_root = gr.Textbox(label="Project root", placeholder="/path/to/code")
            out_dir = gr.Textbox(label="Output dir", value="/tmp/cgx_index")
            embedder = gr.Textbox(label='Embedder "module:attr"', value="st_embedder:make_model")
            metric = gr.Dropdown(["cosine","l2","ip"], value="cosine", label="Metric")
            index_type = gr.Dropdown(["flat","ivf","hnsw"], value="flat", label="Index type")
            btn_i = gr.Button("🚀 Run Index")
            status_i = gr.Textbox(label="Status", interactive=False)

        # ---------------- Ask (LLM) Tab ----------------
        with gr.Tab("Ask (LLM)"):
            index_dir = gr.Textbox(label="Index dir", value="/tmp/cgx_index/indices")
            records = gr.Textbox(label="Records path", value="/tmp/cgx_index/records.jsonl")
            question = gr.Textbox(label='Question', placeholder='e.g., What does this function "prepare_embedding_corpus" do?')
            embedder_q = gr.Textbox(label='Embedder "module:attr" (same as index)', value="st_embedder:make_model")
            with gr.Row():
                prov = gr.Radio(["ollama","openai-compat"], value="ollama", label="Provider")
                model = gr.Textbox(label="Model", value="qwen2.5:7b-instruct")
                base_url = gr.Textbox(label="Base URL (for openai-compat)", value="http://localhost:11434/v1")
                api_key = gr.Textbox(label="API Key (for openai-compat)", type="password")
            with gr.Row():
                max_chars = gr.Slider(200, 20000, value=4000, step=100, label="Max characters per source (debug view)")
                full_text = gr.Checkbox(value=False, label="Show full chunk text (no truncation)")
            btn_q = gr.Button("💬 Answer with LLM")
            answer_md = gr.Markdown(label="Answer")
            prompt_md = gr.Markdown(label="Prompt sent to LLM (preview)")
            meta = gr.JSON(label="LLM meta (JSON)")
            sources_json = gr.JSON(label="SOURCES passed to LLM (debug)")
            hits_json = gr.JSON(label="Hybrid hits (raw scores & views)")
            files_json = gr.JSON(label="Top files aggregation")
            classes_json = gr.JSON(label="Top classes aggregation")
            anchors_json = gr.JSON(label="Suggested insertion points")

        # ---------------- Code Plan Tab ----------------
        with gr.Tab("Code Plan (LLM)"):
            index_dir2 = gr.Textbox(label="Index dir", value="/tmp/cgx_index/indices")
            records2 = gr.Textbox(label="Records path", value="/tmp/cgx_index/records.jsonl")
            task = gr.Textbox(label="Task", placeholder="Add CSV export to reports")
            with gr.Row():
                prov2 = gr.Radio(["ollama","openai-compat"], value="ollama", label="Provider")
                model2 = gr.Textbox(label="Model", value="qwen2.5:7b-instruct")
                base_url2 = gr.Textbox(label="Base URL (for openai-compat)", value="http://localhost:11434/v1")
                api_key2 = gr.Textbox(label="API Key (for openai-compat)", type="password")
            btn_p = gr.Button("🛠️ Generate Plan + Diffs")
            plan_md = gr.Markdown()
            meta2 = gr.JSON()

        # ---------------- Handlers ----------------
        def do_index(project_root, embedder, out_dir, metric, index_type):
            try:
                # Ensure deps are present
                from cgx.answer.providers import requests  # noqa: F401
                import numpy as np  # noqa: F401

                # load embedder spec dynamically
                import importlib, inspect
                mod_name, attr = embedder.split(":", 1)
                obj = getattr(importlib.import_module(mod_name), attr)
                if inspect.isclass(obj):
                    model_obj = obj()
                elif callable(obj) and not hasattr(obj, "encode"):
                    model_obj = obj()  # factory
                else:
                    model_obj = obj     # pre-instantiated
                run_index_auto(project_root, model_obj, out_dir, metric=metric, index_type=index_type)
                return "Index OK (indices/records saved under out_dir)"
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

        def _provider(kind, model, base_url, api_key):
            if kind == "ollama":
                base = (base_url or "http://localhost:11434").replace("/v1","").rstrip("/")
                return OllamaProvider(model=model, base_url=base)
            else:
                return OpenAICompatProvider(model=model, base_url=(base_url or "").rstrip("/"), api_key=api_key or None)

        def _build_sources_from_hits(index_dir, hits, *, max_chars=4000, full_text=False):
            # Re-create SOURCES and include full hit dict for provenance transparency.
            from cgx.io.persist import load_indices
            indices = load_indices(index_dir)
            # chunk map
            cmap = {}
            for name in ["intent", "impl"]:
                vw = (indices.get("views") or {}).get(name) or {}
                for r in (vw.get("rows") or []):
                    cid = r.get("chunk_id")
                    if cid:
                        cmap[str(cid)] = r

            def _trim(txt, n):
                if full_text:
                    return str(txt or "")
                if txt is None: 
                    return ""
                t = str(txt)
                return t if len(t) <= n else t[:n-3] + "..."

            out = []
            for h in (hits or []):
                cid = str(h.get("chunk_id"))
                row = cmap.get(cid) or {}
                text = row.get("text", "") if isinstance(row, dict) else ""
                path, kind, symbol = _split_chunk_id_engine(cid)
                out.append({
                    "chunk_id": cid,
                    "path": path,
                    "kind": kind,
                    "symbol": symbol,
                    "text": _trim(text, max_chars),
                    "hit": h,  # full provenance incl. per-view ranks/scores/rrf/etc if present
                })
            return out

        def _prompt_preview(index_dir, question, sources_dbg):
            # Build the exact USER content that the engine sends (minus minor variations).
            from cgx.io.persist import load_indices
            indices = load_indices(index_dir)
            root = _guess_root_engine(indices)
            readme = _read_readme_engine(root)
            q = (question or "").strip()
            user = "QUESTION:\n" + q + "\n\n"
            # We only include README lead when not a symbol_explain; here we cannot perfectly route without engine, but preview both:
            if readme:
                lead_lines = [ln for ln in readme.splitlines() if ln.strip()][:12]
                user += "README (lead):\n" + "\n".join(lead_lines) + "\n\n"
            def fmt_source(s):
                return f"- {s['chunk_id']} :: {s['path']} :: {s['kind']} :: {s['symbol']}\n  " + (s.get("text","") or "")
            user += "SOURCES:\n" + "\n".join(fmt_source(s) for s in sources_dbg)
            # System is taken directly from engine.SYSTEm; show both parts
            md = "### SYSTEM\n\n````\n" + ENGINE_SYSTEM + "\n````\n\n### USER\n\n````\n" + user + "\n````"
            return md

        def do_answer(index_dir, records, question, kind, model, base_url, api_key, embedder_spec=None, max_chars=4000, full_text=False):
            try:
                prov_obj = _provider(kind, model, base_url, api_key)

                # Build embedder for retrieval
                emb = None
                if embedder_spec:
                    import importlib, inspect
                    mod_name, attr = embedder_spec.split(":", 1)
                    obj = getattr(importlib.import_module(mod_name), attr)
                    if inspect.isclass(obj): emb = obj()
                    elif callable(obj) and not hasattr(obj, 'encode'): emb = obj()
                    else: emb = obj

                # Infer sibling artifact paths (optional)
                out_dir = Path(index_dir).parent
                chunks_path = str(out_dir / "chunks.jsonl")
                graph_path  = str(out_dir / "graph.json")

                # Run hybrid retrieval with lexical + graph (if artifacts exist)
                res = run_query_auto(
                    index_dir=index_dir,
                    records_path=records,
                    embedder=emb,
                    query=question,
                    chunks_path=chunks_path if os.path.exists(chunks_path) else None,
                    graph_path=graph_path  if os.path.exists(graph_path)  else None,
                    top_k_per_view=12,
                    neighbor_depth=1,
                    use_lexical=True,
                )
                hits = res.get('hits', [])
                files = res.get('top_files', [])
                classes = res.get('top_classes', [])
                anchors = res.get('anchors', [])

                # Build SOURCES exactly as used
                sources_dbg = _build_sources_from_hits(index_dir, hits, max_chars=max_chars, full_text=full_text)

                # Prompt preview
                prompt = _prompt_preview(index_dir, question, sources_dbg)

                # Ask LLM with grounded SOURCES
                out = answer_with_llm(index_dir, records, question, prov_obj, hits=hits)

                # Normalize for Markdown component
                ans = out.get("answer_md", "")
                if not isinstance(ans, str):
                    if isinstance(ans, dict):
                        ans = ans.get("content") or ans.get("text") or ans.get("markdown") or ans.get("md") or json.dumps(ans, ensure_ascii=False)
                    elif isinstance(ans, list):
                        ans = "\n".join(str(x) for x in ans)
                    else:
                        ans = str(ans)
                return ans, prompt, out, sources_dbg, hits, files, classes, anchors
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}", "", {"error": str(e)}, [], [], [], []

        def do_plan(index_dir, records, task, kind, model, base_url, api_key):
            try:
                prov_obj = _provider(kind, model, base_url, api_key)
                out = generate_code_plan(index_dir, records, task, prov_obj)
                plan = out.get("plan_md","")
                if not isinstance(plan, str):
                    if isinstance(plan, dict):
                        plan = plan.get("content") or plan.get("text") or plan.get("markdown") or plan.get("md") or json.dumps(plan, ensure_ascii=False)
                    elif isinstance(plan, list):
                        plan = "\n".join(str(x) for x in plan)
                    else:
                        plan = str(plan)
                return plan, out
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}", {"error": str(e)}

        # ---------------- Wire events ----------------
        btn_i.click(do_index, inputs=[project_root, embedder, out_dir, metric, index_type], outputs=[status_i])
        btn_q.click(
            do_answer,
            inputs=[index_dir, records, question, prov, model, base_url, api_key, embedder_q, max_chars, full_text],
            outputs=[answer_md, prompt_md, meta, sources_json, hits_json, files_json, classes_json, anchors_json],
        )
        btn_p.click(do_plan, inputs=[index_dir2, records2, task, prov2, model2, base_url2, api_key2], outputs=[plan_md, meta2])

        gr.Markdown("> Tip: For **ollama**, set Base URL to `http://localhost:11434/v1` or leave default; for **openai-compat**, set the real endpoint and key.")
    return demo

if __name__ == "__main__":
    ui_build().launch()