#!/usr/bin/env python3
"""
Gradio app with local LLM synthesis (Ollama or OpenAI-compatible endpoint).
Provides a Codebase RAG pipeline with hybrid retrieval (semantic + lexical + graph),
backed by FAISS indices built via `run_index_auto` and queried via `run_query_auto`.

Exposes three main tabs:
1. Index: Build indices from a codebase (AST parse → graph → records → embeddings).
2. Ask (Codebase Q&A): Run hybrid retrieval via HybridRetriever (semantic + lexical + graph).
   Then ground an answer with SOURCES (sometimes LLM, sometimes deterministic).
3. Code Plan: Generate code modification plans with diffs from the indexed codebase.
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
from cgx.answer.engine import _split_chunk_id as _split_chunk_id_engine
from cgx.answer.engine import _read_readme as _read_readme_engine
from cgx.answer.engine import _guess_root as _guess_root_engine
from cgx.answer.engine import SYSTEM as ENGINE_SYSTEM


def ui_build():
    with gr.Blocks(title="Averix") as demo:
        gr.Markdown("# 🧠 Codebase RAG + Smart Q&A (Local or Remote) — with full retrieval debug")

        # ---------------- Index Tab ----------------
        with gr.Tab("Index"):
            project_root = gr.Textbox(label="Project root", placeholder="/path/to/code")
            out_dir = gr.Textbox(label="Output dir", value="/tmp/cgx_index")
            model_name_i = gr.Textbox(
                label="Embedding Model Name", value="jinaai/jina-embeddings-v2-base-code"
            )
            metric = gr.Dropdown(["cosine", "l2", "ip"], value="cosine", label="Metric")
            index_type = gr.Dropdown(["flat", "ivf", "hnsw"], value="flat", label="Index type")
            btn_i = gr.Button("🚀 Run Index")
            status_i = gr.Textbox(label="Status", interactive=False)

        # ---------------- Ask (Codebase Q&A) Tab ----------------
        with gr.Tab("Ask (Codebase Q&A)"):
            index_dir = gr.Textbox(label="Index dir", value="/tmp/cgx_index/indices")
            records = gr.Textbox(label="Records path", value="/tmp/cgx_index/records.jsonl")
            question = gr.Textbox(
                label="Question",
                placeholder='e.g., What does this function "prepare_embedding_corpus" do?',
            )
            model_name_q = gr.Textbox(
                label="Embedding Model Name", value="jinaai/jina-embeddings-v2-base-code"
            )
            with gr.Row():
                prov = gr.Radio(["ollama", "openai-compat"], value="ollama", label="Provider")
                model = gr.Textbox(label="Model", value="qwen2.5:7b-instruct")
                base_url = gr.Textbox(
                    label="Base URL (for openai-compat)", value="http://localhost:11434/v1"
                )
                api_key = gr.Textbox(label="API Key (for openai-compat)", type="password")
            with gr.Row():
                max_chars = gr.Slider(
                    200, 20000, value=4000, step=100, label="Max characters per source (debug view)"
                )
                full_text = gr.Checkbox(value=False, label="Show full chunk text (no truncation)")
            btn_q = gr.Button("💬 Ask")
            answer_md = gr.Markdown(label="Answer")
            prompt_md = gr.Markdown(label="Prompt sent to LLM (preview)")
            intent_json = gr.JSON(label="Debug intent/mode")
            meta = gr.JSON(label="LLM meta (JSON)")
            sources_json = gr.JSON(label="SOURCES passed to LLM (debug)")
            hits_json = gr.JSON(label="Hybrid hits (ranks, scores, provenance)")
            provenance_json = gr.JSON(label="Retrieval provenance per chunk")
            files_json = gr.JSON(label="Top files aggregation")
            classes_json = gr.JSON(label="Top classes aggregation")
            anchors_json = gr.JSON(label="Suggested insertion points")

        # ---------------- Code Plan Tab ----------------
        with gr.Tab("Code Plan (LLM)"):
            index_dir2 = gr.Textbox(label="Index dir", value="/tmp/cgx_index/indices")
            records2 = gr.Textbox(label="Records path", value="/tmp/cgx_index/records.jsonl")
            task = gr.Textbox(label="Task", placeholder="Add CSV export to reports")
            with gr.Row():
                prov2 = gr.Radio(["ollama", "openai-compat"], value="ollama", label="Provider")
                model2 = gr.Textbox(label="Model", value="qwen2.5:7b-instruct")
                base_url2 = gr.Textbox(
                    label="Base URL (for openai-compat)", value="http://localhost:11434/v1"
                )
                api_key2 = gr.Textbox(label="API Key (for openai-compat)", type="password")
            btn_p = gr.Button("🛠️ Generate Plan + Diffs")
            plan_md = gr.Markdown()
            meta2 = gr.JSON()

        # ---------------- Handlers ----------------
        def do_index(project_root, out_dir, model_name, metric, index_type):
            try:
                run_index_auto(
                    project_root=project_root,
                    out_dir=out_dir,
                    metric=metric,
                    index_type=index_type,
                    model_name=model_name,
                )
                return "Index OK (indices/records saved under out_dir)"
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

        def _provider(kind, model, base_url, api_key):
            if kind == "ollama":
                base = (base_url or "http://localhost:11434").replace("/v1", "").rstrip("/")
                return OllamaProvider(model=model, base_url=base)
            else:
                return OpenAICompatProvider(
                    model=model, base_url=(base_url or "").rstrip("/"), api_key=api_key or None
                )

        def _build_sources_from_hits(index_dir, hits, *, max_chars=4000, full_text=False):
            from cgx.io.persist import load_indices
            indices = load_indices(index_dir)
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
                return t if len(t) <= n else t[: n - 3] + "..."

            out = []
            for i, h in enumerate(hits or []):
                if not isinstance(h, dict):
                    raise TypeError(
                        f"_build_sources_from_hits: expected dict in hits, got {type(h)} at index {i}: {h}"
                    )
                cid = str(h.get("chunk_id"))
                row = cmap.get(cid) or {}
                text = row.get("text", "") if isinstance(row, dict) else ""
                path, kind, symbol = _split_chunk_id_engine(cid)
                out.append(
                    {
                        "chunk_id": cid,
                        "path": path,
                        "kind": kind,
                        "symbol": symbol,
                        "text": _trim(text, max_chars),
                        "hit": h,
                    }
                )
            return out


        def _prompt_preview(index_dir, question, sources_dbg):
            from cgx.io.persist import load_indices
            indices = load_indices(index_dir)
            root = _guess_root_engine(indices)
            readme = _read_readme_engine(root)
            q = (question or "").strip()
            user = "QUESTION:\n" + q + "\n\n"
            if readme:
                lead_lines = [ln for ln in readme.splitlines() if ln.strip()][:12]
                user += "README (lead):\n" + "\n".join(lead_lines) + "\n\n"

            def fmt_source(s):
                return f"- {s['chunk_id']} :: {s['path']} :: {s['kind']} :: {s['symbol']}\n  " + (
                    s.get("text", "") or ""
                )

            user += "SOURCES:\n" + "\n".join(fmt_source(s) for s in sources_dbg)
            md = (
                "### SYSTEM\n\n````\n"
                + ENGINE_SYSTEM
                + "\n````\n\n### USER\n\n````\n"
                + user
                + "\n````"
            )
            return md

        def do_answer(
            index_dir,
            records,
            question,
            kind,
            model,
            base_url,
            api_key,
            model_name,
            max_chars=4000,
            full_text=False,
        ):
            try:
                # print("DEBUG: entering do_answer")
                prov_obj = _provider(kind, model, base_url, api_key)
                # print("DEBUG: provider object built ->", type(prov_obj))

                out_dir = Path(index_dir).parent
                chunks_path = str(out_dir / "chunks.jsonl")
                graph_path = str(out_dir / "graph.json")

                # print("DEBUG: calling run_query_auto")
                res = run_query_auto(
                    index_dir=index_dir,
                    records_path=records,
                    query=question,
                    model_name=model_name,
                    chunks_path=chunks_path if os.path.exists(chunks_path) else None,
                    graph_path=graph_path if os.path.exists(graph_path) else None,
                    top_k_per_view=3,
                    neighbor_depth=1,
                    use_lexical=True,
                )
                print("DEBUG: run_query_auto returned keys ->", list(res.keys()))
                print("DEBUG: run_query_auto returned ->", res)
                
                hits = res.get("hits", [])
                files = res.get("top_files", [])
                classes = res.get("top_classes", [])
                anchors = res.get("anchors", [])

                # print("DEBUG: hits type ->", type(hits), "len ->", len(hits))
                # if hits:
                #     print("DEBUG: first hit sample ->", hits[0], "type ->", type(hits[0]))
                # print("DEBUG: files type ->", type(files), "len ->", len(files))
                # print("DEBUG: classes type ->", type(classes), "len ->", len(classes))
                # print("DEBUG: anchors type ->", type(anchors), "len ->", len(anchors))

                provenance = []
                for i, h in enumerate(hits):
                    # print(f"DEBUG: provenance loop idx={i}, type(h)={type(h)}")
                    if not isinstance(h, dict):
                        raise TypeError(f"do_answer: expected dict in hits, got {type(h)} at index {i}: {h}")
                    provenance.append({
                        "chunk_id": h.get("chunk_id"),
                        "provenance": h.get("provenance", {})
                    })

                # print("DEBUG: provenance built ->", provenance[:2])

                sources_dbg = _build_sources_from_hits(
                    index_dir, hits, max_chars=max_chars, full_text=full_text
                )
                # print("DEBUG: sources_dbg ->", sources_dbg[:2])

                prompt = _prompt_preview(index_dir, question, sources_dbg)
                # print("DEBUG: prompt built, length ->", len(prompt))

                out = answer_with_llm(index_dir, records, question, prov_obj, hits=hits)
                # print("DEBUG: answer_with_llm out keys ->", list(out.keys()) if isinstance(out, dict) else type(out))

                ans = out.get("answer_md", "")
                if not isinstance(ans, str):
                    if isinstance(ans, dict):
                        ans = (
                            ans.get("content")
                            or ans.get("text")
                            or ans.get("markdown")
                            or ans.get("md")
                            or json.dumps(ans, ensure_ascii=False)
                        )
                    elif isinstance(ans, list):
                        ans = "\n".join(str(x) for x in ans)
                    else:
                        ans = str(ans)

                if not prompt:
                    prompt = "(No LLM prompt used — deterministic answer)"

                def _json_safe(obj):
                    try:
                        json.dumps(obj)  # quick test
                        return obj
                    except TypeError:
                        if isinstance(obj, dict):
                            return {k: _json_safe(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [_json_safe(x) for x in obj]
                        elif hasattr(obj, "item"):  # numpy scalar
                            return obj.item()
                        return str(obj)

                # print("DEBUG: sanitizing outputs")
                hits = _json_safe(hits)
                files = _json_safe(files)
                classes = _json_safe(classes)
                anchors = _json_safe(anchors)
                sources_dbg = _json_safe(sources_dbg)
                provenance = _json_safe(provenance)

                debug = out.get("debug", {}) if isinstance(out, dict) else {}
                if not isinstance(debug, dict):
                    debug = {}
                intent_info = {"mode": debug.get("mode")}

                # print("DEBUG: preparing return tuple")
                return (
                    ans,
                    prompt,
                    out,
                    intent_info,
                    sources_dbg,
                    hits,
                    provenance,
                    files,
                    classes,
                    anchors,
                )

            except Exception as e:
                print("DEBUG: exception caught in do_answer ->", repr(e))
                err = {"error": f"{type(e).__name__}: {e}"}
                return (
                    f"ERROR: {type(e).__name__}: {e}",  # answer_md
                    "",                                # prompt_md
                    err,                               # meta JSON
                    err,                               # intent JSON
                    [],                                # sources_json
                    [],                                # hits_json
                    [],                                # provenance_json
                    [],                                # files_json
                    [],                                # classes_json
                    [],                                # anchors_json
                )

        def do_plan(index_dir, records, task, kind, model, base_url, api_key):
            try:
                prov_obj = _provider(kind, model, base_url, api_key)
                out = generate_code_plan(index_dir, records, task, prov_obj)
                plan = out.get("plan_md", "")
                if not isinstance(plan, str):
                    if isinstance(plan, dict):
                        plan = (
                            plan.get("content")
                            or plan.get("text")
                            or plan.get("markdown")
                            or plan.get("md")
                            or json.dumps(plan, ensure_ascii=False)
                        )
                    elif isinstance(plan, list):
                        plan = "\n".join(str(x) for x in plan)
                    else:
                        plan = str(plan)
                return plan, out
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}", {"error": str(e)}

        # ---------------- Wire events ----------------
        btn_i.click(
            do_index,
            inputs=[project_root, out_dir, model_name_i, metric, index_type],
            outputs=[status_i],
        )
        btn_q.click(
            do_answer,
            inputs=[
                index_dir,
                records,
                question,
                prov,
                model,
                base_url,
                api_key,
                model_name_q,
                max_chars,
                full_text,
            ],
            outputs=[
                answer_md,
                prompt_md,
                meta,
                intent_json,
                sources_json,
                hits_json,
                provenance_json,
                files_json,
                classes_json,
                anchors_json,
            ],
        )
        btn_p.click(
            do_plan,
            inputs=[index_dir2, records2, task, prov2, model2, base_url2, api_key2],
            outputs=[plan_md, meta2],
        )

        gr.Markdown(
            "> Tip: For **ollama**, set Base URL to `http://localhost:11434/v1` or leave default; "
            "for **openai-compat**, set the real endpoint and key."
        )
    return demo


if __name__ == "__main__":
    ui_build().launch()
