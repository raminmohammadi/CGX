"""Regenerate the diagrams embedded in ``docs/book.md``.

Each figure is a clean, technical box-and-arrow diagram (not decorative art)
that mirrors the chapter it illustrates. Run with::

    python docs/generate_book_images.py

Output PNGs land in ``docs/images/`` (image.png, chapter_1..7.png). The style
matches the dark, muted palette used by the mermaid diagrams in
``docs/flowcharts.md`` so the two documents read as one set.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")

# ---- palette (muted slate + accents; light ink reads on the dark panel) ----
BG = "#0f1729"
INK = "#e6ecf7"
MUTED = "#93a3ba"
BOX = "#1b2740"
SKY = "#38bdf8"
VIOLET = "#a78bfa"
EMERALD = "#34d399"
AMBER = "#fbbf24"
ROSE = "#fb7185"
SLATE = "#8296b1"

plt.rcParams["font.family"] = "DejaVu Sans"


def _fig():
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, title, subs=None, accent=SKY, fill=BOX, ts=13, ss=10):
    """Rounded box centered at (cx, cy) with a title and optional sub-lines."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.6,rounding_size=2.2",
            linewidth=2, edgecolor=accent, facecolor=fill, zorder=2,
        )
    )
    subs = subs or []
    if subs:
        ax.text(cx, cy + h / 2 - 4.4, title, ha="center", va="center",
                color=INK, fontsize=ts, fontweight="bold", zorder=3)
        for i, line in enumerate(subs):
            ax.text(cx, cy + h / 2 - 9.4 - i * 4.4, line, ha="center",
                    va="center", color=MUTED, fontsize=ss, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center", color=INK,
                fontsize=ts, fontweight="bold", zorder=3)


def arrow(ax, p1, p2, color=SLATE, lw=2.2, ls="-", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=18, linewidth=lw,
            color=color, linestyle=ls, zorder=1,
            connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        )
    )


def title(ax, text, sub=None):
    ax.text(6, 84, text, ha="left", va="center", color=INK, fontsize=17,
            fontweight="bold")
    if sub:
        ax.text(6, 78.5, sub, ha="left", va="center", color=MUTED, fontsize=11)


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), facecolor=BG,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_hero():
    """image.png -- one request's journey end to end."""
    fig, ax = _fig()
    title(ax, "One request, end to end",
          "repo  ->  index  ->  retrieve  ->  prompt  ->  model  ->  tested patch on disk")
    top_y, bot_y = 60, 24
    box(ax, 22, top_y, 30, 20, "Repository",
        ["source tree", ".gitignore aware"], accent=EMERALD)
    box(ax, 58, top_y, 30, 20, "Index",
        ["chunks + graph", "two-view embeddings"], accent=SKY)
    box(ax, 94, top_y, 30, 20, "Retrieve",
        ["semantic + lexical", "+ graph -> RRF"], accent=SKY)
    box(ax, 130, top_y, 30, 20, "Prompt",
        ["tiered Code Map", "budgeted to window"], accent=VIOLET)
    box(ax, 130, bot_y, 30, 20, "Model",
        ["Ollama (local)", "or opt-in cloud"], accent=AMBER)
    box(ax, 86, bot_y, 30, 20, "Validate & test",
        ["parse diffs, syntax", "sandbox pytest"], accent=VIOLET)
    box(ax, 42, bot_y, 30, 20, "Apply to disk",
        ["backups + undo", ".cgx-backups/"], accent=EMERALD)
    arrow(ax, (37, top_y), (43, top_y))
    arrow(ax, (73, top_y), (79, top_y))
    arrow(ax, (109, top_y), (115, top_y))
    arrow(ax, (130, top_y - 10), (130, bot_y + 10), color=VIOLET)
    arrow(ax, (115, bot_y), (101, bot_y), color=AMBER)
    arrow(ax, (71, bot_y), (57, bot_y))
    save(fig, "image.png")


def fig_ch1():
    """chapter_1.png -- the local-first trust boundary."""
    fig, ax = _fig()
    title(ax, "Local-first by default",
          "everything runs on your machine; the cloud is strictly opt-in")
    ax.add_patch(FancyBboxPatch((6, 12), 104, 64,
                 boxstyle="round,pad=0.6,rounding_size=3", linewidth=2.4,
                 edgecolor=EMERALD, facecolor="#12241d", zorder=0))
    ax.text(12, 71, "YOUR MACHINE  -  no network required", color=EMERALD,
            fontsize=12, fontweight="bold", ha="left", va="center")
    for cx, cy, t in [
        (30, 55, "Parse\n& graph"), (58, 55, "Embed\n& index"),
        (86, 55, "Hybrid\nretrieval"), (30, 30, "Local LLM\n(Ollama)"),
        (58, 30, "Codegen\n& tests"), (86, 30, "Apply\n+ backups"),
    ]:
        box(ax, cx, cy, 22, 15, t, accent=SKY, ts=11)
    box(ax, 138, 55, 34, 20, "Cloud LLM",
        ["opt-in only", "prompt + snippets", "sent per turn"], accent=VIOLET,
        fill="#1e1b3a")
    arrow(ax, (97, 40), (122, 52), color=VIOLET, ls="--", rad=-0.15)
    ax.text(120, 33, "the repo, index &\nsessions never leave", color=MUTED,
            fontsize=9.5, ha="center", va="center")
    save(fig, "chapter_1.png")


def fig_ch2():
    """chapter_2.png -- from repo to records."""
    fig, ax = _fig()
    title(ax, "From repo to records",
          "extension-dispatched parsers -> three-tier chunks -> graph + two-view corpus")
    box(ax, 20, 50, 28, 22, "Files",
        ["walk tree", "size cap + ignores"], accent=EMERALD)
    box(ax, 58, 50, 34, 30, "Parser registry",
        ["Python (ast)  always", "Markdown  always",
         "JS/TS/TSX  tree-sitter", "incremental cache"], accent=SKY, ss=9.5)
    box(ax, 104, 66, 34, 18, "Chunks (3-tier)",
        ["file / class / function"], accent=VIOLET)
    box(ax, 104, 40, 34, 18, "Knowledge graph",
        ["calls / module", "attr / defined_in"], accent=VIOLET)
    box(ax, 146, 40, 26, 26, "Two views",
        ["intent + impl", "FAISS x2", "+ .npz cache"], accent=AMBER, ts=11)
    arrow(ax, (34, 50), (40, 50))
    arrow(ax, (75, 55), (86, 63), rad=-0.1)
    arrow(ax, (75, 45), (86, 41), rad=0.1)
    arrow(ax, (121, 40), (132, 40))
    save(fig, "chapter_2.png")


def fig_ch3():
    """chapter_3.png -- hybrid retrieval."""
    fig, ax = _fig()
    title(ax, "The retrieval pipeline",
          "three retrievers run in parallel, fused by Reciprocal Rank Fusion")
    box(ax, 20, 45, 24, 16, "Query", accent=EMERALD)
    box(ax, 62, 66, 34, 16, "Semantic",
        ["intent + impl FAISS"], accent=SKY, ts=12)
    box(ax, 62, 45, 34, 16, "Lexical (BM25)",
        ["exact-symbol recall"], accent=SKY, ts=12)
    box(ax, 62, 24, 34, 16, "Graph expansion",
        ["callers / classmates"], accent=SKY, ts=12)
    box(ax, 108, 45, 24, 22, "RRF fuse",
        ["1/(k+rank)", "rank-based"], accent=VIOLET)
    box(ax, 146, 45, 24, 22, "Rerank",
        ["cross-encoder", "cloud: auto-on"], accent=AMBER, ts=11)
    for y in (66, 45, 24):
        arrow(ax, (32, 45), (45, y), rad=0.0 if y == 45 else (0.12 if y > 45 else -0.12))
        arrow(ax, (79, y), (96, 45), rad=0.0 if y == 45 else (-0.12 if y > 45 else 0.12))
    arrow(ax, (120, 45), (134, 45), color=VIOLET)
    save(fig, "chapter_3.png")


def fig_ch4():
    """chapter_4.png -- the tiered Code Map."""
    fig, ax = _fig()
    title(ax, "Assembling the prompt",
          "graph_depth splits hits into full-body primaries and one-line neighbours")
    box(ax, 24, 45, 30, 20, "Ranked hits",
        ["carry graph_depth"], accent=EMERALD)
    box(ax, 78, 64, 40, 20, "Primary tier",
        ["graph_depth == 0", "full body (focus-windowed)"], accent=SKY, ss=9.5)
    box(ax, 78, 26, 40, 20, "Neighbour tier",
        ["graph_depth >= 1", "name(sig) -- docstring stub"], accent=VIOLET, ss=9.5)
    box(ax, 134, 45, 30, 24, "Budget",
        ["model_caps by window", "primaries + neighbours", "deterministic order"],
        accent=AMBER, ts=12, ss=9.5)
    arrow(ax, (39, 45), (57, 62), rad=-0.12)
    arrow(ax, (39, 45), (57, 28), rad=0.12)
    arrow(ax, (98, 62), (120, 48), rad=-0.12, color=SKY)
    arrow(ax, (98, 28), (120, 42), rad=0.12, color=VIOLET)
    save(fig, "chapter_4.png")


def fig_ch5():
    """chapter_5.png -- providers behind one interface."""
    fig, ax = _fig()
    title(ax, "Talking to the model",
          "one chat() interface; rate-limit and profile resolution wrap every call")
    box(ax, 24, 45, 26, 18, "Engine", ["answer / plan"], accent=EMERALD)
    box(ax, 66, 45, 30, 22, "LLMProvider",
        ["chat(messages)", "provider-agnostic"], accent=VIOLET)
    box(ax, 118, 66, 34, 14, "OllamaProvider", ["local, loopback"], accent=SKY, ts=11)
    box(ax, 118, 45, 34, 14, "OpenAICompat", ["llama.cpp / vLLM / cloud"], accent=SKY, ts=11)
    box(ax, 118, 24, 34, 14, "GeminiProvider", ["Google REST API"], accent=SKY, ts=11)
    ax.text(66, 24, "rate limit + retry\n+ profile secrets", color=MUTED,
            fontsize=9.5, ha="center", va="center")
    ax.add_patch(FancyBboxPatch((49, 15), 34, 44,
                 boxstyle="round,pad=0.4,rounding_size=2", linewidth=1.4,
                 edgecolor=SLATE, facecolor="none", linestyle="--", zorder=0))
    arrow(ax, (37, 45), (51, 45))
    for y in (66, 45, 24):
        arrow(ax, (81, 45), (101, y), rad=0.0 if y == 45 else (0.1 if y > 45 else -0.1))
    save(fig, "chapter_5.png")


def fig_ch6():
    """chapter_6.png -- validate, test, and write to disk."""
    fig, ax = _fig()
    title(ax, "Writing to disk",
          "diffs are validated and tested in a sandbox before the real tree is touched")
    steps = [
        (20, "Plan + diffs", ["fenced blocks"], VIOLET),
        (54, "Apply in memory", ["+ syntax validate"], SKY),
        (90, "Preflight install", ["map imports->PyPI"], SKY),
        (126, "Sandbox tests", ["impact-aware pytest"], SKY),
    ]
    for cx, t, s, c in steps:
        box(ax, cx, 60, 30, 18, t, s, accent=c, ts=12, ss=9.5)
    for a, b in zip(steps, steps[1:]):
        arrow(ax, (a[0] + 15, 60), (b[0] - 15, 60))
    box(ax, 126, 30, 30, 18, "CodegenReport",
        ["overall_ok gate"], accent=AMBER, ts=12)
    box(ax, 66, 30, 40, 20, "Disk apply",
        [".cgx-backups mirror", "rollback / undo"], accent=EMERALD)
    arrow(ax, (126, 51), (126, 39), color=AMBER)
    arrow(ax, (111, 30), (87, 30), color=EMERALD)
    save(fig, "chapter_6.png")


def fig_ch7():
    """chapter_7.png -- the session agent loop."""
    fig, ax = _fig()
    title(ax, "The agent",
          "a checkpointed task DAG persisted in SQLite; store + runner + router + executors")
    box(ax, 26, 66, 32, 16, "Runner", ["claims READY task"], accent=SKY, ts=12)
    box(ax, 26, 40, 32, 16, "Router", ["TASK_SUCCESSOR", "retry / repair"], accent=VIOLET, ts=12, ss=9.5)
    box(ax, 26, 14, 32, 16, "Store (SQLite)", [".cgx/sessions.db"], accent=EMERALD, ts=12)
    arrow(ax, (26, 58), (26, 48), color=SLATE, rad=0)
    arrow(ax, (26, 32), (26, 22), color=SLATE, rad=0)
    ex = [
        (70, 66, "EXPLORE", SKY), (104, 66, "INVESTIGATE", SKY),
        (138, 66, "RECOMMEND", SKY), (70, 40, "CLARIFY", VIOLET),
        (104, 40, "DECOMPOSE", VIOLET), (138, 40, "SCAFFOLD", VIOLET),
        (87, 15, "APPLY", EMERALD), (121, 15, "VERIFY / REPAIR", AMBER),
    ]
    for cx, cy, t, c in ex:
        box(ax, cx, cy, 30, 13, t, accent=c, ts=11)
    ax.text(104, 53, "explore chain (indexed repo)  ·  greenfield chain (new project)",
            color=MUTED, fontsize=9.5, ha="center", va="center")
    arrow(ax, (85, 66), (89, 66)); arrow(ax, (119, 66), (123, 66))
    arrow(ax, (85, 40), (89, 40)); arrow(ax, (119, 40), (123, 40))
    arrow(ax, (102, 15), (106, 15), color=EMERALD)
    save(fig, "chapter_7.png")


def build_all():
    for fn in (fig_hero, fig_ch1, fig_ch2, fig_ch3, fig_ch4, fig_ch5,
               fig_ch6, fig_ch7):
        fn()


if __name__ == "__main__":
    build_all()
    print("wrote images to", OUT_DIR)
