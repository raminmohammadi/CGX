import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BuildResult, latestVerifyReport, revisionArtifacts } from "./BuildResult";
import type { AgentSessionState, ArtifactDTO, TaskNodeDTO } from "../../lib/api";

function art(kind: ArtifactDTO["kind"], content: Record<string, any>, created_at: number, id = `a${created_at}`): ArtifactDTO {
  return { artifact_id: id, session_id: "s1", produced_by_task_id: "t1", kind, content, created_at };
}

function task(kind: TaskNodeDTO["kind"], status: TaskNodeDTO["status"], inputs: Record<string, any> = {}): TaskNodeDTO {
  return {
    task_id: `t-${kind}-${status}`, session_id: "s1", kind, name: kind, description: "",
    parent_task_id: null, status, inputs, outputs: null, error: null, blockers: [], children: [],
    consumed_decision_ids: [], produced_artifact_id: null, created_at: 0, started_at: 0, completed_at: 1,
  };
}

const report = art("swarm_verify_report", {
  verify_ok: true,
  summary: "verified: built 21/21 files; tests passed; design ok; runtime ok (2 routes render)",
  runtime: { ran: true, ok: true, routes: [
    { route: "/", mounted: true, errors: [], screenshot: "/p/.cgx/screenshots/index.png" },
    { route: "/settings", mounted: false, errors: ["TypeError: x"], screenshot: "/p/.cgx/screenshots/settings.png" },
  ] },
}, 100);

const revision = art("swarm_revision", {
  revision_index: 1, status: "applied", feedback: "make the top bar span the full width",
  note: "Widened the top bar.", changed: ["src/components/TopBar.jsx"], unchanged: [], failed: [],
  changes: [{ path: "src/components/TopBar.jsx", instruction: "remove max-width from .topbar",
              why: "renders the header", outcome: "changed", diff: "--- a\n+++ b\n-old\n+new" }],
  candidates: [{ path: "src/components/TopBar.jsx", score: 13, reasons: ["mentions \"top bar\""] }],
}, 200);

function state(extra: Partial<AgentSessionState> = {}): AgentSessionState {
  return {
    session: {
      session_id: "s1", title: "t", original_objective: "build a chatbot", status: "completed",
      mode: "swarm", current_focus: null, root_task_id: null, project_root: "/p",
      created_at: 0, updated_at: 0,
    },
    tasks: [task("swarm_verify", "done")],
    artifacts: [report, revision],
    facts: [], decisions: [],
    ...extra,
  };
}

describe("BuildResult", () => {
  it("picks the newest verify report and lists revisions newest first", () => {
    const older = art("swarm_verify_report", { verify_ok: false, summary: "partial" }, 50, "old");
    expect(latestVerifyReport([older, report])?.artifact_id).toBe(report.artifact_id);
    expect(latestVerifyReport([])).toBeNull();
    const r2 = art("swarm_revision", { revision_index: 2, status: "applied", feedback: "x" }, 300, "r2");
    expect(revisionArtifacts([revision, r2]).map((a) => a.artifact_id)).toEqual(["r2", revision.artifact_id]);
  });

  it("shows the verdict, one screenshot per route with its status, and the revision history", () => {
    render(<BuildResult sid="s1" state={state()} />);
    expect(screen.getByText(/verified: built 21\/21 files/)).toBeTruthy();
    expect(screen.getByText("verified")).toBeTruthy();
    const imgs = screen.getAllByRole("img") as HTMLImageElement[];
    expect(imgs).toHaveLength(2);
    expect(imgs[0].getAttribute("src")).toBe("/api/agent-session/s1/screenshots/index.png?v=100");
    expect(screen.getByText("renders")).toBeTruthy();
    expect(screen.getByText("blank")).toBeTruthy();
    // revision row: feedback + changed files; details open on click
    expect(screen.getByText(/make the top bar span the full width/)).toBeTruthy();
    expect(screen.getByText(/changed src\/components\/TopBar.jsx/)).toBeTruthy();
    fireEvent.click(screen.getByText(/make the top bar span the full width/));
    expect(screen.getByText("remove max-width from .topbar")).toBeTruthy();
    expect(screen.getByText(/why: renders the header/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Open live preview/ })).toBeTruthy();
  });

  it("says a revision is in flight and disables the preview while running", () => {
    const s = state({ tasks: [task("swarm_revise", "in_progress", { feedback: "align the Send button" })] });
    render(<BuildResult sid="s1" state={s} />);
    expect(screen.getByText(/Applying your feedback/)).toBeTruthy();
    const btn = screen.getByRole("button", { name: /Open live preview/ }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("opens the live preview via the API and renders it inline", async () => {
    const start = vi.fn().mockResolvedValue({ url: "http://127.0.0.1:4321/", port: 4321, dist: "/p/dist", host: "127.0.0.1" });
    const api = await import("../../lib/api");
    const spy = vi.spyOn(api.api, "agentSessionPreviewStart").mockImplementation(start);
    render(<BuildResult sid="s1" state={state()} />);
    fireEvent.click(screen.getByRole("button", { name: /Preview here/ }));
    const frame = await screen.findByTitle("Live preview of the built app");
    expect(frame.getAttribute("src")).toBe("http://127.0.0.1:4321/");
    expect(start).toHaveBeenCalledWith("s1");
    spy.mockRestore();
  });

  it("shows the screenshot review verdict and auto follow-up marker on a revision", () => {
    const rev = art("swarm_revision", {
      revision_index: 2, status: "applied", feedback: "make the top bar full width", auto_round: 1,
      changed: ["src/styles/app.css"], unchanged: [], failed: [], changes: [],
      visual_check: { satisfied: false, verdict: "not_satisfied", evidence: "header still narrow", model: "qwen3-vl:8b" },
    }, 300, "r2");
    render(<BuildResult sid="s1" state={state({ artifacts: [report, rev] })} />);
    expect(screen.getByText(/NOT visibly addressed — header still narrow/)).toBeTruthy();
    expect(screen.getByText(/automatic follow-up #1/)).toBeTruthy();
  });

  it("renders nothing without a report or revision", () => {
    const { container } = render(<BuildResult sid="s1" state={state({ artifacts: [] })} />);
    expect(container.firstChild).toBeNull();
  });
});
