import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AskUserForm } from "./AskUserForm";
import type { ArtifactDTO, TaskNodeDTO } from "../../lib/api";

// Minimal TaskNode fixture; only the fields AskUserForm reads matter,
// but the full DTO is supplied so TS doesn't drift if the shape grows.
function mkTask(expectedKind: string, inputs: Record<string, any> = {}): TaskNodeDTO {
  return {
    task_id: "t1", session_id: "s1", kind: "ask_user",
    name: "ask", description: "", parent_task_id: null,
    status: "in_progress",
    inputs: { expected_kind: expectedKind, ...inputs },
    outputs: null, error: null, blockers: [], children: [],
    consumed_decision_ids: [], produced_artifact_id: null,
    created_at: 0, started_at: 0, completed_at: null,
  };
}

function mkArtifact(kind: ArtifactDTO["kind"], content: Record<string, any>): ArtifactDTO {
  return {
    artifact_id: "a1", session_id: "s1", produced_by_task_id: "t0",
    kind, content, created_at: 0,
  };
}

describe("AskUserForm — choose_path", () => {
  it("posts the option's chunk_id + title when an option is clicked", () => {
    const linked = mkArtifact("directions_list", {
      options: [
        { chunk_id: "src/foo.py::fn::bar", title: "Refactor bar", rationale: "..." },
        { chunk_id: "src/baz.py::class::Quux", title: "Touch Quux", rationale: "..." },
      ],
    });
    const onDecide = vi.fn();
    render(
      <AskUserForm task={mkTask("choose_path")} linked={linked}
                   onDecide={onDecide} pending={false} />,
    );
    fireEvent.click(screen.getByText("Touch Quux"));
    expect(onDecide).toHaveBeenCalledWith({
      chosen: { anchor_chunk_id: "src/baz.py::class::Quux", title: "Touch Quux" },
    });
  });
});

describe("AskUserForm — choose_recommendation", () => {
  it("posts the rec id, title, rationale, kind (+ optional anchor)", () => {
    const linked = mkArtifact("recommendation_list", {
      recommendations: [
        { id: "r1", title: "Investigate further", rationale: "needs depth",
          kind: "investigate_more", anchor_chunk_id: "src/x.py::fn::y" },
        { id: "r2", title: "Wrap up", rationale: "done here", kind: "done" },
      ],
    });
    const onDecide = vi.fn();
    render(
      <AskUserForm task={mkTask("choose_recommendation")} linked={linked}
                   onDecide={onDecide} pending={false} />,
    );
    fireEvent.click(screen.getByText("Investigate further"));
    expect(onDecide).toHaveBeenCalledWith({
      chosen: {
        id: "r1", title: "Investigate further", rationale: "needs depth",
        kind: "investigate_more", anchor_chunk_id: "src/x.py::fn::y",
      },
    });
  });
});

describe("AskUserForm — approve", () => {
  it("posts approved=true on Approve and false on Reject", () => {
    const linked = mkArtifact("code_change_plan", {
      plan_md: "## Plan\n\n- step 1",
      diffs: [{ file: "src/x.py", patch: "diff --git a/x b/x\n+ added\n" }],
      confidence: 0.7,
    });
    const onDecide = vi.fn();
    render(
      <AskUserForm task={mkTask("approve")} linked={linked}
                   onDecide={onDecide} pending={false} />,
    );
    fireEvent.click(screen.getByText(/approve & apply/i));
    expect(onDecide).toHaveBeenLastCalledWith({ chosen: { approved: true } });
    fireEvent.click(screen.getByText(/reject/i));
    expect(onDecide).toHaveBeenLastCalledWith({ chosen: { approved: false } });
  });

  it("renders the plan confidence chip and an empty-state when no plan content", () => {
    const onDecide = vi.fn();
    render(
      <AskUserForm task={mkTask("approve")}
                   linked={mkArtifact("code_change_plan", {})}
                   onDecide={onDecide} pending={false} />,
    );
    expect(screen.getByText(/plan artifact is empty/i)).toBeInTheDocument();
  });
});

describe("AskUserForm — freeform", () => {
  it("disables Send on empty text and posts the trimmed reply otherwise", async () => {
    const user = userEvent.setup();
    const onDecide = vi.fn();
    render(
      <AskUserForm task={mkTask("freeform")} linked={null}
                   onDecide={onDecide} pending={false} />,
    );
    const send = screen.getByRole("button", { name: /send/i });
    expect(send).toBeDisabled();
    await user.type(screen.getByPlaceholderText(/type your answer/i),
                    "  hello world  ");
    expect(send).toBeEnabled();
    await user.click(send);
    expect(onDecide).toHaveBeenCalledWith({ chosen: { text: "hello world" } });
  });
});

describe("AskUserForm — pending flag", () => {
  it("disables every action button while a request is in flight", () => {
    const linked = mkArtifact("directions_list", {
      options: [{ chunk_id: "x", title: "T", rationale: "r" }],
    });
    render(
      <AskUserForm task={mkTask("choose_path")} linked={linked}
                   onDecide={vi.fn()} pending={true} />,
    );
    expect(screen.getByText("T").closest("button")).toBeDisabled();
  });
});
