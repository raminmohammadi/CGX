import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TaskTree } from "./TaskTree";
import type { TaskKind, TaskNodeDTO, TaskNodeStatus } from "../../lib/api";

function mk(
  task_id: string, kind: TaskKind, status: TaskNodeStatus,
  parent: string | null = null, name?: string, ts = 0,
): TaskNodeDTO {
  return {
    task_id, session_id: "s1", kind, name: name || task_id,
    description: "", parent_task_id: parent, status,
    inputs: {}, outputs: null, error: null, blockers: [], children: [],
    consumed_decision_ids: [], produced_artifact_id: null,
    created_at: ts, started_at: null, completed_at: null,
  };
}

describe("TaskTree", () => {
  it("renders an empty-state hint when there are no tasks", () => {
    render(<TaskTree tasks={[]} rootTaskId={null} selectedId={null} onSelect={vi.fn()} />);
    expect(screen.getByText(/no tasks yet/i)).toBeInTheDocument();
  });

  it("groups children under their declared parent and pins the root first", () => {
    const tasks = [
      mk("child-a", "investigate", "done", "root", "child-a", 2),
      mk("orphan",  "explore",     "ready", null,   "orphan",  3),
      mk("root",    "explore",     "done",  null,   "root",    1),
      mk("child-b", "ask_user",    "in_progress", "root", "child-b", 4),
    ];
    render(<TaskTree tasks={tasks} rootTaskId="root" selectedId={null} onSelect={vi.fn()} />);
    const buttons = screen.getAllByRole("button");
    // Root is pinned first, then its children, then the orphan top-level.
    expect(buttons[0]).toHaveTextContent("root");
    expect(buttons[1]).toHaveTextContent("child-a");
    expect(buttons[2]).toHaveTextContent("child-b");
    expect(buttons[3]).toHaveTextContent("orphan");
  });

  it("indents children below the parent (depth-based padding)", () => {
    const tasks = [
      mk("root",  "explore",     "done", null,   "root",  1),
      mk("child", "investigate", "ready", "root", "child", 2),
    ];
    render(<TaskTree tasks={tasks} rootTaskId="root" selectedId={null} onSelect={vi.fn()} />);
    const rootBtn  = screen.getByText("root").closest("button")!;
    const childBtn = screen.getByText("child").closest("button")!;
    const rootPad  = parseInt((rootBtn.style.paddingLeft || "0").replace("px", ""), 10);
    const childPad = parseInt((childBtn.style.paddingLeft || "0").replace("px", ""), 10);
    expect(childPad).toBeGreaterThan(rootPad);
  });

  it("invokes onSelect with the task_id when a node is clicked", () => {
    const onSelect = vi.fn();
    const tasks = [mk("only", "explore", "ready", null, "only-task")];
    render(<TaskTree tasks={tasks} rootTaskId="only" selectedId={null} onSelect={onSelect} />);
    fireEvent.click(screen.getByText("only-task"));
    expect(onSelect).toHaveBeenCalledWith("only");
  });

  it("highlights the selected node with the emerald ring class", () => {
    const tasks = [
      mk("a", "explore", "done", null, "alpha"),
      mk("b", "explore", "done", null, "beta"),
    ];
    render(<TaskTree tasks={tasks} rootTaskId="a" selectedId="b" onSelect={vi.fn()} />);
    const betaBtn = screen.getByText("beta").closest("button")!;
    expect(betaBtn.className).toMatch(/ring-emerald-500/);
  });

  it("re-parents orphaned tasks (missing parent) to the top level", () => {
    const tasks = [
      mk("root",   "explore",     "done",  null,        "root",   1),
      mk("ghost",  "investigate", "ready", "no-such-id", "ghost",  2),
    ];
    render(<TaskTree tasks={tasks} rootTaskId="root" selectedId={null} onSelect={vi.fn()} />);
    // Ghost would otherwise be hidden under its missing parent; ensure
    // it still surfaces at the top level so nothing gets silently dropped.
    expect(screen.getByText("ghost")).toBeInTheDocument();
    const ghostBtn = screen.getByText("ghost").closest("button")!;
    const ghostPad = parseInt((ghostBtn.style.paddingLeft || "0").replace("px", ""), 10);
    expect(ghostPad).toBe(8);
  });
});
