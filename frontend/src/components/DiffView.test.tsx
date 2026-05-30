import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DiffView } from "./DiffView";

const TWO_FILE_DIFF = `diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,2 @@
-old
+new
diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -10,1 +10,2 @@
 ctx
+added
`;

describe("DiffView", () => {
  it("renders an empty-state hint when the diff is blank", () => {
    render(<DiffView diff="" />);
    expect(screen.getByText(/no diff produced/i)).toBeInTheDocument();
  });

  it("splits multi-file diffs into per-file blocks with filenames", () => {
    render(<DiffView diff={TWO_FILE_DIFF} />);
    expect(screen.getByText("foo.py")).toBeInTheDocument();
    expect(screen.getByText("bar.py")).toBeInTheDocument();
    // Each block ships a Split/Unified mode toggle.
    expect(screen.getAllByText("Split")).toHaveLength(2);
    expect(screen.getAllByText("Unified")).toHaveLength(2);
  });

  it("color-codes added / removed / hunk lines in unified mode", () => {
    const { container } = render(
      <DiffView diff={TWO_FILE_DIFF} defaultMode="unified" />,
    );
    const minusLine = Array.from(container.querySelectorAll("div")).find(
      (el) => el.textContent === "-old",
    );
    const plusLine = Array.from(container.querySelectorAll("div")).find(
      (el) => el.textContent === "+new",
    );
    const hunkLine = Array.from(container.querySelectorAll("div")).find(
      (el) => el.textContent?.startsWith("@@"),
    );
    expect(minusLine?.className).toMatch(/text-red-400/);
    expect(plusLine?.className).toMatch(/text-emerald-400/);
    expect(hunkLine?.className).toMatch(/text-purple-300/);
  });

  it("renders before / after panes in split mode (default)", () => {
    const { container } = render(<DiffView diff={TWO_FILE_DIFF} />);
    // The old/new content surface without the unified-diff +/- prefix:
    // "old" appears in a red-tinted cell, "new" in a green-tinted one.
    const oldCell = Array.from(container.querySelectorAll("div")).find(
      (el) => /bg-red-950/.test(el.className) && el.textContent?.includes("old"),
    );
    const newCell = Array.from(container.querySelectorAll("div")).find(
      (el) => /bg-emerald-950/.test(el.className) && el.textContent?.includes("new"),
    );
    expect(oldCell).toBeDefined();
    expect(newCell).toBeDefined();
  });

  it("falls back to a 'diff' label when no diff --git header is present", () => {
    const lone = `--- a/x.ts
+++ b/x.ts
@@ -1 +1 @@
-a
+b
`;
    render(<DiffView diff={lone} />);
    expect(screen.getByText("x.ts")).toBeInTheDocument();
  });
});
