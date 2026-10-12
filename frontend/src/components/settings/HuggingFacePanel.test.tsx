import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { api, type HfHubModel, type HfModelsResponse } from "../../lib/api";
import { HuggingFacePanel } from "./HuggingFacePanel";

function model(id: string): HfHubModel {
  return { id, downloads: 1200, likes: 34, gated: false, pull_tag: `hf.co/${id}`, quants: [] };
}

function page(models: HfHubModel[], next_cursor: string | null): HfModelsResponse {
  return { models, next_cursor, has_more: next_cursor != null, limit: 40, error: null };
}

const A = model("org/model-a");
const B = model("org/model-b");
const C = model("org/model-c");

describe("HuggingFacePanel", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows the first page and says more is available when the Hub returned a cursor", async () => {
    const spy = vi.spyOn(api, "hfModels").mockResolvedValueOnce(page([A, B], "cur-1"));
    render(<HuggingFacePanel />);
    expect(await screen.findByText("org/model-a")).toBeInTheDocument();
    expect(screen.getByText("org/model-b")).toBeInTheDocument();
    expect(screen.getByText("Showing 2 GGUF repos · more available")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Load more/ })).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith({ search: "", sort: "trending_score", limit: 40 });
  });

  it("loads the next page with the cursor and appends it without duplicates", async () => {
    const spy = vi
      .spyOn(api, "hfModels")
      .mockResolvedValueOnce(page([A, B], "cur-1"))
      .mockResolvedValueOnce(page([B, C], null));
    render(<HuggingFacePanel />);
    await screen.findByText("org/model-a");
    fireEvent.click(screen.getByRole("button", { name: /Load more/ }));
    expect(await screen.findByText("org/model-c")).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy.mock.calls[1][0]).toEqual({ search: "", sort: "trending_score", limit: 40, cursor: "cur-1" });
    // B straddled the page boundary and is listed once; the list is now exhausted.
    expect(screen.getAllByText("org/model-b")).toHaveLength(1);
    expect(screen.getAllByText(/^org\/model-/)).toHaveLength(3);
    expect(screen.getByText("Showing 3 GGUF repos")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
  });

  it("surfaces a Hub error returned alongside an empty page", async () => {
    vi.spyOn(api, "hfModels").mockResolvedValueOnce({ models: [], error: "Hugging Face Hub returned HTTP 429" });
    render(<HuggingFacePanel />);
    expect(await screen.findByText("Hugging Face Hub returned HTTP 429")).toBeInTheDocument();
    expect(screen.queryByText("No models found.")).toBeNull();
    expect(screen.queryByText(/Showing/)).toBeNull();
  });

  it("drops the cursor when the search changes so the next request starts over", async () => {
    const spy = vi
      .spyOn(api, "hfModels")
      .mockResolvedValueOnce(page([A, B], "cur-1"))
      .mockResolvedValueOnce(page([C], null));
    render(<HuggingFacePanel />);
    await screen.findByText("org/model-a");
    const input = screen.getByPlaceholderText(/Search the Hub/);
    fireEvent.change(input, { target: { value: "qwen" } });
    // Typing alone invalidates the cursor: paging the stale list on is no longer offered.
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
    fireEvent.submit(input.closest("form")!);
    expect(await screen.findByText("org/model-c")).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy.mock.calls[1][0]).toEqual({ search: "qwen", sort: "trending_score", limit: 40 });
    expect(spy.mock.calls[1][0]).not.toHaveProperty("cursor");
    expect(screen.queryByText("org/model-a")).toBeNull();
    expect(screen.getByText("Showing 1 GGUF repos")).toBeInTheDocument();
  });

  it("re-fetches the first page without a cursor when the sort changes", async () => {
    const spy = vi
      .spyOn(api, "hfModels")
      .mockResolvedValueOnce(page([A], "cur-1"))
      .mockResolvedValueOnce(page([B], null));
    render(<HuggingFacePanel />);
    await screen.findByText("org/model-a");
    fireEvent.change(screen.getByDisplayValue("Trending"), { target: { value: "downloads" } });
    expect(await screen.findByText("org/model-b")).toBeInTheDocument();
    expect(spy.mock.calls[1][0]).toEqual({ search: "", sort: "downloads", limit: 40 });
    expect(screen.queryByText("org/model-a")).toBeNull();
  });
});
