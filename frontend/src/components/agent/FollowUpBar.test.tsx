import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { FollowUpBar } from "./FollowUpBar";
import type { ProfileSummary, ProviderConfig } from "../../lib/api";

const fast: ProfileSummary = {
  name: "fast", kind: "ollama", model: "qwen2.5-coder:7b",
  base_url: "http://localhost:11434", has_api_key: false,
  temperature: 0.2, num_predict: 1024,
};
const big: ProfileSummary = {
  name: "big", kind: "ollama", model: "qwen2.5-coder:32b",
  base_url: "http://localhost:11434", has_api_key: false,
  temperature: 0.1, num_predict: 4096,
};

const provider: ProviderConfig = {
  use_profile: true, profile_name: "fast", kind: "ollama", model: "qwen2.5-coder:7b",
  base_url: "http://localhost:11434", temperature: 0.2, num_predict: 1024,
};

type Props = Parameters<typeof FollowUpBar>[0];

// The bar links to /settings when there are no profiles, so every render
// gets a router even though most cases never reach that branch.
function renderBar(over: Partial<Props> = {}) {
  const props: Props = {
    reply: "", setReply: vi.fn(), onSend: vi.fn(), pending: false,
    profiles: [fast, big], provider, onPickProfile: vi.fn(),
    ...over,
  };
  const { unmount } = render(<MemoryRouter><FollowUpBar {...props} /></MemoryRouter>);
  return { props, unmount };
}

const picker = () =>
  screen.getByLabelText("Model for the next message") as HTMLSelectElement;

describe("FollowUpBar", () => {
  it("selects the active profile and lists every profile as 'name · model'", () => {
    renderBar();
    const select = picker();
    expect(select.value).toBe("fast");
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      "fast · qwen2.5-coder:7b", "big · qwen2.5-coder:32b",
    ]);
    expect(screen.getByText("next message uses qwen2.5-coder:7b")).toBeInTheDocument();
  });

  it("hands the picked profile summary to onPickProfile", () => {
    const { props } = renderBar();
    fireEvent.change(picker(), { target: { value: "big" } });
    expect(props.onPickProfile).toHaveBeenCalledTimes(1);
    expect(props.onPickProfile).toHaveBeenCalledWith(big);
  });

  it("reads 'Revise' for a finished build and 'Send' otherwise", () => {
    const { unmount } = renderBar({ revisable: true, reply: "wider header" });
    expect(screen.getByRole("button", { name: /revise/i })).toBeInTheDocument();
    unmount();
    renderBar({ reply: "wider header" });
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /revise/i })).toBeNull();
  });

  it("shows the model as text with a Settings link when there are no profiles", () => {
    renderBar({ profiles: [] });
    expect(screen.queryByLabelText("Model for the next message")).toBeNull();
    expect(screen.getByText("next message uses qwen2.5-coder:7b")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /add a profile/i }))
      .toHaveAttribute("href", "/settings");
  });

  it("selects a synthetic 'Inline · model' entry for an inline provider and keeps it on re-pick", () => {
    const { props } = renderBar({
      provider: { ...provider, use_profile: false, profile_name: null, model: "llama3:8b" },
    });
    const select = picker();
    expect(select.value).toBe("__inline__");
    expect(select.selectedOptions[0].textContent).toBe("Inline · llama3:8b");
    expect(select.options).toHaveLength(3);
    fireEvent.change(select, { target: { value: "__inline__" } });
    expect(props.onPickProfile).not.toHaveBeenCalled();
  });
});
