import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("../lib/api", () => ({
  api: { status: vi.fn() },
}));

import { useConnection } from "../store/connection";
import ConnectionBanner from "./ConnectionBanner";

const PRISTINE = useConnection.getState();

beforeEach(() => {
  vi.clearAllMocks();
  useConnection.setState(
    {
      ...PRISTINE,
      status: null,
      offline: false,
      error: null,
      lastCheck: null,
      consecutiveFailures: 0,
      isChecking: false,
    },
    true,
  );
});

describe("ConnectionBanner", () => {
  it("renders nothing while online", () => {
    const { container } = render(<ConnectionBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the offline state with failure count and error text", () => {
    useConnection.setState({
      offline: true,
      error: "fetch failed",
      consecutiveFailures: 3,
    });
    render(<ConnectionBanner />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/API Offline/i)).toBeInTheDocument();
    expect(screen.getByText(/3 failed checks/i)).toBeInTheDocument();
    expect(screen.getByText(/fetch failed/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeEnabled();
  });

  it("Retry button calls refresh() on the store", async () => {
    const refresh = vi.fn();
    useConnection.setState({
      offline: true,
      error: "down",
      consecutiveFailures: 2,
      refresh,
    });
    render(<ConnectionBanner />);
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("disables Retry while a check is in-flight", () => {
    useConnection.setState({
      offline: true,
      error: null,
      consecutiveFailures: 2,
      isChecking: true,
    });
    render(<ConnectionBanner />);
    const btn = screen.getByRole("button", { name: /checking/i });
    expect(btn).toBeDisabled();
  });
});
