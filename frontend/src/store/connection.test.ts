import { beforeEach, describe, expect, it, vi } from "vitest";

// Replace the api module before the store imports it so refresh() uses our mock.
vi.mock("../lib/api", () => ({
  api: { status: vi.fn() },
}));

import { api } from "../lib/api";
import { useConnection } from "./connection";

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

const fakeStatus = {
  app: "Averix",
  version: "0.2.0",
  ollama: { ok: true, base_url: "http://localhost:11434" },
  hardware: { ram_gb: 16, gpu_vram_gb: 8 },
  telemetry_enabled: false,
  profile_count: 1,
  session_count: 0,
  default_model: "qwen2.5:7b-instruct",
};

describe("useConnection.refresh", () => {
  it("stores the status payload and clears error on success", async () => {
    (api.status as any).mockResolvedValue(fakeStatus);
    await useConnection.getState().refresh();
    const s = useConnection.getState();
    expect(s.status).toEqual(fakeStatus);
    expect(s.offline).toBe(false);
    expect(s.error).toBeNull();
    expect(s.consecutiveFailures).toBe(0);
    expect(s.lastCheck).not.toBeNull();
  });

  it("does not flip offline on a single failure (debounces transient blips)", async () => {
    (api.status as any).mockRejectedValue(new Error("boom"));
    await useConnection.getState().refresh();
    const s = useConnection.getState();
    expect(s.offline).toBe(false);
    expect(s.consecutiveFailures).toBe(1);
    expect(s.error).toContain("boom");
  });

  it("flips offline after two consecutive failures", async () => {
    (api.status as any).mockRejectedValue(new Error("nope"));
    await useConnection.getState().refresh();
    await useConnection.getState().refresh();
    const s = useConnection.getState();
    expect(s.offline).toBe(true);
    expect(s.consecutiveFailures).toBe(2);
  });

  it("auto-recovers (clears offline + failures) when a poll succeeds again", async () => {
    (api.status as any).mockRejectedValueOnce(new Error("a"));
    (api.status as any).mockRejectedValueOnce(new Error("b"));
    (api.status as any).mockResolvedValueOnce(fakeStatus);

    await useConnection.getState().refresh();
    await useConnection.getState().refresh();
    expect(useConnection.getState().offline).toBe(true);

    await useConnection.getState().refresh();
    const s = useConnection.getState();
    expect(s.offline).toBe(false);
    expect(s.error).toBeNull();
    expect(s.consecutiveFailures).toBe(0);
    expect(s.status).toEqual(fakeStatus);
  });

  it("is reentrancy-safe: a second refresh while in-flight is dropped", async () => {
    let resolveFn: (v: any) => void = () => {};
    (api.status as any).mockReturnValue(
      new Promise((r) => {
        resolveFn = r;
      }),
    );

    const p1 = useConnection.getState().refresh();
    // Second call should short-circuit because isChecking is true.
    const p2 = useConnection.getState().refresh();
    expect(useConnection.getState().isChecking).toBe(true);

    resolveFn(fakeStatus);
    await Promise.all([p1, p2]);

    expect(api.status).toHaveBeenCalledTimes(1);
    expect(useConnection.getState().isChecking).toBe(false);
  });
});
