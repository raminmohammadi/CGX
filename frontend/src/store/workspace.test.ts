import { beforeEach, describe, expect, it } from "vitest";
import { useWorkspace } from "./workspace";

// The store persists into localStorage; reset both the snapshot and the
// browser storage between tests so cross-test bleed can't happen.
const PRISTINE = useWorkspace.getState();

beforeEach(() => {
  // localStorage may not be available on opaque-origin jsdom; clearing it
  // is best-effort. The real reset is the setState(.., replace=true) below.
  try {
    globalThis.localStorage?.clear();
  } catch {
    /* opaque origin -- ignore */
  }
  useWorkspace.setState(
    {
      ...PRISTINE,
      provider: { ...PRISTINE.provider },
      index: { ...PRISTINE.index },
      projectRoot: "",
      selectedSessionId: null,
    },
    true,
  );
});

describe("useWorkspace", () => {
  it("ships with the documented Ollama defaults", () => {
    const s = useWorkspace.getState();
    expect(s.provider.kind).toBe("ollama");
    expect(s.provider.model).toBe("qwen2.5-coder:3b");
    expect(s.provider.base_url).toBe("http://localhost:11434");
    expect(s.provider.use_profile).toBe(false);
    expect(s.selectedSessionId).toBeNull();
  });

  it("setProvider merges patches without dropping unrelated fields", () => {
    useWorkspace.getState().setProvider({ model: "llama3:8b", temperature: 0.7 });
    const p = useWorkspace.getState().provider;
    expect(p.model).toBe("llama3:8b");
    expect(p.temperature).toBe(0.7);
    // unrelated fields untouched
    expect(p.base_url).toBe("http://localhost:11434");
    expect(p.kind).toBe("ollama");
  });

  it("setIndex merges patches", () => {
    useWorkspace
      .getState()
      .setIndex({ embed_model: "BAAI/bge-small-en-v1.5" });
    const idx = useWorkspace.getState().index;
    expect(idx.embed_model).toBe("BAAI/bge-small-en-v1.5");
    expect(idx.index_dir).toBe("/tmp/cgx_index/indices");
  });

  it("setProjectRoot replaces the stored root", () => {
    useWorkspace.getState().setProjectRoot("/home/user/repo");
    expect(useWorkspace.getState().projectRoot).toBe("/home/user/repo");
  });

  it("applyProfile flips use_profile and copies fields", () => {
    useWorkspace.getState().applyProfile({
      name: "prod",
      kind: "openai-compat",
      model: "gpt-4o-mini",
      base_url: "https://api.example/v1",
      temperature: 0.3,
      num_predict: 2048,
    });
    const p = useWorkspace.getState().provider;
    expect(p.use_profile).toBe(true);
    expect(p.profile_name).toBe("prod");
    expect(p.kind).toBe("openai-compat");
    expect(p.model).toBe("gpt-4o-mini");
    expect(p.base_url).toBe("https://api.example/v1");
    expect(p.temperature).toBe(0.3);
    expect(p.num_predict).toBe(2048);
  });

  it("setSelectedSession stores and clears the id", () => {
    useWorkspace.getState().setSelectedSession("abc-123");
    expect(useWorkspace.getState().selectedSessionId).toBe("abc-123");
    useWorkspace.getState().setSelectedSession(null);
    expect(useWorkspace.getState().selectedSessionId).toBeNull();
  });
});
