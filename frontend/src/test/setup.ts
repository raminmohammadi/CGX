import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom 25 dropped its built-in localStorage; the zustand persist
// middleware uses it on first call, so install a minimal in-memory shim
// when one isn't present.
function ensureLocalStorage() {
  if (typeof window === "undefined") return;
  const probe = (() => {
    try {
      return typeof window.localStorage?.setItem === "function";
    } catch {
      return false;
    }
  })();
  if (probe) return;
  const store = new Map<string, string>();
  const shim: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (k) => (store.has(k) ? (store.get(k) as string) : null),
    key: (i) => Array.from(store.keys())[i] ?? null,
    removeItem: (k) => {
      store.delete(k);
    },
    setItem: (k, v) => {
      store.set(k, String(v));
    },
  };
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: shim,
  });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: shim,
  });
}

ensureLocalStorage();

// Tear down React trees and reset DOM between tests so component state
// from a previous test never leaks across cases.
afterEach(() => {
  cleanup();
});
