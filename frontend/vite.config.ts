import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Build straight into the FastAPI static dir so `python app.py` serves
// the freshest bundle without any extra copy step. During `npm run dev`
// the dev server proxies /api → uvicorn on :8765.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../src/cgx/webui/static"),
    emptyOutDir: true,
    sourcemap: false,
    target: "es2020",
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: {
          // React core stays separate so route-level chunks can share it
          // without duplication.
          "react-vendor": ["react", "react-dom", "react-router-dom"],
          // Heaviest dep tree (markdown + GFM + highlight pipeline) only
          // pulled in on routes that render LLM output.
          markdown: [
            "react-markdown",
            "remark-gfm",
            "rehype-highlight",
            "highlight.js",
          ],
          // Small stable libs grouped to avoid per-route micro-chunks.
          ui: ["lucide-react", "clsx", "tailwind-merge", "zustand"],
        },
      },
    },
  },
});
