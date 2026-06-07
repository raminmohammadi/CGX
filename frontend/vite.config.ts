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
        // Function form is required by Rolldown (vite 8+) — the legacy
        // record form (`{ "react-vendor": [...] }`) was dropped. The
        // patterns below match by `node_modules/<pkg>/` so transitive
        // deps of `react-markdown` (remark-*, mdast-*, micromark*, …)
        // land in the markdown chunk instead of leaking into the main
        // bundle.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return;
          // React core stays separate so route-level chunks can share it
          // without duplication.
          if (
            /[\\/]node_modules[\\/](react|react-dom|react-is|react-router|react-router-dom|@remix-run|scheduler)[\\/]/.test(id)
          ) {
            return "react-vendor";
          }
          // Heaviest dep tree (markdown + GFM + highlight pipeline) only
          // pulled in on routes that render LLM output.
          if (
            /[\\/]node_modules[\\/](react-markdown|remark-[^\\/]+|rehype-[^\\/]+|highlight\.js|unified|vfile[^\\/]*|hast-[^\\/]+|mdast-[^\\/]+|micromark[^\\/]*|unist-[^\\/]+|bail|trough|character-entities[^\\/]*|decode-named-character-reference|comma-separated-tokens|space-separated-tokens|property-information|html-void-elements|web-namespaces|zwitch|ccount|markdown-table|is-plain-obj|devlop|estree-util-[^\\/]+)[\\/]/.test(id)
          ) {
            return "markdown";
          }
          // Small stable libs grouped to avoid per-route micro-chunks.
          if (
            /[\\/]node_modules[\\/](lucide-react|clsx|tailwind-merge|zustand)[\\/]/.test(id)
          ) {
            return "ui";
          }
        },
      },
    },
  },
});
