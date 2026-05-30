/** @type {import('tailwindcss').Config} */
// Design tokens lifted from the reference mockup (index.html in repo root):
//   --bg-base #030712, --bg-surface #0B1329, --bg-header #090F1F
//   --color-neon #10B981 (emerald), warning #F59E0B, error #EF4444
// Typography: Plus Jakarta Sans for UI, JetBrains Mono for code/labels.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#030712",
        surface: "#0B1329",
        header: "#090F1F",
        inset: "#02040A",
        neon: {
          DEFAULT: "#10B981",
          dim: "rgba(16, 185, 129, 0.3)",
          glow: "rgba(16, 185, 129, 0.08)",
        },
      },
      borderColor: {
        muted: "rgba(255, 255, 255, 0.06)",
        bright: "rgba(16, 185, 129, 0.3)",
      },
      fontFamily: {
        sans: ["Plus Jakarta Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        neon: "0 0 20px rgba(16, 185, 129, 0.08)",
        "neon-md": "0 0 0 1px rgba(16, 185, 129, 0.3), 0 8px 24px rgba(16, 185, 129, 0.12)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(2px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-in": "fade-in 200ms cubic-bezier(0.4, 0, 0.2, 1)",
        shimmer: "shimmer 2.5s linear infinite",
      },
    },
  },
  plugins: [],
};
