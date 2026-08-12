/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        surface: {
          DEFAULT: "#0f1117",
          raised: "#1a1d27",
          border: "#2a2d3a",
        },
        accent: {
          blue: "#3b82f6",
          purple: "#8b5cf6",
          green: "#10b981",
          amber: "#f59e0b",
          red: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
