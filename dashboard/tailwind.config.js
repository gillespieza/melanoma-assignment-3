/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Clinical-white SaaS palette
        clinical: {
          bg: "#f6f8fb",
          panel: "#ffffff",
          border: "#e6ebf1",
          ink: "#0f2033",
          muted: "#5b6b7c",
          teal: "#0ea5a4",
          tealdark: "#0f766e",
          blue: "#2563eb",
          bluedark: "#1e40af",
          amber: "#d97706",
          rose: "#e11d48",
          green: "#16a34a",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,32,51,0.04), 0 8px 24px rgba(16,32,51,0.06)",
        lift: "0 8px 30px rgba(16,32,51,0.12)",
      },
      keyframes: {
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
