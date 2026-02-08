/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0c10',
        surface: '#12151c',
        'surface-lighter': '#1a1f29',
        border: '#2a313e',
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        accent: '#f59e0b',
        success: '#10b981',
        danger: '#ef4444',
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'glow': '0 0 15px rgba(14, 165, 233, 0.3)',
      },
      backgroundImage: {
        'gradient-medical': 'radial-gradient(circle at top right, #1a1f29, #0a0c10)',
      }
    },
  },
  plugins: [],
}
