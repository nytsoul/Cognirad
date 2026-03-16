/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--color-background) / <alpha-value>)',
        surface: 'hsl(var(--color-surface) / <alpha-value>)',
        'surface-lighter': 'hsl(var(--color-surface-lighter) / <alpha-value>)',
        border: 'hsl(var(--color-border) / <alpha-value>)',
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
        medical: {
          // Healthcare-specific professional colors
          teal: '#0891b2',
          cyan: '#06b6d4',
          blue: '#0284c7',
          green: '#059669',
          indigo: '#4f46e5',
        },
        accent: '#f59e0b',
        success: '#10b981',
        danger: '#ef4444',
        warning: '#f59e0b',
        info: '#0284c7',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica Neue', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'glow': '0 0 15px rgba(14, 165, 233, 0.3)',
        'medical': '0 4px 12px 0 rgba(0, 0, 0, 0.08)',
        'medical-lg': '0 8px 24px 0 rgba(0, 0, 0, 0.12)',
      },
      backgroundImage: {
        'gradient-medical': 'radial-gradient(circle at top right, hsl(var(--color-surface)), hsl(var(--color-background)))',
        'gradient-hospital': 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
      }
    },
  },
  plugins: [],
}
