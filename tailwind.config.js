/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
    './Dashboard.jsx',
  ],
  theme: {
    extend: {
      colors: {
        bg:      '#f6f6fb',
        surface: '#ffffff',
        border:  '#ececf3',
        text:    { DEFAULT: '#1f2937', muted: '#6b7280' },
        accent:  { DEFAULT: '#6d5ce7', hover: '#5b46d6', light: '#efedfd', soft: '#a78bfa', deep: '#4c3bb5' },
        ok:      '#16a34a',
        warn:    '#d97706',
        danger:  '#dc2626',
        // 4-class customer colors
        'cls-am': '#FF9600',
        'cls-af': '#9314FF',
        'cls-mm': '#00C800',
        'cls-mf': '#00A5FF',
      },
      fontFamily: {
        sans: ['Noto Sans KR', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
