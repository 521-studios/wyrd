import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// wyrd-z3lp: Vite + Svelte 5 config for the SPA rewrite.
//
// Dev server proxies /api/* to the Flask app on :5000 so the Svelte
// dev server can fetch the manifest + roll without a CORS dance.
// Build output goes to dist/ — deploy.yml will eventually upload
// from spa-next/dist/ instead of spa/ (the wyrd-20pz cutover PR).
export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5000',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
