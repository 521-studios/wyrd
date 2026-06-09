import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// wyrd-z3lp: Vite + Svelte 5 config for the SPA rewrite.
//
// Dev server proxies /api/* to the Flask app on :5000 so the Svelte
// dev server can fetch the manifest + roll without a CORS dance.
// Build output goes to dist/ — deploy.yml uploads from spa-next/dist/.

// wyrd-etvd: stamp the build's git revision into the bundle so the deployed
// SPA can surface it (Header hover over the wordmark). Lets operators confirm
// WHICH build is live without diffing minified bundles — a recurring pain when
// a flag/UX change "isn't showing up". BUILD_SHA env wins (CI can pass
// github.sha); else `git rev-parse` at build time; else 'dev' for a checkout
// without git.
function buildSha() {
  if (process.env.BUILD_SHA) return process.env.BUILD_SHA;
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
  } catch {
    return 'dev';
  }
}

export default defineConfig({
  plugins: [svelte()],
  define: {
    'import.meta.env.VITE_BUILD_SHA': JSON.stringify(buildSha()),
  },
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
