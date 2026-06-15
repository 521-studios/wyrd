import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
// wyrd-200v: the Svelte-5 component-test harness. svelteTesting() wires the
// `browser` resolve-condition (so Svelte compiles to its CLIENT runtime under
// vitest, making mount + $effect/$derived actually run) and auto-cleanup between
// tests. No-op for `vite build`/`dev` — it only engages under vitest.
import { svelteTesting } from '@testing-library/svelte/vite';

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
  plugins: [svelte(), svelteTesting()],
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
  // wyrd-200v: jsdom so component tests can mount + assert DOM; the existing
  // pure-logic unit tests run unchanged under it. setupFiles adds the jest-dom
  // matchers (toBeInTheDocument, …). `npm test` (vitest run) — already the SPA
  // CI job's test step — picks these up with no workflow change.
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest-setup.js'],
    // Auto-restore every vi.spyOn (e.g. the pipeline.run / pipeline.setSwap spies
    // in the component tests) to its original after each test, so a spy can't leak
    // across tests — no per-file afterEach(restoreAllMocks) boilerplate needed.
    restoreMocks: true,
  },
});
