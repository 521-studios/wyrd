<script>
  // wyrd-14hn: app shell. Header above, per-generator workspace
  // below. The 3-column layout that used to live here is now in
  // workspaces/KenningWorkspace; non-kenning generators fall back
  // to workspaces/DefaultWorkspace (form + result list, no col 3
  // inspector). Adding a new workspace: build it under
  // workspaces/, register in lib/workspaces.js — App needs no
  // changes per generator.
  //
  // The boot-restore for ?s= share-links lives here (was added in
  // wyrd-tz35) because it has to run regardless of which workspace
  // is about to mount — the decoded payload sets selectedGenerator,
  // which then drives workspace selection.
  import { onMount } from 'svelte';
  import { appState } from './lib/appState.svelte.js';
  import { pipeline } from './lib/pipeline.svelte.js';
  import { fetchManifest } from './lib/api.js';
  import {
    decodeWorkspace,
    readShareParam,
    clearShareParam,
  } from './lib/shareLink.js';
  import { workspaceFor } from './lib/workspaces.js';
  import Header from './components/Header.svelte';
  import SavedDrawer from './components/SavedDrawer.svelte';

  let restoreNote = $state('');
  let workspaceRef = $state();

  // wyrd-14hn round 2 (frontend LOW): single matchMedia listener
  // for the whole app. Writes to appState.isMobileViewport so
  // KenningWorkspace + ConfigureColumn + Header read the same
  // source of truth. Pre-fix each ran its own listener.
  $effect(() => {
    const mq = window.matchMedia('(max-width: 899px)');
    appState.isMobileViewport = mq.matches;
    const onChange = (e) => (appState.isMobileViewport = e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  });

  // wyrd-14hn round 2 (frontend LOW): lift manifest fetch to App
  // so it's not gated on ConfigureColumn mounting. Pre-fix the
  // fetch lived in ConfigureColumn.onMount — a future workspace
  // without ConfigureColumn would never load the manifest. Header
  // needs manifest.generators for the picker, so the fetch needs
  // to happen at app boot regardless of which workspace mounts.
  onMount(async () => {
    if (appState.manifest) return; // already loaded (HMR)
    try {
      const manifest = await fetchManifest();
      appState.manifest = manifest;
      if (manifest.generators.length > 0 && !appState.selectedGeneratorName) {
        appState.selectedGeneratorName = manifest.generators[0].name;
      }
    } catch (err) {
      appState.manifestError = err.message;
    }
  });

  // wyrd-tz35: boot restore for ?s= share-link. Unchanged from
  // pre-wyrd-14hn except for moving alongside the new shell.
  onMount(() => {
    const encoded = readShareParam();
    if (!encoded) return;
    const payload = decodeWorkspace(encoded);
    if (!payload) {
      restoreNote = 'Share link was invalid or used an old format.';
      clearShareParam();
      setTimeout(() => (restoreNote = ''), 4000);
      return;
    }
    const entry = JSON.parse(JSON.stringify(payload));
    pipeline.clear();
    for (const step of entry.pipeline || []) {
      pipeline.addStep(step.kind, step.params);
    }
    appState.isLoadingSavedWorkspace = true;
    appState.selectedGeneratorName = entry.generator;
    // wyrd-hia1: restore both fields — seed (input) so user could
    // re-roll the exact result if they wanted, AND lastSeed (meta
    // display) so the result card shows the seed the loaded result
    // was rolled with.
    appState.seed = entry.seed;
    appState.lastSeed = entry.seed;
    if (entry.params) {
      appState.paramsByGenerator[entry.generator] = entry.params;
    }
    appState.results = [
      {
        result: entry.original.name,
        morphemes_by_word: entry.original.morphemes_by_word || [],
        explanation: entry.original.explanation || '',
        components: [],
      },
    ];
    appState.resultsGenerator = entry.generator;
    appState.currentResultIndex = 0;
    restoreNote = `Loaded shared workspace: ${entry.original.name}`;
    clearShareParam();
    setTimeout(() => (restoreNote = ''), 4000);
  });

  // wyrd-14hn round 2 (clarity P3): comment was stale — Svelte 5
  // uses the variable-as-component syntax (<Workspace />); the
  // older <svelte:component> is deprecated in v5.
  //
  // Re-derive the active workspace component when the picker
  // changes; <Workspace> mounts the new one. Each workspace that
  // wants a mobile drawer exposes openMenu() as an instance method;
  // App reads workspaceRef?.openMenu to gate the hamburger so
  // workspaces without a drawer (DefaultWorkspace) don't show a
  // dead button (wyrd-14hn round 2 frontend MED fix).
  let Workspace = $derived(workspaceFor(appState.selectedGeneratorName));

  let openMenu = $derived(workspaceRef?.openMenu);
</script>

{#if restoreNote}
  <div class="restore-banner" role="status">{restoreNote}</div>
{/if}

<div class="app">
  <Header
    onMenuToggle={openMenu}
    isMobileViewport={appState.isMobileViewport}
  />
  <Workspace bind:this={workspaceRef} />
  <SavedDrawer />
</div>

<style>
  .restore-banner {
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--accent);
    color: #1a1a1c;
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  }
  .app {
    display: flex;
    flex-direction: column;
    height: 100dvh;
  }
</style>
