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
  import {
    decodeWorkspace,
    readShareParam,
    clearShareParam,
  } from './lib/shareLink.js';
  import { workspaceFor } from './lib/workspaces.js';
  import Header from './components/Header.svelte';

  let restoreNote = $state('');
  let isMobileViewport = $state(false);
  let workspaceRef = $state();

  // Track viewport so Header can hide the seed/Roll cluster + show
  // the hamburger at narrow widths. The actual drawer mechanics
  // live inside each workspace (KenningWorkspace has them; the
  // default doesn't need them).
  $effect(() => {
    const mq = window.matchMedia('(max-width: 899px)');
    isMobileViewport = mq.matches;
    const onChange = (e) => (isMobileViewport = e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
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
    appState.seed = entry.seed;
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

  // Re-derive the active workspace component when the picker
  // changes. <svelte:component> mounts the new one + the Header's
  // hamburger calls workspaceRef.openMenu() so each workspace
  // controls its own drawer.
  let Workspace = $derived(workspaceFor(appState.selectedGeneratorName));

  function openMenu() {
    workspaceRef?.openMenu?.();
  }
</script>

{#if restoreNote}
  <div class="restore-banner" role="status">{restoreNote}</div>
{/if}

<div class="app">
  <Header onMenuToggle={openMenu} {isMobileViewport} />
  <Workspace bind:this={workspaceRef} />
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
