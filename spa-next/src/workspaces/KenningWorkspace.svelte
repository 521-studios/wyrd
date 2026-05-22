<script>
  // wyrd-14hn: the morpheme-inspector + transform-pipeline workspace.
  // Wraps the columns shipped via wyrd-z3lp through wyrd-tz35:
  // ConfigureColumn (now form-only — picker + seed + Roll moved to
  // Header), OutputColumn (result list with Saved toggle), and
  // InspectorColumn (morpheme detail + pipeline + Save/Share).
  //
  // The 3-column grid + mobile drawer/sheet behavior lives here
  // (used to be in App.svelte pre-this-PR). App.svelte now renders
  // Header + the active workspace, leaving layout decisions to
  // the workspace itself.
  import { onMount } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import ConfigureColumn from '../columns/ConfigureColumn.svelte';
  import OutputColumn from '../columns/OutputColumn.svelte';
  import InspectorColumn from '../columns/InspectorColumn.svelte';

  let col1Open = $state(false);
  let col3Open = $state(false);
  let isMobileViewport = $state(false);

  // Mobile auto-open of col 3 when a result is tapped, and resize-
  // dismiss of any open drawer/sheet — same logic that lived in
  // App.svelte pre-wyrd-14hn.
  $effect(() => {
    const mq = window.matchMedia('(max-width: 899px)');
    isMobileViewport = mq.matches;
    const onChange = (e) => {
      isMobileViewport = e.matches;
      if (!e.matches) {
        col1Open = false;
        col3Open = false;
      }
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  });

  $effect(() => {
    if (isMobileViewport && appState.currentResultIndex !== null) {
      col3Open = true;
    }
  });

  $effect(() => {
    if (!col1Open && !col3Open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') closeAll();
    };
    document.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  });

  function closeAll() {
    col1Open = false;
    col3Open = false;
  }

  // wyrd-14hn: exposed via the workspace's host (App.svelte) — when
  // it's the active workspace and the user taps the header's ☰,
  // open this workspace's col 1 drawer. Default workspace exposes
  // its own openMenu via the same prop contract.
  export function openMenu() {
    col1Open = true;
  }
</script>

<div class="layout" class:col1Open class:col3Open>
  <div
    class="col col-1"
    role={isMobileViewport ? 'dialog' : undefined}
    aria-modal={isMobileViewport && col1Open ? 'true' : undefined}
    aria-label={isMobileViewport ? 'Configure' : undefined}
    aria-hidden={isMobileViewport && !col1Open ? 'true' : undefined}
  >
    <button
      class="close-btn"
      type="button"
      onclick={() => (col1Open = false)}
      aria-label="Close configure">✕</button>
    <ConfigureColumn />
  </div>

  <div class="col col-2">
    <OutputColumn />
  </div>

  <div
    class="col col-3"
    role={isMobileViewport ? 'dialog' : undefined}
    aria-modal={isMobileViewport && col3Open ? 'true' : undefined}
    aria-label={isMobileViewport ? 'Inspect and Transform' : undefined}
    aria-hidden={isMobileViewport && !col3Open ? 'true' : undefined}
  >
    <button
      class="close-btn"
      type="button"
      onclick={() => (col3Open = false)}
      aria-label="Close inspector">✕</button>
    <InspectorColumn />
  </div>

  {#if col1Open || col3Open}
    <button class="backdrop" type="button" aria-label="Close" onclick={closeAll}></button>
  {/if}
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--col-1-width) var(--col-2-width) 1fr;
    /* wyrd-14hn: subtract the 52px Header height so the columns
       fit the remaining viewport without scrolling the page. */
    height: calc(100dvh - 52px);
  }
  .col {
    position: relative;
    display: contents;
  }
  .close-btn { display: none; }
  .backdrop { display: none; }

  @media (max-width: 899px) {
    .layout {
      display: block;
      height: calc(100dvh - 48px); /* 48px mobile Header height */
      overflow: hidden;
    }
    .col { display: block; }
    .col-1 {
      position: fixed;
      top: 48px; /* under the mobile Header */
      left: 0;
      width: min(80vw, 320px);
      height: calc(100dvh - 48px);
      background: var(--bg);
      transform: translateX(-100%);
      transition: transform 220ms ease;
      z-index: 30;
      padding-top: 8px;
      overflow-y: auto;
    }
    .col-1 :global(.column),
    .col-3 :global(.column) {
      border-right: none;
    }
    .col-3 {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 75dvh;
      background: var(--bg);
      transform: translateY(100%);
      transition: transform 240ms ease;
      z-index: 30;
      border-top: 1px solid var(--border);
      border-radius: 12px 12px 0 0;
      overflow-y: auto;
    }
    .col-2 {
      height: calc(100dvh - 48px);
      overflow-y: auto;
    }
    .layout.col1Open .col-1,
    .layout.col3Open .col-3 {
      transform: translate(0, 0);
    }
    .close-btn {
      display: block;
      position: absolute;
      top: 0;
      right: 0;
      background: transparent;
      border: none;
      color: var(--fg-muted);
      font-size: 20px;
      cursor: pointer;
      min-width: 44px;
      min-height: 44px;
      line-height: 1;
      z-index: 1;
    }
    .close-btn:hover { color: var(--fg); }
    .backdrop {
      display: block;
      position: fixed;
      top: 48px; /* below header */
      left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      border: none;
      z-index: 20;
      cursor: pointer;
    }
  }
</style>
