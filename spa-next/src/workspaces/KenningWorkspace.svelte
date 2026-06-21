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
  import { appState } from '../lib/appState.svelte.js';
  import ConfigureColumn from '../columns/ConfigureColumn.svelte';
  import OutputColumn from '../columns/OutputColumn.svelte';
  import InspectorColumn from '../columns/InspectorColumn.svelte';

  let col1Open = $state(false);
  let col3Open = $state(false);

  // wyrd-f5if: bind:this on each mobile panel so the focus trap can scope its
  // Tab cycle to the open one.
  let col1El = $state();
  let col3El = $state();

  // wyrd-14hn round 2 (frontend LOW): viewport tracking lifted to
  // appState.isMobileViewport (App.svelte owns the matchMedia
  // listener). Resize from mobile→desktop still needs to dismiss
  // any open drawer/sheet, but as a reaction to the upstream flag.
  $effect(() => {
    if (!appState.isMobileViewport) {
      col1Open = false;
      col3Open = false;
    }
  });

  // Mobile auto-open of col 3 when a result is tapped.
  $effect(() => {
    if (appState.isMobileViewport && appState.currentResultIndex !== null) {
      col3Open = true;
    }
  });

  // wyrd-f5if: focus trap + restore for the open mobile panel. Mirrors the
  // proven DefectModal pattern (stash activeElement on open, Esc + Tab cycle,
  // restore on close). The effect re-runs whenever col1Open/col3Open changes,
  // so triggerBefore is captured per run — a panel swap (a result tapped while
  // the drawer is up) briefly restores then re-stashes, landing focus in the
  // foreground panel either way.
  $effect(() => {
    if (!col1Open && !col3Open) return;
    // Foreground panel the trap scopes to: the sheet (col3) wins when both are
    // open, since it's the interaction that surfaced it (a tapped result).
    const trapEl = col3Open ? col3El : col1El;
    const triggerBefore = document.activeElement;

    const onKey = (e) => {
      if (e.key === 'Escape') {
        closeAll();
        return;
      }
      if (e.key !== 'Tab' || !trapEl) return;
      const focusables = trapEl.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !trapEl.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !trapEl.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Focus the foreground panel's close button so keyboard/AT users land
    // inside the dialog rather than behind the backdrop.
    queueMicrotask(() => trapEl?.querySelector('.close-btn')?.focus?.());

    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      if (triggerBefore && typeof triggerBefore.focus === 'function') {
        try {
          triggerBefore.focus();
        } catch {
          /* opener detached from the DOM */
        }
      }
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
    bind:this={col1El}
    role={appState.isMobileViewport ? 'dialog' : undefined}
    aria-modal={appState.isMobileViewport && col1Open ? 'true' : undefined}
    aria-label={appState.isMobileViewport ? 'Configure' : undefined}
    aria-hidden={appState.isMobileViewport && !col1Open ? 'true' : undefined}
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
    bind:this={col3El}
    role={appState.isMobileViewport ? 'dialog' : undefined}
    aria-modal={appState.isMobileViewport && col3Open ? 'true' : undefined}
    aria-label={appState.isMobileViewport ? 'Inspect and Transform' : undefined}
    aria-hidden={appState.isMobileViewport && !col3Open ? 'true' : undefined}
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
    /* wyrd-vith: pin the single row to the grid height (minmax(0,1fr), NOT
       the default content-sized `auto` row) so a tall column can't stretch
       the grid past the viewport; the column's own overflow-y:auto then
       scrolls internally and no second page scrollbar appears. */
    grid-template-rows: minmax(0, 1fr);
    overflow: hidden;
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
