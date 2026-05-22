<script>
  // wyrd-z3lp + wyrd-jh75: 3-column layout.
  //
  // Desktop (≥900px): 3-column grid (280 / 360 / 1fr).
  // Mobile (<900px): col 2 fills the screen; col 1 slides in from
  // the left as a drawer (☰ button at top of col 2); col 3 slides
  // up from the bottom as a sheet (auto-opens when the user taps
  // a result; close-X or backdrop-tap dismisses).
  //
  // State lives here rather than appState because no other column
  // needs to react to drawer/sheet open state; if mobile-state ever
  // needs to drive other behavior (deep-link landing rules, etc.)
  // promote to appState.
  import { appState } from './lib/appState.svelte.js';
  import ConfigureColumn from './columns/ConfigureColumn.svelte';
  import OutputColumn from './columns/OutputColumn.svelte';
  import InspectorColumn from './columns/InspectorColumn.svelte';

  let col1Open = $state(false);
  let col3Open = $state(false);
  let isMobileViewport = $state(false);

  // wyrd-jh75 round 2 (frontend MED): track viewport size so the
  // auto-open effect only fires on mobile. Resize between mobile
  // and desktop also auto-dismisses any open drawer/sheet (their
  // state becomes irrelevant on desktop).
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

  // Auto-open the inspector sheet when the user taps a result on
  // mobile. The viewport guard means desktop never sets col3Open
  // (which would be a no-op anyway since CSS hides the sheet
  // chrome on desktop, but the state stays clean).
  $effect(() => {
    if (isMobileViewport && appState.currentResultIndex !== null) {
      col3Open = true;
    }
  });

  // wyrd-jh75 round 2 (frontend HIGH): Esc closes any open
  // drawer/sheet; body scroll-lock when a sheet is open so iOS
  // Safari touchmove doesn't bleed through to col-2 underneath.
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
</script>

<div class="layout" class:col1Open class:col3Open>
  <!-- Mobile-only ☰ + ✕ trigger row, hidden on desktop via CSS. -->
  <header class="mobile-bar">
    <button
      class="bar-btn menu"
      type="button"
      onclick={() => (col1Open = !col1Open)}
      aria-label="Configure menu"
      aria-expanded={col1Open}
    >☰</button>
    <span class="bar-title">wyrd</span>
  </header>

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
      aria-label="Close configure"
    >✕</button>
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
      aria-label="Close inspector"
    >✕</button>
    <InspectorColumn />
  </div>

  <!-- Backdrop: visible only on mobile when a drawer/sheet is open;
       click to dismiss. -->
  {#if col1Open || col3Open}
    <button
      class="backdrop"
      type="button"
      aria-label="Close"
      onclick={closeAll}
    ></button>
  {/if}
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: var(--col-1-width) var(--col-2-width) 1fr;
    height: 100dvh;
  }
  .mobile-bar {
    display: none;
  }
  /* On desktop, .col is a transparent wrapper around the column
     component (which has its own .column with border-right + scroll).
     On mobile (<900px), .col gets positioning + transform via the
     media query below. */
  .col {
    position: relative;
    display: contents;
  }
  .close-btn {
    display: none;
  }
  .backdrop {
    display: none;
  }

  /* wyrd-jh75 mobile layout: <900px viewport.
     - Layout collapses to a single column (col 2 as home).
     - mobile-bar shows at the top with the ☰ trigger.
     - col 1 fixed-position slide-in from left.
     - col 3 fixed-position slide-up sheet from bottom (75% height).
     - backdrop is a full-viewport tap-target behind both drawers. */
  @media (max-width: 899px) {
    .layout {
      display: block;
      height: 100dvh;
      overflow: hidden;
    }
    .mobile-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      height: 44px;
      padding: 0 12px;
      background: var(--bg);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .bar-btn {
      /* wyrd-jh75 round 2 (frontend HIGH): 44x44 minimum touch
         target per WCAG / iOS HIG. Pre-fix the visible icon at
         20px font + 4/8 padding gave ~28x28. */
      background: transparent;
      border: none;
      color: var(--fg);
      font-size: 20px;
      cursor: pointer;
      min-width: 44px;
      min-height: 44px;
      line-height: 1;
    }
    .bar-title {
      font-weight: 600;
      letter-spacing: 0.04em;
    }
    .col {
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .col {
      display: block;
    }
    .col-1 {
      position: fixed;
      top: 0;
      left: 0;
      width: min(80vw, 320px);
      height: 100dvh;
      background: var(--bg);
      transform: translateX(-100%);
      transition: transform 220ms ease;
      z-index: 30;
      padding-top: 8px;
      overflow-y: auto;
    }
    /* Inner column components have their own border-right; on mobile
       they're drawer/sheet not grid columns, so suppress. */
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
      height: calc(100dvh - 44px);
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
      /* wyrd-jh75 round 2 (frontend HIGH): 44x44 min touch target. */
      min-width: 44px;
      min-height: 44px;
      line-height: 1;
      z-index: 1;
    }
    .close-btn:hover {
      color: var(--fg);
    }
    .backdrop {
      display: block;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      border: none;
      z-index: 20;
      cursor: pointer;
    }
  }
</style>
