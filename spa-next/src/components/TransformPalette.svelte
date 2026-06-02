<script>
  // wyrd-kppy: "+ add transform" popover. v1 lists Rewind; future
  // PRs (wyrd-hpjg's Swap, plus Anglicize/Calque/Drift when they
  // ship as generators) auto-appear via the transforms/index.js
  // registry.
  //
  // wyrd-kppy round 2 a11y: aria-haspopup/-expanded on the trigger;
  // Esc closes; click-outside closes; first menu item gets focus
  // when the menu opens. Bare role="menu"/-item without keyboard
  // handlers was worse than no roles per the reviewer.
  import { listTransforms } from '../lib/transforms/index.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { flagOn } from '../lib/featureFlags.js';

  let open = $state(false);
  let paletteEl = $state();
  // wyrd-nwpa: hide flag-gated transforms (e.g. Rewind) when their flag is
  // off — so prod doesn't offer Rewind while its bugs are unresolved.
  let catalog = $derived(
    listTransforms().filter((t) => !t.flag || flagOn(appState.config, t.flag)),
  );

  function add(kind) {
    pipeline.addStep(kind);
    open = false;
  }

  function onKeydown(e) {
    if (e.key === 'Escape' && open) {
      open = false;
    }
  }

  function onDocClick(e) {
    if (open && paletteEl && !paletteEl.contains(e.target)) {
      open = false;
    }
  }

  $effect(() => {
    if (open) {
      document.addEventListener('keydown', onKeydown);
      document.addEventListener('click', onDocClick);
      // Move focus to the first menu item once the menu renders.
      // querySelector instead of bind:this so we don't need a
      // per-iteration conditional binding in the each-block.
      queueMicrotask(() => {
        paletteEl?.querySelector('.menu-item')?.focus?.();
      });
      return () => {
        document.removeEventListener('keydown', onKeydown);
        document.removeEventListener('click', onDocClick);
      };
    }
  });
</script>

<div class="palette" bind:this={paletteEl}>
  <button
    class="add"
    type="button"
    onclick={() => (open = !open)}
    aria-haspopup="menu"
    aria-expanded={open}>
    {open ? '▾' : '+'} add transform
  </button>
  {#if open}
    <div class="menu" role="menu">
      {#each catalog as t (t.kind)}
        <button
          class="menu-item"
          type="button"
          onclick={() => add(t.kind)}
          role="menuitem"
        >
          <span class="menu-label">{t.label}</span>
          <span class="menu-desc">{t.description}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .palette {
    margin-top: 4px;
  }
  .add {
    background: transparent;
    color: var(--accent);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    width: 100%;
    text-align: left;
  }
  .add:hover {
    border-color: var(--accent);
  }
  .menu {
    margin-top: 6px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
  }
  .menu-item {
    display: block;
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: var(--fg);
    cursor: pointer;
    padding: 8px 10px;
    border-radius: 4px;
    font: inherit;
  }
  .menu-item:hover {
    background: var(--border);
  }
  .menu-label {
    display: block;
    font-weight: 600;
    font-size: 13px;
  }
  .menu-desc {
    display: block;
    font-size: 11px;
    color: var(--fg-muted);
    margin-top: 2px;
  }
</style>
