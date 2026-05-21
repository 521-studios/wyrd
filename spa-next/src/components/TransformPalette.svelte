<script>
  // wyrd-kppy: "+ add transform" popover. v1 lists Rewind; future
  // PRs (wyrd-hpjg's Swap, plus Anglicize/Calque/Drift when they
  // ship as generators) auto-appear via the transforms/index.js
  // registry.
  import { listTransforms } from '../lib/transforms/index.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let open = $state(false);
  let catalog = listTransforms();

  function add(kind) {
    pipeline.addStep(kind);
    open = false;
  }
</script>

<div class="palette">
  <button class="add" type="button" onclick={() => (open = !open)}>
    {open ? '▾' : '+'} add transform
  </button>
  {#if open}
    <div class="menu" role="menu">
      {#each catalog as t (t.kind)}
        <button class="menu-item" type="button" onclick={() => add(t.kind)} role="menuitem">
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
