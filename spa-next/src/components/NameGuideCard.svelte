<script>
  // wyrd-2ien: reusable "name + pronunciation/gloss guide" card. Rendered ONCE
  // for a plain (modern) name, or TWICE for an era render — the generated era
  // name + a dimmed modern card beside it. Pure presentation: the parent builds
  // the per-morpheme rows (surface / reader / ipa / gloss) for whichever form
  // the card represents, so the same component composes both.
  let { name, label = '', rows = [], dim = false, headingId = undefined } = $props();
</script>

<div class="name-guide" class:dim>
  <div class="ng-head">
    {#if headingId}
      <h3 class="ng-name" id={headingId}>{name}</h3>
    {:else}
      <span class="ng-name">{name}</span>
    {/if}
    {#if label}<span class="ng-label">{label}</span>{/if}
  </div>
  {#if rows.length}
    <div class="ng-guide" aria-label="pronunciation guide">
      {#each rows as r}
        <span class="ng-col">
          <span class="ng-surface">{r.surface}</span>
          <span class="ng-reader">{r.reader || '·'}</span>
          {#if r.ipa}<span class="ng-ipa">{r.ipa}</span>{/if}
          {#if r.gloss}<span class="ng-gloss" title={r.gloss}>{r.gloss}</span>{/if}
        </span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .name-guide {
    flex: 1 1 280px;
    min-width: 0;
  }
  .ng-head {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .ng-name {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    color: var(--fg);
    letter-spacing: -0.01em;
  }
  /* the modern card is the anchor, not the headline — dim it a touch. */
  .name-guide.dim .ng-name {
    opacity: 0.55;
    font-weight: 600;
  }
  .ng-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1px 5px;
    white-space: nowrap;
  }
  .ng-guide {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
  }
  .ng-col {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .ng-surface {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
    color: var(--fg);
  }
  .ng-reader {
    font-size: 11px;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .ng-ipa {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 10px;
    color: var(--fg-muted);
  }
  .ng-gloss {
    font-size: 10px;
    color: var(--fg-muted);
    max-width: 14ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
