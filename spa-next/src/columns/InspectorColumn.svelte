<script>
  // wyrd-yxf6: col 3 — morpheme detail inspector.
  //
  // Reads appState.currentResult (set by OutputColumn when the user
  // clicks a result card). Renders the rendered name + per-morpheme
  // cards carrying sources/IPA/reader_pronunciation/original_script
  // — the wyrd-cp2d + wyrd-03cx + wyrd-pr9g data now flowing through
  // the API envelope's morphemes_by_word field.
  //
  // The "Transforms" section below the morphemes is a stub; wyrd-kppy
  // (PR #4) wires the pipeline editor + Rewind transform.
  import { appState } from '../lib/appState.svelte.js';
  import MorphemeCard from '../components/MorphemeCard.svelte';

  let result = $derived(appState.currentResult);

  // Flatten morphemes_by_word into a single ordered list. The
  // per-card layout doesn't currently visually emphasize word
  // boundaries (the rendered name already shows the space-shape).
  let allMorphemes = $derived.by(() => {
    if (!result?.morphemes_by_word) return [];
    const out = [];
    result.morphemes_by_word.forEach((word, wi) => {
      word.forEach((m) => out.push({ ...m, _wordIndex: wi }));
    });
    return out;
  });
</script>

<section class="column">
  <h2>Inspect &amp; Transform</h2>

  {#if !result}
    <p class="placeholder">
      Click a result in the middle column to inspect its morphemes,
      etymology, and pronunciation.
    </p>
  {:else}
    <header class="head">
      <h3 class="name">{result.result}</h3>
      {#if result.morphemes_by_word?.length > 0}
        <p class="breakdown">
          {result.morphemes_by_word
            .map((word) => word.map((m) => m.usage).join(' '))
            .join(' · ')}
        </p>
      {/if}
    </header>

    <section class="morphemes">
      <h4 class="section-head">
        Morphemes ({allMorphemes.length})
      </h4>

      {#if allMorphemes.length === 0}
        <p class="placeholder">
          This generator doesn't expose per-morpheme metadata.
        </p>
      {:else}
        {#each allMorphemes as morpheme, i (i)}
          <MorphemeCard {morpheme} />
        {/each}
      {/if}
    </section>

    <section class="transforms">
      <h4 class="section-head">Transforms</h4>
      <p class="placeholder">
        Pipeline editor + Rewind / Swap / Anglicize transforms land
        in PR #4 (wyrd-kppy).
      </p>
    </section>
  {/if}
</section>

<style>
  .head {
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .name {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: var(--fg);
  }
  .breakdown {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  .section-head {
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 12px;
  }
  .transforms {
    margin-top: 24px;
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
    line-height: 1.5;
  }
</style>
