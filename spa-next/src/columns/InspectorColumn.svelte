<script>
  // wyrd-yxf6 + wyrd-kppy: col 3 — morpheme detail inspector +
  // transform pipeline.
  //
  // Reads appState.currentResult (set by OutputColumn when the user
  // clicks a result card). Renders the CURRENT name (post-pipeline
  // if steps exist, else the original) + per-morpheme cards carrying
  // sources/IPA/reader_pronunciation/original_script.
  //
  // wyrd-kppy: the pipeline engine (lib/pipeline.svelte.js) tracks
  // a user-editable stack of transform steps. An \$effect below
  // re-runs the pipeline whenever the original (col-2 selection)
  // or the step list changes — the "editable recipe" model from
  // the design session (vs an undo/redo snapshot history).
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import MorphemeCard from '../components/MorphemeCard.svelte';
  import TransformStack from '../components/TransformStack.svelte';

  let result = $derived(appState.currentResult);

  // The "original" state the pipeline feeds — the col-2 selection.
  // Snapshotting to { name, morphemes_by_word } pins the shape the
  // pipeline engine expects (transforms return the same shape).
  let original = $derived.by(() => {
    if (!result) return null;
    return {
      name: result.result,
      morphemes_by_word: result.morphemes_by_word || [],
    };
  });

  // wyrd-kppy: clear the pipeline when the inspector subject
  // changes. Without this, the user's steps would silently re-run
  // against a different name when they click a new result. v2
  // (wyrd-34tn save/load) lets users explicitly preserve pipelines
  // across switches.
  let lastResultIndex = $state(null);
  $effect(() => {
    if (appState.currentResultIndex !== lastResultIndex) {
      lastResultIndex = appState.currentResultIndex;
      pipeline.clear();
    }
  });

  // Run the pipeline reactively. Dependencies: original (when col-2
  // selection changes) + pipeline.steps (when user edits the stack).
  // The engine itself handles cancellation of stale runs.
  $effect(() => {
    if (!original) return;
    // Touch the deps so the effect re-runs on either change.
    const stepsDep = pipeline.steps;
    pipeline.run(original);
    // Reference stepsDep so Svelte tracks it; the run uses its own
    // snapshot inside.
    void stepsDep;
  });

  // The currently-displayed state — post-pipeline if any, else the
  // original. Drives the head + morpheme cards.
  let displayState = $derived(pipeline.currentState);

  let allMorphemes = $derived.by(() => {
    if (!displayState?.morphemes_by_word) return [];
    const out = [];
    displayState.morphemes_by_word.forEach((word, wi) => {
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
  {:else if !displayState}
    <p class="placeholder">Computing…</p>
  {:else}
    <header class="head">
      <h3 class="name">
        {displayState.name}
        {#if pipeline.isRunning}
          <span class="pending-flag" title="pipeline running">…</span>
        {/if}
      </h3>
      {#if displayState.morphemes_by_word?.length > 0}
        <p class="breakdown">
          {displayState.morphemes_by_word
            .map((word) => word.map((m) => m.usage).join(' '))
            .join(' · ')}
        </p>
      {/if}
      {#if pipeline.steps.length > 0 && displayState.name !== original.name}
        <p class="provenance">
          from <span class="orig">{original.name}</span>
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
        {#each allMorphemes as morpheme, i (morpheme._wordIndex + ':' + morpheme.usage + ':' + i)}
          <MorphemeCard {morpheme} />
        {/each}
      {/if}
    </section>

    <section class="transforms">
      <h4 class="section-head">Transforms</h4>
      <TransformStack />
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
  .pending-flag {
    font-size: 14px;
    color: var(--fg-muted);
    margin-left: 4px;
  }
  .breakdown {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  .provenance {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .provenance .orig {
    color: var(--fg);
    font-style: normal;
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
