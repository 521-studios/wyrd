<script>
  // wyrd-rogd / wyrd-qc0g: col 3 — Inspect & Transform, rebuilt on the
  // family × era axis the backend models (era_grid, wyrd-lftl) instead of the
  // old per-language renderings axis.
  //
  // Structure (skeleton; sections refined in wyrd-yrf9 / wyrd-zw1f / wyrd-410t):
  //   1. Name cards — the active (as-generated) name + a de-accented line +
  //      placement-consistent breakdown, beside a modern paragon.
  //   2. (time-warp button bar — wyrd-410t, not yet built)
  //   3. Per-morpheme blocks — gloss + tags + the family × era variant grid.
  //
  // The pipeline engine (lib/pipeline.svelte.js) is unchanged and still owns
  // the working state (swap steps + a flag-gated rewind step) so save/share
  // keep working; this view only renders it. The active card reads the working
  // pipeline.currentState (`displayState`); the paragon is the one thing
  // intentionally pinned to the pre-pipeline `original`. That's the only
  // remaining two-state split — gone is the old column's convoluted
  // per-derived original-vs-displayState juggling (the lossless-accent gamble,
  // era-reads-one / modern-reads-the-other) that drove the swap bugs.
  import { untrack } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { languageLabel } from '../lib/languageLabels.js';
  import { pronunciationFor } from '../lib/variants.js';
  import { deAccent, cellForSurface, hasEraGrid } from '../lib/era.js';
  import NameGuideCard from '../components/NameGuideCard.svelte';
  import MorphemeGrid from '../components/MorphemeGrid.svelte';
  import DefectModal from '../components/DefectModal.svelte';

  let result = $derived(appState.currentResult);

  // "Report defective" flags the reproducible generator output (currentResult),
  // not the post-swap state. Track the specific result so the modal closes
  // synchronously when the inspected result changes (derived, no flash).
  let flaggedResult = $state(null);
  $effect(() => {
    const current = result;
    untrack(() => {
      if (flaggedResult !== null && flaggedResult !== current) flaggedResult = null;
    });
  });

  // The pre-pipeline result the pipeline runs against + the paragon pins to.
  let original = $derived(
    result ? { name: result.result, morphemes_by_word: result.morphemes_by_word || [] } : null,
  );

  // Subject-change + pipeline-run effect (engine; carried over from wyrd-kppy):
  // clearing the pipeline on a new result is atomic with kicking off the run.
  let lastResultIndex = $state(null);
  $effect(() => {
    const stepsSnapshot = $state.snapshot(pipeline.steps);
    const idx = appState.currentResultIndex;
    const isLoad = untrack(() => {
      const val = appState.isLoadingSavedWorkspace;
      if (val) appState.isLoadingSavedWorkspace = false;
      return val;
    });
    if (idx !== lastResultIndex) {
      lastResultIndex = idx;
      if (!isLoad) pipeline.clear();
    }
    if (!original) return;
    pipeline.run(original);
    void stepsSnapshot;
  });

  // The working state — post-pipeline if any step ran, else the original.
  let displayState = $derived(pipeline.currentState);

  let allMorphemes = $derived.by(() => {
    const out = [];
    (displayState?.morphemes_by_word || []).forEach((word, wi) => {
      word.forEach((m, mi) => out.push({ ...m, _wordIndex: wi, _morphemeIndex: mi }));
    });
    return out;
  });

  // A morpheme's primary gloss — first sense of the first group, else first
  // flat meaning.
  const primaryDef = (m) => m?.meaning_groups?.[0]?.[0] || m?.meanings?.[0] || '';

  // The pronunciation slot for a morpheme's CURRENT surface: the matching
  // era_grid cell (the right axis) first, then the era render's own pron, then
  // the legacy per-language fallback (kept until the grid fully owns
  // pronunciation in wyrd-zw1f). Sparse — coverage grows with mining.
  function pronFor(m, surface) {
    return cellForSurface(m, surface)?.cell || m.rendered_pron || pronunciationFor(m) || {};
  }

  // Build the surface / reader / ipa / gloss rows for a morpheme list. Surfaces
  // keep their placement dashes (tre- / -bȳ / hall) — never stripped.
  function rowsFor(words, surfaceOf) {
    return (words || []).flatMap((word) =>
      word
        .filter((m) => m.usage?.trim())
        .map((m) => {
          const surface = surfaceOf(m);
          const slot = pronFor(m, surface);
          return {
            surface,
            reader: slot.reader_pronunciation,
            ipa: slot.ipa,
            gloss: primaryDef(m),
          };
        }),
    );
  }

  // Active card: the name as generated + each morpheme's live (era) surface.
  let activeRows = $derived.by(() =>
    rowsFor(displayState?.morphemes_by_word, (m) => m.rendered || m.usage),
  );
  // Era badge: the era a render targets, else "as generated". (Mixed-era
  // detection after per-morpheme swaps is a wyrd-yrf9 refinement.)
  let eraLanguage = $derived(
    (original?.morphemes_by_word || []).flat().find((m) => m.rendered_language)?.rendered_language,
  );
  let eraLabel = $derived(eraLanguage ? languageLabel(eraLanguage) : 'as generated');

  // Paragon: the stable, de-accented version of the ORIGINAL name (pinned —
  // unaffected by swaps). A true modern-reflex paragon awaits cleaner modern
  // data (wyrd-rogd.1 / wyrd-yrf9); de-accent is the reliable skeleton form.
  let paragonName = $derived(deAccent(original?.name || ''));
  let paragonRows = $derived.by(() =>
    rowsFor(original?.morphemes_by_word, (m) => deAccent(m.usage)),
  );
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
      <div class="head-top">
        {#if pipeline.isRunning}
          <span class="pending-flag" title="pipeline running">…</span>
        {/if}
        <button
          type="button"
          class="flag"
          onclick={() => (flaggedResult = result)}
          title="Report defective"
        ><span aria-hidden="true">⚑</span> Report defective</button>
      </div>

      <div class="name-cards">
        <NameGuideCard
          headingId="inspect-result-name"
          name={displayState.name}
          altName={deAccent(displayState.name)}
          label={eraLabel}
          rows={activeRows}
        />
        {#if paragonRows.length}
          <NameGuideCard name={paragonName} label="modern" dim rows={paragonRows} />
        {/if}
      </div>
    </header>

    <section class="morphemes">
      <h4 class="section-head">Morphemes ({allMorphemes.length})</h4>

      {#if allMorphemes.length === 0}
        <p class="placeholder">This generator doesn't expose per-morpheme metadata.</p>
      {:else}
        {#each allMorphemes as m (m._wordIndex + ':' + m._morphemeIndex)}
          <article class="morpheme">
            <div class="m-head">
              <span class="m-usage">{m.rendered || m.usage}</span>
              {#if m.tags?.length}
                <span class="m-tags">
                  {#each m.tags as tag}<span class="tag">{tag}</span>{/each}
                </span>
              {/if}
            </div>

            {#if m.meaning_groups?.length}
              <div class="meaning-groups">
                {#each m.meaning_groups as group}
                  <p class="meaning-group">{group.join(', ')}</p>
                {/each}
              </div>
            {:else if m.meanings?.length}
              <p class="meanings">{m.meanings.join(', ')}</p>
            {/if}

            {#if hasEraGrid(m)}
              <MorphemeGrid morpheme={m} />
            {:else}
              <p class="no-grid">No era variants yet for this morpheme.</p>
            {/if}
          </article>
        {/each}
      {/if}
    </section>

    <DefectModal
      open={flaggedResult === result}
      {result}
      onclose={() => (flaggedResult = null)}
    />
  {/if}
</section>

<style>
  .head {
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .head-top {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: 10px;
  }
  .name-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px 24px;
    align-items: start;
    margin-top: 4px;
  }
  .flag {
    flex-shrink: 0;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--fg-muted);
    cursor: pointer;
    font: inherit;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 3px;
    white-space: nowrap;
  }
  .flag:hover {
    color: #e06c6c;
    border-color: #e06c6c;
  }
  .flag:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .pending-flag {
    font-size: 14px;
    color: var(--fg-muted);
    margin-left: 4px;
  }
  .section-head {
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 12px;
  }
  .morpheme {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  .m-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 6px;
  }
  .m-usage {
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .m-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .tag {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    background: var(--border);
    color: var(--fg-muted);
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .meaning-groups {
    margin: 0 0 8px;
  }
  .meaning-group {
    margin: 0;
    padding-left: 8px;
    border-left: 2px solid var(--border);
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
  }
  .meaning-group + .meaning-group {
    margin-top: 4px;
  }
  .meanings {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
  }
  .no-grid {
    margin: 6px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
    line-height: 1.5;
  }
</style>
