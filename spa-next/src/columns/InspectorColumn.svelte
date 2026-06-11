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
  import { tick, untrack } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { languageLabel } from '../lib/languageLabels.js';
  import {
    deAccent,
    hasEraGrid,
    pronForSurface,
    eraBadge,
    primaryGloss,
    glossForSurface,
    isGlossDrift,
  } from '../lib/era.js';
  import { showModernCompanion } from '../lib/render.js';
  import { flagOn } from '../lib/featureFlags.js';
  import NameGuideCard from '../components/NameGuideCard.svelte';
  import MorphemeGrid from '../components/MorphemeGrid.svelte';
  import TransformStep from '../components/TransformStep.svelte';
  import DefectModal from '../components/DefectModal.svelte';

  let result = $derived(appState.currentResult);

  // wyrd-y0lx: per-morpheme regenerate. Kenning-only (the endpoint re-rolls
  // against the kenning vector path) and flag-gated per the wyrd-nwpa
  // convention (prod hides it until WYRD_FF_REGENERATE_MORPHEME / FF_ALL).
  let regenEnabled = $derived(
    appState.resultsGenerator === 'kenning' &&
      flagOn(appState.config, 'regenerate-morpheme'),
  );

  function regenerate(m) {
    // Bake the roll's generation context into the step at click time so
    // pipeline re-runs + restored workspaces replay against the params that
    // produced the result (resultsParams is the roll-time snapshot; fall
    // back to the live form bag for loaded workspaces that predate it).
    const gen = appState.resultsGenerator;
    const context = $state.snapshot(
      appState.resultsParams || appState.paramsByGenerator[gen] || {},
    );
    delete context.count;
    delete context.seed;
    pipeline.setRegenerate({
      wordIndex: m._wordIndex,
      morphemeIndex: m._morphemeIndex,
      context,
      // The slot's CURRENT surface, so the step card reads
      // "Stān morph[0,1] → re-roll" instead of a bare index.
      from: m.rendered || m.usage,
    });
  }

  // Reset the inspector's scroll region to the top whenever a different result
  // is selected — otherwise clicking a new name lands you mid-list at the old
  // scroll offset. Only ``.morphemes`` scrolls (the name cards are pinned), so
  // we reset that element. ``tick()`` waits for the new morphemes to render so
  // scrollTop=0 applies to the new content's layout.
  let morphemesEl = $state(null);
  $effect(() => {
    const idx = appState.currentResultIndex; // track: re-run on selection change
    void idx;
    tick().then(() => {
      if (morphemesEl) morphemesEl.scrollTop = 0;
    });
  });

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
      (word || []).forEach((m, mi) => out.push({ ...m, _wordIndex: wi, _morphemeIndex: mi }));
    });
    return out;
  });

  // Build the surface / reader / ipa / gloss rows for a morpheme list. Surfaces
  // keep their placement dashes (tre- / -bȳ / hall) — never stripped. The
  // pronunciation slot comes from pronForSurface (era_grid cell → rendered_pron
  // → legacy fallback, skipping pronless cells).
  //
  // wyrd-rogd.1: when `driftAware`, the gloss TRACKS the live variant — a
  // swapped era cell carries its OWN meaning (a cognate that may have drifted),
  // so the active card shows the variant's gloss and flags `drift` when it
  // differs from the morpheme's base meaning. The paragon stays base (not
  // drift-aware).
  function rowsFor(words, surfaceOf, driftAware = false) {
    return (words || []).flatMap((word) =>
      (word || [])
        .filter((m) => m?.usage?.trim())
        .map((m) => {
          const surface = surfaceOf(m);
          const slot = pronForSurface(m, surface);
          const base = primaryGloss(m);
          const variant = driftAware ? glossForSurface(m, surface) : '';
          const gloss = variant || base;
          return {
            surface,
            reader: slot.reader_pronunciation,
            ipa: slot.ipa,
            gloss,
            drift: isGlossDrift(base, variant),
          };
        }),
    );
  }

  // Active card: the name as generated + each morpheme's live (era) surface,
  // gloss tracking the swapped variant (drift-aware).
  let activeRows = $derived.by(() =>
    rowsFor(displayState?.morphemes_by_word, (m) => m.rendered || m.usage, true),
  );
  // wyrd-yrf9: the active card's era badge, resolved over the WORKING state
  // (displayState) so it tracks per-morpheme grid swaps — "Old English" when
  // every morpheme shares an era, "Mixed (…)" for a blend, "as generated" for
  // an untouched modern roll. Logic lives in eraBadge (lib/era.js, unit-tested).
  let eraLabel = $derived(eraBadge(displayState?.morphemes_by_word, languageLabel));

  // Paragon: the MODERN companion of the as-generated name, pinned to the
  // ORIGINAL (unaffected by swaps). wyrd-24s6 (D41): the backend now renders
  // BOTH surfaces, so `result_modern` is the true modern reflex — used directly,
  // no longer the de-accented native skeleton. Only the FALLBACK path (a
  // generator that doesn't distinguish the two renderings, so `result_modern` is
  // absent) de-accents the native name as a best-effort modern skeleton. Per
  // morpheme the modern surface is `usage` (the modern bucket key), de-accented
  // in `paragonRows` below (modern reflexes are ASCII anyway).
  let paragonName = $derived(result?.result_modern || deAccent(original?.name || ''));
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
        <!-- wyrd-swh2: only show the modern paragon when it differs from the
             native (result_modern !== result), matching the Output column —
             a native==modern roll would otherwise duplicate the card. -->
        {#if paragonRows.length && showModernCompanion(result)}
          <NameGuideCard name={paragonName} label="modern" dim rows={paragonRows} />
        {/if}
      </div>
    </header>

    <!-- wyrd-y0lx: the transform stack, re-mounted after the wyrd-qc0g col-3
         rebuild dropped it. Each step card carries its × remove control —
         removing a step (e.g. a regenerate) re-runs the pipeline without it,
         which is the undo affordance for per-morpheme regeneration. The
         palette stays unmounted (wyrd-410t's concern); steps are added via
         direct manipulation (grid-cell swaps, the ⟳ regenerate button). -->
    {#if pipeline.steps.length > 0}
      <section class="transforms">
        <h4 class="section-head">Transforms ({pipeline.steps.length})</h4>
        {#each pipeline.steps as step, i (step.id)}
          <TransformStep {step} index={i} />
        {/each}
      </section>
    {/if}

    <section class="morphemes" bind:this={morphemesEl}>
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
              <!-- wyrd-y0lx: re-roll JUST this morpheme in the context of the
                   others. Adds (or re-rolls in place) a regenerate step on the
                   pipeline; remove the step from the Transforms stack to undo. -->
              {#if regenEnabled}
                <button
                  type="button"
                  class="m-regen"
                  disabled={pipeline.isRunning}
                  onclick={() => regenerate(m)}
                  aria-label="Regenerate this morpheme"
                  title="Regenerate this morpheme (re-roll it in the context of the others)"
                >⟳</button>
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

            <!-- wyrd-rogd.14: collapsed-by-default scholarly-source citations,
                 restored after the col-3 era-axis rebuild (wyrd-qc0g) dropped
                 the wyrd-bvwu disclosure. Only morphemes with >=1 citation get
                 the fold (matches the promotion-threshold model). -->
            {#if m.citations?.length}
              <details class="citations">
                <summary>Citations ({m.citations.length})</summary>
                <ul>
                  {#each m.citations as src}<li>{src}</li>{/each}
                </ul>
              </details>
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
  /* wyrd-rogd.8: pin the name cards — the h2 + the head (active/paragon cards)
     stay put; only the morpheme cards scroll. Overrides the app.css whole-
     column `overflow-y: auto` with a flex column whose .morphemes is the lone
     scroll region. (Scoped, so only col 3 changes.) */
  .column {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .morphemes {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  .head {
    flex-shrink: 0;
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
  /* wyrd-y0lx: per-morpheme regenerate button, pinned top-right of the card
     head (margin-left auto pushes it past the usage + tags). */
  .m-regen {
    margin-left: auto;
    align-self: center;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--fg-muted);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    line-height: 1;
    padding: 3px 7px;
  }
  .m-regen:hover:not(:disabled) {
    color: var(--accent);
    border-color: var(--accent);
  }
  .m-regen:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .m-regen:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  /* wyrd-y0lx: the re-mounted transform stack — pinned (non-scroll) between
     the name cards and the scrolling morpheme list, capped so a long stack
     scrolls itself rather than crowding out the morphemes. */
  .transforms {
    flex-shrink: 0;
    max-height: 30%;
    overflow-y: auto;
    margin-bottom: 12px;
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
  /* wyrd-rogd.14: collapsed-by-default scholarly-citation disclosure. */
  .citations {
    margin-top: 10px;
    font-size: 11px;
    color: var(--fg-muted);
  }
  .citations summary {
    cursor: pointer;
    user-select: none;
    letter-spacing: 0.02em;
  }
  .citations ul {
    margin: 6px 0 0;
    padding-left: 18px;
    list-style: disc;
  }
  .citations li {
    margin: 2px 0;
    word-break: break-word;
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
