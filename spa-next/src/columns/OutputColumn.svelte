<script>
  // wyrd-hcmc + wyrd-8jjx: col 2 — current roll output. wyrd-8jjx
  // dropped the Saved view-toggle since the Saved library now opens
  // as a header-triggered drawer (universal across workspaces).
  // Star icon on each result stays inline (per-result quick save).
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import StarToggle from '../components/StarToggle.svelte';
  import {
    representativeMeanings,
    isNameMorpheme,
  } from '../lib/morphemeGloss.js';
  import { accentedName } from '../lib/accents.js';
  // wyrd-24s6 (D41): native/modern surface predicates live in lib/render.js so
  // they're unit-testable (svelte components have no test harness here).
  import { nativeSurface, modernSurface, showModernCompanion } from '../lib/render.js';

  function selectResult(i) {
    appState.currentResultIndex =
      i === appState.currentResultIndex ? null : i;
  }

  // wyrd-z3fl: "report defective" moved to the Inspect & Transform column
  // (col 3) — you flag the result you're inspecting there, not per-row here.

  // The 1–2 word gloss shown under a morpheme. Falls back to a dim "name"
  // marker for proper-name elements (manorial families etc.) — but only when
  // the surface is capitalized, so lowercase connectives (and/et/be) whose
  // pooled meanings happen to include "a personal name" stay blank.
  function glossFor(morph) {
    const g = representativeMeanings(morph.meanings);
    if (g.length) return g.join(' · ');
    const looksProper = /^[A-Z]/.test(morph.usage || '');
    return looksProper && isNameMorpheme(morph) ? 'name' : '';
  }

  // wyrd-y0lx (operator decision): the row being inspected LIVE-RENDERS the
  // pipeline's current state — every transform (regenerate / swap / rewind)
  // shows in the Output column, not just in col 3 — so the transformed name
  // is what the ★ bookmarks. Gated on steps.length so an untouched row keeps
  // its accent-upgraded original rendering; other rows are never affected.
  function liveStateFor(i) {
    if (i !== appState.currentResultIndex || pipeline.steps.length === 0) {
      return null;
    }
    return pipeline.currentState;
  }
</script>

<section class="column">
  <h2>Output</h2>

  {#if appState.results.length === 0}
    <p class="placeholder">
      No rolls yet. Configure a generator on the left + hit Roll.
    </p>
  {:else}
    <p class="meta">
      {appState.results.length}
      {appState.results.length === 1 ? 'result' : 'results'}
      from <strong>{appState.resultsGenerator}</strong>
    </p>
    <ul class="results">
      {#each appState.results as r, i (i)}
        {@const live = liveStateFor(i)}
        {@const mbw = live?.morphemes_by_word?.length
          ? live.morphemes_by_word
          : r.morphemes_by_word}
        <li>
          <div class="result-row">
            <button
              class="result"
              class:selected={appState.currentResultIndex === i}
              onclick={() => selectResult(i)}
            >
              <span class="name-line">
                <span class="name">{live ? live.name : accentedName(r)}</span>
                <!-- wyrd-24s6 (D41): the modern companion, in darker secondary
                     lettering to the right. Shown only when it differs from the
                     native canonical (a plain/force-modern roll has native ==
                     modern, so the companion would be noise). wyrd-c6o1.1: a
                     transform nulls result_modern in the committed store (the
                     original reflex no longer describes the edited name), so
                     showModernCompanion alone hides it — the prior
                     live.name===r.result guard is now always true (the pipeline
                     commits its name into r.result) and was dropped. -->
                {#if showModernCompanion(r)}
                  <span class="name-modern" title="modern reflex">{r.result_modern}</span>
                {/if}
              </span>
              {#if mbw?.length}
                <span class="etymology">
                  {#each mbw as word}
                    <span class="word-group">
                      {#each word as morph}
                        {@const g = glossFor(morph)}
                        {@const nat = nativeSurface(morph)}
                        {@const mod = modernSurface(morph)}
                        <span class="morph-col">
                          <!-- wyrd-24s6 (D41): native surface primary, modern
                               reflex in darker lettering beneath it when the two
                               differ (it matches the name's native + modern pair). -->
                          <span class="surface">{nat}</span>
                          {#if mod && mod !== nat}<span class="surface-modern" title="modern reflex">{mod}</span>{/if}
                          {#if g}<span class="gloss">{g}</span>{/if}
                        </span>
                      {/each}
                    </span>
                  {/each}
                </span>
              {:else if r.explanation}
                <span class="explanation">{r.explanation}</span>
              {/if}
            </button>
            <StarToggle result={r} />
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .meta {
    font-size: 11px;
    color: var(--fg-muted);
    margin: 0 0 12px;
  }
  .results {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .results li {
    margin-bottom: 6px;
  }
  .result-row {
    display: flex;
    align-items: stretch;
    gap: 4px;
  }
  .result {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    text-align: left;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 12px;
    color: var(--fg);
    cursor: pointer;
    font: inherit;
  }
  .result:hover {
    border-color: var(--accent);
  }
  .result.selected {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
  }
  /* wyrd-24s6 (D41): native name primary, with the modern reflex companion
     stacked directly beneath it (wyrd-swh2) in darker, lighter-weight type. */
  .name-line {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
  }
  .name {
    font-size: 15px;
    font-weight: 600;
  }
  .name-modern {
    font-size: 13px;
    font-weight: 400;
    color: var(--fg-muted);
  }
  /* wyrd-2b50: aligned etymology grid — each morpheme's surface form sits
     directly above its 1–2 word gloss; words are spaced apart, wrap as a
     group. Replaces the old multi-line meaning dump. */
  .etymology {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .word-group {
    display: flex;
    gap: 10px;
  }
  .morph-col {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1px;
  }
  .surface {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 12px;
    color: var(--fg);
  }
  /* wyrd-24s6 (D41): the modern reflex of a morpheme, dimmer + smaller, sitting
     directly under its native surface. */
  .surface-modern {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
    color: var(--fg-muted);
  }
  .gloss {
    font-size: 11px;
    color: var(--fg-muted);
    line-height: 1.3;
  }
  .explanation {
    margin-top: 4px;
    font-size: 11px;
    color: var(--fg-muted);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
    padding: 24px 0;
  }
</style>
