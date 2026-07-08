<script>
  // wyrd-hcmc + wyrd-8jjx: col 2 — current roll output. wyrd-8jjx
  // dropped the Saved view-toggle since the Saved library now opens
  // as a header-triggered drawer (universal across workspaces).
  // Star icon on each result stays inline (per-result quick save).
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import StarToggle from '../components/StarToggle.svelte';
  import { glossFor, glossesByResults } from '../lib/morphemeGloss.js';
  import { accentedName } from '../lib/accents.js';
  // wyrd-24s6 (D41): native/modern surface predicates live in lib/render.js so
  // they're unit-testable (svelte components have no test harness here).
  import { nativeSurface, modernSurface, isNativeRender } from '../lib/render.js';

  function selectResult(i) {
    appState.currentResultIndex =
      i === appState.currentResultIndex ? null : i;
  }

  // wyrd-z3fl: "report defective" moved to the Inspect & Transform column
  // (col 3) — you flag the result you're inspecting there, not per-row here.

  // wyrd-myof: glosses are a pure function of each result's morphemes_by_word,
  // so compute them ONCE per roll (keyed off appState.results) instead of inline
  // in the template, where glossFor() recomputed for every result on every
  // reactive render — including selection changes via currentResultIndex.
  // Shape mirrors morphemes_by_word: glossesByResult[i][wordIdx][morphIdx].
  const glossesByResult = $derived(glossesByResults(appState.results));

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
                     lettering to the right. Shown for every NON-MODERN render
                     (isNativeRender: the render carries a native era), so a
                     non-modern name whose morphemes coincidentally spell the same
                     as their modern reflex still shows it — consistent with the
                     inspector's era badge. A plain/force-modern ('as generated')
                     roll has no native era, so it's hidden (no noise).
                     wyrd-c6o1.1: a transform nulls result_modern in the committed
                     store (the original reflex no longer describes the edited
                     name), so the `r.result_modern &&` guard keeps hiding it
                     post-edit (isNativeRender alone would leave an empty span). -->
                {#if r.result_modern && isNativeRender(r.morphemes_by_word, r.result, r.result_modern)}
                  <span class="name-modern" title="modern reflex">{r.result_modern}</span>
                {/if}
              </span>
              {#if mbw?.length}
                <span class="etymology">
                  {#each mbw as word, wi}
                    <span class="word-group">
                      {#each word as morph, mi}
                        <!-- wyrd-myof: static rows read the per-roll memoized
                             glosses; the live (selected + transformed) row's
                             morphemes differ from r.morphemes_by_word, so it
                             computes glossFor inline (one row, recomputed only
                             on a transform). -->
                        {@const g = live ? glossFor(morph) : (glossesByResult[i]?.[wi]?.[mi] ?? '')}
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
