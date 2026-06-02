<script>
  // wyrd-hcmc + wyrd-8jjx: col 2 — current roll output. wyrd-8jjx
  // dropped the Saved view-toggle since the Saved library now opens
  // as a header-triggered drawer (universal across workspaces).
  // Star icon on each result stays inline (per-result quick save).
  import { appState } from '../lib/appState.svelte.js';
  import StarToggle from '../components/StarToggle.svelte';
  import {
    representativeMeanings,
    isNameMorpheme,
  } from '../lib/morphemeGloss.js';
  import { accentedUsage, accentedName } from '../lib/accents.js';

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
        <li>
          <div class="result-row">
            <button
              class="result"
              class:selected={appState.currentResultIndex === i}
              onclick={() => selectResult(i)}
            >
              <span class="name">{accentedName(r)}</span>
              {#if r.morphemes_by_word?.length}
                <span class="etymology">
                  {#each r.morphemes_by_word as word}
                    <span class="word-group">
                      {#each word as morph}
                        {@const g = glossFor(morph)}
                        <span class="morph-col">
                          <!-- wyrd-de5t: show the accented surface (bȳ) when a
                               rendering supplies one. wyrd-2ien: for an era
                               render show the era form (matching the name); the
                               modern breakdown lives in the inspector's two-card
                               view, not here. -->
                          <span class="surface">{morph.rendered || accentedUsage(morph) || morph.usage || ''}</span>
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
  .name {
    font-size: 15px;
    font-weight: 600;
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
