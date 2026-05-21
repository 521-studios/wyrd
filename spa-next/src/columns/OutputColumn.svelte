<script>
  // wyrd-hcmc: col 2 — list of results from the most recent roll.
  //
  // Each result renders as a card with the rendered name, the
  // explanation, and the seed badge. Click → selects the result
  // for col 3 inspection (wired in wyrd-yxf6 PR #3 — for now the
  // click handler just toggles a 'selected' visual state for
  // verification).
  //
  // Saved/Current toggle + star-icon save affordance lands in
  // wyrd-34tn (PR #6).
  import { appState } from '../lib/appState.svelte.js';

  // wyrd-yxf6: selection state lives in appState so InspectorColumn
  // reads it reactively. The re-roll-reset of currentResultIndex
  // lives in ConfigureColumn.roll() (next to the results
  // assignment) rather than an OutputColumn $effect, so the reset
  // survives OutputColumn unmounting (mobile drawer per wyrd-jh75).

  function selectResult(i) {
    appState.currentResultIndex =
      i === appState.currentResultIndex ? null : i;
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
      {#if appState.seed != null}
        <span class="seed">seed {appState.seed}</span>
      {/if}
    </p>
    <ul class="results">
      {#each appState.results as r, i (i)}
        <li>
          <button
            class="result"
            class:selected={appState.currentResultIndex === i}
            onclick={() => selectResult(i)}
          >
            <span class="name">{r.result}</span>
            {#if r.explanation}
              <span class="explanation">{r.explanation}</span>
            {/if}
          </button>
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
  .seed {
    margin-left: 6px;
    padding: 1px 6px;
    background: var(--bg-elev);
    border-radius: 3px;
    font-variant-numeric: tabular-nums;
  }
  .results {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .results li {
    margin-bottom: 6px;
  }
  .result {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    width: 100%;
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
    font-size: 14px;
    font-weight: 600;
  }
  .explanation {
    margin-top: 4px;
    font-size: 11px;
    color: var(--fg-muted);
    line-height: 1.4;
    /* Long etymological explanations from KenningExplain can be
       multi-line — clip with ellipsis at 3 lines so cards stay
       skimmable. Click → col 3 (PR #3) shows full detail.
       The unprefixed line-clamp companion appeases svelte-check
       (a11y/standards lint) — browsers without -webkit- still
       honor the standard property when it ships. */
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
