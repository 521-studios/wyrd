<script>
  // wyrd-hcmc + wyrd-8jjx: col 2 — current roll output. wyrd-8jjx
  // dropped the Saved view-toggle since the Saved library now opens
  // as a header-triggered drawer (universal across workspaces).
  // Star icon on each result stays inline (per-result quick save).
  import { appState } from '../lib/appState.svelte.js';
  import StarToggle from '../components/StarToggle.svelte';

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
      {#if appState.lastSeed != null}
        <!-- wyrd-hia1: meta shows the seed that PRODUCED this
             roll's results (lastSeed). appState.seed is the input
             value and may be blank (random-each-roll mode). -->
        <span class="seed">seed {appState.lastSeed}</span>
      {/if}
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
              <span class="name">{r.result}</span>
              {#if r.explanation}
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
    font-size: 14px;
    font-weight: 600;
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
