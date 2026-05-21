<script>
  // wyrd-hcmc + wyrd-34tn: col 2 — toggle between the most recent
  // roll output and the Saved bookmark library. Each result in the
  // roll view gets a ★ toggle (StarToggle) for inline save. The
  // Saved view (SavedList) shows persisted entries; clicking one
  // rehydrates all 3 columns to that workspace state.
  import { appState } from '../lib/appState.svelte.js';
  import { savedStore } from '../lib/savedStore.svelte.js';
  import StarToggle from '../components/StarToggle.svelte';
  import SavedList from '../components/SavedList.svelte';

  // Local-only view toggle. Lives in OutputColumn (not appState)
  // because no other column needs to react to it; if a future
  // feature needs to navigate to Saved from elsewhere (e.g., share-
  // link landing in wyrd-tz35), promote to appState.
  let view = $state('current'); // 'current' | 'saved'

  function selectResult(i) {
    appState.currentResultIndex =
      i === appState.currentResultIndex ? null : i;
  }
</script>

<section class="column">
  <h2>Output</h2>

  <div class="view-toggle" role="tablist">
    <button
      role="tab"
      class="toggle-btn"
      class:active={view === 'current'}
      aria-selected={view === 'current'}
      onclick={() => (view = 'current')}
    >Current roll</button>
    <button
      role="tab"
      class="toggle-btn"
      class:active={view === 'saved'}
      aria-selected={view === 'saved'}
      onclick={() => (view = 'saved')}
    >Saved ({savedStore.entries.length})</button>
  </div>

  {#if view === 'saved'}
    <SavedList />
  {:else if appState.results.length === 0}
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
  .view-toggle {
    display: flex;
    gap: 2px;
    margin-bottom: 16px;
    background: var(--bg-elev);
    border-radius: 4px;
    padding: 2px;
  }
  .toggle-btn {
    flex: 1;
    background: transparent;
    color: var(--fg-muted);
    border: none;
    border-radius: 3px;
    padding: 6px 8px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    font-weight: 600;
  }
  .toggle-btn.active {
    background: var(--bg);
    color: var(--fg);
  }
  .toggle-btn:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
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
