<script>
  // wyrd-34tn: "save this workspace" button in col 3. Captures the
  // full pipeline state — generator + params + seed + original +
  // pipeline steps + their params. Distinct from the col 2 star
  // (which captures just the original).
  //
  // Loading a workspace save restores all 3 columns: col 1's params
  // form re-populates, col 2 shows the original name + its row
  // highlighted, col 3 rehydrates the inspector + replays the
  // pipeline.
  import { savedStore } from '../lib/savedStore.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let savedNote = $state('');

  function save() {
    const r = appState.currentResult;
    if (!r) return;
    // wyrd-34tn round 2 (frontend MED): dedupe — if this generator
    // + original-name is already saved AND has no pipeline steps
    // beyond what we'd save, skip the duplicate (the existing entry
    // covers it). Pipeline-bearing saves can be duplicated since
    // each pipeline is its own work.
    const existingId = savedStore.findId({
      generator: appState.resultsGenerator,
      originalName: r.result,
    });
    if (existingId && pipeline.steps.length === 0) {
      savedNote = `Already saved`;
      setTimeout(() => {
        savedNote = '';
      }, 2500);
      return;
    }
    const { error } = savedStore.add({
      generator: appState.resultsGenerator,
      params: appState.currentParams,
      seed: appState.seed,
      original: {
        name: r.result,
        morphemes_by_word: r.morphemes_by_word || [],
        // wyrd-34tn round 2 (Gemini MED): preserve etymological
        // explanation so SavedList load restores the full output
        // card, not a stripped version.
        explanation: r.explanation || '',
      },
      pipeline: pipeline.steps.map((s) => ({
        kind: s.kind,
        params: { ...s.params },
      })),
    });
    // wyrd-34tn round 3 (Gemini HIGH): surface localStorage write
    // failures (quota / private browsing) so the user knows the save
    // didn't persist — the in-memory state still works for this
    // session, but they should export soon.
    savedNote = error
      ? `Save failed: ${error.message || 'storage error'}`
      : `Saved as "${r.result}"`;
    setTimeout(() => {
      savedNote = '';
    }, 2500);
  }
</script>

<button class="save" type="button" onclick={save}>
  💾 Save workspace
</button>
{#if savedNote}
  <p class="note">{savedNote}</p>
{/if}

<style>
  .save {
    background: transparent;
    color: var(--fg);
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    width: 100%;
    text-align: left;
    margin-top: 8px;
  }
  .save:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .save:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .note {
    margin: 6px 0 0;
    font-size: 11px;
    color: var(--accent);
    font-style: italic;
  }
</style>
