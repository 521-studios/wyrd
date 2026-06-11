<script>
  // wyrd-34tn: inline save-toggle for col 2 result cards. Clicking
  // a filled star REMOVES the save (toggle semantics); empty star
  // adds. wyrd-y0lx: starring the INSPECTED result also captures the
  // live transform pipeline (the Output row renders the transformed
  // state, so the bookmark restores it); starring any other row
  // captures just the original.
  import { savedStore } from '../lib/savedStore.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let { result } = $props();

  let saved = $derived(
    savedStore.isSaved({
      generator: appState.resultsGenerator,
      originalName: result.result,
    }),
  );

  function toggle(e) {
    // Stop propagation so the parent <button class="result"> click
    // (which selects the result for col 3) doesn't also fire.
    e.stopPropagation();
    const id = savedStore.findId({
      generator: appState.resultsGenerator,
      originalName: result.result,
    });
    if (id) {
      savedStore.remove(id);
      return;
    }
    // wyrd-8jjx round 2 (Gemini HIGH): pull params from
    // paramsByGenerator[resultsGenerator] rather than
    // appState.currentParams (which is keyed by selectedGenerator).
    // The picker might have changed since the roll; the params
    // that PRODUCED this result are the ones we need to save.
    const generator = appState.resultsGenerator;
    // wyrd-y0lx (operator decision): when the starred result is the one
    // being inspected, capture the live transform pipeline alongside the
    // original — the Output row renders the transformed state, so the
    // bookmark must restore it (SavedList.load replays the steps). Step
    // ids are stripped: addStep regenerates them on restore. Starring a
    // non-inspected row keeps the empty-pipeline behavior.
    const isInspected =
      appState.currentResultIndex !== null &&
      appState.results[appState.currentResultIndex] === result;
    const steps = isInspected
      ? $state.snapshot(pipeline.steps).map(({ kind, params }) => ({ kind, params }))
      : [];
    savedStore.add({
      generator,
      params: appState.paramsByGenerator[generator] || {},
      original: {
        name: result.result,
        morphemes_by_word: result.morphemes_by_word || [],
        // wyrd-34tn round 2: preserve etymological explanation so
        // SavedList load restores the full output card.
        explanation: result.explanation || '',
      },
      pipeline: steps,
    });
  }
</script>

<button
  type="button"
  class="star"
  class:saved
  onclick={toggle}
  aria-label={saved ? 'remove from saved' : 'save'}
  title={saved ? 'Saved (click to remove)' : 'Save'}
>{saved ? '★' : '☆'}</button>

<style>
  .star {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: 18px;
    padding: 0 4px;
    line-height: 1;
  }
  .star:hover {
    color: var(--accent);
  }
  .star.saved {
    color: var(--accent);
  }
  .star:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 2px;
  }
</style>
