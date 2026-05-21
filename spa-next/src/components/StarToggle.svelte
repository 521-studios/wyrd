<script>
  // wyrd-34tn: inline save-toggle for col 2 result cards. Clicking
  // a filled star REMOVES the save (toggle semantics); empty star
  // adds. Captures just the result's original metadata — for the
  // current pipeline state too, the col 3 save button is the
  // larger-affordance "save current workspace" path.
  //
  // The star is in addition to col 3's save: col 2 is "save this
  // generated name quickly without losing my place"; col 3 is
  // "save this workspace including transforms". Both flow into the
  // same savedStore.
  import { savedStore } from '../lib/savedStore.svelte.js';
  import { appState } from '../lib/appState.svelte.js';

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
    savedStore.add({
      generator: appState.resultsGenerator,
      params: appState.currentParams,
      seed: appState.seed,
      original: {
        name: result.result,
        morphemes_by_word: result.morphemes_by_word || [],
      },
      // No pipeline — col 2 saves just the original. Col 3's save
      // captures the whole workspace including transforms.
      pipeline: [],
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
