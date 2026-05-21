// wyrd-hcmc: global app state, Svelte 5 runes idiom.
//
// A single class with $state fields, exported as a singleton.
// Cross-component reactivity comes for free: any component that
// imports `appState` and reads a field gets re-rendered when that
// field changes. No Svelte stores, no subscribers, no setup.
//
// As the SPA grows past PR #2 this file gains:
//   - currentResultIndex (which result is selected for col 3) — PR #3
//   - pipeline (transform stack) — PR #4
//   - saved (localStorage-backed bookmark list) — PR #6

class AppState {
  // Loaded once at app boot from /api/manifest. null while in-flight.
  manifest = $state(null);
  manifestError = $state(null);

  // The generator the user is currently configuring (e.g. 'kenning',
  // 'kenning-rewind'). Drives ConfigureColumn's form rendering.
  selectedGeneratorName = $state(null);

  // Per-generator param state. Keyed by generator name so switching
  // generators preserves prior form values. Values are whatever the
  // field built — string, number, boolean, array of strings (tag grid).
  paramsByGenerator = $state({});

  // Reproducibility seed. Null = "let the server pick a fresh one"
  // (the response echoes back the seed actually used).
  seed = $state(null);

  // Most recent roll output. results[i].result + .explanation +
  // .morphemes_by_word + .components etc.
  results = $state([]);
  resultsGenerator = $state(null); // which generator produced `results`

  // Roll in flight. Lets the button disable + show a pending state.
  isRolling = $state(false);
  rollError = $state(null);

  /** Look up the currently-selected generator's manifest entry. */
  get selectedGenerator() {
    if (!this.manifest || !this.selectedGeneratorName) return null;
    return this.manifest.generators.find(
      (g) => g.name === this.selectedGeneratorName,
    );
  }

  /** Get-or-init the params object for the current generator.
   *  Lazy-init lives in the getter for atomicity: child Fields
   *  render before any $effect can run, so any deferred init
   *  produces a null-read race. Returns null only when no
   *  generator is selected (manifest hasn't loaded yet) — at that
   *  point no Field is rendered either, so the null return is
   *  unobservable in practice but defends downstream callers
   *  against an unexpected fresh {}.
   *
   *  Reviewer note (round 2): a prior version of this getter was
   *  pure (lazy init moved to an $effect in ConfigureColumn) per
   *  a "render-time tracked-field mutation risks reactive-loop
   *  warnings" concern. In practice no loop materialized, but
   *  the deferred init opened a render-before-effect race where
   *  child Fields read undefined → null and threw. Reverted to
   *  the lazy-init pattern; the Svelte 5 warning would surface
   *  IF triggered, and we'd revisit then. */
  get currentParams() {
    if (!this.selectedGeneratorName) return null;
    if (!this.paramsByGenerator[this.selectedGeneratorName]) {
      this.paramsByGenerator[this.selectedGeneratorName] = {};
    }
    return this.paramsByGenerator[this.selectedGeneratorName];
  }
}

export const appState = new AppState();
