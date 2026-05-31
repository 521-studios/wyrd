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

  // No reproducibility seed: a seed only reproduces names against ONE
  // bundle version, so it silently breaks the instant a new dataset
  // ships. Every Roll is a fresh random draw; saved/shared workspaces
  // persist the actual RESULT (a frozen name + etymology), not a seed
  // to re-derive it.

  // Most recent roll output. results[i].result + .explanation +
  // .morphemes_by_word + .components etc.
  results = $state([]);
  resultsGenerator = $state(null); // which generator produced `results`

  // wyrd-yxf6: which result is currently selected for col 3 inspection.
  // null = nothing selected, show placeholder. Lifted out of
  // OutputColumn's local state so InspectorColumn can read it
  // reactively. Cleared on re-roll via the OutputColumn $effect.
  currentResultIndex = $state(null);

  // wyrd-14hn round 2 (frontend LOW): single matchMedia source of
  // truth for viewport breakpoint. Pre-fix three components (App,
  // KenningWorkspace, ConfigureColumn) each ran their own listener
  // for the same query. App.svelte's onMount registers the listener
  // and writes this field; consumers read it reactively.
  isMobileViewport = $state(false);

  // wyrd-8jjx: open state for the Saved library drawer. Header's
  // 📚 Saved button toggles it; SavedList.load closes it after
  // restoring a workspace (so the user sees the loaded result
  // immediately). Universal so future surfaces (Esc handler, share-
  // link landing, mobile bottom-sheet alternate) all toggle the
  // same field.
  savedDrawerOpen = $state(false);

  // wyrd-34tn: gate for SavedList load() flow. InspectorColumn's
  // subject-change effect normally clears the pipeline on
  // currentResultIndex change; a load() restores both the result
  // AND the saved pipeline, so we need to suppress that one
  // auto-clear. SavedList sets true → mutates appState → the
  // effect observes the change, skips the clear, and resets the
  // flag.
  isLoadingSavedWorkspace = $state(false);

  // Roll in flight. Lets the button disable + show a pending state.
  isRolling = $state(false);
  rollError = $state(null);

  /** The result that col 3 is currently inspecting, or null. */
  get currentResult() {
    if (this.currentResultIndex === null) return null;
    return this.results[this.currentResultIndex] || null;
  }

  /** Look up the currently-selected generator's manifest entry. */
  get selectedGenerator() {
    if (!this.manifest || !this.selectedGeneratorName) return null;
    return this.manifest.generators.find(
      (g) => g.name === this.selectedGeneratorName,
    );
  }

  /** Pure getter — returns the params object for the current
   *  generator, or null if none selected. Initialization is the
   *  caller's job: use ``ensureParams(name)`` before reads.
   *
   *  wyrd-o7lp (PR #314 Gemini HIGH): pre-fix, this getter
   *  lazy-init'd ``paramsByGenerator[name] = {}`` on read — a
   *  $state mutation during a render path, which Svelte 5
   *  flags as an anti-pattern (risks reactive loops + warnings).
   *  Earlier attempts to move init into a sibling $effect raced
   *  with child Fields (rendered before the parent's effect
   *  ran, read null, threw). The fix: ConfigureColumn calls
   *  ``ensureParams`` via ``$effect.pre`` — which runs BEFORE
   *  child effects, so init lands before any Field reads. */
  get currentParams() {
    if (!this.selectedGeneratorName) return null;
    return this.paramsByGenerator[this.selectedGeneratorName] || null;
  }

  /** Explicit init: idempotent. Call from a parent $effect.pre()
   *  before any child Field reads currentParams. */
  ensureParams(generatorName) {
    if (!generatorName) return;
    if (!this.paramsByGenerator[generatorName]) {
      this.paramsByGenerator[generatorName] = {};
    }
  }
}

export const appState = new AppState();
