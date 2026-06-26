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

import { initialFieldValue } from './featureFlags.js';
import { HIDDEN_FIELDS } from './headlineFields.js';

// Fields sent on every Roll even when still at their default — the SPA's value
// must be authoritative rather than relying on the server's default matching the
// form: ``culture`` is the primary generation selector, and ``count`` must equal
// what the form displays (the user expects exactly that many results).
const ALWAYS_SENT_FIELDS = new Set(['culture', 'count']);

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

  // wyrd-dsl5: the seed + bundle version the current `results` were rolled
  // against. Not used to reproduce in the SPA (a seed only reproduces
  // against one bundle) — captured solely so a "mark defective" report
  // carries the full reproduction context for operator triage. null when
  // results came from a loaded saved workspace (no live roll metadata).
  resultsSeed = $state(null);
  resultsBundleVersion = $state(null);
  // Snapshot of the params that PRODUCED `results`, frozen at roll time. A
  // defect report sends this (not the live form state, which the user may
  // have edited after rolling — e.g. changing `count`), so the stored
  // parameters actually reproduce the flagged name.
  resultsParams = $state(null);

  // wyrd-yxf6: which result is currently selected for col 3 inspection.
  // null = nothing selected, show placeholder. Lifted out of
  // OutputColumn's local state so InspectorColumn can read it
  // reactively. Set by OutputColumn.selectResult (click a row) and reset to
  // null on every re-roll in roll.js (rollCurrent) — there is no OutputColumn
  // $effect — so the index never dangles past the replaced results array.
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

  /** wyrd-0gou: env-resolved SPA feature-flag config from the manifest
   *  ({all, flags, defaults}), or null on a legacy manifest with no config
   *  block. Consumers pass this to lib/featureFlags helpers. */
  get config() {
    return this.manifest?.config || null;
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

  /** Explicit init: idempotent (backfill-only — never clobbers). Call from a
   *  parent $effect.pre() before any child Field reads/binds currentParams.
   *
   *  wyrd-b6hd: the store OWNS field initialization. It walks the selected
   *  generator's schema and BACKFILLS every still-missing non-hidden field with
   *  its default (config.defaults override → schema default → type-empty, via
   *  ``initialFieldValue``) — present values (a restored share-link / saved
   *  workspace) are left untouched. Seeding here — before Fields render — means
   *  the form binds already-populated values, so there's no per-component lazy
   *  seed racing the ``<select>`` bind (the wyrd-etvd class of bug, for EVERY
   *  field, not just scoring_mode). Backfill (vs no-op-if-exists) also covers a
   *  stale cross-version bookmark whose dict predates this seeding.
   *
   *  The schema + config come from the loaded manifest; if the manifest isn't
   *  loaded yet (e.g. a share-link set selectedGeneratorName before the fetch
   *  resolved) we return WITHOUT touching the bag — a later call (once the
   *  manifest lands) backfills it. The caller's $effect.pre tracks ``manifest``
   *  so it re-runs at that point. */
  ensureParams(generatorName) {
    if (!generatorName) return;
    const generator = this.manifest?.generators?.find((g) => g.name === generatorName);
    // Manifest not loaded yet → can't seed from the schema; do NOT lock in an
    // empty bag. Retry when the manifest lands (the caller's $effect.pre tracks
    // it). A pre-manifest share-link restore leaves its bag in place untouched.
    if (!generator) return;
    const properties = generator.input_schema?.properties || {};
    // BACKFILL, don't replace: seed only fields that are still missing, so a
    // restored bag (share-link / saved workspace) keeps its values AND any
    // field it didn't carry (e.g. a stale cross-version bookmark from before
    // this refactor seeded every field) gets a defined default — so no
    // <select> ever binds an undefined value (wyrd-etvd write-back).
    const params = this.paramsByGenerator[generatorName] || {};
    for (const [key, prop] of Object.entries(properties)) {
      if (HIDDEN_FIELDS.has(key)) continue;
      if (params[key] === undefined) {
        params[key] = initialFieldValue(this.config, key, prop);
      }
    }
    this.paramsByGenerator[generatorName] = params;
  }

  /** The subset of a generator's params the user actually CHANGED from its
   *  seeded default (config.defaults override → schema default → type-empty).
   *  The Roll request sends ONLY these, so an untouched field falls through to
   *  the SERVER's default rather than the SPA pinning a value the server owns —
   *  "default should mean don't include". Without this, the form seeds every
   *  field (incl. config.defaults like scoring_mode=vector) and POSTs them all,
   *  which both bloats the request and silently overrides server-side defaults.
   *
   *  Deep-compares via JSON so empty arrays/strings ([], "") equal their
   *  defaults by VALUE, not reference. A field with no schema entry is kept
   *  (we can't know its default); HIDDEN_FIELDS (seed) are dropped.
   *
   *  ALWAYS_SENT fields (``culture``, ``count``) are included even at their
   *  default: culture is the primary generation selector and count must equal
   *  what the form displays, so the SPA's value must be authoritative rather
   *  than relying on the server's default matching the form. */
  changedParams(generatorName) {
    const params = this.paramsByGenerator[generatorName];
    if (!params) return {};
    const generator = this.manifest?.generators?.find((g) => g.name === generatorName);
    const properties = generator?.input_schema?.properties || {};
    const config = this.config; // hoist the getter out of the loop
    const changed = {};
    for (const [key, value] of Object.entries(params)) {
      if (HIDDEN_FIELDS.has(key)) continue;
      const prop = properties[key];
      if (
        !ALWAYS_SENT_FIELDS.has(key) &&
        prop &&
        // JSON-stringify equality is safe only while every param is a scalar or
        // array of scalars (order-stable). If an OBJECT-valued param is ever
        // added, key-order differences would compare unequal and over-send —
        // switch to a structural deep-equal then.
        JSON.stringify(value) === JSON.stringify(initialFieldValue(config, key, prop))
      ) {
        continue; // unchanged from its seeded default → let the server own it
      }
      changed[key] = value;
    }
    return changed;
  }
}

export const appState = new AppState();
