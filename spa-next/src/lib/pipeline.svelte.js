// wyrd-kppy: pipeline engine + reactive state.
//
// User-decided design (2026-05-21 design session): a transform
// pipeline is an EDITABLE RECIPE, not a history of snapshots.
// Editing step 2 re-runs steps 3-N. Adding / removing / reordering
// re-runs from that point onward. Each step is an instruction
// (rewind→OE, swap X→Y, ...) with parameters the user can tweak
// inline on the step card.
//
// The engine is a singleton tied to the current inspector subject
// (appState.currentResult). Switching to a different result resets
// the pipeline; long-term preservation lands in wyrd-34tn (PR #6,
// save/load to localStorage).
//
// SHAPE of a "step": { kind: 'rewind', params: { era: 'oe-late' } }
// SHAPE of a "state": { name: 'Hēl on Mort byrġ', morphemes_by_word: [[...]] }
//   - `name` is what col 3's head displays
//   - `morphemes_by_word` is what the morpheme cards iterate
//
// Pipeline runs are sequential async/await. Each call to run()
// bumps an internal #runToken; the run only commits its state
// writes if its token still matches when the loop finishes — so
// a stale slow run can't clobber a newer fast run's output.
// "Newest-wins" is the real cancellation semantics (vs the
// last-finished-wins shape v0 had).

import { appState } from './appState.svelte.js';
import { flagOn } from './featureFlags.js';
import { getTransform } from './transforms/index.js';

class PipelineState {
  // The list of user-added steps. Mutable; UI binds to this.
  steps = $state([]);

  // Per-step result (states[0] = original; states[i+1] = after step i).
  // Empty until a result is selected + the engine has run at least
  // once. UI reads states[states.length-1] for the current name +
  // morpheme cards.
  states = $state([]);

  // Per-step error indexed parallel to steps: errors[i] is the
  // failure on steps[i] if non-null. Lets the step card show its
  // own error inline rather than a global banner.
  errors = $state([]);

  // True while a run is in flight. Step cards + the inspector head
  // can show a pending state. wyrd-kppy round 2: actually
  // implemented via #runToken — a stale slow run's results are
  // dropped + its isRunning reset is suppressed; the newest
  // run's reset is the only one that commits. Module-header
  // comment pre-round-2 claimed "cancellation isn't implemented";
  // it is.
  isRunning = $state(false);

  // wyrd-kppy: cancellation token. Bumped on each run start; the
  // run only commits results (or resets isRunning) if its token
  // matches the current value, so a stale run that finishes after
  // a newer run started doesn't clobber state.
  #runToken = 0;

  /** The current effective state — bottom of the pipeline if
   *  populated, otherwise the original. Null if no result is
   *  selected in col 3. */
  get currentState() {
    if (this.states.length > 0) {
      return this.states[this.states.length - 1];
    }
    const r = appState.currentResult;
    if (!r) return null;
    return {
      name: r.result,
      morphemes_by_word: r.morphemes_by_word || [],
    };
  }

  /** Add a new step at the end of the pipeline. Each step gets a
   *  stable id (monotonic counter) so each-block keying is robust
   *  to reorder / insert-at-middle. Subsequent reactive re-run
   *  picks it up via the $effect in InspectorColumn.
   *
   *  wyrd-hpjg: optional `paramOverrides` lets callers pre-populate
   *  step params at create time. Used by the direct-manipulation
   *  Swap UX (MorphemeGrid cell click bakes in wordIndex / morphemeIndex
   *  / to). The palette path doesn't pass overrides and gets the
   *  transform's defaultParams. */
  addStep(kind, paramOverrides = {}) {
    const t = getTransform(kind);
    this.#nextStepId += 1;
    this.steps = [
      ...this.steps,
      {
        id: this.#nextStepId,
        kind,
        params: { ...t.defaultParams, ...paramOverrides },
      },
    ];
  }

  #nextStepId = 0;

  /** Direct-manipulation swap (wyrd-2b50 follow-up): maintain AT MOST
   *  ONE swap step per (wordIndex, morphemeIndex) cell. Clicking a
   *  variant in the era grid updates that cell's swap in place
   *  instead of stacking a new step (10 clicks → 1 step, not 10).
   *  Clicking the cell's ORIGINAL generated form removes the swap —
   *  i.e. clicking the original is "revert". `original` is the col-2
   *  result's usage for that cell. Dashes + case are normalized away
   *  for the revert comparison ("-wald-" vs "Wald") — matching
   *  MorphemeGrid.swap's folding so the two never disagree — but
   *  accents are kept, so adding an accent to a plain original is a swap. */
  setSwap({ wordIndex, morphemeIndex, to, original, language, cellId }) {
    const norm = (s) => (s || '').replace(/^-+|-+$/g, '').toLowerCase();
    const idx = this.steps.findIndex(
      (s) =>
        s.kind === 'swap' &&
        s.params.wordIndex === wordIndex &&
        s.params.morphemeIndex === morphemeIndex,
    );
    // Value-based revert: clicking a form whose surface equals the original
    // is a no-op revert — UNLESS a language is pinned (wyrd-thhb). For a
    // cross-language homograph the surface CAN equal the original ("don" in
    // Old French vs the generated Old English "don") while still being a real
    // selection, so a language pin suppresses the surface-only revert.
    if (!language && norm(to) === norm(original)) {
      if (idx !== -1) this.removeStep(idx);
      return;
    }
    if (idx !== -1) {
      const next = [...this.steps];
      next[idx] = {
        ...next[idx],
        params: { ...next[idx].params, to, language, cellId },
      };
      this.steps = next;
    } else {
      this.addStep('swap', { wordIndex, morphemeIndex, to, language, cellId });
    }
  }

  /** Remove the swap step for a (wordIndex, morphemeIndex) cell, if any
   *  — an identity-based revert. setSwap's value-based revert can be
   *  unreachable when the generated surface is ASCII but every form row
   *  carries an accented original_script (no clickable row ever
   *  normalizes back to the bare original). Clicking the CURRENT form
   *  routes here so "click to revert" always works. No-op if no swap. */
  clearSwap({ wordIndex, morphemeIndex }) {
    const idx = this.steps.findIndex(
      (s) =>
        s.kind === 'swap' &&
        s.params.wordIndex === wordIndex &&
        s.params.morphemeIndex === morphemeIndex,
    );
    if (idx !== -1) this.removeStep(idx);
  }

  /** Direct-manipulation regenerate (wyrd-y0lx): maintain AT MOST ONE
   *  regenerate step per (wordIndex, morphemeIndex) slot — operator
   *  decision: a second click on the same morpheme re-rolls IN PLACE
   *  (a fresh seed on the existing step), so removing the step always
   *  reverts straight to the pre-regenerate morpheme. `context` is the
   *  generation-params snapshot the caller takes at click time and is
   *  refreshed on every click (the caller computes it fresh; silently
   *  keeping the first click's copy would discard it). `from` is the
   *  slot's pre-regenerate surface, kept from the FIRST click so the
   *  step card keeps labeling the original morpheme being re-rolled.
   *
   *  wyrd-6p2u supersession rule: a regenerate REPLACES the slot's
   *  morpheme IDENTITY, so any swap step already targeting the slot is
   *  meaningless (it picked a form of a morpheme that no longer exists)
   *  and is dropped here — whether the swap predates the first
   *  regenerate or was layered on one (a fresh re-roll supersedes a
   *  regenerate→swap chain too). The reverse never holds: a swap added
   *  AFTER a regenerate refines the NEW morpheme's form and stacks
   *  normally (setSwap is deliberately untouched). */
  setRegenerate({ wordIndex, morphemeIndex, context = {}, from = '' }) {
    // Number.MAX_SAFE_INTEGER keeps the seed in the JS-safe int range the
    // server's sub-seed contract already uses (wyrd-aof8).
    const seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const sameSlot = (s) =>
      s.params.wordIndex === wordIndex && s.params.morphemeIndex === morphemeIndex;
    // Drop superseded swaps + locate the slot's regenerate against ONE new
    // array, committed in a single assignment (no intermediate reactive
    // state between the removal and the add/update).
    const next = this.steps.filter((s) => !(s.kind === 'swap' && sameSlot(s)));
    const idx = next.findIndex((s) => s.kind === 'regenerate-morpheme' && sameSlot(s));
    if (idx !== -1) {
      next[idx] = { ...next[idx], params: { ...next[idx].params, seed, context } };
      this.steps = next;
    } else {
      const t = getTransform('regenerate-morpheme');
      this.#nextStepId += 1;
      this.steps = [
        ...next,
        {
          id: this.#nextStepId,
          kind: 'regenerate-morpheme',
          params: { ...t.defaultParams, wordIndex, morphemeIndex, seed, context, from },
        },
      ];
    }
  }

  /** Replace step at index i with a new params object (callers
   *  mutate the step's params in-place via bind:value; this
   *  helper is for callers that want explicit replacement). */
  updateStep(i, params) {
    const next = [...this.steps];
    next[i] = { ...next[i], params: { ...params } };
    this.steps = next;
  }

  /** Remove step at index i. */
  removeStep(i) {
    this.steps = this.steps.filter((_, idx) => idx !== i);
  }

  /** Reset the entire pipeline. Called when the inspector subject
   *  changes (user picks a different result in col 2).
   *
   *  wyrd-o7lp (PR #316 Gemini HIGH): bumps #runToken so any
   *  run already in flight commits no state when it finishes
   *  (its token no longer matches), AND resets isRunning so the
   *  pending indicator doesn't latch across the subject change.
   *  Pre-fix, a fast subject switch with a slow run in flight
   *  could commit the prior subject's pipeline output against
   *  the new subject — concurrency corruption under fast clicks. */
  clear() {
    this.#runToken += 1;
    this.steps = [];
    this.states = [];
    this.errors = [];
    this.isRunning = false;
  }

  /** Run the pipeline against the supplied original state. Sequential
   *  apply, error-aware. Called from an $effect in InspectorColumn
   *  whenever this.steps or appState.currentResult changes. */
  async run(original) {
    this.#runToken += 1;
    const myToken = this.#runToken;
    this.isRunning = true;
    const stepsSnapshot = this.steps.map((s) => ({
      kind: s.kind,
      params: { ...s.params },
    }));
    const nextStates = [original];
    const nextErrors = [];
    let state = original;
    let halted = false;
    for (let i = 0; i < stepsSnapshot.length; i += 1) {
      const step = stepsSnapshot[i];
      try {
        const t = getTransform(step.kind);
        // wyrd-nwpa: a flag-gated transform (e.g. Rewind) whose flag is off is
        // skipped as a pass-through, so a restored/shared workspace can't run
        // it in prod (where the palette also hides it).
        if (t.flag && !flagOn(appState.config, t.flag)) {
          nextStates.push(state);
          nextErrors.push(null);
          continue;
        }
        state = await t.apply(state, step.params);
        nextStates.push(state);
        nextErrors.push(null);
      } catch (err) {
        nextErrors.push(err.message || String(err));
        halted = true;
        break;
      }
    }
    // Pad errors so length matches steps even when we halted early.
    while (nextErrors.length < stepsSnapshot.length) {
      nextErrors.push(null);
    }
    // Commit only if we're still the freshest run. wyrd-kppy
    // round 2: isRunning reset moved inside the token guard so a
    // stale slow run can't clear isRunning while a newer fast
    // run is still in flight. The newer run's own reset is the
    // single source of truth for the spinner clearing.
    if (myToken === this.#runToken) {
      this.states = nextStates;
      this.errors = nextErrors;
      this.isRunning = false;
    }
    return !halted;
  }
}

export const pipeline = new PipelineState();
