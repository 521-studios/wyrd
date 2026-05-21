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
   *  stable id (monotonic counter) so each-block keying is robust to
   *  reorder / insert-at-middle (which is on the wyrd-hpjg roadmap
   *  for the swap-step UX). Subsequent reactive re-run picks it up
   *  via the $effect in InspectorColumn. */
  addStep(kind) {
    const t = getTransform(kind);
    this.#nextStepId += 1;
    this.steps = [
      ...this.steps,
      { id: this.#nextStepId, kind, params: { ...t.defaultParams } },
    ];
  }

  #nextStepId = 0;

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
