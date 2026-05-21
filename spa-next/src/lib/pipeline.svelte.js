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
// Pipeline run is debounced by the natural async/await — we
// snapshot the steps + original at the start of a run, do the
// API calls sequentially, and update results at the end. If a
// new edit lands mid-run we'd ideally cancel, but for v1 the
// last-finished-wins behavior is acceptable (network calls are
// ~50ms locally; user edits arrive slower).

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
  // can show a pending state. Run-cancellation isn't implemented —
  // a new run just starts; the older one's writes are discarded.
  isRunning = $state(false);

  // wyrd-kppy: cancellation token. Bumped on each run start; the
  // run only commits results if its token matches the current
  // value (so a stale run that finishes after a newer run started
  // doesn't clobber the newer's output).
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

  /** Add a new step at the end of the pipeline. Subsequent reactive
   *  re-run picks it up via the $effect in InspectorColumn. */
  addStep(kind) {
    const t = getTransform(kind);
    this.steps = [
      ...this.steps,
      { kind, params: { ...t.defaultParams } },
    ];
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
   *  changes (user picks a different result in col 2). */
  clear() {
    this.steps = [];
    this.states = [];
    this.errors = [];
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
    // Commit only if we're still the freshest run.
    if (myToken === this.#runToken) {
      this.states = nextStates;
      this.errors = nextErrors;
      this.isRunning = false;
    }
    return !halted;
  }
}

export const pipeline = new PipelineState();
