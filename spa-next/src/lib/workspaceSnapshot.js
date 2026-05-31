// wyrd-8jjx round 2: single-source workspace payload builder.
//
// Pre-fix, Header / SaveWorkspaceButton / ShareWorkspaceButton /
// StarToggle each constructed the save/share payload inline — and
// each pulled params from appState.currentParams, which is keyed
// by selectedGeneratorName. That's a BUG: when the user changes
// the generator picker WITHOUT rolling, selectedGeneratorName !=
// resultsGenerator and the saved payload would carry the WRONG
// params (the new picker's defaults instead of the params that
// actually produced the current result).
//
// Fix: pull params from paramsByGenerator[resultsGenerator] —
// the result and its params come from the same generator
// regardless of what the picker currently says. Extracted here so
// every consumer uses the same correct shape.

import { appState } from './appState.svelte.js';
import { pipeline } from './pipeline.svelte.js';

/**
 * Returns a workspace snapshot ready for savedStore.add() or
 * encodeWorkspace(): { generator, params, original, pipeline }.
 * Returns null when no result is currently selected (caller should
 * disable the save/share action in that case).
 *
 * A snapshot freezes the actual RESULT (original) — no seed, because a
 * seed only reproduces against one bundle version. Restoring a snapshot
 * shows the frozen name + etymology; it does not re-roll.
 */
export function currentWorkspaceSnapshot() {
  const r = appState.currentResult;
  if (!r) return null;
  const generator = appState.resultsGenerator;
  return {
    generator,
    // wyrd-8jjx round 2 (Gemini HIGH): params keyed by
    // resultsGenerator, not currentParams. They were the params
    // that produced THIS result; selectedGeneratorName may have
    // since changed via the picker without a re-roll.
    params: appState.paramsByGenerator[generator] || {},
    original: {
      name: r.result,
      morphemes_by_word: r.morphemes_by_word || [],
      explanation: r.explanation || '',
    },
    // Shallow pipeline.steps copy is fine — savedStore.add deep-
    // clones the whole payload via JSON round-trip (wyrd-34tn
    // round 3 fix). Same applies to encodeWorkspace which
    // JSON.stringify's anyway.
    pipeline: pipeline.steps.map((s) => ({
      kind: s.kind,
      params: { ...s.params },
    })),
  };
}
