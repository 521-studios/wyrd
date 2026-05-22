// wyrd-14hn: extracted Roll handler. Pre-fix lived inside
// ConfigureColumn.roll() — but the button now lives in Header and
// per-generator workspaces may want to wire their own Roll
// affordance too. Single source of truth.
//
// Calls /api/<gen> with currentParams + seed; writes results into
// appState; clears the inspector subject (wyrd-yxf6 round 2
// inline-with-results-assignment fix).

import { appState } from './appState.svelte.js';
import { rollGenerator } from './api.js';

export async function rollCurrent() {
  if (!appState.selectedGeneratorName) return;
  appState.isRolling = true;
  appState.rollError = null;
  try {
    const envelope = await rollGenerator(
      appState.selectedGeneratorName,
      appState.currentParams,
      appState.seed,
    );
    appState.results = envelope.results;
    appState.resultsGenerator = envelope.generator;
    appState.seed = envelope.seed; // server echoes back the seed used
    appState.currentResultIndex = null;
  } catch (err) {
    appState.rollError = err.message;
  } finally {
    appState.isRolling = false;
  }
}
