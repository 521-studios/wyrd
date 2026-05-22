// wyrd-14hn: extracted Roll handler. Pre-fix lived inside
// ConfigureColumn.roll() — but the button now lives in Header and
// per-generator workspaces may want to wire their own Roll
// affordance too. Single source of truth.
//
// wyrd-hia1: do NOT write envelope.seed back to appState.seed.
// Pre-fix this assignment locked the seed input to whatever value
// the server picked for the first random-seed roll, so every
// subsequent roll produced the same name. Now: the seed input
// stays as the user typed it (blank = random each time); the seed
// the server actually used lands in appState.lastSeed for display
// in the result meta + payload capture in save / share / star.

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
    // wyrd-hia1: lastSeed is the seed the server USED (whether the
    // operator supplied it or it was randomly chosen). seed (the
    // input) stays as the user typed it — blank for random rolls
    // so each Roll picks a fresh seed.
    appState.lastSeed = envelope.seed;
    appState.currentResultIndex = null;
  } catch (err) {
    appState.rollError = err.message;
  } finally {
    appState.isRolling = false;
  }
}
