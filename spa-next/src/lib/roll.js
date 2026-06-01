// wyrd-14hn: extracted Roll handler. Pre-fix lived inside
// ConfigureColumn.roll() — but the button now lives in Header and
// per-generator workspaces may want to wire their own Roll
// affordance too. Single source of truth.
//
// Every Roll is a fresh random draw — no seed is sent or tracked (a
// seed only reproduces against one bundle version).

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
    );
    appState.results = envelope.results;
    appState.resultsGenerator = envelope.generator;
    // wyrd-dsl5: stash roll metadata for defect reports (not for replay).
    appState.resultsSeed = envelope.seed ?? null;
    appState.resultsBundleVersion = envelope.bundle_version ?? null;
    appState.currentResultIndex = null;
  } catch (err) {
    appState.rollError = err.message;
  } finally {
    appState.isRolling = false;
  }
}
