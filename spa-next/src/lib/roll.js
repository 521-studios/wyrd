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
    // Send ONLY the params the user changed from their default — an untouched
    // field falls through to the SERVER's default ("default = don't include").
    const rolledParams = appState.changedParams(appState.selectedGeneratorName);
    const envelope = await rollGenerator(
      appState.selectedGeneratorName,
      rolledParams,
    );
    appState.results = envelope.results;
    appState.resultsGenerator = envelope.generator;
    // wyrd-dsl5: stash roll metadata for defect reports (not for replay).
    appState.resultsSeed = envelope.seed ?? null;
    appState.resultsBundleVersion = envelope.bundle_version ?? null;
    // Freeze the FULL seeded params (everything the form showed) for a defect
    // report's reproduction context. We must NOT use the request body or the
    // server echo: the request now omits defaults, and the server echoes only
    // the sent params minus count/seed (it doesn't write resolved defaults
    // back), so neither is a complete record. currentParams IS the complete
    // local snapshot. Fall back to what we sent if it's somehow null.
    const fullParams = appState.currentParams;
    appState.resultsParams = fullParams ? { ...fullParams } : { ...rolledParams };
    appState.currentResultIndex = null;
  } catch (err) {
    appState.rollError = err.message;
  } finally {
    appState.isRolling = false;
  }
}
