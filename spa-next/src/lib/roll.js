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
    // Freeze the EFFECTIVE params (server-resolved, echoed back) for a defect
    // report's reproduction context — now that the request omits defaults, the
    // response's full parameter set is the accurate record. Fall back to what
    // we sent if the server doesn't echo them.
    appState.resultsParams = envelope.parameters
      ? { ...envelope.parameters }
      : { ...rolledParams };
    appState.currentResultIndex = null;
  } catch (err) {
    appState.rollError = err.message;
  } finally {
    appState.isRolling = false;
  }
}
