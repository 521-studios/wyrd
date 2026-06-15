// wyrd-200v: pins the InspectorColumn reactive wiring that was Playwright-only
// (PR #628 / wyrd-c6o1.1). The $effect snapshots the pipeline BASE via untrack
// and runs the stack; the commit-to-store writes appState.results. If `original`
// were a live $derived of appState.currentResult again (the reverted bug shape),
// the commit would re-trigger the $effect → pipeline.run in an unbounded loop.
// We mount the column, let effects settle, and assert run() fired a BOUNDED
// number of times — a reintroduced loop blows that (or hangs the test).
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import InspectorColumn from './InspectorColumn.svelte';
import { appState } from '../lib/appState.svelte.js';
import { pipeline } from '../lib/pipeline.svelte.js';

function seedResult() {
  appState.results = [
    { result: 'Dūnton', morphemes_by_word: [[{ usage: 'dūn-', _wordIndex: 0, _morphemeIndex: 0 }]] },
  ];
  appState.currentResultIndex = 0;
  appState.resultsGenerator = 'kenning';
  // appState.config is a manifest-derived getter (null here) — flagOn handles
  // null → the regenerate affordance just stays hidden, fine for this test.
}

beforeEach(() => {
  pipeline.clear();
  seedResult();
});

describe('InspectorColumn reactive wiring (wyrd-c6o1.1 / wyrd-200v)', () => {
  it('does not loop: the commit-to-store does not re-trigger the run $effect', async () => {
    // run is async + drives the stack; we only care HOW MANY TIMES the $effect
    // invokes it. The fix (untracked base snapshot) → bounded; the loop bug →
    // runaway. Stub it so it can't do real work but we can count.
    const run = vi.spyOn(pipeline, 'run').mockResolvedValue(undefined);
    render(InspectorColumn, { props: {} });
    await tick();
    await tick();
    expect(run.mock.calls.length).toBeGreaterThan(0); // the effect did run the base
    expect(run.mock.calls.length).toBeLessThanOrEqual(2); // …but did NOT loop
  });

  it('re-snapshots + re-runs the base when the selected result changes', async () => {
    const run = vi.spyOn(pipeline, 'run').mockResolvedValue(undefined);
    render(InspectorColumn, { props: {} });
    await tick();
    const afterMount = run.mock.calls.length;

    appState.results = [
      ...appState.results,
      { result: 'Other', morphemes_by_word: [[{ usage: 'oth-', _wordIndex: 0, _morphemeIndex: 0 }]] },
    ];
    appState.currentResultIndex = 1; // switch subject
    await tick();
    await tick();
    expect(run.mock.calls.length).toBeGreaterThan(afterMount); // re-ran on switch
    expect(run.mock.calls.length).toBeLessThanOrEqual(afterMount + 2); // still bounded
  });
});
