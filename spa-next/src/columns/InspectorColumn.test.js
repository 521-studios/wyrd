// wyrd-200v: pins the InspectorColumn reactive wiring that was Playwright-only
// (PR #628 / wyrd-c6o1.1). The $effect snapshots the pipeline BASE via untrack
// and runs the stack; pipeline.run commits its output back into appState.results
// (#commitToStore). If `base` were read TRACKED off appState.currentResult again
// (the pre-c6o1.1 bug shape), that commit would re-fire the $effect → pipeline.run
// in an unbounded loop.
//
// CRUCIAL: we let pipeline.run CALL THROUGH (not a no-op stub) so the
// run→commit→store write-back — the exact edge the loop closes on — is live.
// Stubbing run to a no-op would sever that edge and the test would pass whether
// or not the bug exists. We then assert run() fired a BOUNDED number of times:
// the fix → once per mount/switch; a reintroduced loop → the count blows the
// bound (or the loop spins until vitest's 5s timeout fails the test).
import { fireEvent, render } from '@testing-library/svelte';
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

// Let the reactive graph + the async run()/commit settle. A reintroduced loop
// re-arms the $effect on each commit, so it keeps scheduling run() across these
// flushes → the bounded-count assertion (or the 5s timeout) catches it.
async function settle() {
  for (let i = 0; i < 5; i += 1) await tick();
}

beforeEach(() => {
  pipeline.clear();
  seedResult();
});

describe('InspectorColumn reactive wiring (wyrd-c6o1.1 / wyrd-200v)', () => {
  it('does not loop: run() CONVERGES — the commit-to-store write-back does not re-fire the $effect', async () => {
    const run = vi.spyOn(pipeline, 'run'); // spy that CALLS THROUGH — commit edge live
    render(InspectorColumn, { props: {} });
    await settle();
    const afterFirst = run.mock.calls.length;
    await settle(); // keep flushing — a reintroduced loop would keep scheduling run()
    const afterSecond = run.mock.calls.length;

    expect(afterFirst).toBeGreaterThan(0); // the effect ran the base
    // The decisive assertion: the count reached a FIXED POINT. The fix's untracked
    // base means the commit can't re-fire the effect → run() stops. The bug shape
    // (base read tracked off currentResult) re-fires on every commit → the count
    // keeps climbing across the extra flushes (or the loop hangs → 5s timeout).
    expect(afterSecond).toBe(afterFirst); // converged — NOT looping
  });

  it('re-snapshots + re-runs the base when the selected result changes (then re-converges)', async () => {
    const run = vi.spyOn(pipeline, 'run');
    render(InspectorColumn, { props: {} });
    await settle();
    const afterMount = run.mock.calls.length;

    appState.results = [
      ...appState.results,
      { result: 'Other', morphemes_by_word: [[{ usage: 'oth-', _wordIndex: 0, _morphemeIndex: 0 }]] },
    ];
    appState.currentResultIndex = 1; // switch subject
    await settle();
    const afterSwitch = run.mock.calls.length;
    await settle();

    expect(afterSwitch).toBeGreaterThan(afterMount); // the switch re-ran the new base
    expect(run.mock.calls.length).toBe(afterSwitch); // …and re-converged (no loop)
  });
});

// wyrd-410t: the time-warp button bar (Section 2). One button per era stage,
// flag-gated behind 'rewind'; pressing a stage drives pipeline.setRewind.
describe('InspectorColumn time-warp bar (wyrd-410t)', () => {
  it('is hidden when the rewind flag is off (prod default)', async () => {
    appState.manifest = null; // config → null → flagOn false
    const { queryByRole } = render(InspectorColumn, { props: {} });
    await settle();
    expect(queryByRole('group', { name: /time-warp/i })).toBeNull();
  });

  it('renders one button per era stage when the flag is on', async () => {
    appState.manifest = { config: { all: true } };
    const { getByRole, getByText } = render(InspectorColumn, { props: {} });
    await settle();
    expect(getByRole('group', { name: /time-warp/i })).toBeTruthy();
    // The three rewind-exposed stages, short-labelled (date range → title only).
    expect(getByText('Old English')).toBeTruthy();
    expect(getByText('Middle English')).toBeTruthy();
    expect(getByText('Modern')).toBeTruthy();
  });

  it('pressing a stage adds a single front rewind step + marks it active; pressing it again clears', async () => {
    appState.manifest = { config: { all: true } };
    const setRewind = vi.spyOn(pipeline, 'setRewind');
    const { getByText } = render(InspectorColumn, { props: {} });
    await settle();

    await fireEvent.click(getByText('Middle English'));
    expect(setRewind).toHaveBeenCalledWith('me');
    expect(pipeline.steps.filter((s) => s.kind === 'rewind')).toHaveLength(1);
    expect(pipeline.rewindEra).toBe('me');
    await settle();
    // active stage reflects aria-pressed
    expect(getByText('Middle English').getAttribute('aria-pressed')).toBe('true');

    await fireEvent.click(getByText('Middle English')); // press active → clear
    expect(pipeline.rewindEra).toBe(null);
  });
});
