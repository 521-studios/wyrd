// wyrd-200v: component-test harness smoke + the MorphemeGrid reactive behavior
// that was Playwright-only before (PR #628 / wyrd-c6o1.1). Pins `originalUsage`:
// it reads the pipeline BASE (states[0]) and FALLS BACK to morpheme.usage when
// the stack is empty — the swap's `original` (revert/placement reference) must
// track that, so a future $derived revert can't silently regress it.
//
// era.js is mocked to feed one deterministic clickable cell, isolating
// MorphemeGrid's own reactive deps (pipeline.states) from the era-axis machinery.
import { fireEvent, render } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../lib/era.js', () => ({
  eraAxis: () => [
    { language: 'old-english', stage: { forms: [{ form: 'tun', gloss: 'town', source: 'attested', is_usage: false }] } },
  ],
  cellForSurface: () => null, // no "current" cell → a click is a swap, not a revert
  primaryGloss: () => 'town',
  isGlossDrift: () => false,
}));

import MorphemeGrid from './MorphemeGrid.svelte';
import { appState } from '../lib/appState.svelte.js';
import { pipeline } from '../lib/pipeline.svelte.js';

const MORPHEME = {
  _wordIndex: 0,
  _morphemeIndex: 0,
  usage: '-ton',
  rendered: null,
  era_grid: [{ family: 'english' }],
};

beforeEach(() => {
  pipeline.clear();
  appState.results = [];
  appState.currentResultIndex = 0;
});

function clickFirstCell() {
  const { container } = render(MorphemeGrid, { props: { morpheme: { ...MORPHEME } } });
  const cell = container.querySelector('button.cell');
  expect(cell, 'a clickable era-grid cell rendered (harness smoke)').toBeInTheDocument();
  return { cell, setSwap: vi.spyOn(pipeline, 'setSwap') };
}

describe('MorphemeGrid originalUsage (wyrd-c6o1.1 / wyrd-200v)', () => {
  it('falls back to morpheme.usage when the pipeline stack is empty', async () => {
    const { cell, setSwap } = clickFirstCell();
    await fireEvent.click(cell);
    expect(setSwap).toHaveBeenCalledWith(expect.objectContaining({ original: '-ton' }));
  });

  it('reads the pipeline BASE (states[0]), NOT the committed appState.currentResult', async () => {
    // The discriminating case: the stack BASE and the committed currentResult hold
    // DIFFERENT surfaces (the pipeline overwrites currentResult with the edited
    // form so edits persist — wyrd-c6o1.1). originalUsage must track the BASE
    // (states[0]); the reverted bug read currentResult and would yield 'EDITED-ton'.
    // Direct `pipeline.states =` assignment seeds the base without driving a real
    // run — the field is plain $state, so this is the minimal fixture.
    pipeline.states = [{ morphemes_by_word: [[{ usage: 'BASE-ton' }]] }];
    appState.results = [{ result: 'EDITED', morphemes_by_word: [[{ usage: 'EDITED-ton' }]] }];
    appState.currentResultIndex = 0;
    const { cell, setSwap } = clickFirstCell();
    await fireEvent.click(cell);
    expect(setSwap).toHaveBeenCalledWith(expect.objectContaining({ original: 'BASE-ton' }));
  });
});
