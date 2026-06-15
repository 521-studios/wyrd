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

  it('reads the pipeline BASE (states[0]) surface when the stack is populated', async () => {
    // The base snapshot the stack rebases on — NOT appState.currentResult (which
    // the pipeline now overwrites with the committed/edited surface).
    pipeline.states = [{ morphemes_by_word: [[{ usage: 'BASE-ton' }]] }];
    const { cell, setSwap } = clickFirstCell();
    await fireEvent.click(cell);
    expect(setSwap).toHaveBeenCalledWith(expect.objectContaining({ original: 'BASE-ton' }));
  });
});
