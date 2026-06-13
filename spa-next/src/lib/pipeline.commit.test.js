// wyrd-c6o1.1: the pipeline continuously commits its OUTPUT into the canonical
// store (appState.results[currentResultIndex]) so a reroll/reflex-swap PERSISTS
// — survives blur / result-switch / save, and downstream ops read the edited
// data. These pin that contract. The transform module is mocked here so a step
// produces a deterministic "edited" state without real era/swap fixtures.
import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('./transforms/index.js', () => ({
  getTransform: () => ({
    defaultParams: {},
    apply: async () => ({ name: 'Edited', morphemes_by_word: [[{ usage: 'B' }]] }),
  }),
}));

import { pipeline } from './pipeline.svelte.js';
import { appState } from './appState.svelte.js';

const base = () => ({
  name: 'Orig',
  morphemes_by_word: [[{ usage: 'A' }]],
  result_modern: 'OrigMod',
});

describe('pipeline commits output to the canonical store (wyrd-c6o1.1)', () => {
  beforeEach(() => {
    pipeline.clear();
    appState.results = [{ result: 'Orig', morphemes_by_word: [[{ usage: 'A' }]], result_modern: 'OrigMod' }];
    appState.currentResultIndex = 0;
  });

  it('a transform commits the edited name + morphemes into the store', async () => {
    pipeline.addStep('swap'); // mocked transform → { name: 'Edited', ... }
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Edited');
    expect(appState.results[0].morphemes_by_word).toEqual([[{ usage: 'B' }]]);
  });

  it('nulls the now-stale modern companion on a transform', async () => {
    pipeline.addStep('swap');
    await pipeline.run(base());
    // the original reflex no longer describes the edited name → hidden
    expect(appState.results[0].result_modern).toBeNull();
  });

  it('reverting to base (step removed) restores name + morphemes + modern', async () => {
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Edited');

    pipeline.removeStep(0);
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Orig');
    expect(appState.results[0].morphemes_by_word).toEqual([[{ usage: 'A' }]]);
    expect(appState.results[0].result_modern).toBe('OrigMod');
  });

  it('commits to the SELECTED result only', async () => {
    appState.results = [
      { result: 'One', morphemes_by_word: [[{ usage: 'A' }]] },
      { result: 'Two', morphemes_by_word: [[{ usage: 'C' }]] },
    ];
    appState.currentResultIndex = 1;
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('One'); // untouched
    expect(appState.results[1].result).toBe('Edited');
  });

  it('a stale run (subject switched mid-flight) commits nothing', async () => {
    pipeline.addStep('swap');
    const p = pipeline.run(base()); // in flight
    pipeline.clear(); // subject switch bumps the token
    await p;
    expect(appState.results[0].result).toBe('Orig'); // never clobbered
  });
});
