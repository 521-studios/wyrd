// wyrd-c6o1.1: the pipeline continuously commits its OUTPUT into the canonical
// store (appState.results[currentResultIndex]) so a reroll/reflex-swap PERSISTS
// — survives blur / result-switch / save, and downstream ops read the edited
// data. These pin that contract. The transform module is mocked here so a step
// produces a deterministic "edited" state without real era/swap fixtures.
//
// The mock's apply is INPUT-DEPENDENT (appends '+'), so a multi-step pipeline's
// bottom state genuinely differs from the first step's — letting us verify the
// commit takes states[N-1] (compounding), not states[1].
import { describe, it, expect, beforeEach, vi } from 'vitest';

const mock = vi.hoisted(() => ({
  apply: async (s) => ({ name: `${s.name}+`, morphemes_by_word: [[{ usage: 'B' }]] }),
}));

vi.mock('./transforms/index.js', () => ({
  getTransform: () => ({ defaultParams: {}, apply: (s, p) => mock.apply(s, p) }),
}));

import { pipeline } from './pipeline.svelte.js';
import { appState } from './appState.svelte.js';

const DEFAULT_APPLY = async (s) => ({ name: `${s.name}+`, morphemes_by_word: [[{ usage: 'B' }]] });
const base = () => ({
  name: 'Orig',
  morphemes_by_word: [[{ usage: 'A' }]],
  result_modern: 'OrigMod',
});

describe('pipeline commits output to the canonical store (wyrd-c6o1.1)', () => {
  beforeEach(() => {
    pipeline.clear();
    mock.apply = DEFAULT_APPLY;
    appState.results = [{ result: 'Orig', morphemes_by_word: [[{ usage: 'A' }]], result_modern: 'OrigMod' }];
    appState.currentResultIndex = 0;
  });

  it('a transform commits the edited name + morphemes into the store', async () => {
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Orig+');
    expect(appState.results[0].morphemes_by_word).toEqual([[{ usage: 'B' }]]);
  });

  it('commits the BOTTOM of a multi-step pipeline (edits compound)', async () => {
    pipeline.addStep('swap');
    pipeline.addStep('swap');
    await pipeline.run(base());
    // input-dependent mock applied twice → states[2], not states[1]
    expect(appState.results[0].result).toBe('Orig++');
  });

  it('nulls the now-stale modern companion on a transform', async () => {
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result_modern).toBeNull();
  });

  it('reverting to base (step removed) restores name + morphemes + modern', async () => {
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Orig+');

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
    expect(appState.results[1].result).toBe('Orig+');
  });

  it('a stale run (subject switched mid-flight) commits nothing', async () => {
    pipeline.addStep('swap');
    const p = pipeline.run(base()); // in flight
    pipeline.clear(); // subject switch bumps the token
    await p;
    expect(appState.results[0].result).toBe('Orig'); // never clobbered
  });

  it('a transform that yields no morphemes_by_word does not clobber the store', async () => {
    mock.apply = async (s) => ({ name: `${s.name}!` }); // no morphemes_by_word
    pipeline.addStep('swap');
    await pipeline.run(base());
    expect(appState.results[0].result).toBe('Orig!');
    // the falsy-guard leaves the existing morphemes intact (not undefined)
    expect(appState.results[0].morphemes_by_word).toEqual([[{ usage: 'A' }]]);
  });

  it('an out-of-bounds currentResultIndex commits nothing (no throw)', async () => {
    appState.currentResultIndex = 99; // no result at this index
    pipeline.addStep('swap');
    await expect(pipeline.run(base())).resolves.toBeDefined();
    expect(appState.results[0].result).toBe('Orig'); // untouched
  });
});
