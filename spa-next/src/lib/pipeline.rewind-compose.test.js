// wyrd-410t (round-1 review): pin the rewind↔swap COMPOSITION through run().
// setRewind keeps the rewind step at the FRONT, so a swap (added earlier, or the
// later per-cell tweak) layers on top + wins. The risk a reviewer flagged: a
// front rewind that DROPS morphemes (wyrd-7cvv subsequence) can leave a
// pre-existing swap's (wordIndex, morphemeIndex) out of bounds. swap.js's
// deliberate loud bounds-check must then surface a per-step ERROR (visible on
// the step card) — never silently swap a different morpheme. These tests pin
// both the happy path (swap wins) and the drop path (loud, not silent).
import { describe, it, expect, beforeEach, vi } from 'vitest';

const mock = vi.hoisted(() => ({
  transforms: {
    rewind: {
      flag: 'rewind',
      defaultParams: { era: 'oe-late' },
      // era 'me' DROPS the last morpheme of each word (unresolvable at that era),
      // rebuilding morphemes_by_word as a subsequence — the wyrd-7cvv behavior.
      // Any other era preserves the structure.
      apply: async (s, p) => {
        if (p.era === 'me') {
          return {
            name: `${s.name}~me`,
            morphemes_by_word: s.morphemes_by_word
              .map((w) => w.slice(0, -1))
              .filter((w) => w.length),
          };
        }
        return { name: `${s.name}~${p.era}`, morphemes_by_word: s.morphemes_by_word };
      },
    },
    swap: {
      // mirrors swap.js's loud bounds check + usage rewrite.
      defaultParams: { wordIndex: 0, morphemeIndex: 0, to: '' },
      apply: async (s, p) => {
        const word = s.morphemes_by_word?.[p.wordIndex];
        if (!word) throw new Error(`word ${p.wordIndex} not in current state`);
        const m = word[p.morphemeIndex];
        if (!m) throw new Error(`morpheme ${p.morphemeIndex} not in word ${p.wordIndex}`);
        const next = s.morphemes_by_word.map((w) => [...w]);
        next[p.wordIndex][p.morphemeIndex] = { ...m, usage: p.to };
        return { name: next.flat().map((x) => x.usage).join(' '), morphemes_by_word: next };
      },
    },
  },
}));

vi.mock('./transforms/index.js', () => ({
  getTransform: (kind) => mock.transforms[kind],
}));

import { pipeline } from './pipeline.svelte.js';
import { appState } from './appState.svelte.js';

const base = () => ({ name: 'Orig', morphemes_by_word: [[{ usage: 'A' }, { usage: 'B' }]] });

describe('pipeline rewind↔swap composition (wyrd-410t)', () => {
  beforeEach(() => {
    pipeline.clear();
    appState.results = [{ result: 'Orig', morphemes_by_word: base().morphemes_by_word }];
    appState.currentResultIndex = 0;
    appState.manifest = { config: { flags: { REWIND: true } } }; // rewind flag ON
  });

  it('swap THEN time-warp: rewind lands at front, the swap layers on top and WINS', async () => {
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: 'A2', original: 'A' });
    pipeline.setRewind('oe-late'); // preserves structure → swap index stays valid
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind', 'swap']);

    await pipeline.run(base());

    expect(pipeline.errors.every((e) => e === null)).toBe(true);
    // Final state: rewound (every morpheme at the era) WITH the swapped cell on top.
    const final = pipeline.states[pipeline.states.length - 1];
    expect(final.morphemes_by_word[0][0].usage).toBe('A2'); // swap won
    expect(final.morphemes_by_word[0][1].usage).toBe('B'); // untouched cell rewound
  });

  it('drop case fails LOUDLY, not silently: rewind drops the swapped morpheme → swap step errors', async () => {
    // Swap targets morpheme (0,1); then time-warp to ME, which drops the last
    // morpheme of the word — so (0,1) is now out of bounds at apply time.
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 1, to: 'B2', original: 'B' });
    pipeline.setRewind('me');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind', 'swap']);

    await pipeline.run(base());

    // Rewind (step 0) succeeded; the swap (step 1) errored LOUDLY — the user sees
    // a ⚠ on that step rather than the edit silently landing on a different morpheme.
    expect(pipeline.errors[0]).toBe(null);
    expect(pipeline.errors[1]).toBeTruthy();
    expect(String(pipeline.errors[1])).toMatch(/morpheme 1 not in word 0/);
  });
});
