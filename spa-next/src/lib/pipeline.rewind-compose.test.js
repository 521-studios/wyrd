// wyrd-410t + swap-clear (2026-06-30): pin the rewind↔swap interaction through
// run(). A time-warp now CLEARS all swap steps (setRewind), so a pre-rewind swap
// is DISCARDED — run() produces the clean rewound result, never the swapped one.
// This retires the old "rewind front-pinned, swap layers on top" composition,
// which broke when the rewound subsequence (wyrd-7cvv) left a pre-existing swap's
// (wordIndex, morphemeIndex) out of bounds (the "swap then time-warp doesn't
// work" report). Re-roll preservation + non-front-pinning is pinned at the
// stack level in pipeline.rewind.test.js.
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

  it('swap THEN time-warp: the swap is CLEARED — run() yields the clean rewound result', async () => {
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: 'A2', original: 'A' });
    pipeline.setRewind('oe-late');
    // The swap step is gone — only the rewind remains in the stack.
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind']);

    await pipeline.run(base());

    expect(pipeline.errors.every((e) => e === null)).toBe(true);
    // Final state is the clean rewound name — the original morpheme, NOT 'A2'.
    const final = pipeline.states[pipeline.states.length - 1];
    expect(final.morphemes_by_word[0][0].usage).toBe('A'); // swap discarded
    expect(final.morphemes_by_word[0][1].usage).toBe('B');
    expect(final.name).toBe('Orig~oe-late');
  });

  it('drop-era time-warp after a swap still runs clean (no out-of-bounds swap to error)', async () => {
    // The old failure mode: a swap on (0,1) + a rewind that drops the last
    // morpheme left the swap out of bounds → loud per-step error. Now the swap
    // is cleared on time-warp, so the drop era runs without any swap to break.
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 1, to: 'B2', original: 'B' });
    pipeline.setRewind('me'); // ME drops the last morpheme of each word
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind']);

    await pipeline.run(base());

    // No swap step → no error; the rewound (dropped-subsequence) result stands.
    expect(pipeline.errors.every((e) => e === null)).toBe(true);
    const final = pipeline.states[pipeline.states.length - 1];
    expect(final.name).toBe('Orig~me');
    expect(final.morphemes_by_word[0].map((m) => m.usage)).toEqual(['A']); // B dropped, no B2
  });
});
