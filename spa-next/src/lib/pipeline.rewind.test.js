// wyrd-410t + swap-clear (2026-06-30): pipeline.setRewind backs the time-warp
// button bar. It maintains AT MOST ONE rewind step. A time-warp CLEARS all swap
// steps (a pre-rewind swap breaks against the rewound subsequence) but PRESERVES
// re-rolls; the rewind is appended at its press-time position (NOT front-pinned),
// so it sits at the appropriate spot among prior + future re-rolls. Pressing a
// stage adds/switches it; pressing the active stage clears it.
import { describe, it, expect, beforeEach } from 'vitest';
import { pipeline } from './pipeline.svelte.js';

const rewindSteps = () => pipeline.steps.filter((s) => s.kind === 'rewind');

describe('pipeline.setRewind', () => {
  beforeEach(() => {
    pipeline.clear();
  });

  it('adds a single rewind step with the pressed era', () => {
    pipeline.setRewind('me');
    expect(rewindSteps()).toHaveLength(1);
    expect(rewindSteps()[0].params.era).toBe('me');
    expect(pipeline.rewindEra).toBe('me');
  });

  it('switches era in place on a different stage (no second step)', () => {
    pipeline.setRewind('oe-late');
    const firstId = rewindSteps()[0].id;
    pipeline.setRewind('modern');
    expect(rewindSteps()).toHaveLength(1);
    expect(rewindSteps()[0].params.era).toBe('modern');
    expect(rewindSteps()[0].id).toBe(firstId); // same step, mutated in place
    expect(pipeline.rewindEra).toBe('modern');
  });

  it('pressing the ACTIVE stage clears it (back to as-generated)', () => {
    pipeline.setRewind('me');
    expect(pipeline.rewindEra).toBe('me');
    pipeline.setRewind('me');
    expect(rewindSteps()).toHaveLength(0);
    expect(pipeline.rewindEra).toBe(null);
  });

  it('rewindEra is null with no rewind step', () => {
    expect(pipeline.rewindEra).toBe(null);
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: '-hām', original: '-ton' });
    expect(pipeline.rewindEra).toBe(null); // a swap is not a rewind
  });

  it('a time-warp CLEARS all swaps from the stack', () => {
    // Swap two cells, then time-warp: both swaps are discarded (a pre-rewind
    // swap targets the as-generated cards + breaks against the rewound
    // subsequence). Only the rewind remains.
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: '-hām', original: '-ton' });
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 1, to: '-by', original: '-ford' });
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['swap', 'swap']);
    pipeline.setRewind('oe-late');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind']);
    // Switching era keeps it single + swap-free.
    pipeline.setRewind('modern');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind']);
    expect(pipeline.steps[0].params.era).toBe('modern');
  });

  it('a time-warp PRESERVES re-rolls and is NOT front-pinned (sits among them)', () => {
    // Re-roll, then time-warp: the rewind is appended AFTER the prior re-roll,
    // not front-pinned — it keeps its chronological position.
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 0 });
    pipeline.setRewind('me');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['regenerate-morpheme', 'rewind']);
    // A FUTURE re-roll appends after the rewind (operates on the rewound state).
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 1 });
    expect(pipeline.steps.map((s) => s.kind)).toEqual([
      'regenerate-morpheme',
      'rewind',
      'regenerate-morpheme',
    ]);
    // Pressing the active stage clears ONLY the rewind; both re-rolls remain.
    pipeline.setRewind('me');
    expect(pipeline.steps.map((s) => s.kind)).toEqual([
      'regenerate-morpheme',
      'regenerate-morpheme',
    ]);
  });
});
