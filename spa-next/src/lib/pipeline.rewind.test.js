// wyrd-410t: pipeline.setRewind backs the time-warp button bar. It maintains
// AT MOST ONE rewind step, kept at the FRONT of the pipeline (the global era
// floor) so per-slot swaps added later layer on top and win. Pressing a stage
// adds/switches it; pressing the active stage clears it.
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

  it('keeps the rewind step at the FRONT so later swaps win (era floor)', () => {
    // Swap first, then time-warp: rewind must land BEFORE the swap so the
    // swapped cell layers on top of the global era floor (→ Mixed badge).
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: '-hām', original: '-ton' });
    pipeline.setRewind('oe-late');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind', 'swap']);
    // Switching era keeps the front position.
    pipeline.setRewind('modern');
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['rewind', 'swap']);
    expect(pipeline.steps[0].params.era).toBe('modern');
  });

  it('clearing the rewind leaves other steps untouched', () => {
    pipeline.setSwap({ wordIndex: 0, morphemeIndex: 0, to: '-hām', original: '-ton' });
    pipeline.setRewind('me');
    pipeline.setRewind('me'); // clear
    expect(pipeline.steps.map((s) => s.kind)).toEqual(['swap']);
  });
});
