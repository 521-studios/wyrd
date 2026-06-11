// wyrd-y0lx: pipeline.setRegenerate maintains AT MOST ONE regenerate step
// per (wordIndex, morphemeIndex) slot — a second click re-rolls IN PLACE
// (fresh seed on the existing step) instead of stacking, so removing the
// step always reverts straight to the pre-regenerate morpheme.
import { describe, it, expect, beforeEach } from 'vitest';
import { pipeline } from './pipeline.svelte.js';

const regenSteps = () => pipeline.steps.filter((s) => s.kind === 'regenerate-morpheme');

describe('pipeline.setRegenerate', () => {
  beforeEach(() => {
    pipeline.clear();
  });

  it('adds one step per slot, baking seed + context + from', () => {
    pipeline.setRegenerate({
      wordIndex: 0,
      morphemeIndex: 1,
      context: { culture: 'english' },
      from: '-tūn',
    });
    expect(regenSteps()).toHaveLength(1);
    const step = regenSteps()[0];
    expect(step.params.wordIndex).toBe(0);
    expect(step.params.morphemeIndex).toBe(1);
    expect(step.params.context).toEqual({ culture: 'english' });
    expect(step.params.from).toBe('-tūn');
    expect(Number.isInteger(step.params.seed)).toBe(true);
  });

  it('re-rolls in place on a second click (fresh seed + context, no new step)', () => {
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 1, context: { novelty: 0 } });
    const firstSeed = regenSteps()[0].params.seed;
    const firstId = regenSteps()[0].id;
    // Math.random seeds collide with ~2^-53 probability; loop a few clicks
    // so the assertion can't flake on a one-in-the-universe collision.
    let seeds = new Set([firstSeed]);
    for (let i = 0; i < 3; i += 1) {
      pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 1, context: { novelty: 1 } });
      seeds.add(regenSteps()[0].params.seed);
    }
    expect(regenSteps()).toHaveLength(1);
    expect(regenSteps()[0].id).toBe(firstId);
    expect(seeds.size).toBeGreaterThan(1);
    // The caller snapshots a fresh context every click; the re-roll must
    // honor it rather than silently keeping the first click's copy.
    expect(regenSteps()[0].params.context).toEqual({ novelty: 1 });
  });

  it('distinct slots get distinct steps', () => {
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 0, context: {} });
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 1, context: {} });
    pipeline.setRegenerate({ wordIndex: 1, morphemeIndex: 0, context: {} });
    expect(regenSteps()).toHaveLength(3);
  });

  it('removing the step reverts (the undo affordance)', () => {
    pipeline.setRegenerate({ wordIndex: 0, morphemeIndex: 1, context: {} });
    pipeline.removeStep(0);
    expect(regenSteps()).toHaveLength(0);
  });
});
