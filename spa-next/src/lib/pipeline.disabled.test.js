// wyrd-obvc: a flag-gated transform (e.g. Rewind) whose feature flag is OFF is
// skipped as a pass-through (wyrd-nwpa) — but it must be MARKED disabled so the
// step card shows a "disabled (feature off)" marker instead of a "→ <name>"
// preview of the unchanged name (which looked as if the step had run).
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Kind-keyed transforms so a pipeline can mix a flag-gated step (rewind), a
// plain non-flag-gated step (swap), and a throwing step (boom) — exercising the
// disabled-marker, the no-flag branch, and the halt-on-error pad alignment.
const mock = vi.hoisted(() => ({
  transforms: {
    rewind: {
      flag: 'rewind',
      defaultParams: {},
      apply: async (s) => ({ name: `${s.name}!`, morphemes_by_word: [[{ usage: 'X' }]] }),
    },
    swap: {
      // no `flag` → never disabled
      defaultParams: {},
      apply: async (s) => ({ name: `${s.name}+`, morphemes_by_word: [[{ usage: 'Y' }]] }),
    },
    boom: {
      defaultParams: {},
      apply: async () => {
        throw new Error('kaboom');
      },
    },
  },
}));

vi.mock('./transforms/index.js', () => ({
  getTransform: (kind) => mock.transforms[kind],
}));

import { pipeline } from './pipeline.svelte.js';
import { appState } from './appState.svelte.js';

const base = () => ({ name: 'Orig', morphemes_by_word: [[{ usage: 'A' }]] });

describe('pipeline disabled-step marker (wyrd-obvc)', () => {
  beforeEach(() => {
    pipeline.clear();
    appState.results = [{ result: 'Orig', morphemes_by_word: [[{ usage: 'A' }]] }];
    appState.currentResultIndex = 0;
  });

  it('marks a flag-OFF step disabled and leaves the state unchanged', async () => {
    appState.manifest = { config: { flags: {} } }; // rewind absent → off
    pipeline.addStep('rewind');
    await pipeline.run(base());

    expect(pipeline.disabled[0]).toBe(true);
    expect(pipeline.errors[0]).toBe(null);
    // Skipped as pass-through: the step's output state equals its input.
    expect(pipeline.states[1].name).toBe('Orig');
  });

  it('does NOT mark a flag-ON step disabled (it runs normally)', async () => {
    appState.manifest = { config: { flags: { REWIND: true } } }; // envify('rewind')==='REWIND'
    pipeline.addStep('rewind');
    await pipeline.run(base());

    expect(pipeline.disabled[0]).toBe(false);
    expect(pipeline.states[1].name).toBe('Orig!'); // apply() ran
  });

  it('does NOT mark a NON-flag-gated step disabled (no flag → always runs)', async () => {
    appState.manifest = { config: { flags: {} } }; // no flags on at all
    pipeline.addStep('swap'); // swap has no `flag`
    await pipeline.run(base());

    expect(pipeline.disabled[0]).toBe(false);
    expect(pipeline.states[1].name).toBe('Orig+'); // ran despite all-flags-off
  });

  it('keeps disabled[] aligned with steps across a mixed pipeline', async () => {
    // Two flag-off steps → both marked disabled, both pass-through.
    appState.manifest = { config: { flags: {} } };
    pipeline.addStep('rewind');
    pipeline.addStep('rewind');
    await pipeline.run(base());

    expect(pipeline.disabled).toEqual([true, true]);
    expect(pipeline.states.map((s) => s.name)).toEqual(['Orig', 'Orig', 'Orig']);
  });

  it('pads disabled[] aligned with steps when a later step ERRORS (halt path)', async () => {
    // A flag-off (disabled) step BEFORE a throwing step: the catch breaks
    // before pushing to disabled, so the pad loop must backfill the erroring
    // step as false (an error is NOT a disable). disabled stays length-aligned.
    appState.manifest = { config: { flags: {} } };
    pipeline.addStep('rewind'); // flag off → disabled
    pipeline.addStep('boom'); // throws
    await pipeline.run(base());

    expect(pipeline.disabled).toEqual([true, false]);
    expect(pipeline.errors[0]).toBe(null);
    expect(pipeline.errors[1]).toBe('kaboom');
    expect(pipeline.disabled).toHaveLength(pipeline.steps.length);
  });

  it('clear() resets the disabled array', async () => {
    appState.manifest = { config: { flags: {} } };
    pipeline.addStep('rewind');
    await pipeline.run(base());
    expect(pipeline.disabled).toHaveLength(1);
    pipeline.clear();
    expect(pipeline.disabled).toEqual([]);
  });
});
