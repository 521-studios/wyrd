// wyrd-obvc: a flag-gated transform (e.g. Rewind) whose feature flag is OFF is
// skipped as a pass-through (wyrd-nwpa) — but it must be MARKED disabled so the
// step card shows a "disabled (feature off)" marker instead of a "→ <name>"
// preview of the unchanged name (which looked as if the step had run).
import { describe, it, expect, beforeEach, vi } from 'vitest';

// A flag-gated transform whose apply() mutates the name (appends '!'), so the
// "skipped" assertion (name unchanged) is meaningful vs the "ran" case.
const mock = vi.hoisted(() => ({
  transform: {
    flag: 'rewind',
    defaultParams: {},
    apply: async (s) => ({ name: `${s.name}!`, morphemes_by_word: [[{ usage: 'X' }]] }),
  },
}));

vi.mock('./transforms/index.js', () => ({
  getTransform: () => mock.transform,
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

  it('keeps disabled[] aligned with steps across a mixed pipeline', async () => {
    // Two flag-off steps → both marked disabled, both pass-through.
    appState.manifest = { config: { flags: {} } };
    pipeline.addStep('rewind');
    pipeline.addStep('rewind');
    await pipeline.run(base());

    expect(pipeline.disabled).toEqual([true, true]);
    expect(pipeline.states.map((s) => s.name)).toEqual(['Orig', 'Orig', 'Orig']);
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
