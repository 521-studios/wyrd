// wyrd-51rv: regression tests for currentWorkspaceSnapshot() — the single
// source of the save/share payload. Pins (1) the exact captured field set,
// (2) falsy-value survival in params (seed:0 / count:0 / cohesion:false must
// not be dropped), and (3) the load-bearing wyrd-8jjx fix: params come from
// paramsByGenerator[resultsGenerator], NOT the picker's selectedGeneratorName.
import { describe, it, expect, beforeEach } from 'vitest';
import { appState } from './appState.svelte.js';
import { pipeline } from './pipeline.svelte.js';
import { currentWorkspaceSnapshot } from './workspaceSnapshot.js';

beforeEach(() => {
  appState.results = [];
  appState.currentResultIndex = null;
  appState.resultsGenerator = null;
  appState.paramsByGenerator = {};
  appState.selectedGeneratorName = 'kenning';
  pipeline.steps = [];
});

describe('currentWorkspaceSnapshot', () => {
  it('returns null when no result is selected', () => {
    expect(currentWorkspaceSnapshot()).toBeNull();
  });

  it('captures the full field set { generator, params, original, pipeline }', () => {
    appState.results = [
      { result: 'Bryntir', morphemes_by_word: [['bryn', 'tir']], explanation: 'hill-land' },
    ];
    appState.currentResultIndex = 0;
    appState.resultsGenerator = 'kenning';
    appState.paramsByGenerator = { kenning: { culture: 'welsh', count: 5 } };
    pipeline.steps = [{ kind: 'rewind', params: { era: 'oe-late' } }];

    const snap = currentWorkspaceSnapshot();
    expect(Object.keys(snap).sort()).toEqual(['generator', 'original', 'params', 'pipeline']);
    expect(snap.generator).toBe('kenning');
    expect(snap.params).toEqual({ culture: 'welsh', count: 5 });
    expect(snap.original).toEqual({
      name: 'Bryntir',
      morphemes_by_word: [['bryn', 'tir']],
      explanation: 'hill-land',
    });
    expect(snap.pipeline).toEqual([{ kind: 'rewind', params: { era: 'oe-late' } }]);
  });

  it('preserves falsy params (seed:0 / count:0 / cohesion:false / packs:[])', () => {
    appState.results = [{ result: 'Ton', explanation: '' }];
    appState.currentResultIndex = 0;
    appState.resultsGenerator = 'kenning';
    appState.paramsByGenerator = {
      kenning: { seed: 0, count: 0, novelty: 0, cohesion: false, packs: [] },
    };

    const snap = currentWorkspaceSnapshot();
    expect(snap.params).toEqual({ seed: 0, count: 0, novelty: 0, cohesion: false, packs: [] });
    // Defaults for a result missing optional fields — pinned, not dropped.
    expect(snap.original).toEqual({ name: 'Ton', morphemes_by_word: [], explanation: '' });
  });

  it('pulls params from resultsGenerator, NOT the picker (wyrd-8jjx)', () => {
    // The user rolled with 'kenning', then switched the picker to 'kenning-rewind'
    // WITHOUT re-rolling. The snapshot must carry the params that produced the
    // current result (kenning's), not the picker's.
    appState.results = [{ result: 'Bryntir' }];
    appState.currentResultIndex = 0;
    appState.resultsGenerator = 'kenning';
    appState.selectedGeneratorName = 'kenning-rewind';
    appState.paramsByGenerator = {
      kenning: { culture: 'welsh', count: 5 },
      'kenning-rewind': { culture: 'english', count: 99 },
    };

    const snap = currentWorkspaceSnapshot();
    expect(snap.generator).toBe('kenning');
    expect(snap.params).toEqual({ culture: 'welsh', count: 5 });
  });

  it('falls back to an empty params object when the generator has none yet', () => {
    appState.results = [{ result: 'X' }];
    appState.currentResultIndex = 0;
    appState.resultsGenerator = 'kenning';
    appState.paramsByGenerator = {};
    expect(currentWorkspaceSnapshot().params).toEqual({});
  });
});
