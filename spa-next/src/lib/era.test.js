// wyrd-qc0g: unit tests for the family × era reflex-grid helpers. These are
// pure functions on the era_grid payload (wyrd-lftl); svelte-check only
// type-checks, so the fold-match + de-accent behavior is pinned here.
import { describe, it, expect } from 'vitest';
import { deAccent, cellForSurface, hasEraGrid } from './era.js';

// A morpheme shaped like the backend ships (era_grid: families → stages →
// forms). Keyed by hyphenated canonical language tags.
const STAN = {
  usage: 'Stan-',
  era_grid: [
    {
      family: 'english',
      stages: [
        {
          language: 'old-english',
          forms: [{ form: 'stān', source: 'cluster', ipa: '/stɑːn/', reader_pronunciation: 'STAAN' }],
        },
        {
          language: 'middle-english',
          forms: [{ form: 'ston', source: 'cluster' }],
        },
      ],
    },
    {
      family: 'norse',
      stages: [{ language: 'old-norse', forms: [{ form: 'steinn', source: 'descent' }] }],
    },
  ],
};

describe('deAccent', () => {
  it('strips diacritics but preserves case and dashes', () => {
    expect(deAccent('Trebȳ')).toBe('Treby');
    expect(deAccent('-bȳ')).toBe('-by');
    expect(deAccent('-tūn-')).toBe('-tun-');
    expect(deAccent('Stāntūn')).toBe('Stantun');
  });
  it('is a no-op on plain ASCII and handles empty input', () => {
    expect(deAccent('Stoneton')).toBe('Stoneton');
    expect(deAccent('')).toBe('');
    expect(deAccent(null)).toBe('');
  });
});

describe('cellForSurface', () => {
  it('matches a surface to its cell folding accent + dash + case', () => {
    // accented original_script form, surface comes in de-accented + dashed
    const hit = cellForSurface(STAN, '-stān');
    expect(hit?.language).toBe('old-english');
    expect(hit?.cell.ipa).toBe('/stɑːn/');
    // de-accented + lowercased surface still matches the accented cell form
    expect(cellForSurface(STAN, 'stan')?.language).toBe('old-english');
  });
  it('resolves the right stage/family for a later-era form', () => {
    expect(cellForSurface(STAN, 'ston')?.language).toBe('middle-english');
    const norse = cellForSurface(STAN, 'steinn');
    expect(norse?.family).toBe('norse');
    expect(norse?.language).toBe('old-norse');
  });
  it('returns null for a surface in no cell, and for missing grid/surface', () => {
    expect(cellForSurface(STAN, 'nowhere')).toBeNull();
    expect(cellForSurface(STAN, '')).toBeNull();
    expect(cellForSurface({}, 'stan')).toBeNull();
    expect(cellForSurface(null, 'stan')).toBeNull();
  });
});

describe('hasEraGrid', () => {
  it('is true only when the morpheme carries at least one stage', () => {
    expect(hasEraGrid(STAN)).toBe(true);
    expect(hasEraGrid({ usage: 'x' })).toBe(false);
    expect(hasEraGrid({ usage: 'x', era_grid: [] })).toBe(false);
    expect(hasEraGrid({ usage: 'x', era_grid: [{ family: 'english', stages: [] }] })).toBe(false);
    expect(hasEraGrid(null)).toBe(false);
  });
});
