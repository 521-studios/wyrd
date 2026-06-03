// wyrd-qc0g: unit tests for the family × era reflex-grid helpers. These are
// pure functions on the era_grid payload (wyrd-lftl); svelte-check only
// type-checks, so the fold-match + de-accent behavior is pinned here.
import { describe, it, expect } from 'vitest';
import { deAccent, cellForSurface, hasEraGrid, pronForSurface } from './era.js';

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
  it('prefers the pinned _lang stage for a homograph surface', () => {
    // same surface "don" present in two families/stages; the pin disambiguates.
    const homograph = {
      usage: 'don',
      _lang: 'old-french',
      era_grid: [
        { family: 'english', stages: [{ language: 'old-english', forms: [{ form: 'don' }] }] },
        { family: 'norman-french', stages: [{ language: 'old-french', forms: [{ form: 'don' }] }] },
      ],
    };
    expect(cellForSurface(homograph, 'don')?.language).toBe('old-french');
    // with no pin, falls back to the first match (English, listed first).
    expect(cellForSurface({ ...homograph, _lang: undefined }, 'don')?.language).toBe('old-english');
    // pin is a SOFT preference: a pin whose stage has no matching cell still
    // falls back to the first match (surface drifted out of the pinned stage).
    expect(cellForSurface({ ...homograph, _lang: 'old-norse' }, 'don')?.language).toBe(
      'old-english',
    );
  });

  it('returns null for a surface in no cell, and for missing grid/surface', () => {
    expect(cellForSurface(STAN, 'nowhere')).toBeNull();
    expect(cellForSurface(STAN, '')).toBeNull();
    expect(cellForSurface({}, 'stan')).toBeNull();
    expect(cellForSurface(null, 'stan')).toBeNull();
  });
});

describe('pronForSurface', () => {
  it('returns the matching cell when it carries sound', () => {
    expect(pronForSurface(STAN, 'stān')).toEqual({
      form: 'stān',
      source: 'cluster',
      ipa: '/stɑːn/',
      reader_pronunciation: 'STAAN',
    });
  });

  it('falls THROUGH a pronless matched cell to rendered_pron', () => {
    // The surface matches the ME cell ('ston'), but that cell has no ipa/reader
    // — must not blank the guide; fall through to the era render pron.
    const m = { ...STAN, rendered_pron: { ipa: '/stoːn/' } };
    expect(pronForSurface(m, 'ston')).toEqual({ ipa: '/stoːn/' });
  });

  it('returns an empty object (never null) when nothing has pron', () => {
    // matched cell pronless, no rendered_pron, no renderings to fall back on.
    expect(pronForSurface(STAN, 'ston')).toEqual({});
    expect(pronForSurface({ usage: 'x' }, 'x')).toEqual({});
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
