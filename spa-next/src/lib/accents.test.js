// wyrd-rogd.6: the scholarly '*' reconstructed/unattested-form marker must not
// leak into rendered surfaces — it grafts away on display + swap, and folds
// away for matching. Plus the core accentFold / graftPosition behavior.
import { describe, it, expect } from 'vitest';
import { accentFold, graftPosition } from './accents.js';

describe('accentFold', () => {
  it('folds case, diacritics, dashes — and the reconstructed * marker', () => {
    expect(accentFold('bȳ')).toBe('by');
    expect(accentFold('By')).toBe('by');
    expect(accentFold('-tūn-')).toBe('tun');
    // a stripped surface folds to the same key as its starred cell form, so the
    // highlight + pronunciation still match after the * is grafted away.
    expect(accentFold('*ur')).toBe('ur');
    expect(accentFold('*ur')).toBe(accentFold('ur'));
    expect(accentFold('*bearwe')).toBe('bearwe');
  });
});

describe('graftPosition', () => {
  it('strips the reconstructed * marker from the grafted surface', () => {
    // bare slot keeps case; * gone (this is the Tray*bearwe leak).
    expect(graftPosition('East', '*ur')).toBe('ur');
    expect(graftPosition('Bearwe', '*bearwe')).toBe('bearwe');
    // a * mid/anywhere is removed.
    expect(graftPosition('x', 'be*arwe')).toBe('bearwe');
  });
  it('keeps placement dashes + applies the position case rule', () => {
    // inner/post slot ('-X-' / '-X') lowercases; dashes preserved.
    expect(graftPosition('-tūn-', '*Bearwe')).toBe('-bearwe-');
    expect(graftPosition('-ham', 'Hamm')).toBe('-hamm');
    // pre/bare keeps capitalization; diacritics survive.
    expect(graftPosition('Stan-', 'Stān')).toBe('Stān-');
  });
  it('falls back to the original slot when the surface is empty (or only *)', () => {
    expect(graftPosition('-ton', '*')).toBe('-ton');
    expect(graftPosition('-ton', '')).toBe('-ton');
  });
});
