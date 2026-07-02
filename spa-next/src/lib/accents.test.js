// wyrd-rogd.6: the scholarly '*' reconstructed/unattested-form marker must not
// leak into rendered surfaces — it grafts away on display + swap, and folds
// away for matching. Plus the core accentFold / graftPosition behavior.
import { describe, it, expect } from 'vitest';
import { accentForm, accentFold, graftPosition, accentedUsage } from './accents.js';

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

  it('drops category-Mn marks in ANY block, keeps spacing (Mc) matras — parity with the Python folds (wyrd-nndd)', () => {
    // Latin accents (Mn, U+0300-U+036F) — unchanged by the widening.
    expect(accentFold('\u00e9')).toBe('e'); // e-acute (Mn U+0301)
    expect(accentFold('\u0101')).toBe('a'); // a-macron (Mn U+0304)
    // Mn marks OUTSIDE U+0300-U+036F: the old codepoint range KEPT these (the
    // documented divergence vs Python); the \p{Mn} test now DROPS them, matching
    // bundle/_subject._surface_fold + runtime/proportions._grid_match_key.
    expect(accentFold('n\u05b4')).toBe('n'); // Hebrew point hiriq (Mn, outside U+0300-036F)
    expect(accentFold('b\u064e')).toBe('b'); // Arabic fatha (Mn)
    expect(accentFold('c\u1ab0')).toBe('c'); // Combining Diacritical Marks Extended (Mn)
    // Spacing combining mark (Mc, ccc==0): NOT Mn, so KEPT by both sides — this
    // is why we unify on \p{Mn}, never \p{M} (which would corrupt Indic matras).
    expect(accentFold('k\u093e')).toBe('k\u093e'); // Devanagari vowel sign AA (Mc, spacing) - KEPT
    // Distinguishes the JS change too: old range kept U+0902, \p{Mn} drops it.
    expect(accentFold('k\u0902')).toBe('k'); // Devanagari anusvara (Mn, ccc==0)
  });
});

describe('accentForm (wyrd-rogd.17: Inspect grid cell matches Output accents)', () => {
  // A morpheme whose stored reflex form is ASCII ("Tongby") but whose
  // rendering carries the accented original_script ("Tongbȳ") — the exact
  // Output/Inspect divergence the ticket reports.
  const morph = {
    renderings: {
      old_english: {
        Tongby: { original_script: 'Tongbȳ' },
        Treton: { original_script: 'Tretōn' },
      },
    },
  };

  it('upgrades an ASCII cell form to its accented original_script', () => {
    expect(accentForm('Tongby', morph)).toBe('Tongbȳ');
    expect(accentForm('Treton', morph)).toBe('Tretōn');
  });

  it('folds (case/dash) when matching the rendering — not raw equality', () => {
    expect(accentForm('tongby', morph)).toBe('Tongbȳ');
    expect(accentForm('-Tongby', morph)).toBe('Tongbȳ');
  });

  it('returns the form unchanged when no rendering supplies an accent', () => {
    expect(accentForm('Halton', morph)).toBe('Halton');
    // a rendering with no diacritic in original_script is not an "upgrade".
    const plain = { renderings: { old_english: { stone: { original_script: 'stone' } } } };
    expect(accentForm('stone', plain)).toBe('stone');
  });

  it('is safe on missing/empty inputs and null lang buckets', () => {
    expect(accentForm('', morph)).toBe('');
    expect(accentForm('Tongby', null)).toBe('Tongby');
    expect(accentForm('Tongby', {})).toBe('Tongby');
    expect(accentForm('Tongby', { renderings: { old_english: null } })).toBe('Tongby');
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

describe('accentedUsage applies the slot position case rule (like graftPosition)', () => {
  // A capitalized accented original_script ("Bȳ", "Rōm") in a dashed (post/inner)
  // slot must render lowercase — col 2 (which renders accentedUsage raw, with no
  // renderName title-case) otherwise leaks the capital ("-Bȳ") where the position
  // rule and col 3 (via graftPosition) both say "-bȳ".
  it('lowercases the accented surface in a post/inner slot', () => {
    const post = { usage: '-by', renderings: { oe: { by: { original_script: 'Bȳ' } } } };
    expect(accentedUsage(post)).toBe('-bȳ');
    const inner = { usage: '-rom-', renderings: { oe: { rom: { original_script: 'Rōm' } } } };
    expect(accentedUsage(inner)).toBe('-rōm-');
  });

  it('keeps the original_script case in a pre/bare slot (no leading dash)', () => {
    const pre = { usage: 'Cornel-', renderings: { oe: { cornel: { original_script: 'Córnel' } } } };
    expect(accentedUsage(pre)).toBe('Córnel-');
    const bare = { usage: 'Rom', renderings: { oe: { rom: { original_script: 'Rōm' } } } };
    expect(accentedUsage(bare)).toBe('Rōm');
  });

  it('is a no-op when the original_script is already lowercase (the common case)', () => {
    const m = { usage: '-by', renderings: { oe: { by: { original_script: 'bȳ' } } } };
    expect(accentedUsage(m)).toBe('-bȳ');
  });

  it('returns null when no rendering supplies an accented surface', () => {
    const m = { usage: '-by', renderings: { oe: { by: { original_script: 'by' } } } };
    expect(accentedUsage(m)).toBeNull();
  });
});
