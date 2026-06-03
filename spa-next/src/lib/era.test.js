// wyrd-qc0g: unit tests for the family × era reflex-grid helpers. These are
// pure functions on the era_grid payload (wyrd-lftl); svelte-check only
// type-checks, so the fold-match + de-accent behavior is pinned here.
import { describe, it, expect } from 'vitest';
import {
  deAccent,
  cellForSurface,
  hasEraGrid,
  pronForSurface,
  eraBadge,
  primaryGloss,
  glossForSurface,
  isGlossDrift,
} from './era.js';

// A stand-in label fn matching the production languageLabel for the tags under
// test: 'modern-english' → 'Modern' (so it aliases the default fold-in, as it
// does live), everything else title-cased ('old-english' → 'Old English').
const label = (t) =>
  t === 'modern-english'
    ? 'Modern'
    : t.replace(/(^|-)([a-z])/g, (_, sep, c) => (sep ? ' ' : '') + c.toUpperCase());

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

describe('eraBadge', () => {
  const word = (...ms) => [ms]; // one word of morphemes

  it('is "as generated" when no morpheme carries an explicit era', () => {
    expect(eraBadge(word({ usage: 'Stan-' }, { usage: '-ton' }), label)).toBe('as generated');
    expect(eraBadge([], label)).toBe('as generated');
    expect(eraBadge(null, label)).toBe('as generated');
  });

  it('shows the single era when every morpheme shares one', () => {
    // an era render: all morphemes rendered_language old-english
    expect(
      eraBadge(word({ usage: 'Stan-', rendered_language: 'old-english' }, { usage: '-tun', rendered_language: 'old-english' }), label),
    ).toBe('Old English');
    // a swap pins via _lang
    expect(eraBadge(word({ usage: '-tun', _lang: 'middle-english' }), label)).toBe('Middle English');
  });

  it('is Mixed when eras differ, and folds in Modern for still-default morphemes', () => {
    // one morpheme swapped to OE, the other still default (modern)
    expect(eraBadge(word({ usage: 'Stan-', _lang: 'old-english' }, { usage: '-ton' }), label)).toBe(
      'Mixed (Old English, Modern)',
    );
    // two distinct explicit eras, none default
    expect(
      eraBadge(word({ usage: 'a', _lang: 'old-english' }, { usage: 'b', _lang: 'middle-english' }), label),
    ).toBe('Mixed (Old English, Middle English)');
  });

  it('ignores blank morphemes when resolving the badge', () => {
    expect(eraBadge(word({ usage: '  ' }, { usage: '-tun', _lang: 'old-english' }), label)).toBe(
      'Old English',
    );
  });

  it('collapses to the single era when the labels dedup to one', () => {
    // a modern-english pin maps to 'Modern' — same as the default fold-in, so
    // it is NOT 'Mixed (Modern)', just 'Modern'.
    expect(eraBadge(word({ usage: 'a', _lang: 'modern-english' }, { usage: 'b' }), label)).toBe(
      'Modern',
    );
  });

  it('lets _lang (a grid swap) win over rendered_language (an earlier render)', () => {
    expect(
      eraBadge(word({ usage: 'x', _lang: 'middle-english', rendered_language: 'old-english' }), label),
    ).toBe('Middle English');
  });

  it('resolves across multiple words (flattens morphemes_by_word)', () => {
    expect(
      eraBadge([[{ usage: 'Stan-', _lang: 'old-english' }], [{ usage: 'ton' }]], label),
    ).toBe('Mixed (Old English, Modern)');
    expect(
      eraBadge([[{ usage: 'a', _lang: 'old-english' }], [{ usage: 'b', _lang: 'old-english' }]], label),
    ).toBe('Old English');
  });
});

describe('primaryGloss', () => {
  it('takes the first sense of the first group, else the first flat meaning', () => {
    expect(primaryGloss({ meaning_groups: [['farm', 'enclosure'], ['hill']] })).toBe('farm');
    expect(primaryGloss({ meanings: ['stone'] })).toBe('stone');
    expect(primaryGloss({})).toBe('');
    expect(primaryGloss(null)).toBe('');
  });
});

describe('glossForSurface', () => {
  // a morpheme whose OE cell carries a (drifted) gloss
  const WORTH = {
    usage: '-worth',
    meanings: ['enclosure'],
    era_grid: [
      {
        family: 'english',
        stages: [
          {
            language: 'old-english',
            forms: [
              { form: 'worþ', source: 'cluster', gloss: 'enclosure' },
              { form: 'hirdels', source: 'cluster', gloss: 'hurdle' },
            ],
          },
        ],
      },
    ],
  };

  it('returns the matching cell gloss (the variant meaning)', () => {
    expect(glossForSurface(WORTH, 'worþ')).toBe('enclosure');
    expect(glossForSurface(WORTH, 'hirdels')).toBe('hurdle'); // the drifted cognate
  });
  it('returns empty string when the cell has no gloss or no cell matches', () => {
    expect(glossForSurface(STAN, 'stān')).toBe(''); // STAN cells carry no gloss
    expect(glossForSurface(WORTH, 'nowhere')).toBe('');
    expect(glossForSurface(null, 'x')).toBe('');
  });
});

describe('isGlossDrift', () => {
  it('is true only when both glosses are non-empty AND differ (fold-compared)', () => {
    expect(isGlossDrift('enclosure', 'hurdle')).toBe(true); // genuine drift
    expect(isGlossDrift('enclosure', 'Enclosure ')).toBe(false); // case/space fold = same
    expect(isGlossDrift('farm', 'farm')).toBe(false);
  });
  it('never flags drift when the base (or variant) gloss is empty', () => {
    // a glossless morpheme must NOT false-flag every variant as drifted.
    expect(isGlossDrift('', 'anything')).toBe(false);
    expect(isGlossDrift('something', '')).toBe(false);
    expect(isGlossDrift(undefined, 'x')).toBe(false);
    expect(isGlossDrift(null, null)).toBe(false);
  });
});
