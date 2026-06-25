// wyrd-myof: unit tests for the pure gloss helpers extracted from
// OutputColumn.svelte (glossFor + the per-roll glossesByResults memo payload).
// svelte components have no harness here, so pinning the gloss logic as pure
// functions is how the Output column's gloss behavior stays tested.
import { describe, it, expect } from 'vitest';
import { glossFor, glossesByResults } from './morphemeGloss.js';

describe('glossFor', () => {
  it('joins both senses of a homograph (distinct repeated stem) with a middot', () => {
    // representativeMeanings only surfaces a 2nd sense when a DISTINCT stem
    // REPEATS (count >= 2) — the fingerprint of two etymons in one family.
    expect(glossFor({ usage: 'Gil', meanings: ['gold', 'gold', 'ravine', 'ravine'] })).toBe(
      'gold · ravine',
    );
  });

  it('shows one gloss when senses each appear once (no repeated 2nd sense)', () => {
    expect(glossFor({ usage: 'ton', meanings: ['farmstead', 'enclosure'] })).toBe('farmstead');
  });

  it('returns a single gloss when only one distinct sense', () => {
    expect(glossFor({ usage: 'dun', meanings: ['hill'] })).toBe('hill');
  });

  it('falls back to "name" for a capitalized proper-name morpheme with no gloss', () => {
    expect(glossFor({ usage: 'Giles', meanings: ['a personal name'] })).toBe('name');
  });

  it('stays blank for a lowercase connective even if a name-sense is pooled', () => {
    // looksProper gate: lowercase surface never shows the "name" marker.
    expect(glossFor({ usage: 'et', meanings: ['a personal name'] })).toBe('');
  });

  it('returns "" when meanings are all junk/empty', () => {
    expect(glossFor({ usage: 'x', meanings: [] })).toBe('');
  });

  it('returns "" for a morph with no meanings key (live-row safety)', () => {
    // The live-row branch calls glossFor on raw pipeline morphemes that may
    // omit `meanings`; representativeMeanings guards non-arrays, so this is "".
    expect(glossFor({ usage: 'x' })).toBe('');
  });
});

describe('glossesByResults', () => {
  it('mirrors morphemes_by_word shape: [result][word][morph]', () => {
    const results = [
      {
        morphemes_by_word: [
          [{ usage: 'ton', meanings: ['farmstead'] }],
          [{ usage: 'dun', meanings: ['hill'] }],
        ],
      },
    ];
    expect(glossesByResults(results)).toEqual([[['farmstead'], ['hill']]]);
  });

  it('memoizes across multiple results and multi-morpheme words ([i] and [mi] axes)', () => {
    // The template indexes glossesByResult[i][wi][mi], so exercise a second
    // result (the per-roll memo axis) and a word with two morphemes.
    const results = [
      { morphemes_by_word: [[{ usage: 'dun', meanings: ['hill'] }]] },
      {
        morphemes_by_word: [
          [
            { usage: 'Stoke', meanings: ['place'] },
            { usage: 'ton', meanings: ['farmstead'] },
          ],
        ],
      },
    ];
    expect(glossesByResults(results)).toEqual([[['hill']], [['place', 'farmstead']]]);
  });

  it('tolerates a result with no morphemes_by_word (explanation-only rolls)', () => {
    expect(glossesByResults([{ explanation: 'opaque name' }])).toEqual([[]]);
  });

  it('returns [] for null/empty input', () => {
    expect(glossesByResults(null)).toEqual([]);
    expect(glossesByResults([])).toEqual([]);
  });

  it('contains a malformed morpheme to its own slot, not the whole column', () => {
    // A single null/non-object morpheme (a corrupt or stale restored/shared
    // workspace) must degrade to '' for THAT slot — not throw out of the nested
    // map and blank every gloss. Pre-fix this raised TypeError on `morph.meanings`
    // and lost the valid 'town' gloss beside it.
    const results = [{ morphemes_by_word: [[{ usage: 'ton', meanings: ['town'] }, null]] }];
    expect(glossesByResults(results)).toEqual([[['town', '']]]);
  });

  it('tolerates a null word (whole sub-array) without dropping sibling words', () => {
    const results = [{ morphemes_by_word: [null, [{ usage: 'dun', meanings: ['hill'] }]] }];
    expect(glossesByResults(results)).toEqual([[[], ['hill']]]);
  });
});

describe('glossFor defensive input', () => {
  it('returns "" for a null / undefined / non-object morpheme instead of throwing', () => {
    expect(glossFor(null)).toBe('');
    expect(glossFor(undefined)).toBe('');
    expect(glossFor('a personal name')).toBe('');
    expect(glossFor(42)).toBe('');
  });
});
