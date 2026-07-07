// wyrd-refl2: the Rewind transform must align each rewound form back to its
// INPUT morpheme by the server's stable `source_index`, keeping the input
// morpheme's era_grid + active_form_id so the inspector still highlights which
// forms are selected — not by matching canonical↔usage strings.

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api.js', () => ({
  rewindWithMorphemes: vi.fn(),
}));

import { rewindWithMorphemes } from '../api.js';
import { rewindTransform } from './rewind.js';

// One result per era; the SPA picks components[0].era === params.era.
function envelope(morphemes, era = 'oe-late', result = 'Rewound') {
  return { results: [{ result, components: [{ era, morphemes }] }] };
}

const state = {
  name: 'Otterton',
  morphemes_by_word: [
    [
      {
        usage: 'otter-',
        active_form_id: 'modern-english:otter',
        era_grid: [{ stages: [1] }],
        renderings: {},
      },
      {
        usage: '-ton',
        active_form_id: 'modern-english:ton',
        era_grid: [{ stages: [1] }],
        renderings: {},
      },
    ],
  ],
};

describe('rewind aligns by stable source_index (wyrd-refl2)', () => {
  beforeEach(() => rewindWithMorphemes.mockReset());

  it('keeps each input morpheme id + era_grid and takes the rewound surface', async () => {
    rewindWithMorphemes.mockResolvedValue(
      envelope([
        { form: 'otor', language: 'old-english', source_index: 0, canonical: 'otter' },
        { form: 'tūn', language: 'old-english', source_index: 1, canonical: 'ton' },
      ]),
    );
    const out = await rewindTransform.apply(state, { era: 'oe-late' });
    const flat = out.morphemes_by_word.flat();
    expect(flat.map((m) => m.usage)).toEqual(['otor', 'tūn']); // rewound surfaces
    // the selection ids survive the rewind → inspector can still highlight
    expect(flat.map((m) => m.active_form_id)).toEqual([
      'modern-english:otter',
      'modern-english:ton',
    ]);
    expect(flat.every((m) => m.era_grid)).toBe(true);
  });

  it('drops an input morpheme the rewind omitted (missing source_index), keeps the rest highlightable', async () => {
    rewindWithMorphemes.mockResolvedValue(
      envelope([{ form: 'tūn', language: 'old-english', source_index: 1, canonical: 'ton' }]),
    );
    const out = await rewindTransform.apply(state, { era: 'oe-late' });
    const flat = out.morphemes_by_word.flat();
    expect(flat.map((m) => m.usage)).toEqual(['tūn']); // only source_index 1 kept
    expect(flat[0].active_form_id).toBe('modern-english:ton');
  });

  it('falls back to canonical-string alignment when the server omits source_index', async () => {
    rewindWithMorphemes.mockResolvedValue(
      envelope([
        { form: 'otor', language: 'old-english', canonical: 'otter' },
        { form: 'tūn', language: 'old-english', canonical: 'ton' },
      ]),
    );
    const out = await rewindTransform.apply(state, { era: 'oe-late' });
    const flat = out.morphemes_by_word.flat();
    expect(flat.map((m) => m.usage)).toEqual(['otor', 'tūn']);
    expect(flat.map((m) => m.active_form_id)).toEqual([
      'modern-english:otter',
      'modern-english:ton',
    ]);
  });
});

describe('time-warp pins the era-grid selection to the warped cell (wyrd-warpsel)', () => {
  beforeEach(() => rewindWithMorphemes.mockReset());

  // A morpheme with a REAL era_grid: a modern cell (as generated) + an OE cell.
  const gridState = {
    name: 'Otterton',
    morphemes_by_word: [
      [
        {
          usage: '-ton',
          active_form_id: 'modern-english:ton',
          era_grid: [
            {
              family: 'english',
              stages: [
                { language: 'modern-english', forms: [{ id: 'modern-english:ton', form: 'ton' }] },
                { language: 'old-english', forms: [{ id: 'old-english:tūn', form: 'tūn' }] },
              ],
            },
          ],
          renderings: {},
        },
      ],
    ],
  };

  it('sets _cellId + _lang to the warped cell so cellForSurface highlights the new form', async () => {
    rewindWithMorphemes.mockResolvedValue(
      envelope([{ form: 'tūn', language: 'old-english', source_index: 0, canonical: 'ton' }]),
    );
    const out = await rewindTransform.apply(gridState, { era: 'oe-late' });
    const m = out.morphemes_by_word.flat()[0];
    expect(m.usage).toBe('tūn');
    // the highlight pin now points at the OE cell, not the original modern one
    expect(m._cellId).toBe('old-english:tūn');
    expect(m._lang).toBe('old-english');
    // active_form_id is still carried (identity preserved), but _cellId wins in
    // cellForSurface's id-first precedence.
    expect(m.active_form_id).toBe('modern-english:ton');
  });

  it('leaves no _cellId pin when the warped form is absent from the era_grid', async () => {
    rewindWithMorphemes.mockResolvedValue(
      envelope([{ form: 'zzz', language: 'old-english', source_index: 0, canonical: 'ton' }]),
    );
    const out = await rewindTransform.apply(gridState, { era: 'oe-late' });
    const m = out.morphemes_by_word.flat()[0];
    expect(m._cellId).toBeUndefined();
  });
});
