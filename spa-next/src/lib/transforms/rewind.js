// wyrd-kppy: Rewind transform. First (and so far only) entry in the
// pipeline's transform catalog.
//
// A "transform" is the unit the pipeline composes:
//   {
//     kind:        unique-string id for this transform
//     label:       UI display name for the palette + step header
//     description: one-line UX hint shown in the palette
//     defaultParams: object — shape of the editable params + their initial values
//     paramSchema: per-key UI metadata for rendering edit controls inline
//                  on a step card. {label, type, options?}.
//     apply(state, params): async fn — takes the previous step's
//                  state ({name, morphemes_by_word}) + this step's
//                  params, returns the new state. Throws on failure.
//   }
//
// Rewind sends state.morphemes_by_word to /api/kenning-rewind (which
// accepts the pre-picked morphemes per wyrd-y9aa, skipping trie
// re-decomposition); the response carries one GenerationResult per
// era stop; we pick the one matching params.era. The new state's
// `name` is the era-rendered string; `morphemes_by_word` is
// preserved unchanged (rewind doesn't swap morpheme picks, it just
// renders them at a different period).

import { rewindWithMorphemes } from '../api.js';

export const rewindTransform = {
  kind: 'rewind',
  label: 'Rewind',
  description: 'Render the name at a historical era stop (OE / ME / modern).',
  defaultParams: { era: 'oe-late' },
  paramSchema: {
    era: {
      label: 'Era',
      type: 'select',
      options: [
        { value: 'oe-late', label: 'Old English (late, 800–1100)' },
        { value: 'me', label: 'Middle English (1100–1500)' },
        { value: 'modern', label: 'Modern (1700–)' },
      ],
    },
  },
  async apply(state, params) {
    const envelope = await rewindWithMorphemes(
      state.name,
      state.morphemes_by_word,
    );
    const picked = envelope.results.find(
      (r) => r.components?.[0]?.era === params.era,
    );
    if (!picked) {
      throw new Error(
        `era '${params.era}' missing from rewind output (got: ` +
          envelope.results.map((r) => r.components?.[0]?.era).join(', ') +
          ')',
      );
    }
    // wyrd-7cvv: rebuild morphemes_by_word from the REWOUND morphemes so the
    // inspector (cards + breakdown + pronunciation guide) matches the rewound
    // NAME — not the original morphemes.
    //
    // The rewind OMITS any input morpheme whose usage is no longer resolvable
    // in the bundle (a meaning_db lookup miss — era-less morphemes still
    // appear, via their canonical form). So its `morphemes` list is a
    // subsequence (in order) of the input, each tagged with `canonical`
    // = the original modern usage. Align by canonical with a two-pointer
    // walk: a matched input morpheme keeps its meanings/sources/tags but
    // takes the rewound surface + respelling; an input morpheme the rewind
    // dropped is OMITTED (so the cards can't disagree with the name — the
    // earlier "Hyrst Enlihtan" head over "hōl- -hurst low" cards bug).
    const rewound = picked.components?.[0]?.morphemes || [];
    const normKey = (s) => (s || '').replace(/^-+|-+$/g, '').toLowerCase();
    // wyrd-q1np: strip the '*' reconstructed-form marker from rewound surfaces
    // and the rendered name. The asterisk is a scholarly "unattested form"
    // convention — fine in an etymon list, but it reads as a glitch in a
    // generated place-name ("Sūþ *fǣre", two '*' in one word). Surfaces are
    // cleaned everywhere the rewind feeds the inspector (name + breakdown +
    // cards + pronunciation-key) so they stay consistent.
    const deStar = (s) => (s || '').replace(/\*/g, '');

    // Inject the rewound respelling so the pronunciation guide (which matches
    // on usage) surfaces it; merge into any existing entry case-insensitively
    // so we don't split data across a casing-variant duplicate.
    const withRespelling = (m, rw) => {
      const langField = (rw.language || '').replace(/-/g, '_');
      if (!langField || !rw.respelling) return m.renderings;
      const renderings = { ...(m.renderings || {}) };
      const langGroup = { ...(renderings[langField] || {}) };
      const cleanForm = deStar(rw.form);
      const key =
        Object.keys(langGroup).find((k) => k.toLowerCase() === cleanForm.toLowerCase()) ||
        cleanForm;
      langGroup[key] = { ...(langGroup[key] || {}), reader_pronunciation: rw.respelling };
      renderings[langField] = langGroup;
      return renderings;
    };

    let ri = 0;
    const aligned = state.morphemes_by_word
      .map((word) => {
        const kept = [];
        for (const m of word) {
          const rw = rewound[ri];
          if (rw && rw.form && normKey(rw.canonical) === normKey(m.usage)) {
            ri += 1;
            kept.push({ ...m, usage: deStar(rw.form), renderings: withRespelling(m, rw) });
          }
          // else: this input morpheme had no rewound counterpart → drop it.
        }
        return kept;
      })
      .filter((word) => word.length > 0);

    let morphemes_by_word;
    if (rewound.length > 0 && ri === rewound.length) {
      // Clean alignment — every rewound form mapped back to an input morpheme
      // (so `aligned` has at least one kept morpheme).
      morphemes_by_word = aligned;
    } else {
      // Alignment didn't fully consume the rewound list (canonical didn't
      // line up — unexpected). Fall back to the rewound morphemes as a single
      // word so the cards still AGREE with the rewound name (degraded: no
      // meanings/sources, but never disjoint). If even that's empty, keep the
      // original morphemes.
      const flat = rewound
        .filter((rw) => rw.form)
        .map((rw) => {
          const surface = deStar(rw.form);
          const m = { usage: surface };
          const langField = (rw.language || '').replace(/-/g, '_');
          if (langField && rw.respelling) {
            m.renderings = { [langField]: { [surface]: { reader_pronunciation: rw.respelling } } };
          }
          return m;
        });
      morphemes_by_word = flat.length ? [flat] : state.morphemes_by_word;
    }
    return { name: deStar(picked.result), morphemes_by_word };
  },
};
