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
    // wyrd-7cvv: re-project the rewound era-forms back onto the per-word
    // structure so the inspector (morpheme cards + the pronunciation guide)
    // reflects the REWOUND word, not the original. Pre-fix this returned
    // state.morphemes_by_word unchanged, so the cards/guide kept showing the
    // original forms + pronunciations even though the rendered name changed.
    //
    // The rewind endpoint returns a FLAT `morphemes` list ({form, language,
    // respelling}) in morpheme order. Re-nest it onto the original word
    // grouping positionally. Etymology (meanings/tags/sources) is preserved
    // — it's the same morpheme at a later era — but `usage` becomes the
    // rewound form and the rewound `respelling` is injected as a rendering
    // keyed by that form so renderingForUsage() surfaces it in the guide.
    // Count mismatch → keep the original morphemes (the rendered name is
    // still the rewound one); never worse than the pre-fix behavior.
    const rewound = picked.components?.[0]?.morphemes || [];
    const total = state.morphemes_by_word.reduce((n, w) => n + w.length, 0);
    let morphemes_by_word = state.morphemes_by_word;
    if (rewound.length === total) {
      let i = 0;
      morphemes_by_word = state.morphemes_by_word.map((word) =>
        word.map((m) => {
          const rw = rewound[i];
          i += 1;
          // Defensive: a malformed entry (missing form) → keep the original
          // morpheme rather than blanking its usage.
          if (!rw || !rw.form) return m;
          const langField = (rw.language || '').replace(/-/g, '_');
          // Only inject a rendering when we have BOTH a language bucket and a
          // respelling — otherwise we'd add a spurious empty language panel
          // to the morpheme card (orderedSourceEntries unions renderings
          // keys). Keep any pre-existing renderings.
          let renderings = m.renderings;
          if (langField && rw.respelling) {
            renderings = { ...(m.renderings || {}) };
            renderings[langField] = {
              ...(renderings[langField] || {}),
              [rw.form]: {
                ...(renderings[langField]?.[rw.form] || {}),
                reader_pronunciation: rw.respelling,
              },
            };
          }
          return { ...m, usage: rw.form, renderings };
        }),
      );
    }
    return { name: picked.result, morphemes_by_word };
  },
};
