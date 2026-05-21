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
    return {
      name: picked.result,
      // wyrd-kppy: morpheme picks are preserved across rewind —
      // rewind doesn't change which morphemes a name is composed
      // of, it changes which era's reflexes they're rendered as.
      // Downstream transforms (Swap in PR #5) operate on the same
      // morpheme stack the user originally generated.
      morphemes_by_word: state.morphemes_by_word,
    };
  },
};
