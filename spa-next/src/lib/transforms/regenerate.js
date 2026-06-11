// wyrd-y0lx: Regenerate-morpheme transform. Direct-manipulation like Swap —
// the user clicks the ⟳ button on a morpheme card in Inspect & Transform,
// which adds (or re-rolls in place, one step per morpheme) a step targeting
// that slot. apply() calls the kenning-regenerate-morpheme endpoint, which
// re-runs the vector path's gate → score → sample for JUST that slot in the
// context of all the others (same hard gates as the roll; cohesion over the
// other slots' tags; in-use morphemes excluded so a re-roll can never create
// a "Hill Hill"), then splices the replacement morpheme into the state.
//
// Splice convention follows Swap (wyrd-qc0g): post-transform, `usage` IS the
// live surface — the replacement's native rendered form grafted onto the
// slot's placement dashes — and the auto-era-render fields are dropped so
// the grid highlight + breakdown rows can't desync from the headline. The
// server's `active_form_id` (the rendered reflex's grid-cell id) is kept so
// the new morpheme's era grid highlights the live cell; `_lang` pins the
// rendered language for pronunciation resolution (the wyrd-thhb mechanism).
//
// `params.seed` is baked at click time (pipeline.setRegenerate) so pipeline
// re-runs and restored workspaces replay the same pick deterministically;
// `params.context` is the generation-params snapshot of the roll (culture /
// tags / mood / era / …) so the replacement honors the original request.

import { regenerateMorpheme } from '../api.js';
import { graftPosition } from '../accents.js';
import { renderName } from './swap.js';

export const regenerateTransform = {
  kind: 'regenerate-morpheme',
  // wyrd-y0lx: gated behind the 'regenerate-morpheme' feature flag (default
  // OFF) per the wyrd-nwpa convention — prod hides the button + skips the
  // step until WYRD_FF_REGENERATE_MORPHEME (or FF_ALL, staging) is set.
  flag: 'regenerate-morpheme',
  label: 'Regenerate morpheme',
  description: 'Re-roll one morpheme in the context of the others.',
  defaultParams: { wordIndex: 0, morphemeIndex: 0, seed: 0, context: {} },
  // Params are baked by the ⟳ button (click-time), not operator-editable
  // inline on the step card — same shape as Swap.
  paramSchema: {},
  summary({ wordIndex, morphemeIndex, from }) {
    const src = from ? `${from} ` : '';
    return `${src}morph[${wordIndex},${morphemeIndex}] → re-roll`;
  },
  async apply(state, params) {
    const { wordIndex, morphemeIndex, seed, context } = params;
    // Defensive bounds check (mirrors Swap): pipeline state can shift under
    // the step when an earlier step changes the morpheme structure.
    const word = state.morphemes_by_word?.[wordIndex];
    if (!word) {
      throw new Error(`word ${wordIndex} not in current state`);
    }
    const morph = word[morphemeIndex];
    if (!morph) {
      throw new Error(`morpheme ${morphemeIndex} not in word ${wordIndex}`);
    }
    const envelope = await regenerateMorpheme({
      ...(context || {}),
      name: state.name,
      words: state.morphemes_by_word,
      word_index: wordIndex,
      morpheme_index: morphemeIndex,
      seed,
    });
    const fresh =
      envelope.results?.[0]?.morphemes_by_word?.[wordIndex]?.[morphemeIndex];
    if (!fresh?.usage) {
      throw new Error('regenerate returned no replacement morpheme');
    }
    // Live surface: the replacement's native form (server `rendered`), or its
    // modern usage when no distinct native render exists — grafted onto the
    // slot's placement dashes + case rule (graftPosition).
    const surface = graftPosition(morph.usage, fresh.rendered || fresh.usage);
    const next = { ...fresh, usage: surface };
    if (fresh.rendered_language) next._lang = fresh.rendered_language;
    delete next.rendered;
    delete next.rendered_language;
    delete next.rendered_pron;
    const nextWords = state.morphemes_by_word.map((w, wi) =>
      w.map((m, mi) => (wi === wordIndex && mi === morphemeIndex ? next : m)),
    );
    return { name: renderName(nextWords), morphemes_by_word: nextWords };
  },
};
