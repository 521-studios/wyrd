// wyrd-hpjg: Swap-morpheme transform. Pure client-side (no API call):
// the user clicks an alternate form on a morpheme card, which adds a
// Swap step targeting that morpheme + that form. apply() rewrites the
// targeted morpheme's `usage` and re-renders the state's `name` by
// joining the (now-mutated) usages per-word + space-joining words.
//
// "Siblings" v1 scope: the alternate inflected/dialectal forms that
// the morpheme's own `sources` + `renderings` already carry (e.g.,
// for -ham: ham / hamm / hamme / hammum). True cluster-siblings
// (different morphemes sharing an etymon) require a backend
// endpoint and land in a follow-up PR.
//
// Pipeline composition with Rewind: when the user has Rewind→OE
// before Swap, the rewind output's morphemes_by_word is preserved
// (rewind doesn't mutate it — only the rendered name); Swap then
// operates on the post-rewind state. If Swap is BEFORE Rewind, the
// downstream Rewind sees the swapped morphemes via the supplied-
// morphemes path (wyrd-y9aa).

export const swapTransform = {
  kind: 'swap',
  label: 'Swap morpheme',
  description: 'Replace a morpheme with an alternate form from its sources.',
  defaultParams: {
    wordIndex: 0,
    morphemeIndex: 0,
    to: '',
  },
  // No paramSchema entries — the params are picked by the click-to-
  // swap UX (MorphemeCard's form rows) and aren't operator-editable
  // inline on the step card.
  paramSchema: {},
  // summary() lets a no-param-schema transform render a meaningful
  // descriptor on the step card. Generic step UI fallback for
  // transforms with paramSchema is the bind:value controls; for
  // swap we just describe the targeted cell + the new form.
  summary({ wordIndex, morphemeIndex, to }) {
    return `morph[${wordIndex},${morphemeIndex}] → ${to}`;
  },
  async apply(state, params) {
    const { wordIndex, morphemeIndex, to } = params;
    if (!to) {
      throw new Error('swap target form is empty');
    }
    // Defensive bounds check — pipeline state can legitimately shift
    // under the step (a previous step changed the morpheme structure).
    // Better to fail loudly than silently swap a different morpheme.
    const word = state.morphemes_by_word?.[wordIndex];
    if (!word) {
      throw new Error(`word ${wordIndex} not in current state`);
    }
    const morph = word[morphemeIndex];
    if (!morph) {
      throw new Error(`morpheme ${morphemeIndex} not in word ${wordIndex}`);
    }
    // Deep-clone morphemes_by_word so the prior state isn't mutated
    // (pipeline.states keeps each step's output separately).
    const nextWords = state.morphemes_by_word.map((w, wi) =>
      w.map((m, mi) =>
        wi === wordIndex && mi === morphemeIndex
          ? { ...m, usage: to }
          : m,
      ),
    );
    // Re-render the name by concat-within-word + space-between-words.
    // Drop dash markers from the joined output (wyrd-2pio behavior
    // matches: '-ham' as a positional suffix shouldn't bleed dashes
    // into the rendered name). Title-case the first letter of each
    // word so 'beggarhamm bridge' renders as 'Beggarhamm Bridge'
    // (mirrors the legacy render_form_particle_pairs title-casing).
    const name = nextWords
      .map((w) => {
        const joined = w
          .map((m) => (m.usage || '').replace(/^-+|-+$/g, ''))
          .join('');
        return joined ? joined[0].toUpperCase() + joined.slice(1) : '';
      })
      .filter((s) => s.trim())
      .join(' ');
    return { name, morphemes_by_word: nextWords };
  },
};
