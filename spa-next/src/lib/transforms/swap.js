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
    // Shallow-clone the outer + inner arrays so the prior state's
    // morphemes_by_word reference isn't aliased; the targeted
    // morpheme is REPLACED with a new object (rest are reused).
    // Nothing downstream mutates morphemes_by_word so the shared
    // references are safe.
    const nextWords = state.morphemes_by_word.map((w, wi) =>
      w.map((m, mi) =>
        wi === wordIndex && mi === morphemeIndex
          ? { ...m, usage: to }
          : m,
      ),
    );
    const name = renderName(nextWords);
    return { name, morphemes_by_word: nextWords };
  },
};

// wyrd-hpjg round 2 (frontend MED): port of the Python rewinder's
// render_form_particle_pairs particle handling. The pre-fix renderer
// concat'd everything in a word — a morpheme whose usage was 'on'
// produced 'Whitonfoo' instead of 'Whit on Foo'. Free particles
// (on / upon / under / of) render as their own lowercase tokens
// surrounded by spaces, matching scribal place-name convention
// (Stratford on Avon, Henley upon Thames).
const FREE_PARTICLES = new Set(['on', 'upon', 'under', 'of']);

function isFreeParticle(usage) {
  if (!usage) return false;
  // Mirrors wyrd-085k _is_free_particle: lowercase membership AND
  // no hyphen markers (a dashed morpheme like '-by' is a settlement
  // suffix even if its bare form would match the particle set).
  if (usage.includes('-')) return false;
  return FREE_PARTICLES.has(usage.toLowerCase());
}

function renderName(wordsList) {
  const wordRenders = wordsList.map((w) => {
    // Per-word: smart-join particles as separate tokens, concat the
    // rest, title-case the non-particle tokens. Mirrors
    // render_form_particle_pairs with smart_join=True.
    const tokens = [];
    let pending = [];
    for (const m of w) {
      const form = (m.usage || '').replace(/^-+|-+$/g, '');
      if (isFreeParticle(m.usage)) {
        if (pending.length > 0) {
          tokens.push(pending.join(''));
          pending = [];
        }
        tokens.push(form.toLowerCase());
      } else {
        pending.push(form);
      }
    }
    if (pending.length > 0) tokens.push(pending.join(''));
    return tokens
      .map((t) => (FREE_PARTICLES.has(t) ? t : t ? t[0].toUpperCase() + t.slice(1) : ''))
      .filter(Boolean)
      .join(' ');
  });
  return wordRenders.filter((s) => s.trim()).join(' ');
}
