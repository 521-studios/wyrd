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
    //
    // wyrd-hpjg round 4 (Gemini MED): title-case non-particle
    // tokens DURING the loop so we don't have to redo the
    // particle-vs-pending classification at the end. Pre-fix, the
    // final .map checked tokens against FREE_PARTICLES — a pending-
    // joined token that happened to spell a particle (e.g. a dashed
    // '-on-' infix whose stripped form is 'on') would be wrongly
    // lowercased.
    const tokens = [];
    let pending = [];
    const flushPending = () => {
      if (pending.length === 0) return;
      const joined = pending.join('');
      // wyrd-hpjg round 5 (Gemini MED): title-case lowers the tail
      // too so case-inconsistent inputs (a user-supplied JSON, a
      // future cluster-sibling endpoint) normalize. Bundle data is
      // already correctly-cased so this is defensive against
      // fabricated inputs; macron letters (ē, ā) are unaffected
      // since toLowerCase is identity on them.
      if (joined) {
        tokens.push(joined[0].toUpperCase() + joined.slice(1).toLowerCase());
      }
      pending = [];
    };
    for (const m of w) {
      const form = (m.usage || '').replace(/^-+|-+$/g, '');
      if (isFreeParticle(m.usage)) {
        flushPending();
        tokens.push(form.toLowerCase());
      } else {
        pending.push(form);
      }
    }
    flushPending();
    return tokens.filter(Boolean).join(' ');
  });
  const name = wordRenders.filter((s) => s.trim()).join(' ');
  // wyrd-hpjg round 3 (Gemini MED): always capitalize the very
  // first letter — a name that happens to lead with a particle
  // (rare in real corpus, but possible after a Swap) reads as
  // 'Of Avon' not 'of Avon'. Diverges slightly from the Python
  // render_form_particle_pairs which leaves leading particles
  // lowercase; defended against the synthetic-input edge case.
  return name ? name[0].toUpperCase() + name.slice(1) : '';
}
