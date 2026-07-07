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
import { accentFold } from '../accents.js';

export const rewindTransform = {
  kind: 'rewind',
  // wyrd-nwpa: gated behind the 'rewind' feature flag (default OFF) so prod
  // can ship with Rewind hidden while its era-rendering bugs (e.g. runic
  // 'ᚦᚩᚱᚾ' leaking out) are fixed. Staging (WYRD_FF_ALL) keeps it on.
  flag: 'rewind',
  label: 'Rewind',
  description: 'Render the name at a historical era stop (OE / ME / modern).',
  defaultParams: { era: 'oe-late' },
  paramSchema: {
    era: {
      label: 'Era',
      type: 'select',
      // `short` is the bare stage name for compact UIs (the wyrd-410t time-warp
      // bar); `label` keeps the date range for the palette select + tooltips.
      // First-class so consumers don't regex-parse the label back apart.
      options: [
        { value: 'oe-late', label: 'Old English (late, 800–1100)', short: 'Old English' },
        { value: 'me', label: 'Middle English (1100–1500)', short: 'Middle English' },
        { value: 'modern', label: 'Modern (1700–)', short: 'Modern' },
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
      // Match an existing rendering key dash-INSENSITIVELY too: the original
      // morpheme's keys may carry position dashes ("-fǣre") that the bare era
      // form ("fǣre") wouldn't otherwise match, which would split the data
      // into a duplicate slot and hide the injected respelling.
      const key =
        Object.keys(langGroup).find((k) => normKey(k) === normKey(cleanForm)) || cleanForm;
      langGroup[key] = { ...(langGroup[key] || {}), reader_pronunciation: rw.respelling };
      renderings[langField] = langGroup;
      return renderings;
    };

    // wyrd-warpsel: build the rewound morpheme AND pin the era-grid selection to
    // the newly-warped cell. The rewound morpheme keeps its ORIGINAL
    // `active_form_id` (the modern cell) via `...m`, so cellForSurface's
    // id-first lookup would keep highlighting the OLD form after a time-warp —
    // "time warp not selecting the newly chosen morpheme". Setting `_cellId`
    // (+ `_lang`) to the warped cell — exactly what a manual grid swap does
    // (swap.js) — moves the highlight to the chosen form. No pin when the warped
    // form isn't in this morpheme's era_grid (leave the id-first fallback).
    const langEq = (a, b) => (a || '').replace(/_/g, '-') === (b || '').replace(/_/g, '-');
    const warpMorpheme = (m, rw) => {
      const next = { ...m, usage: deStar(rw.form), renderings: withRespelling(m, rw) };
      if (rw.language) next._lang = rw.language;
      const target = accentFold(deStar(rw.form));
      for (const section of m.era_grid || []) {
        for (const stage of section?.stages || []) {
          if (!langEq(stage?.language, rw.language)) continue;
          for (const cell of stage?.forms || []) {
            if (cell?.id && accentFold(cell?.form) === target) {
              next._cellId = cell.id;
              return next;
            }
          }
        }
      }
      return next;
    };

    // wyrd-refl2: align each rewound form back to its INPUT morpheme by the
    // STABLE `source_index` the server echoes (the flat position of the input
    // morpheme it came from). Keeping the input morpheme (`...m`) preserves its
    // `era_grid` + `active_form_id` through the rewind, so the inspector still
    // highlights which forms are selected. The prior alignment matched
    // normKey(rw.canonical) === normKey(m.usage) — a string match that drifted
    // under accent/transform edits, dropped morphemes, and fell back to bare
    // morphemes with no id → nothing highlighted. Kept below as the fallback
    // for older server responses that don't carry `source_index`.
    const bySource = new Map();
    for (const rw of rewound) {
      if (rw && rw.form && Number.isInteger(rw.source_index)) bySource.set(rw.source_index, rw);
    }

    let morphemes_by_word;
    if (bySource.size > 0) {
      // The SPA walks state.morphemes_by_word in the same flat order the server
      // indexed (it IS the supplied-words input), so `flat` and source_index
      // stay in lockstep — including for input morphemes the rewind dropped
      // (no matching source_index → null → omitted).
      let flat = 0;
      morphemes_by_word = state.morphemes_by_word
        .map((word) =>
          word
            .map((m) => {
              const rw = bySource.get(flat++);
              return rw
                ? warpMorpheme(m, rw)
                : null;
            })
            .filter(Boolean),
        )
        .filter((word) => word.length > 0);
      if (!morphemes_by_word.length) morphemes_by_word = state.morphemes_by_word;
    } else {
      // Fallback: older server without source_index — the original
      // canonical-string two-pointer alignment.
      let ri = 0;
      const aligned = state.morphemes_by_word
        .map((word) => {
          const kept = [];
          for (const m of word) {
            const rw = rewound[ri];
            if (rw && rw.form && normKey(rw.canonical) === normKey(m.usage)) {
              ri += 1;
              kept.push(warpMorpheme(m, rw));
            }
            // else: this input morpheme had no rewound counterpart → drop it.
          }
          return kept;
        })
        .filter((word) => word.length > 0);
      if (rewound.length > 0 && ri === rewound.length) {
        morphemes_by_word = aligned;
      } else {
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
    }
    return { name: deStar(picked.result), morphemes_by_word };
  },
};
