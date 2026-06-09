// wyrd-thhb / wyrd-0h96: resolve a morpheme's ACTIVE cross-language
// rendering.
//
// A morpheme's surface can exist in several languages' rendering panels
// (e.g. "-by" → Old English bȳ /byː/ AND Old Norse by /biː/; "-don" → OE
// /doːn/, Old French /dun/, Celtic /dɔ̃/). The morpheme carries only the
// bare `usage` string plus `sources` (the top-ranked sibling — the etymon it
// was actually generated from) and `renderings` (slots across ALL ranked
// siblings). The OLD code iterated `renderings` in raw dict order and
// returned the first surface match, so the pronunciation guide showed an
// arbitrary language (often the wrong one).
//
// `activeRendering` resolves the ONE rendering to treat as active, in
// priority order:
//   1. an explicit user pin (`m._lang`, set by clicking an era_grid cell — see
//      MorphemeGrid.swap → pipeline.setSwap),
//   2. the canonical etymon — a language present in `m.sources`,
//   3. fallback — the first rendering match in any language.
// The InspectorColumn pronunciation fallback uses this (now a last-resort
// path behind the era_grid cell match — see lib/era.js).

const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');
const norm = (s) => stripDashes(s).toLowerCase();

/**
 * @returns {{lang: string, form: string, slot: object} | null}
 */
export function activeRendering(m) {
  if (!m) return null;
  const u = norm(m.usage);
  if (!u) return null;
  const renderings = m.renderings || {};

  const matchIn = (lang) => {
    const forms = renderings[lang];
    if (!forms) return null;
    for (const form of Object.keys(forms)) {
      const slot = forms[form] || {};
      if (u === norm(form) || u === norm(slot.original_script)) {
        return { lang, form, slot };
      }
    }
    return null;
  };

  // 1. explicit pin (user clicked a language's row)
  if (m._lang) {
    const r = matchIn(m._lang);
    if (r) return r;
  }
  // 2. canonical: a language in `sources` (the etymon this surface came
  //    from). `sources` is the single top-ranked sibling's languages
  //    (proportions.to_dict: `first.sources`) — in practice one language
  //    key, so first-key-wins is deterministic; if a future bundle ever
  //    carries cognate sources in several languages they're the SAME
  //    etymon, so any of them is an acceptable canonical pick.
  for (const lang of Object.keys(m.sources || {})) {
    const r = matchIn(lang);
    if (r) return r;
  }
  // 3. fallback: first rendering match in any language
  for (const lang of Object.keys(renderings)) {
    const r = matchIn(lang);
    if (r) return r;
  }
  return null;
}

const _hasPron = (slot) => !!(slot && (slot.ipa || slot.reader_pronunciation));

/**
 * wyrd-cxwl: the pronunciation slot to show for a morpheme in the guide.
 * Prefers the active rendering (the matched form), but when the GENERATED
 * surface has no matching form with pronunciation — common for surfaces whose
 * spelling diverges from the etymon form that carries the IPA ("ay" ← Old
 * Norse "ey", "moy" ← Celtic "magh") — falls back to the canonical etymon's
 * pronunciation (first sources language, then any rendering language) so the
 * guide shows the closest available sound instead of a blank '·'. NOTE: the
 * fallback slot may belong to a different FORM of the etymon than the surface
 * (e.g. a nominative vs genitive variant) — an accepted approximation, since
 * the goal is "closest available sound", not an exact per-surface transcription.
 */
export function pronunciationFor(m) {
  if (!m) return null;
  const active = activeRendering(m);
  if (_hasPron(active?.slot)) return active.slot;
  const renderings = m.renderings || {};
  const langs = [...Object.keys(m.sources || {}), ...Object.keys(renderings)];
  for (const lang of langs) {
    const forms = renderings[lang];
    if (!forms) continue;
    for (const form of Object.keys(forms)) {
      if (_hasPron(forms[form])) return forms[form];
    }
  }
  return active?.slot || null;
}
