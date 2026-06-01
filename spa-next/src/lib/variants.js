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
//   1. an explicit user pin (`m._lang`, set by clicking a language's row),
//   2. the canonical etymon — a language present in `m.sources`,
//   3. fallback — the first rendering match in any language.
// Both the pronunciation guide (InspectorColumn) and the card highlight
// (MorphemeCard) use this so they always agree.

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
  // 2. canonical: a language in `sources` (the etymon this surface came from)
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
