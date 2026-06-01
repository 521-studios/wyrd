// wyrd-de5t: shared morpheme accent-upgrade helper.
//
// The bundle's generated `usage` is often the lossy ASCII surface ("by",
// "hy"), while the etymon's renderings carry the accented original_script
// ("bȳ", "hȳ"). `accentedUsage` returns the accented surface — grafted onto
// the usage's dash markers — when a rendering for this morpheme's OWN
// surface supplies one; null otherwise.
//
// Extracted from InspectorColumn (col 3, which already showed accents) so
// the Output column (col 2) can render the same accented surfaces — and the
// same accented NAME — instead of the raw ASCII.

import { renderName } from './transforms/swap.js';

const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');
const norm = (s) => stripDashes(s).toLowerCase();

/**
 * Dedup key that folds BOTH case and diacritics: "bȳ", "by", and "By" all
 * map to "by". Used to collapse case/accent variants of the same form into a
 * single row (keeping the richest), while genuinely-distinct inflections
 * ("byht", "byhtas") stay separate. NFD-decompose, drop combining marks,
 * strip dashes, lowercase.
 */
export function accentFold(s) {
  return stripDashes(s || '')
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase();
}
// A string carries a diacritic if NFD-decomposing it yields combining
// marks (i.e. it changes under decomposition).
const hasAccent = (s) => !!s && s.normalize('NFD') !== s;

/**
 * Return this morpheme's usage upgraded to its accented original_script
 * surface when a rendering for the same (dash-stripped, lowercased) form
 * supplies one with a diacritic; otherwise null. Dash markers on the
 * original usage ("-by") are preserved on the result ("-bȳ").
 */
export function accentedUsage(morph) {
  if (!morph) return null;
  const u = norm(morph.usage);
  if (!u) return null;
  const R = morph.renderings || {};
  for (const lang of Object.keys(R)) {
    const langForms = R[lang];
    if (!langForms) continue; // a null lang bucket would crash Object.keys
    for (const form of Object.keys(langForms)) {
      const os = langForms[form]?.original_script;
      if (os && norm(form) === u && hasAccent(os)) {
        const lead = morph.usage.match(/^-+/)?.[0] || '';
        const trail = morph.usage.match(/-+$/)?.[0] || '';
        return lead + stripDashes(os) + trail;
      }
    }
  }
  return null;
}

/**
 * Accent-upgrade every morpheme in a per-word structure. Returns
 * `{ words, changed }` — `words` with each morpheme's usage swapped to its
 * accented form (when one exists), and `changed` true if anything moved.
 */
export function upgradeAccents(mbw) {
  let changed = false;
  const words = (mbw || []).map((word) =>
    word.map((m) => {
      const acc = accentedUsage(m);
      if (acc && acc !== m.usage) {
        changed = true;
        return { ...m, usage: acc };
      }
      return m;
    }),
  );
  return { words, changed };
}

/**
 * The accent-upgraded display name for a generated result. Re-renders the
 * name from the accent-upgraded morphemes ONLY when renderName is proven
 * lossless for this result (it reproduces the plain bundle name) — otherwise
 * returns the bundle name unchanged, so an imperfect re-render can't corrupt
 * casing/joining (e.g. "HamySide" → "Hamyside"). Mirrors the Inspect
 * column's head-name logic so col 2 and col 3 agree.
 */
export function accentedName(result) {
  if (!result) return '';
  const plain = result.morphemes_by_word || [];
  if (!plain.length) return result.result;
  const up = upgradeAccents(plain);
  if (!up.changed) return result.result;
  return renderName(plain) === result.result ? renderName(up.words) : result.result;
}
