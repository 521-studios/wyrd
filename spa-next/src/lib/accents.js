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

// True for a category-Mn combining mark — matching the Python folds'
// `unicodedata.category(c) == 'Mn'` (bundle/_subject._surface_fold +
// runtime/proportions._grid_match_key), wyrd-nndd. Was a U+0300–U+036F codepoint
// range, which dropped ONLY the main Combining Diacritical Marks block and left
// Mn marks in other blocks (Hebrew niqqud, Arabic harakat, …) that the Python
// side dropped — a documented-parity divergence. `\p{Mn}/u` is a Unicode
// property escape (ES2018+): it covers Mn in every block AND keeps a raw
// combining char out of source (the same safety the codepoint check gave). The
// `g` flag lets accentFold strip them in one native `.replace` pass.
const COMBINING_MARKS_RE = /\p{Mn}/gu;

/**
 * Dedup key that folds BOTH case and diacritics: "bȳ", "by", and "By" all
 * map to "by". Used to collapse case/accent variants of the same form into a
 * single row (keeping the richest), while genuinely-distinct inflections
 * ("byht", "byhtas") stay separate. NFD-decompose, drop category-Mn combining
 * marks, strip dashes, lowercase. Marks are matched by the ``\p{Mn}/u`` property
 * escape (not a raw combining char) so editors/git can't normalize it away.
 *
 * Also drops the scholarly '*' reconstructed/unattested-form marker so a
 * stripped surface ("ur") still matches its starred cell form ("*ur").
 */
export function accentFold(s) {
  // strip the reconstructed '*' marker upfront (consistent with graftPosition),
  // NFD-decompose, drop category-Mn marks in one pass, lowercase.
  return stripDashes((s || '').replace(/\*/g, ''))
    .normalize('NFD')
    .replace(COMBINING_MARKS_RE, '')
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
 *
 * wyrd-refl: the upgrade is scoped to the morpheme's ACTIVE render language
 * (`active_form_id` = "lang:surface"). `original_script` is the NATIVE
 * (historical) spelling of a form; scanning EVERY rendering language grafted
 * that native accent onto a form the generator rendered in a DIFFERENT, later
 * era — e.g. a modern-english reflex "lead" picking up old-english "lēad" — so
 * the modern-era Output column showed a macron the de-accented Inspect era-grid
 * didn't. Restricting to the active language means a modern reflex only upgrades
 * if its OWN (modern) rendering carries an accent, which it doesn't. When
 * `active_form_id` is absent (older payloads / unit fixtures) we fall back to
 * scanning all languages, preserving the prior behavior.
 */
export function accentedUsage(morph) {
  if (!morph) return null;
  const u = norm(morph.usage);
  if (!u) return null;
  const R = morph.renderings || {};
  const activeLang = (morph.active_form_id || '').split(':')[0];
  // Language keys carry hyphens in active_form_id ("modern-english") but
  // underscores in the renderings map ("modern_english") — try both spellings.
  const langKeys = activeLang
    ? [...new Set([activeLang, activeLang.replace(/-/g, '_'), activeLang.replace(/_/g, '-')])]
    : Object.keys(R);
  for (const lang of langKeys) {
    const langForms = R[lang];
    if (!langForms) continue; // a null/absent lang bucket → nothing to upgrade
    for (const form of Object.keys(langForms)) {
      const os = langForms[form]?.original_script;
      if (os && norm(form) === u && hasAccent(os)) {
        const lead = morph.usage.match(/^-+/)?.[0] || '';
        const trail = morph.usage.match(/-+$/)?.[0] || '';
        // Apply the slot's position case rule, mirroring graftPosition: a
        // leading dash marks an inner/post slot, which renders lowercase, while
        // pre/bare keep the original_script's case. Without this a capitalized
        // accented original_script ("Bȳ", "Rōm") in a dashed slot would leak its
        // capital into col 2 ("-Bȳ") where the position rule — and col 3, which
        // routes through graftPosition — both say "-bȳ". Diacritics survive
        // toLowerCase (ā/ȳ/ō unaffected).
        let core = stripDashes(os);
        if (lead) core = core.toLowerCase();
        return lead + core + trail;
      }
    }
  }
  return null;
}

/**
 * wyrd-rogd.17: accent-upgrade an ARBITRARY (bare) form against a morpheme's
 * renderings — the per-cell counterpart of `accentedUsage`. Returns the
 * accented `original_script` when a rendering for the same accent+case-folded
 * form supplies one with a diacritic; otherwise the form unchanged.
 *
 * The Output column (col 2) shows `accentedUsage` ("Tongbȳ", "Tretōn"), but the
 * Inspect era-grid (col 3) historically displayed the plain `cell.form`
 * ("Tongby", "Treton") — so the two surfaces disagreed on macrons whenever a
 * rendering carried an accented original_script the stored reflex form didn't.
 * (#594 fixed the by-ID highlight; this fixes the displayed SURFACE.) Folding
 * by `accentFold` — not a raw equality — is what lets the ASCII cell form find
 * its accented rendering.
 *
 * NOTE: this intentionally folds via `accentFold` (strips diacritics) on BOTH
 * sides, where `accentedUsage` matches via `norm` (case+dash only). The grid
 * keys on arbitrary `cell.form`s (and rendering keys may themselves carry an
 * accent), so the diacritic-fold is the robust match here — don't "unify" the
 * two to the same fold without re-checking both call sites.
 */
export function accentForm(form, morph) {
  const bare = stripDashes(form || '');
  if (!bare || !morph) return form;
  const target = accentFold(bare);
  if (!target) return form;
  const R = morph.renderings || {};
  for (const lang of Object.keys(R)) {
    const langForms = R[lang];
    if (!langForms) continue; // a null lang bucket would crash Object.keys
    for (const f of Object.keys(langForms)) {
      const os = langForms[f]?.original_script;
      if (os && accentFold(f) === target && hasAccent(os)) {
        return stripDashes(os);
      }
    }
  }
  return form;
}

/**
 * wyrd-at53: graft a new surface into a morpheme slot's POSITION, taking the
 * leading/trailing dash markers from `positionFrom` (the slot's generated
 * usage, e.g. "-hām", "-ing-", "Cornel-") and applying the position's case
 * rule: inner ("-X-") and post ("-X") positions render lowercase; pre ("X-")
 * and bare ("X") keep their capitalization. So clicking etymon form "Hamm"
 * for a "-hām" slot yields "-hamm" (dash kept, lowercased), not "Hamm".
 * Diacritics survive toLowerCase (ā/ȳ unaffected).
 *
 * The scholarly '*' reconstructed/unattested-form marker is stripped — it's a
 * citation convention, not part of the surface, and reads as a glitch in a
 * generated name ("Tray*bearwe"). So a starred cell form ("*bearwe") grafts in
 * as a clean "bearwe" both in the grid display and when swapped into the name.
 */
export function graftPosition(positionFrom, surface) {
  const lead = (positionFrom || '').match(/^-+/)?.[0] || '';
  const trail = (positionFrom || '').match(/-+$/)?.[0] || '';
  let s = stripDashes((surface || '').replace(/\*/g, ''));
  if (!s) return positionFrom || ''; // nothing to graft → keep the original
  if (lead) s = s.toLowerCase(); // leading dash ⇒ inner or post ⇒ lowercase
  return lead + s + trail;
}

/**
 * Accent-upgrade every morpheme in a per-word structure. Returns
 * `{ words, changed }` — `words` with each morpheme's usage swapped to its
 * accented form (when one exists), and `changed` true if anything moved.
 */
export function upgradeAccents(mbw) {
  let changed = false;
  const words = (mbw || []).map((word) =>
    (word || []).map((m) => {
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
  const fallback = result.result || '';
  const plain = result.morphemes_by_word || [];
  if (!plain.length) return fallback;
  const up = upgradeAccents(plain);
  if (!up.changed) return fallback;
  return renderName(plain) === fallback ? renderName(up.words) : fallback;
}
