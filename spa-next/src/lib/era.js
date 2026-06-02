// wyrd-qc0g: helpers for the family × era reflex grid (the `era_grid` field
// the backend now ships per morpheme, wyrd-lftl). These replace the old
// per-language `variants.js` resolution as the col-3 axis: the active
// morpheme's pronunciation, the paragon's modern form, and the grid's
// current-cell highlight all derive from era_grid here.
//
// era_grid shape (per morpheme):
//   [{ family: 'english',
//      stages: [{ language: 'old-english',
//                 forms: [{ form, source, reader_pronunciation?, ipa? }, ...] },
//               ...] }, ...]

import { accentFold } from './accents.js';

const stripDashes = (s) => (s || '').replace(/^-+|-+$/g, '');

/**
 * Strip diacritics while PRESERVING case (unlike accentFold, which also
 * lowercases): "Trebȳ" → "Treby", "-bȳ" → "-by". Drives the active card's
 * de-accented "modern form of the exact variant" line and the paragon.
 * Dash markers pass through untouched (only combining marks are dropped).
 */
export function deAccent(s) {
  const decomposed = (s || '').normalize('NFD');
  let out = '';
  for (const ch of decomposed) {
    const cp = ch.codePointAt(0);
    if (cp < 0x300 || cp > 0x36f) out += ch; // keep non-combining-mark chars
  }
  return out;
}

/**
 * The era_grid cell whose form matches a morpheme's current surface (folded on
 * accent + dash + case), or null. One source of truth for "which variant is
 * live": the active-card pronunciation reads it, and the grid highlights it.
 * @returns {{family: string, language: string, cell: object} | null}
 */
export function cellForSurface(morpheme, surface) {
  const target = accentFold(surface);
  if (!target) return null;
  for (const section of morpheme?.era_grid || []) {
    for (const stage of section.stages || []) {
      for (const cell of stage.forms || []) {
        if (accentFold(cell.form) === target) {
          return { family: section.family, language: stage.language, cell };
        }
      }
    }
  }
  return null;
}

/**
 * The morpheme's modern-English reflex cell (for the paragon), or null when
 * the grid carries no English/modern stage. Picks the first form — the modern
 * stage can be noisy (cluster mates), so callers fall back to deAccent(usage)
 * when this is null. (A cleaner modern-lemma pick lands with wyrd-rogd.1.)
 * @returns {object | null}
 */
export function modernReflexCell(morpheme) {
  for (const section of morpheme?.era_grid || []) {
    if (section.family !== 'english') continue;
    for (const stage of section.stages || []) {
      if (stage.language === 'modern-english' && stage.forms?.length) {
        return stage.forms[0];
      }
    }
  }
  return null;
}

/** True when the morpheme exposes any era_grid stages (sparse — coverage gap
 *  tracked in wyrd-32t1). Lets the view skip the grid entirely for bare
 *  morphemes rather than render an empty shell. */
export function hasEraGrid(morpheme) {
  return !!morpheme?.era_grid?.some((s) => s.stages?.length);
}

export { stripDashes };
