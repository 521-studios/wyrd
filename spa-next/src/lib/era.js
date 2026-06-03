// wyrd-qc0g: helpers for the family × era reflex grid (the `era_grid` field
// the backend now ships per morpheme, wyrd-lftl). These supersede the old
// per-language `variants.js` resolution as the col-3 axis — the active
// morpheme's pronunciation and the grid's current-cell highlight derive from
// era_grid here; variants.js stays only as a last-resort pronunciation
// fallback until the grid fully owns it (wyrd-zw1f).
//
// era_grid shape (per morpheme):
//   [{ family: 'english',
//      stages: [{ language: 'old-english',
//                 forms: [{ form, source, reader_pronunciation?, ipa? }, ...] },
//               ...] }, ...]

import { accentFold } from './accents.js';
import { pronunciationFor } from './variants.js';

// A pronunciation slot is useful only if it actually carries sound — mirrors
// variants.js's _hasPron. A cell can exist (matched surface) but be pronless
// (IPA is sparse; many Middle-English cells carry neither ipa nor reader), in
// which case the lookup must FALL THROUGH rather than return a blank slot.
const hasPron = (slot) => !!(slot && (slot.ipa || slot.reader_pronunciation));

/**
 * Strip diacritics while PRESERVING case (unlike accentFold, which also
 * lowercases): "Trebȳ" → "Treby", "-bȳ" → "-by". Drives the active card's
 * de-accented "modern form of the exact variant" line and the paragon.
 * Dash markers pass through untouched (only combining marks are dropped).
 */
export function deAccent(s) {
  const decomposed = String(s || '').normalize('NFD');
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
 *
 * wyrd-zw1f: when the same surface appears in several stages (a homograph —
 * e.g. "don" in OE / OF / Celtic, "by" in OE + Old Norse), a swap pins the
 * chosen stage on `morpheme._lang`; we prefer the cell in that stage so the
 * highlight + pronunciation track the EXACT variant the user picked, not an
 * arbitrary first match. Falls back to the first match both when nothing is
 * pinned AND when a pin is set but its stage carries no matching cell (the
 * pin is a soft preference, not a hard filter — the surface may have drifted
 * out of the pinned stage).
 * @returns {{family: string, language: string, cell: object} | null}
 */
export function cellForSurface(morpheme, surface) {
  const target = accentFold(surface);
  if (!target) return null;
  const pinned = morpheme?._lang;
  let fallback = null;
  for (const section of morpheme?.era_grid || []) {
    for (const stage of section?.stages || []) {
      for (const cell of stage?.forms || []) {
        if (accentFold(cell?.form) === target) {
          const hit = { family: section.family, language: stage.language, cell };
          if (!pinned) return hit; // unpinned: first match wins — stop scanning
          if (stage.language === pinned) return hit; // pinned exact match
          if (!fallback) fallback = hit; // remember first as the soft fallback
        }
      }
    }
  }
  return fallback;
}

/** True when the morpheme exposes any era_grid stages (sparse — coverage gap
 *  tracked in wyrd-32t1). Lets the view skip the grid entirely for bare
 *  morphemes rather than render an empty shell. */
export function hasEraGrid(morpheme) {
  return !!morpheme?.era_grid?.some((s) => s?.stages?.length);
}

/**
 * wyrd-yrf9: the active card's era badge for a WORKING morpheme list (per-word
 * arrays). A morpheme's era is the stage it's pinned to (`_lang`, set on a grid
 * swap), else the era a render targeted (`rendered_language`), else none — a
 * still-default/modern surface. Returns:
 *   • 'as generated'      — no morpheme carries an explicit era
 *   • '<Era>'             — every morpheme shares one era (e.g. 'Old English')
 *   • 'Mixed (A, B, …)'   — a blend of eras, plus 'Modern' when some morphemes
 *                           are still at their default (un-swapped) surface
 * ``labelFn`` maps a stage/language tag to its display label — injected so this
 * stays a pure helper with no dependency on languageLabels.
 */
export function eraBadge(morphemes, labelFn) {
  const morphs = (morphemes || []).flat().filter((m) => m?.usage?.trim());
  const eras = morphs.map((m) => m?._lang || m?.rendered_language || null);
  const explicit = [...new Set(eras.filter(Boolean))];
  if (explicit.length === 0) return 'as generated';
  const hasDefault = eras.some((e) => !e);
  if (explicit.length === 1 && !hasDefault) return labelFn(explicit[0]);
  const labels = [...new Set([...explicit.map(labelFn), ...(hasDefault ? ['Modern'] : [])])];
  // dedup can collapse the labels to one — e.g. an explicit `modern-english`
  // pin maps to the same 'Modern' label the default fold-in adds. That's not a
  // blend, just that single era, so don't dress it up as 'Mixed (Modern)'.
  return labels.length === 1 ? labels[0] : `Mixed (${labels.join(', ')})`;
}

/**
 * The pronunciation slot ({reader_pronunciation?, ipa?}) to show for a
 * morpheme's CURRENT surface, in priority order:
 *   1. the matching era_grid cell (the right axis) — but ONLY if it carries
 *      sound (a pronless matched cell falls through, else the guide blanks);
 *   2. the era render's own pronunciation (rendered_pron);
 *   3. the legacy per-language variants.js fallback (kept until the grid fully
 *      owns pronunciation — wyrd-zw1f).
 * Always returns an object (never null) so callers can read .reader/.ipa
 * unguarded. Extracted from InspectorColumn so it's unit-testable.
 */
export function pronForSurface(morpheme, surface) {
  const cell = cellForSurface(morpheme, surface)?.cell;
  if (hasPron(cell)) return cell;
  if (hasPron(morpheme?.rendered_pron)) return morpheme.rendered_pron;
  return pronunciationFor(morpheme) || {};
}

/** wyrd-rogd.1: a morpheme's OWN primary gloss — the first sense of the first
 *  (deduped) group, else the first flat meaning. The drift baseline. */
export function primaryGloss(morpheme) {
  return morpheme?.meaning_groups?.[0]?.[0] || morpheme?.meanings?.[0] || '';
}

/** wyrd-rogd.1: the gloss of the era_grid cell matching ``surface`` — the
 *  reflex's OWN meaning (which may have drifted from the morpheme's), or ''
 *  when the cell carries no gloss (sparse) / no cell matches. Drives the
 *  active-card gloss tracking the swapped variant. */
export function glossForSurface(morpheme, surface) {
  return cellForSurface(morpheme, surface)?.cell?.gloss || '';
}
