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
 * wyrd-i4jd: find the era_grid cell with a given stable `id` ({family, language,
 * cell} or null). The id-based lookup behind both the swap pin (`_cellId`) and
 * the backend `active_form_id` — opaque id compare, no surface fold.
 * @returns {{family: string, language: string, cell: object} | null}
 */
export function cellById(grid, id) {
  if (!id) return null;
  for (const section of grid || []) {
    for (const stage of section?.stages || []) {
      for (const cell of stage?.forms || []) {
        if (cell?.id === id) {
          return { family: section.family, language: stage.language, cell };
        }
      }
    }
  }
  return null;
}

/**
 * The era_grid cell that is "live" for a morpheme. ID-FIRST (wyrd-i4jd): an
 * explicit user swap (`morpheme._cellId`) wins, then the backend's
 * `active_form_id` (the id of the form the generator actually rendered), and
 * only as a legacy fallback (old bundles / no id) do we match the live surface
 * by accent+dash+case fold — preferring the `_lang` stage for a homograph,
 * else the first match. The id path avoids the surface-fold collision
 * (bǣre/bære, *ur/ur, same spelling across eras) that lit up two cells at once.
 * @returns {{family: string, language: string, cell: object} | null}
 */
export function cellForSurface(morpheme, surface) {
  const grid = morpheme?.era_grid || [];
  // wyrd-i4jd: highlight BY ID, not by surface fold (which collides on
  // same-spelled forms across eras/etymons). Precedence:
  //   1. `_cellId` — an explicit user swap (authoritative).
  //   2. `active_form_id` — the backend's id of the form the generator actually
  //      rendered (the unswapped highlight). The form is guaranteed in this
  //      morpheme's own grid, so no cross-grid string match.
  // A stale/missing id falls through to the legacy surface fold (old bundles).
  const byId = cellById(grid, morpheme?._cellId) || cellById(grid, morpheme?.active_form_id);
  if (byId) return byId;
  const target = accentFold(surface);
  if (!target) return null;
  const pinned = morpheme?._lang;
  let fallback = null;
  for (const section of grid) {
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
 *
 * wyrd-24s6 (D41): the canonical render is NATIVE (each morpheme in its
 * source-era form), with modern as the companion. So an un-pinned render whose
 * morphemes carry a distinct native ``rendered`` form is framed as 'native'
 * (paired with the 'modern' paragon beside it). 'as generated' stays for a roll
 * with no native/modern distinction (a plain roll, or an explicit
 * era=modern-english force-modern, where native == modern).
 */
export function eraBadge(morphemes, labelFn) {
  const morphs = (morphemes || []).flat().filter((m) => m?.usage?.trim());
  const eras = morphs.map((m) => m?._lang || m?.rendered_language || null);
  const explicit = [...new Set(eras.filter(Boolean))];
  if (explicit.length === 0) {
    const isNative = morphs.some((m) => m?.rendered && m.rendered !== m.usage);
    return isNative ? 'native' : 'as generated';
  }
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
 *  sense group (the backend dedupes the groups), else the first flat meaning.
 *  The drift baseline. */
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

/** wyrd-rogd.1: a HEURISTIC "this variant's meaning differs from the
 *  morpheme's" cue, for the ≠ drift marker. BOTH must be non-empty — an empty
 *  base can't drift (so a glossless morpheme never false-flags every cell) —
 *  and they're compared on a whitespace+case fold only, so it's a *possible*
 *  drift: near-synonyms ('farm' vs 'a farm') or multi-sense glosses can still
 *  read as drift. Single source of truth so the active card + grid can't
 *  diverge. */
export function isGlossDrift(base, variant) {
  const fold = (g) => String(g || '').trim().toLowerCase();
  return !!base && !!variant && fold(base) !== fold(variant);
}

/** wyrd-rogd.12: lay a section's populated stages onto its family's FULL fixed
 *  era axis (`eraStages[family]`, oldest→newest from the manifest), so periods
 *  hold the same column positions across morphemes. Returns a list of
 *  `{language, stage}` where a null `stage` is an empty slot (the grid renders
 *  a muted '—'). Falls back to the populated stages as-is for a family the
 *  manifest doesn't know. Any present-but-unmapped stage (defensive — the data
 *  contract says there are none) is appended after the canonical order so it's
 *  never silently dropped.
 *  @param {{family?: string, stages?: Array}} section
 *  @param {Record<string, string[]>} eraStages  manifest era_stages map
 *  @returns {Array<{language: string, stage: object|null}>}
 */
export function eraAxis(section, eraStages) {
  const stages = (section?.stages || []).filter(Boolean);
  const order = eraStages?.[section?.family];
  const byLang = new Map(stages.map((s) => [s?.language, s]));
  if (!order?.length) {
    return stages.map((s) => ({ language: s?.language, stage: s }));
  }
  const axis = order.map((language) => ({ language, stage: byLang.get(language) || null }));
  const known = new Set(order);
  for (const s of stages) {
    if (!known.has(s?.language)) axis.push({ language: s?.language, stage: s });
  }
  return axis;
}
