// wyrd-2b50 follow-up: pick ONE representative gloss per morpheme for the
// compact output row, instead of dumping the whole meaning list.
//
// Why not just meanings[0]? The list arrives ALPHABETICALLY SORTED, so the
// first element is systematically the worst: dictionary articles ("A bee.",
// "An enclosure; a farmstead; …"), quoted grammatical particles ('"at the"'),
// and "a personal name" all sort to the top. And reordering to find the
// "cleanest" word is also wrong — list order encodes curation priority, so
// reordering loses the canonical pick (Chil- "child" -> "well"). The right
// move is: walk in order, CLEAN the cruft, SKIP the junk, take the first
// survivor. The residual misses (-more -> "carrot") are polluted source data
// (wyrd-u9k6), not fixable here.

const ARTICLE = /^(an?|the|of)\s+/i;

// substrings that mark a gloss as grammatical/non-semantic — never a meaning
// a reader wants under a morpheme.
const GRAMMAR = [
  'personal name',
  'particle',
  'suffix',
  'indication',
  'inflection of',
  'possessive',
  'patronymic',
  'connective',
  'place-name forming',
];

// A cleaned gloss that is nothing but a bare article is not a meaning.
const BARE_ARTICLES = new Set(['a', 'an', 'the', 'of']);

// Reduce one raw gloss to its head term: drop a leading article, keep only the
// first clause (before ; , ( or " of "), strip trailing periods/quotes.
// Returns '' for non-strings (defensive — the array comes from upstream JSON).
function clean(raw) {
  if (typeof raw !== 'string') return '';
  let s = raw.trim().replace(/^["']+|["']+$/g, '').trim();
  s = s.split(/\s+of\s+|[;,(]/i)[0].trim();
  s = s.replace(ARTICLE, '').trim().replace(/[.\s]+$/, '').trim();
  return s;
}

// A raw entry to skip outright before cleaning: non-string, or a quoted
// grammatical particle ('"at the"' / "'at the'").
function skippable(raw) {
  return typeof raw !== 'string' || /^['"]/.test(raw.trim());
}

// A cleaned term is junk if it's too short, a bare article, still
// multi-clause/verbose (> 2 words), or grammatical.
function isJunk(cleaned) {
  if (!cleaned || cleaned.length < 2) return true;
  const lo = cleaned.toLowerCase();
  if (BARE_ARTICLES.has(lo)) return true;
  if (GRAMMAR.some((g) => lo.includes(g))) return true;
  if (cleaned.split(/\s+/).length > 2) return true;
  return false;
}

/**
 * The single representative meaning for a morpheme's `meanings` list, or '' if
 * every entry is junk (e.g. pure connectives — caller renders nothing). This is
 * the in-order, curation-priority pick.
 */
export function representativeMeaning(meanings) {
  if (!Array.isArray(meanings) || meanings.length === 0) return '';
  for (const raw of meanings) {
    if (skippable(raw)) continue;
    const c = clean(raw);
    if (!isJunk(c)) return c.toLowerCase();
  }
  return '';
}

// Two cleaned stems are "the same sense" if one contains the other or they
// share a 4-char prefix — cheap dedup for inflections (gold/golden,
// ravine/ravines) without a thesaurus. It will NOT catch true synonyms
// (dale/valley), which is acceptable: showing both is short and informative.
function sameSense(a, b) {
  const k = Math.min(a.length, b.length, 4);
  return a.slice(0, k) === b.slice(0, k) || a.includes(b) || b.includes(a);
}

/**
 * A SHORT representative list (1–2 entries) for a morpheme.
 *
 * Item 1 is the in-order curated canonical (representativeMeaning). Item 2 is
 * added only when a DISTINCT stem is *repeated* (count >= 2) across the pooled
 * meanings — a repeat is the fingerprint of a second etymon merged into the
 * family (a homograph like Gil- = gold + gully). Single-sense morphemes stay at
 * one word; genuine homographs surface both senses (Gil- -> ["gold","ravine"]).
 */
export function representativeMeanings(meanings, max = 2) {
  const item1 = representativeMeaning(meanings);
  if (!item1) return [];
  if (max < 2) return [item1];

  const counts = new Map();
  const order = [];
  for (const raw of meanings) {
    if (skippable(raw)) continue;
    const c = clean(raw).toLowerCase();
    if (isJunk(c)) continue;
    if (!counts.has(c)) order.push(c);
    counts.set(c, (counts.get(c) || 0) + 1);
  }
  const second = order
    .filter((c) => counts.get(c) >= 2 && !sameSense(c, item1))
    .sort((a, b) => counts.get(b) - counts.get(a) || order.indexOf(a) - order.indexOf(b))[0];

  return second ? [item1, second] : [item1];
}

/**
 * True when a morpheme is essentially a proper name (so the row can show a
 * dim "name" marker instead of a blank when representativeMeaning is empty).
 */
export function isNameMorpheme(morph) {
  const tags = morph?.tags || [];
  if (tags.some((t) => /\bname\b/i.test(t))) return true;
  // Anchor to the PRIMARY (first) sense + manorial families, so a buried
  // name-sense in a connective (et/and) doesn't mislabel it as a name.
  const first = String((morph?.meanings || [])[0] || '');
  return /personal name|manorial family|surname|family name/i.test(first);
}

/**
 * The 1–2 word gloss shown under a morpheme. Falls back to a dim "name" marker
 * for proper-name elements (manorial families etc.) — but only when the surface
 * is capitalized, so lowercase connectives (and/et/be) whose pooled meanings
 * happen to include "a personal name" stay blank.
 *
 * Pure (regex + Map build over `morph`), so the caller can memoize it.
 */
export function glossFor(morph) {
  // A null / non-object morph yields no gloss rather than crashing — the array
  // may come from a restored/shared workspace (not just a live API roll), and
  // one malformed entry must degrade to '' for that slot, not throw out of the
  // whole glossesByResults map and blank every gloss in the column. Matches the
  // defensive posture of `clean` (non-string guard) and `isNameMorpheme` (?.).
  if (!morph || typeof morph !== 'object') return '';
  const g = representativeMeanings(morph.meanings);
  if (g.length) return g.join(' · ');
  const looksProper = /^[A-Z]/.test(morph.usage || '');
  return looksProper && isNameMorpheme(morph) ? 'name' : '';
}

/**
 * Precompute the per-morpheme glosses for a list of results, once. Returns a
 * nested array mirroring each result's `morphemes_by_word`:
 * `out[resultIdx][wordIdx][morphIdx]` is the gloss string for that morpheme.
 *
 * wyrd-myof: glosses are a pure function of a result's `morphemes_by_word`, so
 * computing them here (keyed off `appState.results` via a `$derived` in the
 * Output column) runs the regex/Map work once per ROLL — not inline in the
 * template, where it recomputed for every result on every reactive render
 * (including selection changes via `currentResultIndex`).
 */
export function glossesByResults(results) {
  return (results ?? []).map((r) =>
    (r?.morphemes_by_word ?? []).map((word) => (word ?? []).map((morph) => glossFor(morph))),
  );
}
