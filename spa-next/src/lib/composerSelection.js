// wyrd-tbcj.1: selection reducers for the MoodTagComposer.
//
// Moods are SINGLE-select (radio-style) to match the runtime's soft
// "one mood morpheme per name" overlay (wyrd-4rp8 / #565), which changed
// generation to apply at most one mood-accent per name. The composer had
// always been multi-select, so a user could pick several moods while the
// runtime honored only one — this reconciles the UX with that behavior.
//
// Tags stay MULTI-select (toggleMulti). Both reducers return a NEW array
// (never mutate) and keep the field an array — so `params.mood` is still a
// list (now length 0 or 1) and the API / input-schema contract is unchanged.

// Single-select: clicking the active value clears the selection; clicking
// any other value replaces it. Result is a 0-or-1-element array.
export function selectSingle(list, value) {
  return list.includes(value) ? [] : [value];
}

// Multi-select toggle: add the value if absent, remove it if present.
export function toggleMulti(list, value) {
  return list.includes(value) ? list.filter((x) => x !== value) : [...list, value];
}

// Remove a single value (the chip × control), shared by moods and tags.
export function removeValue(list, value) {
  return list.filter((x) => x !== value);
}

// wyrd-ah53: the tag options offered for a given culture. The tag schema carries
// a culture-agnostic `items.enum` (the union, for CLI/API validation) and an
// `x-options-by-culture` map narrowing it to the tags each culture can actually
// satisfy (a --tag reserves one slot from a pool that HAS the tag, D47).
//
// Fallback policy (deliberately NOT the era/stratum `Object.values(map)[0]`
// first-culture fallback): when the schema has no per-culture map, OR the culture
// is missing/undefined (a transient init/transition state), fall back to the FULL
// enum — never another culture's subset. Returning the first culture's tags for an
// undefined culture would wrongly HIDE valid options and PRUNE valid selections
// mid-transition; the union is always safe (it only ever over-offers, which the
// runtime handles, never silently drops the user's choice).
export function tagOptionsForCulture(tagSchema, culture) {
  const enumAll = tagSchema?.items?.enum || [];
  const byCulture = tagSchema?.['x-options-by-culture'];
  // No map, or no resolved culture yet (transient init/transition) → the full
  // union, never another culture's subset. The explicit !culture guard avoids
  // relying on hasOwnProperty's string-coercion of undefined/null.
  if (!byCulture || !culture) return enumAll;
  return Object.prototype.hasOwnProperty.call(byCulture, culture) ? byCulture[culture] : enumAll;
}

// wyrd-ah53: keep only the tags valid for the resolved option set — used to prune
// a selection (workingTags / params.tags) when the culture changes. Order-
// preserving; returns the SAME array reference when nothing is dropped so callers
// can cheaply skip a no-op write.
export function pruneTagsToOptions(tags, options) {
  const allowed = new Set(options);
  const kept = tags.filter((t) => allowed.has(t));
  return kept.length === tags.length ? tags : kept;
}
