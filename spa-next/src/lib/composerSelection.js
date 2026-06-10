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
