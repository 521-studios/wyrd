// wyrd-24s6 (D38): the native/modern surface predicates the Output column
// renders with. Extracted from OutputColumn.svelte so they're unit-testable
// like era.js (svelte components have no test harness here — only lib/*.test.js).
//
// The D38 contract: each result carries a native canonical `result` + a modern
// `result_modern` companion; per morpheme the backend ships `rendered` (native,
// absent when the slot has no distinct native form) + `usage` (the modern
// bucket key).

import { accentedUsage } from './accents.js';

/** The native surface for a morpheme: the backend's native `rendered` form,
 *  then the accented original-script surface (`accentedUsage`, wyrd-de5t — so a
 *  bundle without per-morpheme native renders still shows `-bȳ` rather than the
 *  bare ASCII `-by`), then the plain modern `usage`. */
export function nativeSurface(morph) {
  return morph?.rendered || accentedUsage(morph) || morph?.usage || '';
}

/** The modern surface for a morpheme: the plain `usage` bucket key. */
export function modernSurface(morph) {
  return morph?.usage || '';
}

/** Whether to show the per-morpheme modern reflex beneath its native surface —
 *  only when both are non-empty AND differ (a slot with no distinct native form
 *  has native == modern, so the second line would be noise). */
export function showMorphemeModern(morph) {
  const mod = modernSurface(morph);
  return !!mod && mod !== nativeSurface(morph);
}

/** Whether to show the result-level modern companion beside the native name —
 *  only when `result_modern` exists AND differs from the native `result` (a
 *  plain / force-modern roll has native == modern, so the companion is noise). */
export function showModernCompanion(result) {
  return !!(result?.result_modern && result.result_modern !== result.result);
}
