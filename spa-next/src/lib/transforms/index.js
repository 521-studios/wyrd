// wyrd-kppy: transform registry. Single source of truth for the
// palette + the pipeline engine. New transforms (Swap, Anglicize,
// Calque, Drift) plug in by adding an entry here.

import { rewindTransform } from './rewind.js';
import { swapTransform } from './swap.js';

const TRANSFORMS = {
  [rewindTransform.kind]: rewindTransform,
  [swapTransform.kind]: swapTransform,
};

/** Lookup by kind. Throws if unknown — surfaces transform-kind
 *  typos at the call site rather than swallowing silently. */
export function getTransform(kind) {
  const t = TRANSFORMS[kind];
  if (!t) throw new Error(`unknown transform kind: ${kind}`);
  return t;
}

/** Catalog for the palette UI (TransformPalette.svelte). Returns an
 *  array of { kind, label, description } in declaration order. */
export function listTransforms() {
  return Object.values(TRANSFORMS).map((t) => ({
    kind: t.kind,
    label: t.label,
    description: t.description,
  }));
}

/** Build a fresh step record for a given transform kind, with the
 *  transform's defaultParams (shallow-copied so user edits don't
 *  mutate the template). */
export function newStep(kind) {
  const t = getTransform(kind);
  return {
    kind,
    params: { ...t.defaultParams },
  };
}
