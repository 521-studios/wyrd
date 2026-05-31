// wyrd-hcmc: per-generator headline-field classification.
//
// Progressive disclosure: each generator gets a list of "headline"
// param keys shown unconditionally in col 1. Everything else
// collapses behind a single ▾ Advanced disclosure.
//
// Generators not in this map show all their params unconditionally
// (the safe default — small schemas don't benefit from disclosure).

// wyrd-ppk6: keys that should NOT appear in the col 1 form for any
// generator. 'seed' is kept here defensively: reproducibility-by-seed
// was removed from the SPA (a seed only reproduces against one bundle
// version), so should a generator's schema ever surface a 'seed' param,
// it must not render an input that re-introduces the broken affordance.
export const HIDDEN_FIELDS = new Set(['seed']);

export const HEADLINE_FIELDS = {
  // Kenning's headline knobs: pick a culture, decide how many names,
  // optionally fix the seed, optionally tune the mood. Everything
  // else (novelty, spelling_variety, inflection_density, era,
  // stratum, scoring_mode + weights, packs, cohesion, priors_path,
  // include_fiction) hides until the operator wants depth.
  kenning: ['culture', 'count', 'mood'],

  // kenning-rewind: name is the only headline; words is the
  // continuity-flow input handled by col 3, not the params form.
  'kenning-rewind': ['name'],

  // KenningExplain: bare minimum.
  'kenning-explain': ['name'],
};

/**
 * Return { headline, advanced } — two lists of [key, propSchema] tuples
 * partitioned per the HEADLINE_FIELDS map. Falls back to "all headline"
 * when the generator isn't in the map.
 *
 * Preserves the schema's property order within each list.
 */
export function partitionFields(generatorName, schema) {
  // wyrd-ppk6: HIDDEN_FIELDS drop out entirely before partitioning —
  // 'seed' lives in the Header, not in either headline or advanced.
  const props = Object.entries(schema?.properties || {}).filter(
    ([k]) => !HIDDEN_FIELDS.has(k),
  );
  const headlineKeys = HEADLINE_FIELDS[generatorName];
  if (!headlineKeys) {
    return { headline: props, advanced: [] };
  }
  const headlineSet = new Set(headlineKeys);
  const headline = props.filter(([k]) => headlineSet.has(k));
  const advanced = props.filter(([k]) => !headlineSet.has(k));
  return { headline, advanced };
}
