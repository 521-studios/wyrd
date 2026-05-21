// wyrd-hcmc: thin fetch wrapper around the wyrd Flask API.
//
// During dev the Vite proxy (spa-next/vite.config.js) forwards /api/*
// to Flask on :5000. In production CloudFront fronts both the SPA and
// the Lambda — same-origin, no CORS dance, no base-URL config needed.

/**
 * Fetch /api/manifest — the catalog of generators + their input_schemas.
 * Returns { generators: [{ name, display_name, description, input_schema, ... }, ...] }.
 */
export async function fetchManifest() {
  const resp = await fetch('/api/manifest');
  if (!resp.ok) {
    throw new Error(`manifest fetch failed: HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * wyrd-kppy: POST /api/kenning-rewind with pre-picked morphemes
 * (wyrd-y9aa input-schema addition). Skips trie re-decomposition —
 * uses the supplied morphemes so the era stops respect the
 * generator's actual morpheme picks. Returns the standard envelope:
 *   { results: [{ result, components: [{ era, family, rendered, ... }] }, ...] }
 * with one entry per era stop (oe-late / me / modern).
 */
export async function rewindWithMorphemes(name, morphemesByWord) {
  const resp = await fetch('/api/kenning-rewind', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, words: morphemesByWord }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`rewind failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}

/**
 * Roll one generator with the supplied params + seed. POSTs to
 * /api/<generator>. Returns the envelope: { generator, parameters,
 * seed, results: [{ result, explanation, components, morphemes_by_word, ... }] }.
 */
export async function rollGenerator(generatorName, params, seed) {
  const body = { ...params };
  if (seed !== null && seed !== undefined && seed !== '') {
    body.seed = Number(seed);
  }
  const resp = await fetch(`/api/${generatorName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`roll failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}
