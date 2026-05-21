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
