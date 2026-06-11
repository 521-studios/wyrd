// wyrd-hcmc + wyrd-20pz: thin fetch wrapper around the wyrd Flask /
// Lambda API.
//
// During dev the Vite proxy (spa-next/vite.config.js) forwards /api/*
// to Flask on :5000. In production CloudFront fronts both the SPA
// and the Lambda — same-origin, no CORS dance, no base-URL config
// needed.
//
// wyrd-20pz (POST body signing): CloudFront OAC + Lambda Function
// URL requires POST/PUT clients to send `x-amz-content-sha256` —
// CloudFront does NOT compute the body hash itself; without the
// header Lambda rejects with InvalidSignatureException. The hash
// is computed via crypto.subtle (secure-context API; production is
// HTTPS-only so it's always available there; dev over http://
// localhost also gets it because crypto.subtle has a localhost
// carve-out in evergreen browsers). Ported from spa/app.js's
// sha256Hex helper at cutover time.

async function postSignedJson(url, payload) {
  const body = JSON.stringify(payload);
  const headers = { 'Content-Type': 'application/json' };
  // crypto.subtle absent on http:// in some browsers — skip the
  // header in those cases. Dev gets the localhost carve-out;
  // production is HTTPS-only.
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const buf = new TextEncoder().encode(body);
    const hashBuf = await crypto.subtle.digest('SHA-256', buf);
    headers['x-amz-content-sha256'] = [...new Uint8Array(hashBuf)]
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
  return fetch(url, { method: 'POST', headers, body });
}

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
  const resp = await postSignedJson('/api/kenning-rewind', {
    name,
    words: morphemesByWord,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`rewind failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}

/**
 * wyrd-y0lx: POST /api/kenning-regenerate-morpheme — re-roll ONE morpheme
 * of a generated name while holding the others fixed. The payload carries
 * the current morphemes (`words`), the target `word_index` /
 * `morpheme_index`, a `seed` (baked into the transform step so re-runs are
 * deterministic), and the generation-context knobs the roll used (culture /
 * tags / mood / era / …). Returns the standard envelope; the single
 * result's `morphemes_by_word[word_index][morpheme_index]` is the
 * replacement morpheme dict the transform splices in.
 */
export async function regenerateMorpheme(payload) {
  const resp = await postSignedJson('/api/kenning-regenerate-morpheme', payload);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`regenerate failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}

/**
 * Roll one generator with the supplied params. POSTs to
 * /api/<generator>. No seed is sent — the server picks a fresh random
 * one each call. Returns the envelope: { generator, parameters,
 * results: [{ result, explanation, components, morphemes_by_word, ... }] }.
 */
export async function rollGenerator(generatorName, params) {
  const body = { ...params };
  const resp = await postSignedJson(`/api/${generatorName}`, body);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`roll failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}

/**
 * wyrd-dsl5: flag a generated name as defective. POSTs the full
 * reproduction context plus the user's required free-text reason to
 * /api/defects, which writes it to DynamoDB for operator triage.
 *
 * `reason` must be non-empty (the caller — DefectModal — enforces this in
 * the UI; the server re-validates and 400s a blank reason). Generic across
 * generators: `generator` is just a stored field.
 *
 * Returns { id, status } on success; throws on HTTP error.
 */
export async function reportDefect(report) {
  const resp = await postSignedJson('/api/defects', report);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`defect report failed: HTTP ${resp.status} — ${text.slice(0, 200)}`);
  }
  return resp.json();
}
