// wyrd-tz35: share-link encode/decode.
//
// A shared workspace ships the same shape we persist in localStorage
// (savedStore entries minus id/saved_at/label, plus a stable
// schema_version for future migrations) as base64-encoded JSON in
// the `?s=` URL param. Recipient's SPA detects the param at boot,
// decodes + restores all 3 columns.
//
// v1 uses one opaque `?s=` param for simplicity. Pretty URLs
// (?gen=kenning&seed=42&culture=english) can layer on later as a
// secondary encoding path — the decoder just needs to try both
// shapes at boot.

const SHARE_SCHEMA = 1;

/**
 * Encode a workspace state into a URL-safe base64 string. Caller
 * supplies the same fields as savedStore.add: { generator, params,
 * seed, original, pipeline }. The encoded string goes into the
 * `?s=` query param.
 */
export function encodeWorkspace({ generator, params, seed, original, pipeline }) {
  const payload = {
    schema_version: SHARE_SCHEMA,
    generator,
    params,
    seed,
    original,
    pipeline,
  };
  const json = JSON.stringify(payload);
  return base64UrlEncode(json);
}

/**
 * Decode a `?s=` value back into a workspace payload. Returns null
 * on any failure (malformed base64, bad JSON, wrong schema_version)
 * so callers can fall back to the default empty boot. Failures are
 * console.warn'd; never thrown.
 */
export function decodeWorkspace(encoded) {
  if (!encoded) return null;
  try {
    const json = base64UrlDecode(encoded);
    const payload = JSON.parse(json);
    if (payload.schema_version !== SHARE_SCHEMA) {
      console.warn(
        `share-link: schema ${payload.schema_version} != ${SHARE_SCHEMA}; ignoring`,
      );
      return null;
    }
    return payload;
  } catch (err) {
    console.warn('share-link: decode failed', err);
    return null;
  }
}

/**
 * Build a shareable URL from the encoded payload. Uses window's
 * current origin + pathname so the link works against whatever
 * environment the user is on (dev = localhost:5173, prod =
 * CloudFront domain).
 */
export function buildShareUrl(encoded) {
  const url = new URL(window.location.href);
  url.search = `?s=${encoded}`;
  url.hash = '';
  return url.toString();
}

/** Extract the `?s=` param from the current location, or null. */
export function readShareParam() {
  const params = new URLSearchParams(window.location.search);
  return params.get('s');
}

/**
 * Strip the `?s=` param from the URL after loading, so a refresh
 * doesn't re-restore the share and the user's local workspace
 * mutations don't get attributed to the original share link.
 */
export function clearShareParam() {
  const url = new URL(window.location.href);
  url.searchParams.delete('s');
  window.history.replaceState({}, '', url.toString());
}

// base64url variant: + → -, / → _, trailing = padding stripped. The
// decoder restores the padding from string length.
function base64UrlEncode(str) {
  // btoa works on Latin-1; convert to bytes first so unicode (e.g.
  // OE 'hām' macron, Welsh 'llyn') round-trips correctly.
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function base64UrlDecode(encoded) {
  // Restore base64 padding + reverse URL-safe substitutions.
  const padded = encoded.replace(/-/g, '+').replace(/_/g, '/');
  const padding = padded.length % 4 === 0 ? 0 : 4 - (padded.length % 4);
  const b64 = padded + '='.repeat(padding);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}
