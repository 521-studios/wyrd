// wyrd-0gou: SPA-side feature-flag mapping.
//
// The server ships the resolved config on /api/manifest:
//   config = { all: bool, flags: { <ENVIFIED_NAME>: bool }, defaults: { <opt>: str } }
// (see wyrd/feature_flags.py). This module owns the SPA half: the canonical
// flag NAMES, the flag→field mapping, and the english-always rule. Resolution
// is fail-closed — an option whose controlling flag isn't on is hidden, so
// default (no env set) = a minimal, prod-safe surface.
//
// The flags map is keyed by the env-var suffix the server saw (e.g.
// WYRD_FF_CULTURE_WELSH → "CULTURE_WELSH"); we compute the same suffix from a
// flag name via envify(), so namespaced flags round-trip without the server
// having to reverse-parse 'CULTURE_WELSH' back to 'culture.welsh'.

// Cultures always available regardless of flags — the generator needs a
// culture, and english is the guaranteed baseline.
const ALWAYS_ON_CULTURES = new Set(['english']);

// Advanced fields that ride under another field's flag rather than their own.
// The vector axis-weight knobs are meaningless without the scoring-mode
// selector, so the `scoring_mode` flag gates them as a unit.
const FIELD_FLAG_OVERRIDES = {
  phonological_weight: 'scoring_mode',
  semantic_weight: 'scoring_mode',
  position_weight: 'scoring_mode',
  baseline_weight: 'scoring_mode',
};

/** The flag name controlling an advanced field. Defaults to the field key
 *  itself (1:1); a few fields map to a grouping flag. */
export function fieldFlag(fieldKey) {
  return FIELD_FLAG_OVERRIDES[fieldKey] || fieldKey;
}

/** Envify a flag name into the manifest flags-map key (matches the server's
 *  WYRD_FF_<NAME> suffix): upper-case, '.' and '-' → '_'. */
export function envify(name) {
  return name.toUpperCase().replace(/[.-]/g, '_');
}

/** Resolve a flag against the manifest config. The master override (`all`)
 *  wins; an absent flag is off (fail-closed). `config` may be null/undefined
 *  (a legacy manifest with no config block) → everything off. */
export function flagOn(config, name) {
  if (!config) return false;
  if (config.all) return true;
  return config.flags?.[envify(name)] === true;
}

/** Is an advanced field enabled (its controlling flag on)? */
export function fieldEnabled(config, fieldKey) {
  return flagOn(config, fieldFlag(fieldKey));
}

/** Filter a culture enum down to english (always) + flagged-on cultures,
 *  preserving the schema's order. */
export function visibleCultures(config, cultureEnum) {
  return (cultureEnum || []).filter(
    (c) => ALWAYS_ON_CULTURES.has(c) || flagOn(config, `culture.${c}`),
  );
}

/** Coerce an env-string default override to the field's schema type (env
 *  values arrive as strings; the schema default is already typed). Arrays
 *  aren't supported as default overrides — they fall through unchanged. */
export function coerceToType(raw, prop) {
  if (prop?.type === 'integer' || prop?.type === 'number') {
    // A bad/empty numeric env (e.g. WYRD_DEFAULT_COUNT=abc, or '') would
    // otherwise yield NaN / 0 and silently seed a junk value (NaN serializes
    // to null in the request). Return undefined so the caller falls back to
    // the schema default / type-based empty rather than shipping garbage.
    if (String(raw).trim() === '') return undefined;
    const n = prop.type === 'integer' ? parseInt(raw, 10) : Number(raw);
    return Number.isNaN(n) ? undefined : n;
  }
  if (prop?.type === 'boolean') {
    return ['1', 'true', 'yes', 'on'].includes(String(raw).toLowerCase());
  }
  return raw;
}

/** The seed value for a field: env default-override (coerced) wins over the
 *  schema default. Returns `undefined` when neither is set (caller applies
 *  its type-based empty fallback). */
export function seedDefault(config, fieldKey, prop) {
  const override = config?.defaults?.[fieldKey];
  if (override !== undefined) {
    const coerced = coerceToType(override, prop);
    // A junk override (coerced → undefined, e.g. WYRD_DEFAULT_COUNT=abc)
    // falls through to the schema default rather than seeding nothing.
    if (coerced !== undefined) return coerced;
  }
  return prop?.default;
}
