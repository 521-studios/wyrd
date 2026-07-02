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

// Advanced fields that are unconditionally available regardless of any flag.
// The vector axis-weight knobs are always on now that vector is the only
// scoring path (the scoring_mode selector was retired): there's no mode to
// gate them under, so they ride alongside their always-present sliders.
const ALWAYS_ON_FIELDS = new Set([
  'phonological_weight',
  'semantic_weight',
  'position_weight',
  'baseline_weight',
]);

// Advanced fields that ride under another field's flag rather than their own.
// (None currently — kept as the extension point for grouped gating.)
const FIELD_FLAG_OVERRIDES = {};

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

/** Is an advanced field enabled? Some fields are unconditionally available
 *  (always-on); the rest gate on their controlling flag. */
export function fieldEnabled(config, fieldKey) {
  if (ALWAYS_ON_FIELDS.has(fieldKey)) return true;
  if (!config) return false;
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
 *  values arrive as strings; the schema default is already typed). Array
 *  fields can't be seeded from a scalar env string, so they return undefined
 *  (caller falls back to the schema default / empty array) rather than
 *  seeding a raw string into an array slot. */
export function coerceToType(raw, prop) {
  if (prop?.type === 'array') return undefined;
  if (prop?.type === 'integer' || prop?.type === 'number') {
    // A bad/empty numeric env (e.g. WYRD_DEFAULT_COUNT=abc, or '') would
    // otherwise yield NaN / 0 and silently seed a junk value. Reject anything
    // NON-FINITE, not just NaN: Number('Infinity') and an overflow like
    // Number('1e999') both yield Infinity, which Number.isNaN does NOT catch
    // and which serializes to null in the request — the very junk this guard
    // exists to stop. Number.isFinite rejects NaN AND ±Infinity, matching the
    // server's math.isfinite mirror (feature_flags._coerce_to_schema_type).
    // Return undefined so the caller falls back to the schema default rather
    // than shipping garbage.
    if (String(raw).trim() === '') return undefined;
    const n = Number(raw);
    if (prop.type === 'integer') {
      // Integer parity with the server (_coerce_to_schema_type, wyrd-8uuq):
      // accept integral floats ('5.0'→5), reject non-integers and trailing
      // garbage ('5.5'/'3abc'/'12px'→undefined). Number.isInteger is false for
      // NaN and ±Infinity, so it subsumes the non-finite guard on this path.
      // (Was parseInt(raw,10), which leniently parsed '3abc'→3 — a slider≠
      // generation divergence against the server's strict int(); wyrd-8uuq.)
      return Number.isInteger(n) ? n : undefined;
    }
    return Number.isFinite(n) ? n : undefined;
  }
  if (prop?.type === 'boolean') {
    // .trim() for parity with the server (feature_flags._parse_bool strips) and
    // with this function's own number branch (String(raw).trim()): a padded
    // ' true ' must coerce to true, not silently false.
    return ['1', 'true', 'yes', 'on'].includes(String(raw).trim().toLowerCase());
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

/** wyrd-etvd: snap-to-valid for an enum/option select. Returns the value the
 *  field SHOULD be set to, or `undefined` to mean "leave it alone".
 *
 *  Critically, an `undefined` (not-yet-seeded) value returns `undefined` — the
 *  seed pass (``seedDefault``) owns initial seeding, including any env
 *  ``config.defaults`` override (e.g. WYRD_DEFAULT_COUNT=5).
 *  Snapping an unseeded field to ``schemaDefault`` here would clobber that
 *  override and the seed pass would then skip (value no longer undefined).
 *  Only a DEFINED-but-invalid value (e.g. a culture-filtered option) snaps —
 *  to the schema default if it's valid, else the first option.
 *
 *  @param value the current field value (may be undefined)
 *  @param options the valid options (prop.enum, or culture-filtered options)
 *  @param schemaDefault the schema's default value (prop.default) */
export function snapEnumValue(value, options, schemaDefault) {
  if (value === undefined) return undefined;
  if (options.includes(value)) return undefined;
  return schemaDefault !== undefined && options.includes(schemaDefault)
    ? schemaDefault
    : options[0];
}

/** wyrd-kqyf: the culture-agnostic era-default token (WYRD_DEFAULT_ERA /
 *  terraform feature_flag_defaults). Era stages are per-culture, so the env
 *  default can't name a literal stage; the token means "this culture's
 *  present-day stage" on both sides of the contract (the server translates it
 *  in _resolve_era_param / _resolve_era_render_language). */
export const PRESENT_DAY_ERA = 'present-day';

/** wyrd-kqyf: snap-to-valid for a DEPENDENT select (x-options-by-culture),
 *  layering the present-day translation on snapEnumValue:
 *
 *  - a seeded `present-day` value resolves to the culture's present-day stage
 *    — the LAST option (per-culture lists are chronologically ordered with ''
 *    first), so the env default displays as e.g. "Modern English" instead of
 *    binding an invalid select value;
 *  - when the CONFIG default is the token, a culture switch that invalidates
 *    the prior stage (english's 'modern-english' isn't a welsh option) snaps
 *    to the NEW culture's present-day stage rather than clobbering to the
 *    schema default — without this the env default would silently become
 *    Mixed Era (and be sent as an explicit '') after every culture change.
 *
 *  Returns the value to set, or undefined for "leave it alone" (same contract
 *  as snapEnumValue). */
export function snapDependentValue(value, options, prop, config, fieldKey) {
  if (value === undefined) return undefined;
  if (!options || options.length === 0) return undefined;
  const presentDay = options[options.length - 1];
  if (value === PRESENT_DAY_ERA) return presentDay;
  const target =
    seedDefault(config, fieldKey, prop) === PRESENT_DAY_ERA ? presentDay : prop?.default;
  return snapEnumValue(value, options, target);
}

/** wyrd-b6hd: the value a field should be INITIALIZED to — the env
 *  default-override (config.defaults) or schema default, falling back to a
 *  type-appropriate empty when neither is set. The store seeds every field
 *  with this up front (appState.ensureParams), so the form binds populated
 *  values and no per-component lazy seed races the <select> bind (wyrd-etvd).
 *  Always returns a defined value (never undefined) so the bound state is
 *  never empty at mount. */
export function initialFieldValue(config, fieldKey, prop) {
  const seed = seedDefault(config, fieldKey, prop);
  if (seed !== undefined) return seed;
  if (prop?.type === 'array') return [];
  if (prop?.type === 'boolean') return false;
  return '';
}
