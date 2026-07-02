// wyrd-0gou: unit tests for the SPA feature-flag mapping. These pure
// functions are the fail-closed gate that keeps unvalidated options hidden,
// so they're pinned directly (svelte-check only type-checks; it doesn't
// exercise behavior).
import { describe, it, expect } from 'vitest';
import {
  envify,
  flagOn,
  fieldFlag,
  fieldEnabled,
  visibleCultures,
  coerceToType,
  snapDependentValue,
  PRESENT_DAY_ERA,
  seedDefault,
  snapEnumValue,
  initialFieldValue,
} from './featureFlags.js';

describe('flagOn (fail-closed)', () => {
  it('returns false for a null/legacy config (no config block)', () => {
    expect(flagOn(null, 'novelty')).toBe(false);
    expect(flagOn(undefined, 'novelty')).toBe(false);
  });

  it('returns false for an empty config and for an absent flag', () => {
    expect(flagOn({ all: false, flags: {} }, 'novelty')).toBe(false);
    expect(flagOn({ all: false, flags: { ERA: true } }, 'novelty')).toBe(false);
  });

  it('master override (all) turns every flag on', () => {
    expect(flagOn({ all: true, flags: {} }, 'novelty')).toBe(true);
    expect(flagOn({ all: true, flags: { NOVELTY: false } }, 'novelty')).toBe(true);
  });

  it('requires a strict boolean true (a truthy string/number does not pass)', () => {
    expect(flagOn({ all: false, flags: { NOVELTY: true } }, 'novelty')).toBe(true);
    expect(flagOn({ all: false, flags: { NOVELTY: 'true' } }, 'novelty')).toBe(false);
    expect(flagOn({ all: false, flags: { NOVELTY: 1 } }, 'novelty')).toBe(false);
  });
});

describe('envify (server parity)', () => {
  it('upper-cases and maps . and - to _ (matches WYRD_FF_<NAME>)', () => {
    expect(envify('culture.welsh')).toBe('CULTURE_WELSH');
    expect(envify('priors-path')).toBe('PRIORS_PATH');
  });

  it('namespaced flags resolve through the same suffix the server emits', () => {
    expect(flagOn({ all: false, flags: { CULTURE_WELSH: true } }, 'culture.welsh')).toBe(true);
  });
});

describe('fieldFlag / fieldEnabled', () => {
  it('maps 1:1 by default', () => {
    expect(fieldFlag('novelty')).toBe('novelty');
  });

  it('flag-gated fields follow their controlling flag', () => {
    expect(fieldEnabled({ all: false, flags: {} }, 'novelty')).toBe(false);
    expect(fieldEnabled({ all: false, flags: { NOVELTY: true } }, 'novelty')).toBe(true);
  });

  it('the vector axis-weight knobs are always on (vector is the only scoring path)', () => {
    const weights = ['phonological_weight', 'semantic_weight', 'position_weight', 'baseline_weight'];
    // Always-on even with a null/empty config (no flag controls them).
    for (const w of weights) {
      expect(fieldEnabled(null, w)).toBe(true);
      expect(fieldEnabled({ all: false, flags: {} }, w)).toBe(true);
    }
  });
});

describe('visibleCultures (english guaranteed)', () => {
  const enumv = ['english', 'scottish', 'welsh', 'irish', 'breton'];

  it('keeps english even with null config / no flags', () => {
    expect(visibleCultures(null, enumv)).toEqual(['english']);
    expect(visibleCultures({ all: false, flags: {} }, enumv)).toEqual(['english']);
  });

  it('adds only flagged-on cultures, preserving schema order', () => {
    const cfg = { all: false, flags: { CULTURE_WELSH: true, CULTURE_BRETON: true } };
    expect(visibleCultures(cfg, enumv)).toEqual(['english', 'welsh', 'breton']);
  });

  it('master override shows all; empty/undefined enum → []', () => {
    expect(visibleCultures({ all: true, flags: {} }, enumv)).toEqual(enumv);
    expect(visibleCultures(null, undefined)).toEqual([]);
  });
});

describe('coerceToType', () => {
  it('coerces numeric strings', () => {
    expect(coerceToType('5', { type: 'integer' })).toBe(5);
    expect(coerceToType('1.5', { type: 'number' })).toBe(1.5);
  });

  it('returns undefined for non-numeric / empty numeric env (no NaN / 0 leak)', () => {
    expect(coerceToType('abc', { type: 'integer' })).toBeUndefined();
    expect(coerceToType('', { type: 'number' })).toBeUndefined();
    expect(coerceToType('   ', { type: 'integer' })).toBeUndefined();
  });

  it('rejects non-finite numbers (Infinity / overflow), not just NaN', () => {
    // Number('Infinity') / Number('1e999') yield Infinity, which Number.isNaN
    // does NOT catch and which serializes to null in the request — must fall
    // back to the schema default, matching the server's math.isfinite guard.
    expect(coerceToType('Infinity', { type: 'number' })).toBeUndefined();
    expect(coerceToType('-Infinity', { type: 'number' })).toBeUndefined();
    expect(coerceToType('1e999', { type: 'number' })).toBeUndefined();
    // The integer path uses Number(raw) + Number.isInteger, which is false for
    // NaN and ±Infinity — so 'Infinity' rejects on this path too.
    expect(coerceToType('5', { type: 'integer' })).toBe(5);
    expect(coerceToType('Infinity', { type: 'integer' })).toBeUndefined();
  });

  // The shared server/SPA integer-coercion parity vector (wyrd-8uuq). The
  // server's tests/test_feature_flags.py::test_integer_coercion_parity_vector
  // asserts the SAME (input → outcome) list against _coerce_to_schema_type, so
  // slider (seeded from a WYRD_DEFAULT) and generation never disagree. Here
  // `undefined` == the server's `None` (fall back to the schema default).
  it('converges integer coercion with the server (wyrd-8uuq parity vector)', () => {
    const intp = { type: 'integer' };
    const vector = [
      ['5', 5],
      ['5.0', 5], // integral float → accepted (the decimal-styled WYRD_DEFAULT)
      ['5.5', undefined], // non-integral → rejected (was 5 under lenient parseInt)
      ['3abc', undefined], // trailing garbage → rejected (was 3 under parseInt)
      ['12px', undefined], // trailing garbage → rejected (was 12)
      ['-3.9', undefined], // non-integral → rejected (was -3)
      ['', undefined],
      ['-2', -2],
      [' 7 ', 7],
      ['Infinity', undefined],
      ['NaN', undefined],
      ['.5', undefined], // non-integral → rejected
      ['١٢٣', undefined], // Arabic-Indic 123: Number(raw) → NaN (ASCII-only), reject
      ['१२३', undefined], // Devanagari 123: same ASCII-only parity
    ];
    for (const [raw, expected] of vector) {
      expect(coerceToType(raw, intp), raw).toBe(expected);
    }
  });

  it('parses booleans with the same truthy set as the server', () => {
    for (const t of ['1', 'true', 'TRUE', 'yes', 'on']) {
      expect(coerceToType(t, { type: 'boolean' })).toBe(true);
    }
    for (const f of ['0', 'false', 'off', 'nope', '']) {
      expect(coerceToType(f, { type: 'boolean' })).toBe(false);
    }
  });

  it('trims surrounding whitespace on booleans (server / number-branch parity)', () => {
    // The number branch already trims (String(raw).trim()), and the server's
    // boolean coerce strips via _parse_bool. A padded truthy value like ' true '
    // must coerce to true — otherwise the SPA shows a boolean option OFF while
    // the server (which strips) generates with it ON (a WYRD_DEFAULT_* parity
    // break, the exact divergence feature_flags exists to prevent).
    for (const t of [' true ', '\ttrue', 'yes ', ' 1 ', ' ON ']) {
      expect(coerceToType(t, { type: 'boolean' })).toBe(true);
    }
  });

  it('passes strings through unchanged', () => {
    expect(coerceToType('english', { type: 'string' })).toBe('english');
  });

  it('returns undefined for array fields (no scalar string into an array slot)', () => {
    expect(coerceToType('harsh', { type: 'array' })).toBeUndefined();
  });
});

describe('seedDefault (override > schema default)', () => {
  it('env default-override wins, coerced to the field type', () => {
    const cfg = { defaults: { count: '7' } };
    expect(seedDefault(cfg, 'count', { type: 'integer', default: 5 })).toBe(7);
  });

  it('falls back to the schema default when no override', () => {
    expect(seedDefault({ defaults: {} }, 'count', { type: 'integer', default: 5 })).toBe(5);
    expect(seedDefault(null, 'culture', { type: 'string', default: 'english' })).toBe('english');
  });

  it('returns undefined when neither override nor schema default exists', () => {
    expect(seedDefault({ defaults: {} }, 'novelty', { type: 'number' })).toBeUndefined();
  });

  it('a junk numeric override falls back to the schema default', () => {
    const cfg = { defaults: { count: 'abc' } };
    expect(seedDefault(cfg, 'count', { type: 'integer', default: 5 })).toBe(5);
  });
});

describe('snapEnumValue (wyrd-etvd: never preempt the config-default seed)', () => {
  const prop = ['english', 'welsh'];

  it('returns undefined for an UNSEEDED (undefined) value — the seed pass owns it', () => {
    // THE regression class: snapping undefined → schema default here clobbered
    // a WYRD_DEFAULT_<OPTION> override (the Advanced menu stuck on the schema
    // default despite the manifest serving the override).
    expect(snapEnumValue(undefined, prop, 'english')).toBeUndefined();
  });

  it('returns undefined (no change) for a value already in the options', () => {
    expect(snapEnumValue('welsh', prop, 'english')).toBeUndefined();
    expect(snapEnumValue('english', prop, 'english')).toBeUndefined();
  });

  it('snaps a DEFINED out-of-options value to the schema default when valid', () => {
    expect(snapEnumValue('banana', prop, 'english')).toBe('english');
  });

  it('snaps to the first option when the schema default is also invalid/absent', () => {
    expect(snapEnumValue('banana', ['a', 'b'], 'gone')).toBe('a');
    expect(snapEnumValue('banana', ['a', 'b'], undefined)).toBe('a');
  });

  it("preserves the '' no-filter sentinel when it is a valid option (era/stratum)", () => {
    // era/stratum dependent-select lists always start with '' (no filter), and
    // era's schema default is ''. '' is a real, valid value — it must NEVER be
    // snapped away (e.g. on a culture change). Load-bearing now that era is a
    // headline field.
    expect(snapEnumValue('', ['', 'medieval', 'modern'], '')).toBeUndefined();
  });
});

describe('initialFieldValue (wyrd-b6hd: store-seeded init — override > default > empty)', () => {
  it('uses the config.defaults override (e.g. culture=welsh)', () => {
    const prop = { type: 'string', enum: ['english', 'welsh'], default: 'english' };
    expect(initialFieldValue({ defaults: { culture: 'welsh' } }, 'culture', prop)).toBe('welsh');
  });

  it('falls back to the schema default when no override', () => {
    expect(initialFieldValue({ defaults: {} }, 'count', { type: 'integer', default: 5 })).toBe(5);
  });

  it('falls back to a type-appropriate empty when neither is set', () => {
    expect(initialFieldValue({ defaults: {} }, 'tags', { type: 'array' })).toEqual([]);
    expect(initialFieldValue({ defaults: {} }, 'x', { type: 'boolean' })).toBe(false);
    expect(initialFieldValue({ defaults: {} }, 'name', { type: 'string' })).toBe('');
  });

  it('never returns undefined — bound state is never empty at mount', () => {
    expect(initialFieldValue(null, 'whatever', { type: 'string' })).not.toBeUndefined();
    expect(initialFieldValue(null, 'novelty', { type: 'number' })).not.toBeUndefined();
  });
});

// wyrd-kqyf: the culture-agnostic era-default token. The env default can't
// name a literal stage (stages are per-culture), so 'present-day' means "this
// culture's present-day stage" — the LAST option of the chronologically-
// ordered per-culture list. These pin the two translation duties: seeding the
// token to a real option, and re-snapping to the NEW culture's present-day
// stage on culture switch (instead of clobbering the env default to '').
describe('snapDependentValue (present-day era token)', () => {
  const englishOptions = ['', 'old-english', 'middle-english', 'modern-english'];
  const welshOptions = ['', 'old-welsh', 'middle-welsh', 'welsh'];
  const eraProp = { type: 'string', default: '' };
  const tokenConfig = { all: false, flags: {}, defaults: { era: PRESENT_DAY_ERA } };

  it("translates a seeded token to the culture's present-day stage", () => {
    expect(snapDependentValue(PRESENT_DAY_ERA, englishOptions, eraProp, tokenConfig, 'era')).toBe(
      'modern-english',
    );
    expect(snapDependentValue(PRESENT_DAY_ERA, welshOptions, eraProp, tokenConfig, 'era')).toBe(
      'welsh',
    );
  });

  it('culture switch re-snaps an invalid stage to the NEW present-day stage when the config default is the token', () => {
    // english's modern-english isn't a welsh option → snap to welsh's present-day
    expect(snapDependentValue('modern-english', welshOptions, eraProp, tokenConfig, 'era')).toBe(
      'welsh',
    );
  });

  it('without a token config default, invalid values snap to the schema default (pre-kqyf behavior)', () => {
    const plainConfig = { all: false, flags: {}, defaults: {} };
    expect(snapDependentValue('modern-english', welshOptions, eraProp, plainConfig, 'era')).toBe(
      '',
    );
  });

  it('stays a pure snapEnumValue for non-token dependent selects (stratum parity)', () => {
    // Field.svelte routes EVERY dependent select through snapDependentValue;
    // without a token config default the present-day layer must be inert —
    // identical to snapEnumValue — so stratum's culture-switch snap can't be
    // corrupted by a future edit to the token branch.
    const stratumOptions = ['', 'native-welsh', 'latin-loan'];
    const stratumProp = { type: 'string', default: '' };
    const plainConfig = { all: false, flags: {}, defaults: {} };
    for (const value of [undefined, '', 'native-welsh', 'east-norse']) {
      expect(snapDependentValue(value, stratumOptions, stratumProp, plainConfig, 'stratum')).toBe(
        snapEnumValue(value, stratumOptions, stratumProp.default),
      );
    }
  });

  it('valid values and unseeded fields are left alone', () => {
    expect(
      snapDependentValue('middle-welsh', welshOptions, eraProp, tokenConfig, 'era'),
    ).toBeUndefined();
    expect(snapDependentValue(undefined, welshOptions, eraProp, tokenConfig, 'era')).toBeUndefined();
    // explicit Mixed Era ('') is a valid option — never overridden back to modern
    expect(snapDependentValue('', welshOptions, eraProp, tokenConfig, 'era')).toBeUndefined();
  });
});
