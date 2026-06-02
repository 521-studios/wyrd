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
    expect(envify('scoring_mode')).toBe('SCORING_MODE');
    expect(envify('priors-path')).toBe('PRIORS_PATH');
  });

  it('namespaced flags resolve through the same suffix the server emits', () => {
    expect(flagOn({ all: false, flags: { CULTURE_WELSH: true } }, 'culture.welsh')).toBe(true);
  });
});

describe('fieldFlag / fieldEnabled (grouping)', () => {
  it('maps 1:1 by default', () => {
    expect(fieldFlag('novelty')).toBe('novelty');
  });

  it('groups the vector axis-weight knobs under scoring_mode', () => {
    for (const w of ['phonological_weight', 'semantic_weight', 'position_weight', 'baseline_weight']) {
      expect(fieldFlag(w)).toBe('scoring_mode');
    }
    const cfg = { all: false, flags: { SCORING_MODE: true } };
    expect(fieldEnabled(cfg, 'phonological_weight')).toBe(true);
    expect(fieldEnabled(cfg, 'novelty')).toBe(false);
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

  it('parses booleans with the same truthy set as the server', () => {
    for (const t of ['1', 'true', 'TRUE', 'yes', 'on']) {
      expect(coerceToType(t, { type: 'boolean' })).toBe(true);
    }
    for (const f of ['0', 'false', 'off', 'nope', '']) {
      expect(coerceToType(f, { type: 'boolean' })).toBe(false);
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
  const prop = ['proportions', 'vector'];

  it('returns undefined for an UNSEEDED (undefined) value — the seed pass owns it', () => {
    // THE regression: snapping undefined → schema default here clobbered the
    // WYRD_DEFAULT_SCORING_MODE=vector override (Advanced menu stuck on
    // Proportions despite the manifest serving vector).
    expect(snapEnumValue(undefined, prop, 'proportions')).toBeUndefined();
  });

  it('returns undefined (no change) for a value already in the options', () => {
    expect(snapEnumValue('vector', prop, 'proportions')).toBeUndefined();
    expect(snapEnumValue('proportions', prop, 'proportions')).toBeUndefined();
  });

  it('snaps a DEFINED out-of-options value to the schema default when valid', () => {
    expect(snapEnumValue('banana', prop, 'proportions')).toBe('proportions');
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
  it('uses the config.defaults override (the scoring_mode=vector case)', () => {
    const prop = { type: 'string', enum: ['proportions', 'vector'], default: 'proportions' };
    expect(initialFieldValue({ defaults: { scoring_mode: 'vector' } }, 'scoring_mode', prop)).toBe(
      'vector',
    );
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
