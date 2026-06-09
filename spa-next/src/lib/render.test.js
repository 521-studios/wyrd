// wyrd-24s6 (D38): unit tests for the native/modern Output-column predicates.
import { describe, it, expect } from 'vitest';
import {
  nativeSurface,
  modernSurface,
  showMorphemeModern,
  showModernCompanion,
} from './render.js';

describe('nativeSurface', () => {
  it('prefers the backend native `rendered` form', () => {
    expect(nativeSurface({ rendered: 'cumb', usage: '-combe' })).toBe('cumb');
  });

  it('falls back to the accented original-script surface when no rendered (wyrd-de5t)', () => {
    // accentedUsage upgrades `-by` → `-bȳ` via renderings.old_english.bȳ.original_script.
    const m = {
      usage: '-by',
      renderings: { old_english: { by: { original_script: 'bȳ' } } },
    };
    expect(nativeSurface(m)).toBe('-bȳ');
  });

  it('falls back to the plain usage when neither rendered nor accent data exists', () => {
    expect(nativeSurface({ usage: '-ton' })).toBe('-ton');
    expect(nativeSurface({})).toBe('');
    expect(nativeSurface(null)).toBe('');
  });
});

describe('modernSurface', () => {
  it('is the plain usage bucket key', () => {
    expect(modernSurface({ rendered: 'cumb', usage: '-combe' })).toBe('-combe');
    expect(modernSurface({})).toBe('');
    expect(modernSurface(null)).toBe('');
  });
});

describe('showMorphemeModern', () => {
  it('shows the modern reflex only when it differs from the native surface', () => {
    expect(showMorphemeModern({ rendered: 'cumb', usage: '-combe' })).toBe(true);
  });

  it('hides it when native == modern (no distinct native form)', () => {
    // No rendered, no accent data → native falls back to usage → equal.
    expect(showMorphemeModern({ usage: '-ton' })).toBe(false);
    // rendered equals usage.
    expect(showMorphemeModern({ rendered: 'green', usage: 'green' })).toBe(false);
  });
});

describe('showModernCompanion', () => {
  it('shows the companion when result_modern differs from native result', () => {
    expect(showModernCompanion({ result: 'Bolingcumb', result_modern: 'Boingcombe' })).toBe(true);
  });

  it('hides it when native == modern (plain / force-modern roll)', () => {
    expect(showModernCompanion({ result: 'Bolesby', result_modern: 'Bolesby' })).toBe(false);
  });

  it('hides it when result_modern is absent or empty', () => {
    expect(showModernCompanion({ result: 'X' })).toBe(false);
    expect(showModernCompanion({ result: 'X', result_modern: '' })).toBe(false);
    expect(showModernCompanion(null)).toBe(false);
  });
});
