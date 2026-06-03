// wyrd-rogd.2: the Configure era dropdown now labels its stage options via
// languageLabel (Field.svelte x-option-language), so pin the user-visible
// mappings here — including the non-obvious modern-english → 'Modern' alias.
import { describe, it, expect } from 'vitest';
import { languageLabel } from './languageLabels.js';

describe('languageLabel', () => {
  it('maps the English era stages to readable labels', () => {
    expect(languageLabel('old-english')).toBe('Old English');
    expect(languageLabel('middle-english')).toBe('Middle English');
    expect(languageLabel('modern-english')).toBe('Modern'); // folded alias
  });

  it('maps non-English stages', () => {
    expect(languageLabel('old-welsh')).toBe('Old Welsh');
    expect(languageLabel('welsh')).toBe('Welsh');
  });

  it('title-cases an unknown hyphenated tag as a fallback', () => {
    expect(languageLabel('some-unknown-lang')).toBe('Some Unknown Lang');
  });

  it('returns empty string for nullish/empty input', () => {
    expect(languageLabel('')).toBe('');
    expect(languageLabel(undefined)).toBe('');
    expect(languageLabel(null)).toBe('');
  });
});
