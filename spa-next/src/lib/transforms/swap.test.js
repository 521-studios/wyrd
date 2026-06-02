// wyrd-qc0g: the swap transform must make `usage` authoritative — dropping the
// stale auto-era-render fields on the swapped morpheme. Pins the HIGH fix for
// the col-3 grid: without it, the grid highlight + breakdown surface (which
// read `rendered || usage`) stay pinned to the pre-swap era form on
// era-rendered (WYRD_FF_ERA) names while the headline follows the swap.
import { describe, it, expect } from 'vitest';
import { swapTransform, renderName } from './swap.js';

const eraState = () => ({
  name: 'Stāntūn',
  morphemes_by_word: [
    [
      {
        usage: 'Stan-',
        rendered: 'Stān-',
        rendered_language: 'old-english',
        rendered_pron: { ipa: '/stɑːn/' },
      },
      {
        usage: '-tun',
        rendered: '-tūn',
        rendered_language: 'old-english',
        rendered_pron: { ipa: '/tuːn/' },
      },
    ],
  ],
});

describe('swapTransform.apply', () => {
  it('drops stale era fields on the swapped morpheme so usage is authoritative', async () => {
    const out = await swapTransform.apply(eraState(), {
      wordIndex: 0,
      morphemeIndex: 1,
      to: '-toun',
      language: 'middle-english',
    });
    const swapped = out.morphemes_by_word[0][1];
    expect(swapped.usage).toBe('-toun');
    expect(swapped.rendered).toBeUndefined();
    expect(swapped.rendered_language).toBeUndefined();
    expect(swapped.rendered_pron).toBeUndefined();
    expect(swapped._lang).toBe('middle-english');
    // the headline recomposes from the (now-modern) usages of the word
    expect(out.name).toBe('Stantoun');
  });

  it('leaves the non-swapped morpheme (incl. its era fields) untouched', async () => {
    const out = await swapTransform.apply(eraState(), {
      wordIndex: 0,
      morphemeIndex: 1,
      to: '-toun',
    });
    const other = out.morphemes_by_word[0][0];
    expect(other.usage).toBe('Stan-');
    expect(other.rendered).toBe('Stān-');
    expect(other.rendered_language).toBe('old-english');
  });

  it('a language-less swap clears any pin', async () => {
    const out = await swapTransform.apply(eraState(), {
      wordIndex: 0,
      morphemeIndex: 0,
      to: 'Stone-',
    });
    expect(out.morphemes_by_word[0][0]._lang).toBeUndefined();
  });
});

describe('renderName', () => {
  it('concatenates a compound word and title-cases it, stripping dashes', () => {
    expect(renderName([[{ usage: 'Stan-' }, { usage: '-tun' }]])).toBe('Stantun');
  });

  it('renders free particles as their own lowercase tokens', () => {
    expect(
      renderName([[{ usage: 'Henley' }], [{ usage: 'on' }], [{ usage: 'Thames' }]]),
    ).toBe('Henley on Thames');
  });

  it('capitalizes a leading particle and skips empty words', () => {
    expect(renderName([[{ usage: 'of' }], [{ usage: 'Avon' }]])).toBe('Of Avon');
    expect(renderName([[{ usage: '' }], [{ usage: 'Dale' }]])).toBe('Dale');
  });
});
