// wyrd-y0lx: the regenerate-morpheme transform must splice the server's
// replacement morpheme following the Swap convention — `usage` becomes the
// live (native) surface grafted onto the slot's placement dashes, the
// auto-era-render fields are dropped, `_lang` pins the rendered language,
// and the server's `active_form_id` survives so the grid highlights the
// live cell. Held morphemes pass through untouched.
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api.js', () => ({
  regenerateMorpheme: vi.fn(),
}));

import { regenerateMorpheme } from '../api.js';
import { regenerateTransform } from './regenerate.js';

const state = () => ({
  name: 'Stāntūn',
  morphemes_by_word: [
    [
      { usage: 'Stan-', rendered: 'Stān', rendered_language: 'old-english' },
      { usage: '-tun', rendered: '-tūn', rendered_language: 'old-english' },
    ],
  ],
});

const serverReplacement = (overrides = {}) => ({
  results: [
    {
      morphemes_by_word: [
        [
          { usage: 'Stan-' },
          {
            usage: '-ham',
            rendered: 'hām',
            rendered_language: 'old-english',
            rendered_pron: { ipa: '/hɑːm/' },
            tags: ['architecture'],
            meanings: ['homestead'],
            era_grid: [{ family: 'english', stages: [] }],
            active_form_id: 'old-english:hām',
            ...overrides,
          },
        ],
      ],
    },
  ],
});

beforeEach(() => {
  regenerateMorpheme.mockReset();
});

describe('regenerateTransform.apply', () => {
  it('splices the replacement with the native surface as the live usage', async () => {
    regenerateMorpheme.mockResolvedValue(serverReplacement());
    const out = await regenerateTransform.apply(state(), {
      wordIndex: 0,
      morphemeIndex: 1,
      seed: 7,
      context: { culture: 'english' },
    });
    const fresh = out.morphemes_by_word[0][1];
    // native form grafted onto the slot's post-position dash + lowercase
    expect(fresh.usage).toBe('-hām');
    expect(fresh.rendered).toBeUndefined();
    expect(fresh.rendered_language).toBeUndefined();
    expect(fresh.rendered_pron).toBeUndefined();
    expect(fresh._lang).toBe('old-english');
    expect(fresh.active_form_id).toBe('old-english:hām');
    expect(fresh.tags).toEqual(['architecture']);
    // held morpheme untouched (same object semantics as Swap)
    expect(out.morphemes_by_word[0][0]).toEqual(state().morphemes_by_word[0][0]);
    expect(out.name).toBe('Stanhām');
  });

  it('sends the current state + baked seed/context to the endpoint', async () => {
    regenerateMorpheme.mockResolvedValue(serverReplacement());
    const s = state();
    await regenerateTransform.apply(s, {
      wordIndex: 0,
      morphemeIndex: 1,
      seed: 99,
      context: { culture: 'english', tags: ['water'] },
    });
    expect(regenerateMorpheme).toHaveBeenCalledWith({
      culture: 'english',
      tags: ['water'],
      name: 'Stāntūn',
      words: s.morphemes_by_word,
      word_index: 0,
      morpheme_index: 1,
      seed: 99,
    });
  });

  it('falls back to the modern usage when the server sent no native render', async () => {
    regenerateMorpheme.mockResolvedValue(
      serverReplacement({ rendered: undefined, rendered_language: undefined }),
    );
    const out = await regenerateTransform.apply(state(), {
      wordIndex: 0,
      morphemeIndex: 1,
      seed: 7,
      context: {},
    });
    const fresh = out.morphemes_by_word[0][1];
    expect(fresh.usage).toBe('-ham');
    expect(fresh._lang).toBeUndefined();
  });

  it('fails loudly on a stale slot (bounds check, mirrors Swap)', async () => {
    await expect(
      regenerateTransform.apply(state(), { wordIndex: 3, morphemeIndex: 0, seed: 1 }),
    ).rejects.toThrow('word 3 not in current state');
    await expect(
      regenerateTransform.apply(state(), { wordIndex: 0, morphemeIndex: 9, seed: 1 }),
    ).rejects.toThrow('morpheme 9 not in word 0');
    expect(regenerateMorpheme).not.toHaveBeenCalled();
  });

  it('fails loudly when the response carries no replacement', async () => {
    regenerateMorpheme.mockResolvedValue({ results: [{ morphemes_by_word: [[]] }] });
    await expect(
      regenerateTransform.apply(state(), { wordIndex: 0, morphemeIndex: 1, seed: 1 }),
    ).rejects.toThrow('no replacement morpheme');
  });
});
