// wyrd-51rv: round-trip regression tests for the share-link serialization
// seam. encodeWorkspace → decodeWorkspace is a user-DATA path (a shared
// workspace URL); the latent regression class is falsy-loss — a later
// truthiness gate (`if (seed)` instead of `if (seed !== undefined)`) silently
// dropping seed:0 / count:0 / cohesion:false on restore. These pin the current
// (correct-but-previously-untested) behavior so that regression fails loudly.
import { describe, it, expect, vi } from 'vitest';
import { encodeWorkspace, decodeWorkspace } from './shareLink.js';

// Mirror the module's private base64url encoder so we can craft raw inputs
// (e.g. a wrong-schema payload) without going through encodeWorkspace, which
// always stamps the current schema_version.
function b64url(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// A workspace carrying the zero/false/empty values the regression class
// threatens. Only seed:0 / count:0 / novelty:0 / cohesion:false / era:'' are
// strictly falsy (what a truthiness gate would drop); packs:[] / name:'' etc.
// are empty-but-truthy, pinned alongside for completeness.
const FALSY_WORKSPACE = {
  generator: 'kenning',
  params: { seed: 0, count: 0, novelty: 0, cohesion: false, packs: [], era: '' },
  original: { name: '', morphemes_by_word: [], explanation: '' },
  pipeline: [],
};

describe('shareLink encode → decode round-trip', () => {
  it('round-trips every field of a normal workspace', () => {
    const ws = {
      generator: 'kenning',
      params: { culture: 'welsh', count: 5, novelty: 0.3, packs: ['a', 'b'] },
      original: { name: 'Bryntir', morphemes_by_word: [['bryn', 'tir']], explanation: 'hill-land' },
      pipeline: [{ kind: 'rewind', params: { era: 'oe-late' } }],
    };
    const decoded = decodeWorkspace(encodeWorkspace(ws));
    expect(decoded.schema_version).toBe(1);
    expect(decoded.generator).toBe(ws.generator);
    expect(decoded.params).toEqual(ws.params);
    expect(decoded.original).toEqual(ws.original);
    expect(decoded.pipeline).toEqual(ws.pipeline);
  });

  it('preserves zero/false/empty values (seed:0, cohesion:false, explanation:"", packs:[]) across the round-trip', () => {
    const decoded = decodeWorkspace(encodeWorkspace(FALSY_WORKSPACE));
    expect(decoded.params.seed).toBe(0);
    expect(decoded.params.count).toBe(0);
    expect(decoded.params.novelty).toBe(0);
    expect(decoded.params.cohesion).toBe(false);
    expect(decoded.params.era).toBe('');
    expect(decoded.params.packs).toEqual([]);
    expect(decoded.original.name).toBe('');
    expect(decoded.original.explanation).toBe('');
    // The whole params object must be identical — no key silently dropped.
    expect(decoded.params).toEqual(FALSY_WORKSPACE.params);
  });

  it('round-trips non-ASCII via TextEncoder/TextDecoder (OE macron, Welsh)', () => {
    const ws = {
      generator: 'kenning',
      params: {},
      original: { name: 'Hāmtūn · Llyn Ffynnon', morphemes_by_word: [], explanation: 'macron ā/ū' },
      pipeline: [],
    };
    const decoded = decodeWorkspace(encodeWorkspace(ws));
    expect(decoded.original.name).toBe('Hāmtūn · Llyn Ffynnon');
    expect(decoded.original.explanation).toBe('macron ā/ū');
  });

  it('restores base64url padding for payloads of varied length', () => {
    // Vary the payload length so the base64 lands on each of the 3 padding
    // residues (len % 4 ∈ {2, 3, 0-after-strip}); every one must decode back.
    for (let n = 0; n < 6; n += 1) {
      const ws = { generator: 'kenning', params: { tag: 'x'.repeat(n) }, original: {}, pipeline: [] };
      const encoded = encodeWorkspace(ws);
      expect(encoded).not.toMatch(/=/); // padding stripped in the URL form
      expect(decodeWorkspace(encoded).params.tag).toBe('x'.repeat(n));
    }
  });

  it('returns null (not throw) on a schema_version mismatch, and warns', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const future = b64url(JSON.stringify({ schema_version: 2, generator: 'kenning' }));
    expect(decodeWorkspace(future)).toBeNull();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('returns null (never throws) on malformed / empty input', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(() => decodeWorkspace('%%%not-base64%%%')).not.toThrow();
    expect(decodeWorkspace('%%%not-base64%%%')).toBeNull();
    expect(decodeWorkspace(b64url('{not json'))).toBeNull(); // valid b64, bad JSON
    expect(decodeWorkspace('')).toBeNull();
    expect(decodeWorkspace(null)).toBeNull();
    expect(decodeWorkspace(undefined)).toBeNull();
    warn.mockRestore();
  });
});
