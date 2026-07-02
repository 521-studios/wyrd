// wyrd-34tn: importJSON's contract — it must ALWAYS return
// { added, skipped, error } and never throw, so the Import UI can surface a
// failure instead of crashing. A JSON literal that parses to a non-object
// (`null` most notably) used to throw TypeError on `null.schema_version`.
import { describe, it, expect, beforeEach } from 'vitest';
import { savedStore } from './savedStore.svelte.js';

beforeEach(() => {
  localStorage.clear();
  savedStore.entries = [];
});

describe('savedStore.importJSON contract (always {added, skipped, error}, never throws)', () => {
  it('returns a clean error for a JSON literal that parses to null — does not throw', () => {
    // 'null' is valid JSON → parsed === null → the old `null.schema_version`
    // threw TypeError past the {error} contract (a truncated/garbage export of
    // literally `null` crashed the importer instead of reporting an error).
    expect(() => savedStore.importJSON('null')).not.toThrow();
    expect(savedStore.importJSON('null')).toEqual({
      added: 0,
      skipped: 0,
      error: expect.stringContaining('expected a JSON object'),
    });
  });

  it('returns an error (not a throw) for non-object JSON: number / string / array / boolean', () => {
    for (const j of ['42', '"x"', '[]', 'true']) {
      let r;
      expect(() => {
        r = savedStore.importJSON(j);
      }, j).not.toThrow();
      expect(r.added, j).toBe(0);
      expect(r.error, j).toBeTruthy();
    }
  });

  it('returns a parse error for malformed JSON', () => {
    expect(savedStore.importJSON('{not json').error).toContain('parse failed');
  });

  it('rejects a wrong schema_version distinctly (still an object, just unsupported)', () => {
    expect(savedStore.importJSON('{"schema_version":2,"entries":[]}').error).toContain(
      'schema_version',
    );
  });

  it('still imports a valid export object unchanged', () => {
    const json = JSON.stringify({
      schema_version: 1,
      entries: [{ id: 'a1', original: { name: 'Tonby' }, generator: 'kenning' }],
    });
    expect(savedStore.importJSON(json)).toEqual({ added: 1, skipped: 0, error: null });
    expect(savedStore.get('a1')?.original.name).toBe('Tonby');
  });
});

// wyrd-51rv: save → load round-trip on a user-DATA path. add() deep-clones via
// JSON round-trip and write-throughs to localStorage; a reload reads it back
// via loadFromStorage. The regression class is falsy-loss — a truthiness gate
// (`if (seed)` instead of `if (seed !== undefined)`) silently dropping seed:0 /
// count:0 / cohesion:false on save or restore. Pin that every zero/false/empty
// field survives BOTH the in-memory clone (get) AND the on-disk JSON (what a
// reload loads). (Only 0/false/'' are strictly falsy; packs:[]/pipeline:[] are
// empty-but-truthy, pinned alongside for completeness.)
describe('savedStore save → load round-trip preserves zero/false/empty fields', () => {
  const FALSY_PAYLOAD = {
    generator: 'kenning',
    params: { seed: 0, count: 0, novelty: 0, cohesion: false, packs: [], era: '' },
    original: { name: 'Ton', morphemes_by_word: [], explanation: '' },
    pipeline: [],
  };

  it('add() then get() returns every zero/false/empty field intact (in-memory deep-clone)', () => {
    const { id, error } = savedStore.add(FALSY_PAYLOAD);
    expect(error).toBeNull();
    const e = savedStore.get(id);
    expect(e.params).toEqual(FALSY_PAYLOAD.params);
    expect(e.params.seed).toBe(0);
    expect(e.params.count).toBe(0);
    expect(e.params.cohesion).toBe(false);
    expect(e.params.packs).toEqual([]);
    expect(e.params.era).toBe('');
    expect(e.original.explanation).toBe('');
    expect(e.pipeline).toEqual([]);
  });

  it('persists the zero/false/empty fields to localStorage (what a reload would loadFromStorage)', () => {
    const { id } = savedStore.add(FALSY_PAYLOAD);
    const stored = JSON.parse(localStorage.getItem('wyrd.saved.v1'));
    expect(stored.schema_version).toBe(1);
    const persisted = stored.entries.find((e) => e.id === id);
    expect(persisted.params).toEqual(FALSY_PAYLOAD.params);
    expect(persisted.params.seed).toBe(0);
    expect(persisted.params.cohesion).toBe(false);
    expect(persisted.original.explanation).toBe('');
  });
});
