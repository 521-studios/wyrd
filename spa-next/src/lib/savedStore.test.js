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
