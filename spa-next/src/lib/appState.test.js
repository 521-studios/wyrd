// wyrd-b6hd: the store owns field initialization — ensureParams seeds every
// non-hidden field's default from the manifest schema (config override →
// schema default → type-empty), BEFORE any Field renders. These pin that
// contract (the structural fix for the wyrd-etvd seed-vs-bind race).
import { describe, it, expect, beforeEach } from 'vitest';
import { appState } from './appState.svelte.js';

const SCHEMA = {
  properties: {
    culture: { type: 'string', default: 'english' }, // ALWAYS_SENT — even at default
    scoring_mode: { type: 'string', enum: ['proportions', 'vector'], default: 'proportions' },
    count: { type: 'integer', default: 5 },
    tags: { type: 'array' },
    seed: { type: 'integer' }, // HIDDEN_FIELDS — must NOT be seeded
  },
};

function setManifest({ defaults = {} } = {}) {
  appState.manifest = {
    generators: [{ name: 'gen', input_schema: SCHEMA }],
    config: { all: false, flags: {}, defaults },
  };
}

describe('appState.ensureParams (store-owned seeding)', () => {
  beforeEach(() => {
    appState.manifest = null;
    appState.paramsByGenerator = {};
  });

  it('seeds every non-hidden field: config override > schema default > type-empty', () => {
    setManifest({ defaults: { scoring_mode: 'vector' } });
    appState.ensureParams('gen');
    const p = appState.paramsByGenerator['gen'];
    expect(p.scoring_mode).toBe('vector'); // config.defaults override wins
    expect(p.count).toBe(5); // schema default
    expect(p.tags).toEqual([]); // type-empty
    expect('seed' in p).toBe(false); // HIDDEN_FIELDS dropped
  });

  it('does NOT lock-in an empty bag when the manifest is not loaded yet', () => {
    // share-link can set selectedGeneratorName before the manifest fetch
    // resolves; ensureParams must no-op so a later call (post-manifest) seeds.
    appState.manifest = null;
    appState.ensureParams('gen');
    expect(appState.paramsByGenerator['gen']).toBeUndefined();
    // ...and once the manifest lands, it seeds.
    setManifest({ defaults: { scoring_mode: 'vector' } });
    appState.ensureParams('gen');
    expect(appState.paramsByGenerator['gen'].scoring_mode).toBe('vector');
  });

  it('backfills only MISSING fields — present (restored) values are preserved', () => {
    setManifest();
    // A restored bag (share-link / saved) missing 'tags' — e.g. a stale
    // cross-version bookmark from before the store seeded every field.
    appState.paramsByGenerator['gen'] = { scoring_mode: 'vector', count: 9 };
    appState.ensureParams('gen');
    const p = appState.paramsByGenerator['gen'];
    expect(p.scoring_mode).toBe('vector'); // preserved (NOT clobbered to schema default)
    expect(p.count).toBe(9); // preserved
    expect(p.tags).toEqual([]); // backfilled — was missing, so no <select> binds undefined
    expect('seed' in p).toBe(false); // HIDDEN still dropped
  });
});

describe('appState.changedParams (send only non-default params)', () => {
  beforeEach(() => {
    appState.manifest = null;
    appState.paramsByGenerator = {};
  });

  it('omits default fields but ALWAYS sends culture + count (authoritative selectors)', () => {
    setManifest({ defaults: { scoring_mode: 'vector' } });
    appState.ensureParams('gen');
    // user touched nothing → culture + count still go out; server owns the rest
    expect(appState.changedParams('gen')).toEqual({ culture: 'english', count: 5 });
  });

  it('includes the changed fields plus the always-sent culture + count', () => {
    setManifest({ defaults: { scoring_mode: 'vector' } });
    appState.ensureParams('gen');
    appState.paramsByGenerator['gen'].count = 9; // changed from default 5
    appState.paramsByGenerator['gen'].tags = ['water']; // changed from []
    const out = appState.changedParams('gen');
    expect(out).toEqual({ culture: 'english', count: 9, tags: ['water'] });
    expect('scoring_mode' in out).toBe(false); // still == config default 'vector'
  });

  it('treats a value equal to the config.defaults override as unchanged', () => {
    setManifest({ defaults: { scoring_mode: 'vector' } });
    appState.ensureParams('gen');
    appState.paramsByGenerator['gen'].scoring_mode = 'vector'; // explicitly = override
    expect('scoring_mode' in appState.changedParams('gen')).toBe(false);
    // but flipping to the OTHER value is a real change
    appState.paramsByGenerator['gen'].scoring_mode = 'proportions';
    expect(appState.changedParams('gen')).toEqual({
      culture: 'english',
      count: 5,
      scoring_mode: 'proportions',
    });
  });

  it('sends culture + count even when they equal their defaults', () => {
    setManifest();
    appState.ensureParams('gen');
    const out = appState.changedParams('gen');
    expect(out.culture).toBe('english');
    expect(out.count).toBe(5);
  });

  it('never emits HIDDEN_FIELDS (seed)', () => {
    setManifest();
    appState.ensureParams('gen');
    appState.paramsByGenerator['gen'].seed = 42;
    expect('seed' in appState.changedParams('gen')).toBe(false);
  });

  it('returns {} when the generator has no seeded params bag', () => {
    setManifest();
    // no ensureParams → paramsByGenerator['gen'] is undefined
    expect(appState.changedParams('gen')).toEqual({});
  });

  it('keeps a param that has no schema entry (unknown default → forward it)', () => {
    setManifest();
    appState.ensureParams('gen');
    appState.paramsByGenerator['gen'].mystery = 'x'; // not in SCHEMA.properties
    expect(appState.changedParams('gen').mystery).toBe('x');
  });
});
