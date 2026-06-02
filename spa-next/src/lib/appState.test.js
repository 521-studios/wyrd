// wyrd-b6hd: the store owns field initialization — ensureParams seeds every
// non-hidden field's default from the manifest schema (config override →
// schema default → type-empty), BEFORE any Field renders. These pin that
// contract (the structural fix for the wyrd-etvd seed-vs-bind race).
import { describe, it, expect, beforeEach } from 'vitest';
import { appState } from './appState.svelte.js';

const SCHEMA = {
  properties: {
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
