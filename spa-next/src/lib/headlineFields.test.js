// Unit tests for the headline/advanced field partition (wyrd-hcmc) +
// the curated-order render contract (wyrd-etvd: 'era' above the fold,
// directly under 'culture').
import { describe, it, expect } from 'vitest';
import { partitionFields, HEADLINE_FIELDS } from './headlineFields.js';

function schemaOf(...keys) {
  return { properties: Object.fromEntries(keys.map((k) => [k, { type: 'string' }])) };
}

describe('HEADLINE_FIELDS', () => {
  it("surfaces 'era' in kenning's headline, right after 'culture'", () => {
    expect(HEADLINE_FIELDS.kenning).toEqual(['culture', 'era', 'count', 'mood']);
  });
});

describe('partitionFields', () => {
  it('headline follows the curated order (era under culture), NOT schema order', () => {
    // Schema property order is deliberately scrambled vs the curated order
    // to prove the headline list is ordered by HEADLINE_FIELDS, not the schema.
    const schema = schemaOf('count', 'culture', 'novelty', 'era', 'mood', 'stratum');
    const { headline, advanced } = partitionFields('kenning', schema);
    expect(headline.map(([k]) => k)).toEqual(['culture', 'era', 'count', 'mood']);
    // era moved to the headline → it's no longer in advanced; advanced keeps
    // the schema's own property order.
    expect(advanced.map(([k]) => k)).toEqual(['novelty', 'stratum']);
  });

  it('skips a curated headline key the schema does not define', () => {
    const schema = schemaOf('culture', 'count'); // no era, no mood
    expect(partitionFields('kenning', schema).headline.map(([k]) => k)).toEqual([
      'culture',
      'count',
    ]);
  });

  it("drops HIDDEN_FIELDS ('seed') before partitioning", () => {
    const schema = schemaOf('culture', 'seed', 'count');
    const { headline, advanced } = partitionFields('kenning', schema);
    const all = [...headline, ...advanced].map(([k]) => k);
    expect(all).not.toContain('seed');
  });

  it('falls back to all-headline (schema order) for a generator not in the map', () => {
    const schema = schemaOf('a', 'b');
    const { headline, advanced } = partitionFields('unknown-gen', schema);
    expect(headline.map(([k]) => k)).toEqual(['a', 'b']);
    expect(advanced).toEqual([]);
  });
});
