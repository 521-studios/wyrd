// wyrd-tbcj.1: pins the composer selection reducers — the single- vs
// multi-select distinction is the whole point of the ticket (moods are
// single-select to match the runtime's one-mood-per-name overlay, tags
// stay multi-select), so guard it directly.
import { describe, it, expect } from 'vitest';
import { selectSingle, toggleMulti, removeValue } from './composerSelection.js';

describe('selectSingle (moods — radio-style)', () => {
  it('selecting from empty yields a 1-element array', () => {
    expect(selectSingle([], 'harsh')).toEqual(['harsh']);
  });

  it('selecting a different value REPLACES the current one (never accumulates)', () => {
    expect(selectSingle(['harsh'], 'soft')).toEqual(['soft']);
  });

  it('selecting the already-active value clears it (toggle off)', () => {
    expect(selectSingle(['harsh'], 'harsh')).toEqual([]);
  });

  it('replaces even when the working list somehow holds multiple (legacy params)', () => {
    // A saved/shared config from the old multi-select era could carry >1
    // mood; the first click collapses it to a single selection.
    expect(selectSingle(['harsh', 'soft'], 'bright')).toEqual(['bright']);
  });

  it('never mutates the input array', () => {
    const input = ['harsh'];
    selectSingle(input, 'soft');
    expect(input).toEqual(['harsh']);
  });
});

describe('toggleMulti (tags — unchanged multi-select)', () => {
  it('adds an absent value', () => {
    expect(toggleMulti(['a'], 'b')).toEqual(['a', 'b']);
  });

  it('removes a present value', () => {
    expect(toggleMulti(['a', 'b'], 'a')).toEqual(['b']);
  });

  it('preserves multiple selections (accumulates, unlike selectSingle)', () => {
    expect(toggleMulti(['a', 'b'], 'c')).toEqual(['a', 'b', 'c']);
  });

  it('never mutates the input array', () => {
    const input = ['a'];
    toggleMulti(input, 'b');
    expect(input).toEqual(['a']);
  });
});

describe('removeValue (chip × — shared)', () => {
  it('removes the value', () => {
    expect(removeValue(['a', 'b'], 'a')).toEqual(['b']);
  });

  it('is a no-op for an absent value', () => {
    expect(removeValue(['a'], 'b')).toEqual(['a']);
  });
});
