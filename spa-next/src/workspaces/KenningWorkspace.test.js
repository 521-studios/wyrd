// wyrd-f5if: focus trap + focus restoration on the mobile drawer (col 1) and
// sheet (col 3). jsdom (no Playwright): the trap logic lives in a $effect keyed
// on col1Open/col3Open, so we drive it via appState (isMobileViewport +
// currentResultIndex auto-opens col 3) and the exported openMenu() (col 1).
import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import KenningWorkspace from './KenningWorkspace.svelte';
import { appState } from '../lib/appState.svelte.js';

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function seed() {
  appState.results = [
    { result: 'Dūnton', morphemes_by_word: [[{ usage: 'dūn-', _wordIndex: 0, _morphemeIndex: 0 }]] },
  ];
  appState.currentResultIndex = null;
  appState.resultsGenerator = 'kenning';
  appState.isMobileViewport = true;
}

// Flush the reactive graph AND the queueMicrotask the effect uses to focus the
// panel's close button.
async function settle() {
  for (let i = 0; i < 4; i += 1) await tick();
  await Promise.resolve();
}

let opener;

beforeEach(() => {
  seed();
  opener = document.createElement('button');
  opener.textContent = 'opener';
  document.body.appendChild(opener);
  opener.focus();
});

afterEach(() => {
  appState.isMobileViewport = false;
  appState.currentResultIndex = null;
  opener?.remove();
});

describe('KenningWorkspace mobile focus trap (wyrd-f5if)', () => {
  it('col 3 sheet: focuses its close button on open, restores focus to the opener on close', async () => {
    const { container } = render(KenningWorkspace);
    expect(document.activeElement).toBe(opener);

    appState.currentResultIndex = 0; // auto-opens the sheet
    await settle();

    const col3 = container.querySelector('.col-3');
    const closeBtn = col3.querySelector('.close-btn');
    expect(document.activeElement).toBe(closeBtn);

    await fireEvent.click(closeBtn); // closes col3
    await settle();
    expect(document.activeElement).toBe(opener);
  });

  it('col 3 sheet: Tab wraps within the panel (trap)', async () => {
    const { container } = render(KenningWorkspace);
    appState.currentResultIndex = 0;
    await settle();

    const col3 = container.querySelector('.col-3');
    const focusables = col3.querySelectorAll(FOCUSABLE);
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    last.focus();
    await fireEvent.keyDown(document, { key: 'Tab' });
    expect(document.activeElement).toBe(first);

    first.focus();
    await fireEvent.keyDown(document, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(last);
  });

  it('Esc closes the open panel and restores focus', async () => {
    render(KenningWorkspace);
    appState.currentResultIndex = 0;
    await settle();
    expect(document.activeElement).not.toBe(opener);

    await fireEvent.keyDown(document, { key: 'Escape' });
    await settle();
    expect(document.activeElement).toBe(opener);
  });

  it('both panels open: the trap scopes to the col 3 sheet (foreground)', async () => {
    const result = render(KenningWorkspace);
    result.component.openMenu(); // col 1 drawer
    appState.currentResultIndex = 0; // col 3 sheet
    await settle();

    const col1 = result.container.querySelector('.col-1');
    const col3 = result.container.querySelector('.col-3');

    // Focus a col-1 element, then Tab: the trap (scoped to col3) pulls focus
    // into the sheet rather than cycling within the drawer.
    col1.querySelector('.close-btn').focus();
    await fireEvent.keyDown(document, { key: 'Tab' });
    expect(col3.contains(document.activeElement)).toBe(true);
  });

  it('does not trap or scroll-lock when no panel is open', async () => {
    render(KenningWorkspace);
    await settle();
    expect(document.activeElement).toBe(opener);
    expect(document.body.style.overflow).not.toBe('hidden');
  });

  it('col 1 drawer: focuses its close button on open and restores focus on close', async () => {
    const result = render(KenningWorkspace);
    result.component.openMenu(); // opens the drawer
    await settle();

    const col1 = result.container.querySelector('.col-1');
    const closeBtn = col1.querySelector('.close-btn');
    expect(document.activeElement).toBe(closeBtn);

    await fireEvent.click(closeBtn); // closes col1
    await settle();
    expect(document.activeElement).toBe(opener);
  });

  it('tolerates the opener being detached before close (no throw)', async () => {
    render(KenningWorkspace);
    appState.currentResultIndex = 0; // open the sheet (opener was the trigger)
    await settle();

    opener.remove(); // opener leaves the DOM while the panel is open
    opener = null; // afterEach guard

    // Closing must not throw even though triggerBefore.focus() now targets a
    // detached node (the try/catch in the effect cleanup).
    await fireEvent.keyDown(document, { key: 'Escape' });
    await settle();
    expect(document.body.style.overflow).not.toBe('hidden'); // cleanup completed
  });
});
