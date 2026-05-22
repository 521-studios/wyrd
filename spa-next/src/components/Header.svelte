<script>
  // wyrd-14hn + wyrd-8jjx: top-of-page chrome. wyrd-14hn lifted
  // picker + seed + Roll into here; wyrd-8jjx adds Save / Share /
  // Saved (N) to the .right slot. All universal controls (work
  // across every workspace) live in the header now; workspace-
  // specific UI lives in the per-workspace component.
  //
  // Design rationale: dark-minimal + a single distinctive moment.
  // The wynn rune ᚹ (U+16B9, Anglo-Saxon ancestor of W) replaces
  // generic logo-mark chrome; reinforces the OE/etymological story
  // wyrd tells. Everything else stays in the existing monospace-
  // data idiom (same as MorphemeCard's tables, the existing column
  // section heads) so the header reads as the same surface as the
  // columns below it.
  import { appState } from '../lib/appState.svelte.js';
  import { savedStore } from '../lib/savedStore.svelte.js';
  import { rollCurrent } from '../lib/roll.js';
  import { encodeWorkspace, buildShareUrl } from '../lib/shareLink.js';
  import { currentWorkspaceSnapshot } from '../lib/workspaceSnapshot.js';

  let { onMenuToggle = null, isMobileViewport = false } = $props();

  // Save / Share need a current result to operate on. Header
  // buttons disable when there's nothing to save (no roll done
  // yet, or no result selected).
  let canSaveOrShare = $derived(appState.currentResult !== null);

  let savedCount = $derived(savedStore.entries.length);

  let toastNote = $state('');
  function showToast(msg, ms = 2500) {
    toastNote = msg;
    setTimeout(() => {
      if (toastNote === msg) toastNote = '';
    }, ms);
  }

  function saveCurrent() {
    const payload = currentWorkspaceSnapshot();
    if (!payload) return;
    // wyrd-8jjx round 2 (Gemini MED): dropped the dedupe check.
    // A user may legitimately want multiple saves of the same name
    // with different seed / params (different mood producing the
    // same surname, for instance). Saves get unique ids; users
    // delete duplicates from the drawer if they don't want them.
    const { error } = savedStore.add(payload);
    showToast(error ? `Save failed: ${error.message || 'storage error'}` : `Saved`);
  }

  async function shareCurrent() {
    const payload = currentWorkspaceSnapshot();
    if (!payload) return;
    const url = buildShareUrl(encodeWorkspace(payload));
    try {
      await navigator.clipboard.writeText(url);
      showToast('Share link copied');
    } catch (err) {
      // wyrd-tz35: clipboard requires secure context. Future:
      // surface fallback panel; for now toast tells the user the
      // copy didn't land + the URL is logged for the dev.
      console.warn('share: clipboard write failed', err, url);
      showToast('Copy failed — see console');
    }
  }
</script>

<header class="header" class:mobile={isMobileViewport}>
  <div class="left">
    {#if isMobileViewport && onMenuToggle}
      <button
        class="menu-btn"
        type="button"
        onclick={onMenuToggle}
        aria-label="Open configure menu">☰</button>
    {/if}
    <span class="brand" aria-label="wyrd">
      <span class="rune" aria-hidden="true">ᚹ</span>
      <span class="wordmark">wyrd</span>
    </span>
    {#if appState.manifest}
      <span class="path-sep" aria-hidden="true">/</span>
      <label class="picker-wrap">
        <span class="sr-only">Generator</span>
        <select class="gen-picker" bind:value={appState.selectedGeneratorName}>
          {#each appState.manifest.generators as g (g.name)}
            <option value={g.name}>{g.name}</option>
          {/each}
        </select>
        <span class="picker-caret" aria-hidden="true">▾</span>
      </label>
    {/if}
  </div>

  <div class="center">
    <label class="seed-field">
      <span class="seed-label">seed</span>
      <input
        type="number"
        class="seed-input"
        bind:value={appState.seed}
        placeholder="—"
        aria-label="Seed (blank = random)"
      />
    </label>
    <button
      class="roll-btn"
      type="button"
      onclick={rollCurrent}
      disabled={appState.isRolling || !appState.selectedGeneratorName}
    >
      {appState.isRolling ? '···' : 'Roll'}
    </button>
  </div>

  <div class="right">
    <button
      class="hdr-action"
      type="button"
      onclick={saveCurrent}
      disabled={!canSaveOrShare}
      title="Save current workspace"
      aria-label="Save current workspace"
    >
      <span class="hdr-icon" aria-hidden="true">+</span>
      <span class="hdr-label">Save</span>
    </button>
    <button
      class="hdr-action"
      type="button"
      onclick={shareCurrent}
      disabled={!canSaveOrShare}
      title="Copy share link"
      aria-label="Copy share link"
    >
      <span class="hdr-icon" aria-hidden="true">↗</span>
      <span class="hdr-label">Share</span>
    </button>
    <button
      class="hdr-action saved"
      type="button"
      onclick={() => (appState.savedDrawerOpen = !appState.savedDrawerOpen)}
      title="Saved library"
      aria-label="Open saved library"
      aria-expanded={appState.savedDrawerOpen}
    >
      <span class="hdr-icon" aria-hidden="true">☰</span>
      <span class="hdr-label">Saved</span>
      {#if savedCount > 0}
        <span class="hdr-count">{savedCount}</span>
      {/if}
    </button>
    {#if toastNote}
      <span class="toast" role="status">{toastNote}</span>
    {/if}
  </div>
</header>

<style>
  .header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 24px;
    padding: 0 20px;
    height: 52px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .left {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
  }
  .brand {
    display: inline-flex;
    align-items: baseline;
    gap: 8px;
    user-select: none;
  }
  /* The wynn rune is the one moment of accent color in the header.
     Sized larger than the wordmark so it reads as a logomark, not
     just a leading letter. Slight optical lift via translate to
     sit on the baseline cleanly across runic-capable fonts. */
  .rune {
    font-family: 'Noto Sans Runic', 'Segoe UI Historic', ui-serif, serif;
    font-size: 22px;
    line-height: 1;
    color: var(--accent);
    transform: translateY(2px);
  }
  .wordmark {
    font-family: ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.16em;
    color: var(--fg);
    text-transform: lowercase;
  }
  .path-sep {
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 14px;
  }
  .picker-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
  }
  .gen-picker {
    appearance: none;
    background: transparent;
    border: none;
    color: var(--fg);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
    cursor: pointer;
    padding: 4px 18px 4px 4px;
    border-radius: 3px;
  }
  .gen-picker:hover {
    color: var(--accent);
  }
  .gen-picker:focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: 2px;
  }
  .picker-caret {
    position: absolute;
    right: 4px;
    color: var(--fg-muted);
    font-size: 9px;
    pointer-events: none;
  }
  .center {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .seed-field {
    display: inline-flex;
    align-items: center;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0 10px;
    height: 32px;
    transition: border-color 120ms ease;
  }
  .seed-field:focus-within {
    border-color: var(--accent);
  }
  .seed-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--fg-muted);
    margin-right: 8px;
  }
  .seed-input {
    background: transparent;
    border: none;
    color: var(--fg);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    width: 80px;
    padding: 0;
    outline: none;
  }
  .seed-input::-webkit-outer-spin-button,
  .seed-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .seed-input[type='number'] {
    -moz-appearance: textfield;
    appearance: textfield;
  }
  .seed-input::placeholder {
    color: var(--fg-muted);
  }
  .roll-btn {
    background: var(--accent);
    color: #1a1a1c;
    border: none;
    border-radius: 3px;
    height: 32px;
    padding: 0 22px;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    cursor: pointer;
    transition: filter 120ms ease;
  }
  .roll-btn:hover:not(:disabled) {
    filter: brightness(1.12);
  }
  .roll-btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .roll-btn:focus-visible {
    outline: 2px solid var(--fg);
    outline-offset: 2px;
  }
  .right {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 4px;
    position: relative; /* anchor the toast */
  }

  .hdr-action {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: transparent;
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 6px 10px;
    height: 32px;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease;
  }
  .hdr-action:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--border);
  }
  .hdr-action:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .hdr-action:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .hdr-icon {
    font-size: 14px;
    line-height: 1;
    transform: translateY(-1px);
  }
  /* .hdr-label: full text on desktop, display:none at mobile (see
     media query). No desktop rules needed — inherits from button. */
  .hdr-count {
    background: var(--accent);
    color: #1a1a1c;
    border-radius: 8px;
    padding: 1px 6px;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    margin-left: 2px;
  }
  .toast {
    position: absolute;
    right: 0;
    top: 36px;
    background: var(--bg-elev);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 11px;
    font-style: italic;
    white-space: nowrap;
    z-index: 1;
  }
  .menu-btn {
    background: transparent;
    border: none;
    color: var(--fg);
    font-size: 20px;
    cursor: pointer;
    min-width: 44px;
    min-height: 44px;
    line-height: 1;
    padding: 0;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* Mobile (<900px): collapse to ☰ | ᚹ wyrd / picker | (empty right).
     Seed + Roll move to the col 1 drawer at narrow widths so the
     bar stays uncluttered. The picker stays in the header — it's
     the primary nav. */
  @media (max-width: 899px) {
    .header {
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      padding: 0 12px;
      height: 48px;
    }
    .center {
      display: none;
    }
    .wordmark {
      font-size: 13px;
      letter-spacing: 0.14em;
    }
    .rune {
      font-size: 20px;
    }
    /* Mobile: hide the text labels; icons-only to fit. */
    .hdr-label {
      display: none;
    }
    .hdr-action {
      padding: 6px 8px;
    }
  }
</style>
