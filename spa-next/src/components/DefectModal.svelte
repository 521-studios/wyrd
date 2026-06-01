<script>
  // wyrd-dsl5: "mark this name defective" modal. Opened from a per-result
  // flag button in OutputColumn. Requires a non-empty free-text reason
  // (Submit stays disabled until the user types one), then POSTs the full
  // reproduction context to /api/defects for operator triage.
  //
  // Mirrors MoodTagComposer's center-modal a11y pattern: backdrop click +
  // Esc close, focus trap, body scroll-lock, focus restore on close.

  import { appState } from '../lib/appState.svelte.js';
  import { reportDefect } from '../lib/api.js';

  // `result` is the GenerationResult being flagged (its name + morphemes +
  // explanation). `open` is one-way (parent derives it from which result is
  // being flagged); we ask the parent to close via the `onclose` callback
  // rather than mutating `open`, so the parent's source-of-truth stays put.
  let { open = false, result = null, onclose } = $props();

  let reason = $state('');
  let submitting = $state(false);
  let error = $state(null);
  let done = $state(false);

  let triggerBefore = null;
  let firstFocusEl = $state();
  let modalEl = $state();

  // Submit is gated on a non-empty trimmed reason — the required-text
  // contract. The server re-validates, but blocking here is the UX.
  const canSubmit = $derived(reason.trim().length > 0 && !submitting);

  $effect(() => {
    if (!open) return;
    // Fresh state each time the modal opens.
    reason = '';
    error = null;
    done = false;
    submitting = false;
    triggerBefore = document.activeElement;

    const onKey = (e) => {
      if (e.key === 'Escape') {
        close();
        return;
      }
      if (e.key !== 'Tab' || !modalEl) return;
      const focusables = modalEl.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !modalEl.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !modalEl.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    queueMicrotask(() => firstFocusEl?.focus?.());

    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      if (triggerBefore && typeof triggerBefore.focus === 'function') {
        try { triggerBefore.focus(); } catch (e) { /* trigger gone */ }
      }
      triggerBefore = null;
    };
  });

  function close() {
    onclose?.();
  }

  async function submit() {
    if (!canSubmit || !result) return;
    submitting = true;
    error = null;
    // The generator that PRODUCED this result (resultsGenerator), and the
    // params keyed under it — not the currently-selected generator, which
    // the user may have changed since rolling. Same reasoning as
    // StarToggle's save path.
    const generator = appState.resultsGenerator;
    try {
      await reportDefect({
        generator,
        reason: reason.trim(),
        result: result.result,
        seed: appState.resultsSeed,
        // The params frozen at roll time (resultsParams); fall back to the
        // per-generator form state for a loaded saved workspace.
        parameters: appState.resultsParams || appState.paramsByGenerator?.[generator] || {},
        explanation: result.explanation || '',
        components: result.components || [],
        morphemes_by_word: result.morphemes_by_word || null,
        bundle_version: appState.resultsBundleVersion,
      });
      done = true;
    } catch (err) {
      error = err.message;
    } finally {
      submitting = false;
    }
  }
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="backdrop" onclick={close}></div>

  <div
    bind:this={modalEl}
    class="modal"
    role="dialog"
    aria-modal="true"
    aria-label="Report defective name"
  >
    <header class="hdr">
      <h2>Report defective name</h2>
      <button
        bind:this={firstFocusEl}
        class="close"
        type="button"
        onclick={close}
        aria-label="Close"
      >✕</button>
    </header>

    <div class="body">
      {#if done}
        <p class="ok">Thanks — report submitted. We'll take a look.</p>
      {:else}
        <p class="name">
          <span class="label">Name</span>
          <strong>{result?.result}</strong>
        </p>
        <label class="field">
          <span class="label">What's wrong with it? <span class="req">(required)</span></span>
          <textarea
            bind:value={reason}
            rows="4"
            placeholder="e.g. ungrammatical compound / wrong morpheme / garbled characters / doesn't read as a place name"
            disabled={submitting}
          ></textarea>
        </label>
        {#if error}
          <p class="err">{error}</p>
        {/if}
      {/if}
    </div>

    <footer class="actions">
      {#if done}
        <button type="button" class="btn primary" onclick={close}>Close</button>
      {:else}
        <button type="button" class="btn" onclick={close} disabled={submitting}>Cancel</button>
        <button type="button" class="btn primary" onclick={submit} disabled={!canSubmit}>
          {submitting ? 'Submitting…' : 'Submit report'}
        </button>
      {/if}
    </footer>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 90;
  }
  .modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(520px, calc(100vw - 32px));
    max-height: calc(100vh - 64px);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6);
    z-index: 100;
  }
  .hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
  }
  .hdr h2 {
    margin: 0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--fg-muted);
    font-weight: 600;
  }
  .close {
    background: transparent;
    border: 1px solid transparent;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 4px 8px;
    border-radius: 3px;
  }
  .close:hover,
  .close:focus-visible {
    color: var(--fg);
    border-color: var(--border);
    outline: none;
  }
  .body {
    padding: 16px 18px;
    overflow-y: auto;
  }
  .label {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--fg-muted);
    margin-bottom: 4px;
  }
  .req {
    color: var(--accent);
    letter-spacing: normal;
    text-transform: none;
  }
  .name {
    margin: 0 0 14px;
  }
  .name strong {
    font-size: 15px;
  }
  .field textarea {
    width: 100%;
    box-sizing: border-box;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg);
    font: inherit;
    padding: 8px 10px;
    resize: vertical;
  }
  .field textarea:focus-visible {
    border-color: var(--accent);
    outline: none;
  }
  .err {
    color: #e06c6c;
    font-size: 12px;
    margin: 10px 0 0;
  }
  .ok {
    margin: 0;
    color: var(--fg);
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 12px 18px;
    border-top: 1px solid var(--border);
  }
  .btn {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 6px 14px;
    border-radius: 3px;
    cursor: pointer;
    font: inherit;
  }
  .btn:hover:not(:disabled),
  .btn:focus-visible {
    border-color: var(--accent);
    outline: none;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.primary {
    background: color-mix(in oklab, var(--accent) 22%, var(--bg-elev));
    border-color: var(--accent);
  }
</style>
