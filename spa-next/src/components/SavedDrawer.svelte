<script>
  // wyrd-8jjx: right-side overlay drawer hosting the saved library.
  // Pre-fix the SavedList lived inline as a col 2 view-toggle.
  // Lifting it into a drawer lets us surface the library from any
  // workspace (header button opens it) without consuming column
  // real estate.
  //
  // Mounted by App.svelte; open state lives in
  // appState.savedDrawerOpen (Header's 📚 button toggles, SavedList's
  // load() closes after restore, this component's backdrop/Esc
  // also close).
  import { appState } from '../lib/appState.svelte.js';
  import SavedList from './SavedList.svelte';

  let closeBtnEl = $state();
  let triggerBefore = $state(null);

  // wyrd-8jjx round 2 (Gemini MED): focus management. On open,
  // stash the previously-focused element + move focus to the
  // close button so keyboard users land inside the dialog. On
  // close, return focus to the stash (typically the Header's
  // 'Saved' button that opened it). Esc handler + body scroll-
  // lock happen here too.
  $effect(() => {
    if (!appState.savedDrawerOpen) return;
    triggerBefore = document.activeElement;
    const onKey = (e) => {
      if (e.key === 'Escape') appState.savedDrawerOpen = false;
    };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Move focus into the dialog. queueMicrotask waits one tick
    // so the drawer markup is in the DOM before we focus.
    queueMicrotask(() => closeBtnEl?.focus?.());
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
      // Return focus to the opener. Guard for when the trigger
      // was removed from the DOM mid-session (rare; defensive).
      if (triggerBefore && typeof triggerBefore.focus === 'function') {
        try { triggerBefore.focus(); } catch (e) { /* gone */ }
      }
      triggerBefore = null;
    };
  });

  function close() {
    appState.savedDrawerOpen = false;
  }
</script>

{#if appState.savedDrawerOpen}
  <button
    class="backdrop"
    type="button"
    onclick={close}
    aria-label="Close saved library"
  ></button>
  <!-- div + role=dialog instead of aside+role=dialog: svelte-check
       a11y flags 'non-interactive aside cannot have interactive
       role dialog'. Use a generic div with the role explicitly. -->
  <div
    class="drawer"
    role="dialog"
    aria-modal="true"
    aria-label="Saved library"
  >
    <header>
      <h2>Saved library</h2>
      <button
        bind:this={closeBtnEl}
        class="close-btn"
        type="button"
        onclick={close}
        aria-label="Close">✕</button>
    </header>
    <div class="body">
      <SavedList />
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    border: none;
    z-index: 80;
    cursor: pointer;
  }
  .drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(420px, 90vw);
    background: var(--bg);
    border-left: 1px solid var(--border);
    z-index: 90;
    display: flex;
    flex-direction: column;
    animation: slideIn 220ms ease;
  }
  @keyframes slideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }
  h2 {
    margin: 0;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--fg-muted);
  }
  .close-btn {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    font-size: 18px;
    cursor: pointer;
    min-width: 44px;
    min-height: 44px;
    line-height: 1;
    padding: 0;
  }
  .close-btn:hover {
    color: var(--fg);
  }
  .body {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  @media (max-width: 899px) {
    .drawer {
      width: 100vw;
      border-left: none;
    }
  }
</style>
