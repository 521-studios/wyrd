<script>
  // wyrd-vslw (composer phase 1): center-stage modal that replaces
  // the per-field mood chip-add + tag-grid renderings in
  // ConfigureColumn. Catalog (left) → click to add. Active stack
  // (right) → click chip × to remove. Apply writes back to
  // appState.currentParams; Cancel discards the working copy.
  //
  // The modal is generator-aware via appState.selectedGenerator's
  // input_schema: reads x-pick-from for mood options and items.enum
  // for tag options. Renders sections only when the schema declares
  // the corresponding field — i.e., a hypothetical future generator
  // with mood but no tags only shows the Moods section.
  //
  // Future phases (wyrd-9qsc / wyrd-5vf7 / wyrd-c47y):
  //  - per-mood weight slider for 'harsh:0.5' syntax
  //  - recipes (built-in presets + user-saved combinations)
  //  - export/import recipes as JSON

  import { appState } from '../lib/appState.svelte.js';
  import { flagOn } from '../lib/featureFlags.js';
  import { selectSingle, toggleMulti, removeValue } from '../lib/composerSelection.js';

  let { open = $bindable(false) } = $props();

  // Schema-derived option pools. $derived so they refresh when the
  // operator changes the active generator while the modal is open.
  const schemaProps = $derived(
    appState.selectedGenerator?.input_schema?.properties || {},
  );
  const moodSchema = $derived(schemaProps.mood);
  const tagSchema = $derived(schemaProps.tags);
  const moodOptions = $derived(moodSchema?.['x-pick-from'] || []);
  const tagOptions = $derived(tagSchema?.items?.enum || []);

  // wyrd-0gou: the moods + tags halves are independently feature-flagged.
  // A section shows only when the schema declares it AND its flag is on
  // (the composer trigger already requires at least one to be on).
  const showMoods = $derived(!!moodSchema && flagOn(appState.config, 'moods'));
  const showTags = $derived(!!tagSchema && flagOn(appState.config, 'tags'));

  // Working copy — local to the modal until Apply. Cancel / Esc /
  // backdrop click discards. Initialized on open from currentParams.
  let workingMoods = $state([]);
  let workingTags = $state([]);
  let triggerBefore = null;
  let firstFocusEl = $state();
  let modalEl = $state();

  $effect(() => {
    if (!open) return;
    triggerBefore = document.activeElement;
    // wyrd-vslw round 2 (Gemini HIGH): guard currentParams. The
    // composer trigger only renders when the schema declares mood
    // or tags, and ConfigureColumn's $effect.pre calls ensureParams
    // before children mount — but the derived getter still CAN
    // return null briefly (e.g., manifest race on first paint), so
    // a defensive fallback to {} keeps the initial read safe.
    const params = appState.currentParams || {};
    workingMoods = [...(params.mood || [])];
    workingTags = [...(params.tags || [])];

    const onKey = (e) => {
      if (e.key === 'Escape') {
        cancel();
        return;
      }
      // wyrd-vslw round 2 (Gemini MED): focus trap. Cycle Tab /
      // Shift+Tab between the modal's first + last focusable
      // elements so keyboard users can't tab out into the
      // backgrounded page. Re-querying on each Tab event handles
      // dynamic content (chip count changes as user toggles).
      if (e.key !== 'Tab' || !modalEl) return;
      // wyrd-vslw round 3 (Gemini MED): include <summary> — the
      // catalog uses <details>/<summary> for the Moods / Tags
      // section toggles; those are keyboard-focusable and need to
      // participate in the cycle. Today the close button is first
      // and Apply is last so the omission was latent, but future
      // layout shuffles could put a summary at the boundary.
      const focusables = modalEl.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])'
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

  // wyrd-tbcj.1: moods are single-select (radio-style) to match the
  // runtime's "one mood morpheme per name" overlay (wyrd-4rp8/#565) —
  // selecting another mood replaces the current one, re-clicking clears it.
  // Tags stay multi-select. See lib/composerSelection.js.
  function selectMood(m) {
    workingMoods = selectSingle(workingMoods, m);
  }
  function removeMood(m) {
    workingMoods = removeValue(workingMoods, m);
  }
  function toggleTag(t) {
    workingTags = toggleMulti(workingTags, t);
  }
  function removeTag(t) {
    workingTags = removeValue(workingTags, t);
  }

  function apply() {
    // wyrd-vslw round 2 (Gemini HIGH): guard currentParams during
    // write. If the params dict somehow isn't initialized (manifest
    // race, generator changed between trigger click and apply),
    // ensure via the appState helper before writing.
    if (appState.selectedGeneratorName) {
      appState.ensureParams(appState.selectedGeneratorName);
    }
    const params = appState.currentParams;
    if (!params) {
      // No generator selected — modal shouldn't have been openable
      // in the first place, but bail rather than crash.
      open = false;
      return;
    }
    if (showMoods) params.mood = [...workingMoods];
    if (showTags) params.tags = [...workingTags];
    open = false;
  }
  function cancel() {
    open = false;
  }
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="backdrop" onclick={cancel}></div>

  <div
    bind:this={modalEl}
    class="modal"
    role="dialog"
    aria-modal="true"
    aria-label="Customize moods and tags"
  >
    <header class="hdr">
      <h2>Customize</h2>
      <button
        bind:this={firstFocusEl}
        class="close"
        type="button"
        onclick={cancel}
        aria-label="Close"
      >✕</button>
    </header>

    <div class="panes">
      <section class="catalog" aria-label="Catalog">
        {#if showMoods && moodOptions.length > 0}
          <details open>
            <summary>Moods <span class="count">({moodOptions.length})</span></summary>
            <div class="options">
              {#each moodOptions as m (m)}
                <button
                  type="button"
                  class="opt"
                  class:active={workingMoods.includes(m)}
                  aria-pressed={workingMoods.includes(m)}
                  onclick={() => selectMood(m)}
                >{m}</button>
              {/each}
            </div>
          </details>
        {/if}
        {#if showTags && tagOptions.length > 0}
          <details open>
            <summary>Tags <span class="count">({tagOptions.length})</span></summary>
            <div class="options">
              {#each tagOptions as t (t)}
                <button
                  type="button"
                  class="opt"
                  class:active={workingTags.includes(t)}
                  onclick={() => toggleTag(t)}
                >{t}</button>
              {/each}
            </div>
          </details>
        {/if}
      </section>

      <section class="active" aria-label="Active stack">
        {#if showMoods}
          <h3>Mood <span class="count">(one per name)</span></h3>
          {#if workingMoods.length === 0}
            <p class="empty">No mood selected. Click a mood at left to set one.</p>
          {:else}
            <div class="chips">
              {#each workingMoods as m (m)}
                <span class="chip">
                  {m}
                  <button
                    type="button"
                    class="chip-x"
                    onclick={() => removeMood(m)}
                    aria-label="remove {m}"
                  >×</button>
                </span>
              {/each}
            </div>
          {/if}
        {/if}
        {#if showTags}
          <h3>Tags <span class="count">({workingTags.length})</span></h3>
          {#if workingTags.length === 0}
            <p class="empty">No tags selected. Click a tag at left to add.</p>
          {:else}
            <div class="chips">
              {#each workingTags as t (t)}
                <span class="chip">
                  {t}
                  <button
                    type="button"
                    class="chip-x"
                    onclick={() => removeTag(t)}
                    aria-label="remove {t}"
                  >×</button>
                </span>
              {/each}
            </div>
          {/if}
        {/if}
      </section>
    </div>

    <footer class="actions">
      <button type="button" class="btn" onclick={cancel}>Cancel</button>
      <button type="button" class="btn primary" onclick={apply}>Apply</button>
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
    width: min(900px, calc(100vw - 32px));
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
    color: var(--muted);
    font-weight: 600;
  }
  .close {
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
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

  .panes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    flex: 1;
    overflow: hidden;
  }
  .catalog,
  .active {
    background: var(--bg);
    padding: 16px 18px;
    overflow-y: auto;
  }

  /* Catalog details/summary mimic Advanced disclosure style. */
  .catalog details {
    margin-bottom: 14px;
  }
  .catalog summary {
    cursor: pointer;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 8px;
    list-style: none;
  }
  .catalog summary::-webkit-details-marker { display: none; }
  .catalog summary::before {
    content: '▾ ';
    display: inline-block;
    transform: rotate(-90deg);
    transition: transform 120ms ease;
    margin-right: 4px;
    color: var(--muted);
  }
  .catalog details[open] summary::before {
    transform: rotate(0);
  }
  .count {
    font-weight: 400;
    opacity: 0.65;
  }
  .options {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .opt {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--fg);
    padding: 4px 10px;
    border-radius: 3px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    transition: border-color 120ms ease, background 120ms ease;
  }
  .opt:hover,
  .opt:focus-visible {
    border-color: var(--accent);
    outline: none;
  }
  .opt.active {
    background: color-mix(in oklab, var(--accent) 22%, var(--bg-elev));
    border-color: var(--accent);
  }

  .active h3 {
    margin: 0 0 8px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    font-weight: 600;
  }
  .active h3:not(:first-child) {
    margin-top: 18px;
  }
  .empty {
    color: var(--muted);
    font-size: 13px;
    font-style: italic;
    margin: 0;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: color-mix(in oklab, var(--accent) 18%, var(--bg-elev));
    border: 1px solid var(--accent);
    color: var(--fg);
    padding: 3px 4px 3px 10px;
    border-radius: 3px;
    font-size: 13px;
  }
  .chip-x {
    background: transparent;
    border: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0 6px;
    border-radius: 2px;
  }
  .chip-x:hover,
  .chip-x:focus-visible {
    color: var(--fg);
    background: rgba(255, 255, 255, 0.08);
    outline: none;
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
    font-size: 13px;
  }
  .btn:hover,
  .btn:focus-visible {
    border-color: var(--accent);
    outline: none;
  }
  .btn.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--bg);
    font-weight: 600;
  }
  .btn.primary:hover,
  .btn.primary:focus-visible {
    filter: brightness(1.1);
  }

  /* Mobile: single column. */
  @media (max-width: 700px) {
    .panes {
      grid-template-columns: 1fr;
      grid-template-rows: 1fr 1fr;
    }
  }
</style>
