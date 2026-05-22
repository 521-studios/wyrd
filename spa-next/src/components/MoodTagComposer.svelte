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

  // Working copy — local to the modal until Apply. Cancel / Esc /
  // backdrop click discards. Initialized on open from currentParams.
  let workingMoods = $state([]);
  let workingTags = $state([]);
  let triggerBefore = null;
  let firstFocusEl = $state();

  $effect(() => {
    if (!open) return;
    triggerBefore = document.activeElement;
    workingMoods = [...(appState.currentParams.mood || [])];
    workingTags = [...(appState.currentParams.tags || [])];

    const onKey = (e) => {
      if (e.key === 'Escape') cancel();
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

  function toggleMood(m) {
    workingMoods = workingMoods.includes(m)
      ? workingMoods.filter((x) => x !== m)
      : [...workingMoods, m];
  }
  function removeMood(m) {
    workingMoods = workingMoods.filter((x) => x !== m);
  }
  function toggleTag(t) {
    workingTags = workingTags.includes(t)
      ? workingTags.filter((x) => x !== t)
      : [...workingTags, t];
  }
  function removeTag(t) {
    workingTags = workingTags.filter((x) => x !== t);
  }

  function apply() {
    if (moodSchema) appState.currentParams.mood = [...workingMoods];
    if (tagSchema) appState.currentParams.tags = [...workingTags];
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
        {#if moodOptions.length > 0}
          <details open>
            <summary>Moods <span class="count">({moodOptions.length})</span></summary>
            <div class="options">
              {#each moodOptions as m (m)}
                <button
                  type="button"
                  class="opt"
                  class:active={workingMoods.includes(m)}
                  onclick={() => toggleMood(m)}
                >{m}</button>
              {/each}
            </div>
          </details>
        {/if}
        {#if tagOptions.length > 0}
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
        {#if moodSchema}
          <h3>Moods <span class="count">({workingMoods.length})</span></h3>
          {#if workingMoods.length === 0}
            <p class="empty">No moods selected. Click a mood at left to add.</p>
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
        {#if tagSchema}
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
