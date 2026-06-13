<script>
  // wyrd-hcmc + wyrd-14hn: col 1 — params form ONLY. The picker +
  // seed + Roll button moved into the Header (wyrd-14hn). This
  // column now renders the active generator's params form (manifest-
  // driven) with progressive disclosure: headline knobs visible,
  // advanced collapsed.
  //
  // Manifest fetch + initial generator selection still happen here
  // (mounted at app start regardless of which workspace is active),
  // so this is the de-facto bootstrap surface for appState.manifest.
  import { untrack } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { rollCurrent } from '../lib/roll.js';
  import { partitionFields } from '../lib/headlineFields.js';
  import { fieldEnabled, flagOn, visibleCultures } from '../lib/featureFlags.js';
  import Field from '../components/Field.svelte';
  import MoodTagComposer from '../components/MoodTagComposer.svelte';

  // wyrd-14hn round 2 (frontend LOW): manifest fetch lifted to
  // App.svelte (so it loads regardless of which workspace mounts);
  // viewport tracking lifted to appState.isMobileViewport.

  // wyrd-o7lp (PR #314 Gemini HIGH): init the per-generator params dict via
  // $effect.pre, which runs BEFORE child effects on the same render pass — so
  // Field components see currentParams already populated before their
  // <select>s bind (no render-before-init race).
  // wyrd-b6hd: ensureParams now SEEDS every field's default from the manifest
  // schema (the store owns initialization), so this must (re-)run once the
  // manifest loads — not only when selectedGeneratorName changes. Reading
  // `manifest` registers it as a dependency; ensureParams is a no-op until
  // both the generator + its schema are present.
  $effect.pre(() => {
    void appState.manifest;
    if (appState.selectedGeneratorName) {
      appState.ensureParams(appState.selectedGeneratorName);
    }
  });

  let fieldPartition = $derived.by(() => {
    const gen = appState.selectedGenerator;
    if (!gen) return { headline: [], advanced: [], composerVisible: false };
    const cfg = appState.config;
    const part = partitionFields(gen.name, gen.input_schema);
    // wyrd-vslw: when the schema has a mood field with x-pick-from
    // OR a tags field with items.enum, the MoodTagComposer handles
    // them via the 'Customize moods & tags' trigger — strip from
    // both headline and advanced so they don't double-render.
    const props = gen.input_schema?.properties || {};
    const moodComposable = !!props.mood?.['x-pick-from'];
    const tagsComposable = Array.isArray(props.tags?.items?.enum);
    const drop = new Set();
    if (moodComposable) drop.add('mood');
    if (tagsComposable) drop.add('tags');
    // wyrd-0gou: headline fields always show, but the culture select is
    // filtered to english (guaranteed) + flagged-on cultures.
    const headline = part.headline
      .filter(([k]) => !drop.has(k))
      .map(([k, p]) =>
        k === 'culture' && Array.isArray(p.enum)
          ? [k, { ...p, enum: visibleCultures(cfg, p.enum) }]
          : [k, p],
      );
    // wyrd-0gou: each advanced option is gated by its feature flag (default
    // off). When none are on the list is empty and the Advanced panel below
    // doesn't render at all.
    const advanced = part.advanced.filter(([k]) => !drop.has(k) && fieldEnabled(cfg, k));
    // wyrd-0gou: the moods/tags composer trigger shows only when the schema
    // declares the control AND its feature flag is on.
    const composerVisible =
      (moodComposable && flagOn(cfg, 'moods')) || (tagsComposable && flagOn(cfg, 'tags'));
    return {
      headline,
      advanced,
      composerVisible,
      composerSummary: summarizeComposer(),
    };
  });

  function summarizeComposer() {
    const params = appState.currentParams || {};
    const m = (params.mood || []).length;
    const t = (params.tags || []).length;
    if (m === 0 && t === 0) return null;
    const parts = [];
    if (m) parts.push(`${m} mood${m === 1 ? '' : 's'}`);
    if (t) parts.push(`${t} tag${t === 1 ? '' : 's'}`);
    return parts.join(' · ');
  }

  // wyrd-ah53: prune selected tags the chosen culture can't satisfy when the
  // culture changes — the dependent-select analogue of Field.svelte's era/stratum
  // snap-to-valid, but for the composed (non-Field) tags array. Without this a
  // 'monster' selected under welsh would survive a switch to english and roll
  // 'no eligible name'. Keyed on culture; the params.tags read/write is untracked
  // so a tags edit (composer Apply) can't re-trigger it.
  $effect(() => {
    const byCulture =
      appState.selectedGenerator?.input_schema?.properties?.tags?.['x-options-by-culture'];
    const culture = appState.currentParams?.culture; // the dependency
    if (!byCulture) return;
    const valid = byCulture[culture] || Object.values(byCulture)[0] || [];
    untrack(() => {
      const params = appState.currentParams;
      if (!params?.tags?.length) return;
      const pruned = params.tags.filter((t) => valid.includes(t));
      if (pruned.length !== params.tags.length) params.tags = pruned;
    });
  });

  let showAdvanced = $state(false);
  let composerOpen = $state(false);
</script>

<aside class="column">
  <h2>Configure</h2>

  {#if appState.manifestError}
    <p class="error">Failed to load generators: {appState.manifestError}</p>
  {:else if !appState.manifest}
    <p class="placeholder">Loading…</p>
  {:else if appState.selectedGenerator}
    {#if appState.selectedGenerator.description}
      {#if appState.selectedGenerator.description.length > 140}
        <details class="hint-disclosure" open>
          <summary class="hint hint-summary">
            {appState.selectedGenerator.description.slice(0, 140)}…
          </summary>
          <p class="hint">{appState.selectedGenerator.description}</p>
        </details>
      {:else}
        <p class="hint">{appState.selectedGenerator.description}</p>
      {/if}
    {/if}

    {#each fieldPartition.headline as [key, prop] (key)}
      <Field fieldKey={key} {prop} />
    {/each}

    {#if fieldPartition.composerVisible}
      <!-- wyrd-vslw: composer trigger. Shows a summary of the
           current selection so the operator sees state without
           opening the modal. -->
      <button
        class="composer-trigger"
        type="button"
        onclick={() => (composerOpen = true)}
      >
        <span class="composer-label">Customize moods &amp; tags</span>
        {#if fieldPartition.composerSummary}
          <span class="composer-summary">{fieldPartition.composerSummary}</span>
        {:else}
          <span class="composer-summary muted">none selected</span>
        {/if}
      </button>
    {/if}

    {#if fieldPartition.advanced.length > 0}
      <details class="advanced" bind:open={showAdvanced}>
        <summary>Advanced ({fieldPartition.advanced.length})</summary>
        {#each fieldPartition.advanced as [key, prop] (key)}
          <Field fieldKey={key} {prop} />
        {/each}
      </details>
    {/if}

    {#if appState.isMobileViewport}
      <!-- wyrd-14hn: mobile-only Roll. The header hides it at narrow
           widths; the drawer is where mobile users find it. -->
      <div class="mobile-controls">
        <button
          class="roll-btn-mobile"
          type="button"
          onclick={rollCurrent}
          disabled={appState.isRolling}
        >{appState.isRolling ? 'Rolling…' : 'Roll'}</button>
      </div>
    {/if}

    {#if appState.rollError}
      <p class="error">Roll failed: {appState.rollError}</p>
    {/if}
  {/if}
</aside>

<MoodTagComposer bind:open={composerOpen} />

<style>
  aside {
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 16px 20px;
  }
  h2 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--fg-muted);
    margin: 0 0 16px;
  }
  .hint {
    margin: 0 0 14px;
    font-size: 11px;
    color: var(--fg-muted);
    line-height: 1.5;
  }
  .hint-disclosure {
    margin-bottom: 14px;
  }
  .hint-disclosure summary {
    cursor: pointer;
    list-style: none;
  }
  .hint-disclosure summary::marker {
    display: none;
  }
  .hint-disclosure[open] summary {
    display: none;
  }
  /* wyrd-vslw: composer trigger button. Full-width, looks like a
     field but with a chevron suggesting it opens something. */
  .composer-trigger {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    width: 100%;
    margin: 12px 0 16px;
    padding: 10px 12px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--fg);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    text-align: left;
    transition: border-color 120ms ease;
  }
  .composer-trigger:hover,
  .composer-trigger:focus-visible {
    border-color: var(--accent);
    outline: none;
  }
  .composer-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    font-weight: 600;
  }
  .composer-summary {
    color: var(--fg);
    font-variant-numeric: tabular-nums;
  }
  .composer-summary.muted {
    color: var(--muted);
    font-style: italic;
  }

  .advanced {
    margin: 8px 0 16px;
    border-top: 1px solid var(--border);
    padding-top: 12px;
  }
  .advanced summary {
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    list-style: none;
    margin-bottom: 12px;
  }
  .advanced summary::before {
    content: '▸ ';
  }
  .advanced[open] summary::before {
    content: '▾ ';
  }
  .error {
    color: #ef6f6c;
    font-size: 12px;
    margin: 8px 0;
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
  }
  .mobile-controls {
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }
  .roll-btn-mobile {
    width: 100%;
    background: var(--accent);
    color: #1a1a1c;
    border: none;
    border-radius: 4px;
    padding: 12px;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    cursor: pointer;
  }
  .roll-btn-mobile:disabled {
    opacity: 0.4;
  }
</style>
