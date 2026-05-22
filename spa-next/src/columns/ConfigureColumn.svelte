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
  import { onMount } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { fetchManifest } from '../lib/api.js';
  import { rollCurrent } from '../lib/roll.js';
  import { partitionFields } from '../lib/headlineFields.js';
  import Field from '../components/Field.svelte';

  // wyrd-14hn: when the drawer is the active surface (mobile), the
  // header's seed+Roll cluster is hidden (column space is too
  // narrow). Render them in the drawer instead so mobile users
  // have universal-control access without going back to the
  // header. Tracked locally via matchMedia.
  let isMobileViewport = $state(false);
  $effect(() => {
    const mq = window.matchMedia('(max-width: 899px)');
    isMobileViewport = mq.matches;
    const onChange = (e) => (isMobileViewport = e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  });

  onMount(async () => {
    try {
      const manifest = await fetchManifest();
      appState.manifest = manifest;
      if (manifest.generators.length > 0 && !appState.selectedGeneratorName) {
        appState.selectedGeneratorName = manifest.generators[0].name;
      }
    } catch (err) {
      appState.manifestError = err.message;
    }
  });

  // wyrd-o7lp (PR #314 Gemini HIGH): init the per-generator params
  // dict via $effect.pre, which runs BEFORE child effects on the
  // same render pass. That means Field components see currentParams
  // populated by their own $effect's first read — no render-before-
  // effect race like the earlier failed attempt with plain $effect.
  $effect.pre(() => {
    if (appState.selectedGeneratorName) {
      appState.ensureParams(appState.selectedGeneratorName);
    }
  });

  let fieldPartition = $derived.by(() => {
    const gen = appState.selectedGenerator;
    if (!gen) return { headline: [], advanced: [] };
    return partitionFields(gen.name, gen.input_schema);
  });

  let showAdvanced = $state(false);
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

    {#if fieldPartition.advanced.length > 0}
      <details class="advanced" bind:open={showAdvanced}>
        <summary>Advanced ({fieldPartition.advanced.length})</summary>
        {#each fieldPartition.advanced as [key, prop] (key)}
          <Field fieldKey={key} {prop} />
        {/each}
      </details>
    {/if}

    {#if isMobileViewport}
      <!-- wyrd-14hn: mobile-only seed + Roll. The header hides these
           at narrow widths; drawer is where mobile users find them. -->
      <div class="mobile-controls">
        <label class="seed-row">
          <span class="seed-label">Seed</span>
          <input
            type="number"
            bind:value={appState.seed}
            placeholder="random"
          />
        </label>
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
  .seed-row {
    display: block;
    margin-bottom: 12px;
  }
  .seed-label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--fg-muted);
    margin-bottom: 4px;
  }
  .seed-row input {
    width: 100%;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    font: inherit;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
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
