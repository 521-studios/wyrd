<script>
  // wyrd-hcmc: col 1 — generator picker + params form + roll button.
  //
  // On mount: fetch /api/manifest, populate the generator picker,
  // default-select the first generator. Per-generator form renders
  // from input_schema, partitioned into headline + advanced (the
  // progressive-disclosure design pick from the design session).
  //
  // The roll button POSTs to /api/<generator> with currentParams +
  // seed, writes the response to appState.results.
  import { onMount } from 'svelte';
  import { appState } from '../lib/appState.svelte.js';
  import { fetchManifest, rollGenerator } from '../lib/api.js';
  import { partitionFields } from '../lib/headlineFields.js';
  import Field from '../components/Field.svelte';

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


  let fieldPartition = $derived.by(() => {
    const gen = appState.selectedGenerator;
    if (!gen) return { headline: [], advanced: [] };
    return partitionFields(gen.name, gen.input_schema);
  });

  let showAdvanced = $state(false);

  async function roll() {
    if (!appState.selectedGeneratorName) return;
    appState.isRolling = true;
    appState.rollError = null;
    try {
      const envelope = await rollGenerator(
        appState.selectedGeneratorName,
        appState.currentParams,
        appState.seed,
      );
      appState.results = envelope.results;
      appState.resultsGenerator = envelope.generator;
      appState.seed = envelope.seed; // echo back the seed the server used
      // wyrd-yxf6 round 2: clear the inspector's selection inline
      // with the results assignment. Pre-fix this lived in an
      // OutputColumn $effect — fragile if OutputColumn ever
      // unmounts (mobile drawer per wyrd-jh75 may toggle it).
      appState.currentResultIndex = null;
    } catch (err) {
      appState.rollError = err.message;
    } finally {
      appState.isRolling = false;
    }
  }
</script>

<aside class="column">
  <h2>Configure</h2>

  {#if appState.manifestError}
    <p class="error">Failed to load generators: {appState.manifestError}</p>
  {:else if !appState.manifest}
    <p class="placeholder">Loading…</p>
  {:else}
    <div class="field">
      <label for="gen-select">Generator</label>
      <select id="gen-select" bind:value={appState.selectedGeneratorName}>
        {#each appState.manifest.generators as gen}
          <option value={gen.name}>{gen.display_name || gen.name}</option>
        {/each}
      </select>
      {#if appState.selectedGenerator?.description}
        {#if appState.selectedGenerator.description.length > 140}
          <details class="hint-disclosure">
            <summary class="hint hint-summary">
              {appState.selectedGenerator.description.slice(0, 140)}…
            </summary>
            <p class="hint">{appState.selectedGenerator.description}</p>
          </details>
        {:else}
          <p class="hint">{appState.selectedGenerator.description}</p>
        {/if}
      {/if}
    </div>

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

    <div class="field">
      <label for="seed-input">Seed</label>
      <input
        id="seed-input"
        type="number"
        placeholder="random"
        bind:value={appState.seed}
      />
    </div>

    <button class="roll" onclick={roll} disabled={appState.isRolling}>
      {appState.isRolling ? 'Rolling…' : 'Roll'}
    </button>

    {#if appState.rollError}
      <p class="error">Roll failed: {appState.rollError}</p>
    {/if}
  {/if}
</aside>

<style>
  .field {
    margin-bottom: 14px;
  }
  label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    color: var(--fg);
    margin-bottom: 4px;
    text-transform: capitalize;
  }
  select,
  input[type='number'] {
    width: 100%;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 8px;
    font: inherit;
  }
  select:focus,
  input:focus {
    outline: 1px solid var(--accent);
    outline-offset: -1px;
  }
  .hint {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
    line-height: 1.4;
  }
  .hint-disclosure {
    margin-top: 4px;
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
    display: inline-block;
    transition: transform 0.1s;
  }
  .advanced[open] summary::before {
    content: '▾ ';
  }
  .roll {
    width: 100%;
    background: var(--accent);
    color: #1a1a1c;
    border: none;
    border-radius: 4px;
    padding: 10px;
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    margin-top: 8px;
  }
  .roll:hover:not(:disabled) {
    filter: brightness(1.1);
  }
  .roll:disabled {
    opacity: 0.5;
    cursor: not-allowed;
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
</style>
