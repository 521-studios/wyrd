<script>
  // wyrd-hcmc: render one schema field from a generator's input_schema.
  // Switches on JSON-schema shape — mirrors spa/app.js's _buildField:
  //
  //   x-options-by-culture present  → dependent select
  //   type=string + enum            → select
  //   type=array + items.enum       → tag grid
  //   type=integer | number         → number input
  //   type=boolean                  → checkbox
  //   else                          → text input
  //
  // The component reads + writes appState.currentParams[key] so form
  // values persist across generator switches (paramsByGenerator stores
  // per-generator).
  import { appState } from '../lib/appState.svelte.js';

  let { fieldKey, prop } = $props();

  // Initialize the param from schema default if not already set.
  // Wrapped in $effect because Svelte 5 warns when $props values are
  // read at the top level of <script> — they're tracked reactively
  // and the lint rule guards against capturing-the-initial-value
  // bugs. For us the props never change for a given Field instance,
  // so the effect runs exactly once at mount.
  $effect(() => {
    if (appState.currentParams[fieldKey] === undefined) {
      if (prop.default !== undefined) {
        appState.currentParams[fieldKey] = prop.default;
      } else if (prop.type === 'array') {
        appState.currentParams[fieldKey] = [];
      } else if (prop.type === 'boolean') {
        appState.currentParams[fieldKey] = false;
      } else {
        appState.currentParams[fieldKey] = '';
      }
    }
  });

  function isDependentSelect(prop) {
    return Boolean(prop['x-options-by-culture']);
  }

  // Dependent select: options depend on currently-selected culture.
  // Re-derive the option list whenever culture changes; preserve the
  // current selection when it's still valid for the new culture.
  let dependentOptions = $derived.by(() => {
    if (!isDependentSelect(prop)) return [];
    const culture = appState.currentParams.culture;
    const map = prop['x-options-by-culture'] || {};
    return map[culture] || Object.values(map)[0] || [];
  });

  // If the current value isn't in the new option set, snap to the
  // default or first option. This handles the "switch culture →
  // current era cell is invalid" case from the legacy SPA.
  $effect(() => {
    if (!isDependentSelect(prop)) return;
    const value = appState.currentParams[fieldKey];
    if (dependentOptions.length > 0 && !dependentOptions.includes(value)) {
      if (prop.default !== undefined && dependentOptions.includes(prop.default)) {
        appState.currentParams[fieldKey] = prop.default;
      } else {
        appState.currentParams[fieldKey] = dependentOptions[0];
      }
    }
  });

  // Tag-grid checkbox toggle: array-of-strings field value.
  function toggleTag(tag, checked) {
    const current = appState.currentParams[fieldKey] || [];
    if (checked) {
      appState.currentParams[fieldKey] = [...current, tag];
    } else {
      appState.currentParams[fieldKey] = current.filter((t) => t !== tag);
    }
  }

  // Free-array field (mood: type=array but no items.enum). User types
  // a value + Enter → adds as a chip. Click chip × → removes.
  let chipInput = $state('');
  function addChip() {
    const v = chipInput.trim();
    if (!v) return;
    const current = appState.currentParams[fieldKey] || [];
    if (!current.includes(v)) {
      appState.currentParams[fieldKey] = [...current, v];
    }
    chipInput = '';
  }
  function removeChip(idx) {
    const current = appState.currentParams[fieldKey] || [];
    appState.currentParams[fieldKey] = current.filter((_, i) => i !== idx);
  }
  function onChipKeydown(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      addChip();
    }
  }
</script>

<div class="field">
  <label for="field-{fieldKey}">{fieldKey}</label>

  {#if isDependentSelect(prop)}
    <select id="field-{fieldKey}" bind:value={appState.currentParams[fieldKey]}>
      {#each dependentOptions as opt}
        <option value={opt}>{opt === '' ? '(no filter)' : opt}</option>
      {/each}
    </select>
  {:else if prop.type === 'string' && Array.isArray(prop.enum)}
    <select id="field-{fieldKey}" bind:value={appState.currentParams[fieldKey]}>
      {#each prop.enum as opt}
        <option value={opt}>{opt}</option>
      {/each}
    </select>
  {:else if prop.type === 'array' && prop.items?.enum}
    <div class="tag-grid">
      {#each prop.items.enum as tag}
        <label class="tag">
          <input
            type="checkbox"
            checked={(appState.currentParams[fieldKey] || []).includes(tag)}
            onchange={(e) => toggleTag(tag, e.currentTarget.checked)}
          />
          {tag}
        </label>
      {/each}
    </div>
  {:else if prop.type === 'array'}
    <div class="chips">
      {#each appState.currentParams[fieldKey] || [] as chip, i}
        <span class="chip">
          {chip}
          <button
            type="button"
            class="chip-x"
            onclick={() => removeChip(i)}
            aria-label="remove {chip}">×</button>
        </span>
      {/each}
      <input
        type="text"
        class="chip-input"
        placeholder="add + Enter"
        bind:value={chipInput}
        onkeydown={onChipKeydown}
      />
    </div>
  {:else if prop.type === 'integer' || prop.type === 'number'}
    <input
      id="field-{fieldKey}"
      type="number"
      min={prop.minimum}
      max={prop.maximum}
      step={prop.type === 'integer' ? 1 : 'any'}
      bind:value={appState.currentParams[fieldKey]}
    />
  {:else if prop.type === 'boolean'}
    <input
      id="field-{fieldKey}"
      type="checkbox"
      bind:checked={appState.currentParams[fieldKey]}
    />
  {:else}
    <input
      id="field-{fieldKey}"
      type="text"
      bind:value={appState.currentParams[fieldKey]}
    />
  {/if}

  {#if prop.description}
    {#if prop.description.length > 140}
      <details class="hint-disclosure">
        <summary class="hint hint-summary">{prop.description.slice(0, 140)}…</summary>
        <p class="hint">{prop.description}</p>
      </details>
    {:else}
      <p class="hint">{prop.description}</p>
    {/if}
  {/if}
</div>

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
  label.tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-weight: 400;
    text-transform: none;
    margin: 0 8px 6px 0;
    cursor: pointer;
  }
  select,
  input[type='number'],
  input[type='text'] {
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
  .tag-grid {
    display: flex;
    flex-wrap: wrap;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
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
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px;
    align-items: center;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--border);
    color: var(--fg);
    border-radius: 3px;
    padding: 2px 4px 2px 8px;
    font-size: 12px;
  }
  .chip-x {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    cursor: pointer;
    font-size: 14px;
    padding: 0 4px;
    line-height: 1;
  }
  .chip-x:hover {
    color: #ef6f6c;
  }
  .chip-input {
    flex: 1;
    min-width: 80px;
    background: transparent;
    border: none;
    color: var(--fg);
    font: inherit;
    outline: none;
    padding: 2px 4px;
  }
</style>
