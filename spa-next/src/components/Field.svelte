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
  import { snapDependentValue, snapEnumValue } from '../lib/featureFlags.js';
  import { languageLabel } from '../lib/languageLabels.js';

  let { fieldKey, prop } = $props();

  // wyrd-o7lp round 2 (Gemini MED): single derivation for the
  // humanized label, used by both label branches below.
  let humanLabel = $derived(fieldKey.replace(/_/g, ' '));

  // wyrd-b6hd: NO per-field seeding here. The store owns initialization —
  // appState.ensureParams() seeds every field's default (config.defaults
  // override → schema default → type-empty) BEFORE Fields render, so the form
  // binds already-populated values. This deliberately replaces the old
  // per-Field seed $effect, which raced Svelte's bind_select_value
  // undefined-write-back (wyrd-etvd: a <select> mounted with an undefined value
  // wrote its first <option> back before the seed ran). Centralizing init in
  // the store removes that race for EVERY field. Fields here just bind; the
  // snap effects below only correct a value invalid for the current options.

  function isDependentSelect(prop) {
    return Boolean(prop['x-options-by-culture']);
  }

  // wyrd-rogd.2: option display label — '' is the 'no filter' sentinel; when
  // the options are language tags (x-option-language, the era stages) label
  // them via languageLabel so they read 'Old English' not 'old-english'.
  // wyrd-3vju.2: a field may name its empty sentinel via x-empty-option-label
  // (the era field calls '' "Mixed Era" — the native-per-morpheme default — so
  // it reads as a real, chosen mode, not an absence of one).
  const optionLabel = (opt) =>
    opt === ''
      ? prop['x-empty-option-label'] || '(no filter)'
      : prop['x-option-language']
        ? languageLabel(opt)
        : opt;

  // wyrd-0gou: snap-to-valid for plain string-enum selects. The culture enum
  // is filtered by feature flags (visibleCultures), so a seeded value — incl.
  // a WYRD_DEFAULT_<OPT> override — that isn't in the (filtered) options must
  // snap to a valid one. Otherwise the <select> shows a blank/stale selection
  // and an invisible value gets submitted. Mirrors the dependent-select snap
  // below; skips dependent selects (they have their own) and non-enum fields.
  $effect(() => {
    if (isDependentSelect(prop)) return;
    if (prop.type !== 'string' || !Array.isArray(prop.enum) || prop.enum.length === 0) return;
    const params = appState.currentParams;
    if (!params) return;
    // wyrd-etvd/b6hd: snapEnumValue returns undefined for an unseeded value, so
    // this snap never preempts the store's seeded config.defaults override
    // (e.g. WYRD_DEFAULT_SCORING_MODE=vector). It only corrects a
    // DEFINED-but-invalid value (the culture-filtered case).
    const snapped = snapEnumValue(params[fieldKey], prop.enum, prop.default);
    if (snapped !== undefined) params[fieldKey] = snapped;
  });

  // Dependent select: options depend on currently-selected culture.
  // The derivation + snap-to-valid live in a single effect so we
  // avoid the round-1 footgun where the init effect set a default
  // and the snap effect immediately overwrote it (two renders for
  // one decision).
  let dependentOptions = $derived.by(() => {
    if (!isDependentSelect(prop)) return [];
    const culture = appState.currentParams.culture;
    const map = prop['x-options-by-culture'] || {};
    return map[culture] || Object.values(map)[0] || [];
  });

  $effect(() => {
    if (!isDependentSelect(prop)) return;
    if (dependentOptions.length === 0) return;
    // wyrd-etvd/b6hd: same undefined-guard as the plain-enum snap — an unseeded
    // value is left for the store's seeding (preserving any config.defaults
    // override); only a defined value invalid for the current culture's options
    // snaps. wyrd-kqyf: snapDependentValue adds the 'present-day' translation
    // (env-default era token → the culture's present-day stage, incl. on
    // culture switch).
    const snapped = snapDependentValue(
      appState.currentParams[fieldKey],
      dependentOptions,
      prop,
      appState.config,
      fieldKey,
    );
    if (snapped !== undefined) appState.currentParams[fieldKey] = snapped;
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
  <!-- For inputs with a single underlying element (select / number /
       text / checkbox), `for` associates label with id. For composite
       widgets (tag-grid, chips), we render the label as a generic
       group caption + use aria-labelledby on the group container. -->
  <!-- wyrd-o7lp (PR #314 Gemini MED): humanize snake_case keys
       like 'spelling_variety' to 'spelling variety'. CSS already
       capitalizes; rendered output reads 'Spelling Variety'.
       wyrd-o7lp round 2: DRY'd via the humanLabel $derived
       (defined in <script>) — {@const} can't be a direct child
       of <div> in Svelte 5. -->
  {#if (prop.type === 'array')}
    <span class="label" id="label-{fieldKey}">{humanLabel}</span>
  {:else}
    <label for="field-{fieldKey}">{humanLabel}</label>
  {/if}

  {#if isDependentSelect(prop)}
    <!-- wyrd-rogd.2: when the options are language tags (x-option-language),
         label them via languageLabel (old-english → "Old English") so the era
         dropdown matches the col-3 grid's stage headers. -->
    <select id="field-{fieldKey}" bind:value={appState.currentParams[fieldKey]}>
      {#each dependentOptions as opt}
        <option value={opt}>{optionLabel(opt)}</option>
      {/each}
    </select>
  {:else if prop.type === 'string' && Array.isArray(prop.enum)}
    <select id="field-{fieldKey}" bind:value={appState.currentParams[fieldKey]}>
      {#each prop.enum as opt}
        <option value={opt}>{opt}</option>
      {/each}
    </select>
  {:else if prop.type === 'array' && prop.items?.enum}
    <div class="tag-grid" role="group" aria-labelledby="label-{fieldKey}">
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
    <div class="chips" role="group" aria-labelledby="label-{fieldKey}">
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
        aria-label="add {fieldKey} value"
        bind:value={chipInput}
        onkeydown={onChipKeydown}
      />
    </div>
  {:else if (prop.type === 'number' || prop.type === 'integer') && prop['x-ui-widget'] === 'slider'}
    <!-- wyrd-0k9o: bounded proportions (novelty, …) render as a slider with a
         live value readout. Opt-in via the schema's `x-ui-widget: 'slider'`
         hint (x-prefixed, matching the repo's SPA-extension keys) so other
         numeric knobs keep the plain box. -->
    <div class="slider-row">
      <input
        id="field-{fieldKey}"
        type="range"
        min={prop.minimum ?? 0}
        max={prop.maximum ?? 1}
        step={prop.type === 'integer' ? 1 : (prop['x-ui-step'] ?? 0.05)}
        bind:value={appState.currentParams[fieldKey]}
      />
      <!-- Round for display only: a restored/shared value can carry IEEE-754
           noise (0.05 steps → 0.30000000000000004); the bound param is untouched. -->
      <output class="slider-value" for="field-{fieldKey}">
        {Math.round((appState.currentParams[fieldKey] ?? 0) * 1000) / 1000}
      </output>
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
    <!-- wyrd-yan: render booleans as checkboxes, NOT text inputs.
         A text-input fallback ships stringified "false" to the
         server where bool("false") is True — silent gate inversion. -->
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
  label,
  .label {
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
  /* wyrd-0k9o: slider row — range input + live value readout. */
  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .slider-row input[type='range'] {
    flex: 1;
    accent-color: var(--accent);
  }
  .slider-value {
    min-width: 2.5em;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    color: var(--fg-muted);
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
