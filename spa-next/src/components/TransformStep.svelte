<script>
  // wyrd-kppy: one step card in the pipeline stack. Shows the
  // transform's label + inline-editable params (one control per
  // entry in transform.paramSchema) + the resulting name preview +
  // a remove button + inline error display on failure.
  import { getTransform } from '../lib/transforms/index.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let { step, index } = $props();

  let transform = $derived(getTransform(step.kind));

  // states[i+1] is the output of step i (states[0] = original).
  let stepResult = $derived(pipeline.states[index + 1] || null);
  let stepError = $derived(pipeline.errors[index] || null);

  function remove() {
    pipeline.removeStep(index);
  }
</script>

<article class="step" class:errored={!!stepError}>
  <header>
    <span class="num">{index + 1}.</span>
    <span class="label">{transform.label}</span>
    <button class="remove" type="button" onclick={remove} aria-label="remove step">×</button>
  </header>

  <div class="params">
    {#each Object.entries(transform.paramSchema) as [key, meta] (key)}
      <label class="param">
        <span class="param-label">{meta.label}</span>
        {#if meta.type === 'select'}
          <select bind:value={step.params[key]}>
            {#each meta.options as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </select>
        {:else}
          <input type="text" bind:value={step.params[key]} />
        {/if}
      </label>
    {/each}
    <!-- wyrd-hpjg: transforms with no paramSchema entries (e.g.,
         Swap — its params are baked at click-time, not user-edited
         inline) get a summary() string instead so the step card
         still shows what the step is doing. -->
    {#if Object.keys(transform.paramSchema).length === 0 && typeof transform.summary === 'function'}
      <p class="summary">{transform.summary(step.params)}</p>
    {/if}
  </div>

  {#if stepError}
    <p class="error">⚠ {stepError}</p>
  {:else if stepResult}
    <p class="preview">
      → <span class="name">{stepResult.name}</span>
    </p>
  {:else if pipeline.isRunning}
    <p class="pending">running…</p>
  {/if}
</article>

<style>
  .step {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
  .step.errored {
    border-color: #ef6f6c;
  }
  header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .num {
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
    font-size: 12px;
  }
  .label {
    font-weight: 600;
    color: var(--fg);
    flex: 1;
  }
  .remove {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    font-size: 18px;
    cursor: pointer;
    padding: 0 6px;
    line-height: 1;
  }
  .remove:hover {
    color: #ef6f6c;
  }
  .params {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 6px;
  }
  .param {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .param-label {
    font-size: 11px;
    color: var(--fg-muted);
    min-width: 50px;
  }
  .param select,
  .param input {
    flex: 1;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 3px 6px;
    font: inherit;
    font-size: 12px;
  }
  .preview {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--fg-muted);
  }
  .preview .name {
    color: var(--fg);
    font-weight: 600;
  }
  .pending {
    margin: 6px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
    font-style: italic;
  }
  .summary {
    margin: 0;
    font-size: 11px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  .error {
    margin: 6px 0 0;
    font-size: 12px;
    color: #ef6f6c;
  }
</style>
