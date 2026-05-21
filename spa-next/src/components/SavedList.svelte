<script>
  // wyrd-34tn: list of saved workspaces, shown in col 2 when the
  // operator toggles from "Current roll" to "Saved". Each entry is
  // a card with the label (editable inline), saved-at timestamp,
  // generator name + step count, plus actions: load (click anywhere
  // on the card), rename, delete.
  //
  // Load behavior: write back into appState (selectedGeneratorName,
  // params, seed, results = [original]) + pipeline.steps. The user
  // sees their full workspace exactly as they saved it.
  import { savedStore } from '../lib/savedStore.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let renamingId = $state(null);
  let renameValue = $state('');

  // wyrd-34tn: use:focusOnMount avoids svelte-check's a11y autofocus
  // warning while still focusing the input on rename-start (rename is
  // user-initiated, so a focus jump is expected, not annoying).
  function focusOnMount(node) {
    node.focus();
    node.select();
  }

  function startRename(entry) {
    renamingId = entry.id;
    renameValue = entry.label;
  }
  function commitRename() {
    if (renamingId) {
      savedStore.rename(renamingId, renameValue.trim() || '(unlabeled)');
      renamingId = null;
    }
  }
  function cancelRename() {
    renamingId = null;
  }

  function load(entry) {
    // wyrd-34tn: clear the pipeline + set our own steps BEFORE
    // mutating appState.currentResultIndex, then flag the
    // InspectorColumn effect to skip its auto-clear. Without the
    // gate the effect would wipe our just-set steps on the
    // subject-change. Manual clear is safe even if the load is to
    // the same subject — pipeline.clear() bumps #runToken so any
    // stale run is also dropped.
    pipeline.clear();
    for (const step of entry.pipeline || []) {
      pipeline.addStep(step.kind, step.params);
    }
    appState.isLoadingSavedWorkspace = true;
    // Restore col 1 state.
    appState.selectedGeneratorName = entry.generator;
    appState.seed = entry.seed;
    if (entry.params) {
      // Replace the per-generator params dict so the form re-renders
      // with the saved values.
      appState.paramsByGenerator[entry.generator] = { ...entry.params };
    }
    // Synthesize a single-result roll so col 2's result list shows
    // the saved original highlighted via currentResultIndex=0.
    // Mutating these together queues a single Svelte tick.
    appState.results = [
      {
        result: entry.original.name,
        morphemes_by_word: entry.original.morphemes_by_word || [],
        // wyrd-34tn round 2 (Gemini MED): restore the saved
        // etymological explanation instead of defaulting to empty.
        explanation: entry.original.explanation || '',
        components: [],
      },
    ];
    appState.resultsGenerator = entry.generator;
    appState.currentResultIndex = 0;
  }

  function exportAll() {
    const blob = new Blob([savedStore.exportJSON()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wyrd-saved-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  let importNote = $state('');
  function importFile(e) {
    const file = e.currentTarget.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = savedStore.importJSON(String(reader.result));
      if (result.error) {
        importNote = `Import failed: ${result.error}`;
      } else {
        importNote = `Imported ${result.added}, skipped ${result.skipped} duplicate(s)`;
      }
      setTimeout(() => {
        importNote = '';
      }, 4000);
    };
    reader.readAsText(file);
    e.currentTarget.value = ''; // allow re-import of same file
  }
</script>

<div class="actions">
  <button class="action" type="button" onclick={exportAll}>⬇ Export all</button>
  <label class="action">
    ⬆ Import
    <input type="file" accept="application/json" onchange={importFile} hidden />
  </label>
</div>

{#if importNote}
  <p class="note">{importNote}</p>
{/if}

{#if savedStore.entries.length === 0}
  <p class="placeholder">
    No saves yet. Click the ★ on a result, or use the Save workspace
    button on the right column.
  </p>
{:else}
  <ul class="saved-list">
    {#each savedStore.entries as entry (entry.id)}
      <li>
        <article class="saved-entry">
          <header>
            {#if renamingId === entry.id}
              <input
                class="rename"
                bind:value={renameValue}
                onblur={commitRename}
                onkeydown={(e) => {
                  if (e.key === 'Enter') commitRename();
                  if (e.key === 'Escape') cancelRename();
                }}
                use:focusOnMount
              />
            {:else}
              <button class="label-btn" type="button" onclick={() => load(entry)}>
                <span class="label">{entry.label}</span>
              </button>
              <button
                class="icon-btn"
                type="button"
                onclick={() => startRename(entry)}
                aria-label="rename"
                title="Rename">✎</button>
              <button
                class="icon-btn"
                type="button"
                onclick={() => savedStore.remove(entry.id)}
                aria-label="delete"
                title="Delete">×</button>
            {/if}
          </header>
          <p class="meta">
            {entry.generator}
            {#if entry.pipeline?.length > 0}
              · {entry.pipeline.length} step{entry.pipeline.length === 1 ? '' : 's'}
            {/if}
            · {new Date(entry.saved_at).toLocaleDateString()}
          </p>
        </article>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .actions {
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
  }
  .action {
    flex: 1;
    background: transparent;
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px;
    font: inherit;
    font-size: 11px;
    cursor: pointer;
    text-align: center;
  }
  .action:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .saved-list {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .saved-list li {
    margin-bottom: 6px;
  }
  .saved-entry {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .label-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--fg);
    text-align: left;
    cursor: pointer;
    padding: 0;
    font: inherit;
    font-weight: 600;
  }
  .label-btn:hover .label {
    color: var(--accent);
  }
  .icon-btn {
    background: transparent;
    border: none;
    color: var(--fg-muted);
    cursor: pointer;
    padding: 2px 6px;
    font-size: 14px;
    line-height: 1;
  }
  .icon-btn:hover {
    color: var(--accent);
  }
  .rename {
    flex: 1;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: 3px;
    padding: 2px 6px;
    font: inherit;
    font-weight: 600;
  }
  .meta {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--fg-muted);
  }
  .placeholder {
    color: var(--fg-muted);
    font-style: italic;
    padding: 16px 0;
    text-align: center;
  }
  .note {
    margin: 0 0 8px;
    font-size: 11px;
    color: var(--accent);
    font-style: italic;
  }
</style>
