<script>
  // wyrd-qc0g + wyrd-zw1f: the family × era variant grid for ONE morpheme
  // (Section 3). Renders the morpheme's `era_grid` as a subsection per language
  // family, with that family's era stages as aligned COLUMNS (oldest→newest)
  // and clickable form cells stacked beneath each. Clicking a cell swaps this
  // morpheme in the active name card (pipeline swap) and its pronunciation
  // comes along; the live variant is highlighted.
  //
  // wyrd-zw1f delivered: era-column layout, inferred-source marking
  // (phonology-rule cells render dashed + italic + "~"), and exact-cell
  // homograph tracking (the highlight prefers the pinned `_lang` stage, via
  // cellForSurface). Still deferred: drift-gloss surfacing — it needs the
  // per-reflex gloss the bundle doesn't ship yet (blocked on wyrd-rogd.1).
  import { languageLabel } from '../lib/languageLabels.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { graftPosition, accentFold } from '../lib/accents.js';
  import { cellForSurface } from '../lib/era.js';

  let { morpheme } = $props();

  // Family display labels (the user's "English, Norse, Celtic, …").
  const FAMILY_LABELS = {
    english: 'English',
    norse: 'Norse',
    brythonic: 'Brythonic (Welsh)',
    goidelic: 'Goidelic (Irish)',
    'norman-french': 'Norman French',
    latin: 'Latin',
  };
  const familyLabel = (f) => FAMILY_LABELS[f] || languageLabel(f);

  // The slot's ORIGINAL generated surface (col-2 result) — the source of the
  // placement dashes + the revert target for the pipeline swap.
  let originalUsage = $derived(
    appState.currentResult?.morphemes_by_word?.[morpheme._wordIndex]?.[
      morpheme._morphemeIndex
    ]?.usage || '',
  );

  // The form live in the name right now (era form if rendered, else usage).
  let liveSurface = $derived(morpheme.rendered || morpheme.usage);

  // The single grid cell matching the live surface — exactly one highlights.
  let currentRef = $derived(cellForSurface(morpheme, liveSurface));

  function isCurrent(stage, cell) {
    return (
      !!currentRef &&
      currentRef.language === stage.language &&
      accentFold(currentRef.cell.form) === accentFold(cell.form)
    );
  }

  // Display the cell form WITH the slot's placement dashes (tre- / -bȳ / hall)
  // so every breakdown is placement-consistent.
  const withPlacement = (form) => graftPosition(originalUsage || morpheme.usage, form);

  function swap(stage, cell) {
    const to = graftPosition(originalUsage || morpheme.usage, cell.form);
    if (isCurrent(stage, cell)) {
      // Clicking the live variant reverts to the generated default.
      pipeline.clearSwap({
        wordIndex: morpheme._wordIndex,
        morphemeIndex: morpheme._morphemeIndex,
      });
      return;
    }
    pipeline.setSwap({
      wordIndex: morpheme._wordIndex,
      morphemeIndex: morpheme._morphemeIndex,
      to,
      original: originalUsage,
      language: stage.language,
    });
  }
</script>

<div class="era-grid">
  {#each morpheme?.era_grid || [] as section (section.family)}
    <section class="family">
      <h6 class="family-label">{familyLabel(section.family)}</h6>
      <!-- era-columns: one aligned grid column per stage (oldest→newest),
           the stage label as the column header, form cells stacked beneath. -->
      <div class="stages" style="--era-cols: {(section?.stages || []).length}">
        {#each section?.stages || [] as stage (stage.language)}
          <div class="stage">
            <div class="stage-label">{languageLabel(stage.language)}</div>
            <!-- key on form+index: a stage CAN carry duplicate surface forms
                 (homographs / spelling variants), and a bare cell.form key
                 would throw Svelte's duplicate-key error. -->
            {#each (stage?.forms || []).filter(Boolean) as cell, i (cell?.form + '|' + i)}
              {@const inferred = cell.source === 'phonology-rule:v1'}
              <button
                type="button"
                class="cell"
                class:current={isCurrent(stage, cell)}
                class:inferred
                aria-pressed={isCurrent(stage, cell)}
                onclick={() => swap(stage, cell)}
                title={isCurrent(stage, cell)
                  ? 'Live in the name — click to revert'
                  : inferred
                    ? `Swap to ${cell.form} — inferred via phonology rule (not attested)`
                    : `Swap this morpheme to ${cell.form}`}
              >
                <span class="cell-form"
                  >{withPlacement(cell.form)}{#if inferred}<span
                      class="cell-mark"
                      aria-hidden="true">~</span
                    >{/if}</span
                >
                {#if cell.reader_pronunciation}
                  <span class="cell-reader">{cell.reader_pronunciation}</span>
                {/if}
                {#if cell.ipa}<span class="cell-ipa">{cell.ipa}</span>{/if}
              </button>
            {/each}
          </div>
        {/each}
      </div>
    </section>
  {/each}
</div>

<style>
  .era-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 8px;
  }
  .family-label {
    margin: 0 0 6px;
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  /* wyrd-zw1f: era-columns — one equal-width aligned column per stage. */
  .stages {
    display: grid;
    grid-template-columns: repeat(var(--era-cols, 1), minmax(68px, 1fr));
    gap: 6px 8px;
    align-items: start;
  }
  .stage {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .stage-label {
    font-size: 9px;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-bottom: 3px;
    border-bottom: 1px solid var(--border);
  }
  .cell {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 7px;
    cursor: pointer;
    font: inherit;
    text-align: left;
    min-width: 0;
  }
  .cell:hover {
    border-color: var(--accent);
  }
  .cell:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .cell.current {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
  }
  .cell-form {
    font-size: 13px;
    font-weight: 600;
    color: var(--fg);
    font-variant-numeric: tabular-nums;
  }
  .cell.current .cell-form {
    color: var(--accent);
  }
  /* wyrd-zw1f: inferred (phonology-rule) cells read as less-certain — dashed
     border + italic form + a ~ marker — so users know the form is derived,
     not attested. */
  .cell.inferred {
    border-style: dashed;
  }
  .cell.inferred .cell-form {
    font-style: italic;
    color: var(--fg-muted);
  }
  .cell-mark {
    margin-left: 2px;
    font-size: 10px;
    color: var(--fg-muted);
    vertical-align: super;
  }
  .cell-reader {
    font-size: 10px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  .cell-ipa {
    font-size: 9px;
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
</style>
