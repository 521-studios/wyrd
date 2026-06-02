<script>
  // wyrd-qc0g: the family × era variant grid for ONE morpheme (Section 3,
  // basic). Renders the morpheme's `era_grid`: a subsection per language
  // family, era-stages across the top, clickable form cells. Clicking a cell
  // swaps this morpheme in the active name card (pipeline swap) and the cell's
  // pronunciation comes along. The current variant (the form live in the name)
  // is highlighted.
  //
  // Refinements deferred to wyrd-zw1f: drift-gloss surfacing, exact-cell
  // (homograph) tracking, inferred-source (phonology-rule) marking, the
  // no-matching-cell pinned-extra case.
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
      <div class="stages">
        {#each section?.stages || [] as stage (stage.language)}
          <div class="stage">
            <div class="stage-label">{languageLabel(stage.language)}</div>
            <!-- key on form+index: a stage CAN carry duplicate surface forms
                 (homographs / spelling variants), and a bare cell.form key
                 would throw Svelte's duplicate-key error. -->
            {#each (stage?.forms || []).filter(Boolean) as cell, i (cell?.form + '|' + i)}
              <button
                type="button"
                class="cell"
                class:current={isCurrent(stage, cell)}
                aria-pressed={isCurrent(stage, cell)}
                onclick={() => swap(stage, cell)}
                title={isCurrent(stage, cell)
                  ? 'Live in the name — click to revert'
                  : `Swap this morpheme to ${cell.form}`}
              >
                <span class="cell-form">{withPlacement(cell.form)}</span>
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
  .stages {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    align-items: flex-start;
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
