<script>
  // wyrd-qc0g + wyrd-zw1f: the family × era variant grid for ONE morpheme
  // (Section 3). Renders the morpheme's `era_grid` as a subsection per language
  // family, with that family's era stages as aligned COLUMNS (oldest→newest)
  // and clickable form cells stacked beneath each. Clicking a cell swaps this
  // morpheme in the active name card (pipeline swap) and its pronunciation
  // comes along; the live variant is highlighted.
  //
  // wyrd-zw1f delivered: era-column layout + inferred-source marking
  // (phonology-rule cells render dashed + italic + "~"). wyrd-rogd.1: drift-
  // gloss surfacing — a cell whose gloss differs from the morpheme's own
  // meaning is marked (the cognate drifted), shown on hover. wyrd-rogd.7: the
  // highlight tracks the selected cell by stable `cell.id` (cellForSurface
  // honours `morpheme._cellId`), not surface-fold — so fold-equal forms
  // (bǣre/bære) light exactly one cell, not two.
  import { languageLabel } from '../lib/languageLabels.js';
  import { pipeline } from '../lib/pipeline.svelte.js';
  import { appState } from '../lib/appState.svelte.js';
  import { graftPosition } from '../lib/accents.js';
  import { cellForSurface, primaryGloss, isGlossDrift } from '../lib/era.js';

  let { morpheme } = $props();

  // wyrd-rogd.1: the morpheme's own meaning — the drift baseline a cell's gloss
  // is compared against.
  let baseGloss = $derived(primaryGloss(morpheme));

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

  // wyrd-rogd.12: the canonical per-family era axis from the manifest. Keeping
  // periods in FIXED column positions (empty slots → muted '—') preserves the
  // spatial mapping across morphemes — a morpheme with only Old English data
  // still shows OE in the OE column, not a lone unlabelled column.
  const eraStages = $derived(appState.selectedGenerator?.era_stages || {});

  // Lay a section's populated stages onto its family's full fixed axis. Falls
  // back to the populated stages as-is for a family the manifest doesn't know.
  function axisFor(section) {
    const order = eraStages[section?.family];
    const byLang = new Map((section?.stages || []).map((s) => [s?.language, s]));
    if (!order?.length) {
      return (section?.stages || []).map((s) => ({ language: s?.language, stage: s }));
    }
    return order.map((language) => ({ language, stage: byLang.get(language) || null }));
  }

  // The slot's ORIGINAL generated surface (col-2 result) — the source of the
  // placement dashes + the revert target for the pipeline swap.
  let originalUsage = $derived(
    appState.currentResult?.morphemes_by_word?.[morpheme._wordIndex]?.[
      morpheme._morphemeIndex
    ]?.usage || '',
  );

  // The form live in the name right now (era form if rendered, else usage).
  let liveSurface = $derived(morpheme.rendered || morpheme.usage);

  // The single selected cell — the swapped cell by id (cellForSurface honours
  // morpheme._cellId), else the cell matching the live surface. wyrd-rogd.7:
  // highlight by cell.id, NOT surface-fold, so fold-equal forms (bǣre/bære)
  // light exactly one cell.
  let currentId = $derived(cellForSurface(morpheme, liveSurface)?.cell?.id ?? null);

  function isCurrent(cell) {
    return !!currentId && cell?.id === currentId;
  }

  // Display the cell form WITH the slot's placement dashes (tre- / -bȳ / hall)
  // so every breakdown is placement-consistent.
  const withPlacement = (form) => graftPosition(originalUsage || morpheme.usage, form);

  function swap(stage, cell) {
    const to = graftPosition(originalUsage || morpheme.usage, cell.form);
    if (isCurrent(cell)) {
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
      cellId: cell.id,
    });
  }
</script>

<div class="era-grid">
  {#each morpheme?.era_grid || [] as section (section?.family)}
    {@const axis = axisFor(section)}
    <section class="family">
      <h6 class="family-label">{familyLabel(section?.family)}</h6>
      <!-- era-columns: one aligned grid column per stage (oldest→newest),
           the stage label as the column header, form cells stacked beneath. -->
      <div class="stages" style="--era-cols: {Math.max(1, axis.length)}">
        {#each axis as { language, stage } (language)}
          <div class="stage" class:empty={!stage}>
            <div class="stage-label">{languageLabel(language)}</div>
            <!-- wyrd-rogd.12: a stage with no data holds its column with a
                 muted placeholder so the era axis stays aligned. -->
            {#if !stage}
              <div class="cell-empty" aria-hidden="true">—</div>
            {/if}
            <!-- key on form+index: a stage CAN carry duplicate surface forms
                 (homographs / spelling variants), and a bare cell.form key
                 would throw Svelte's duplicate-key error. -->
            {#each (stage?.forms || []).filter(Boolean) as cell, i (cell?.form + '|' + i)}
              <!-- match the inferred tier by PREFIX so a future phonology-rule:v2
                   stays marked (not just :v1). -->
              {@const inferred = !!cell.source?.startsWith('phonology-rule')}
              <!-- wyrd-rogd.1: the cell's own meaning has DRIFTED from the
                   morpheme's (the cognate no longer means the same thing). -->
              {@const drifted = isGlossDrift(baseGloss, cell.gloss)}
              {@const current = isCurrent(cell)}
              {@const swapTitle = inferred
                ? `Swap to ${cell.form} — inferred via phonology rule (not attested)`
                : `Swap this morpheme to ${cell.form}`}
              {@const liveTitle = inferred
                ? 'Live in the name (inferred via phonology rule) — click to revert'
                : 'Live in the name — click to revert'}
              {@const glossNote = cell.gloss
                ? ` — ${drifted ? 'drifted meaning: ' : 'means: '}${cell.gloss}`
                : ''}
              <button
                type="button"
                class="cell"
                class:current={current}
                class:inferred
                class:drift={drifted}
                aria-pressed={current}
                onclick={() => swap(stage, cell)}
                title={(current ? liveTitle : swapTitle) + glossNote}
              >
                <span class="cell-form"
                  >{withPlacement(cell.form)}{#if inferred}<span
                      class="cell-mark"
                      role="img"
                      aria-label="inferred (phonology rule)">~</span
                    >{/if}{#if drifted}<span
                      class="cell-drift"
                      role="img"
                      aria-label="drifted meaning">≠</span
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
  /* wyrd-rogd.12: an empty era column — label stays, muted placeholder holds
     the slot so the axis reads at a glance across morphemes. */
  .stage.empty .stage-label {
    opacity: 0.55;
  }
  .cell-empty {
    color: var(--fg-muted);
    opacity: 0.5;
    font-size: 13px;
    padding: 4px 2px;
    text-align: center;
    user-select: none;
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
  /* a live (current) cell reads as current even when inferred — current wins
     the form color over the muted inferred style (equal specificity, so this
     must come AFTER .cell.inferred .cell-form). */
  .cell.current.inferred .cell-form {
    color: var(--accent);
  }
  .cell-mark {
    margin-left: 2px;
    font-size: 10px;
    color: var(--fg-muted);
    vertical-align: super;
  }
  /* wyrd-rogd.1: a ≠ marker on cells whose meaning drifted from the morpheme's
     — hover the cell for the drifted gloss. */
  .cell-drift {
    margin-left: 2px;
    font-size: 10px;
    font-weight: 600;
    color: var(--accent);
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
