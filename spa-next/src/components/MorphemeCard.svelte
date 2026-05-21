<script>
  // wyrd-yxf6 + wyrd-hpjg: one card per morpheme in InspectorColumn.
  // Shows usage as the card header + meanings + tags + per-source-
  // language panels carrying form / IPA / reader_pronunciation /
  // original_script / dialect (wyrd-cp2d + wyrd-03cx data).
  //
  // wyrd-hpjg: form rows are clickable buttons — picking one adds a
  // Swap step to the pipeline (replaces this morpheme's usage with
  // the picked form). v1 "siblings" = the alternate forms within
  // THIS morpheme's own sources/renderings; cluster-walking (other
  // morphemes sharing an etymon) requires a backend endpoint and
  // lands in a follow-up.
  import { languageLabel } from '../lib/languageLabels.js';
  import { pipeline } from '../lib/pipeline.svelte.js';

  let { morpheme, morphemeIndex } = $props();

  function swapTo(form) {
    pipeline.addStep('swap', {
      wordIndex: morpheme._wordIndex,
      morphemeIndex,
      to: form,
    });
  }

  // Language display order, mirroring _ANCHOR_LANG_PREFERENCE in
  // wyrd/generators/kenning/era/rewind.py — the etymological-anchor
  // priority that the CLI rewinder uses to pick a source-language
  // for time-rewind. OE first, then Norse (Danelaw), then OF/Celtic,
  // then modern_english as a last-resort anchor. Frequency-wise
  // modern_english is the most common rendering in the bundle, but
  // showing it first would bury the historically-interesting forms
  // — the inspector's point is to surface etymology, not
  // distribution. Unknown languages fall through to alphabetical.
  const LANG_PRIORITY = [
    'old_english',
    'old _english',
    'old_scandinavian',
    'old_scandanavian',
    'old_french',
    'celtic_mix',
    'latin',
    'modern_english',
  ];

  // Union of languages present in EITHER sources (the bundle's
  // per-morpheme picks) OR renderings (cross-language rendering
  // data, often richer — the etymological cluster's OE attestations
  // show up in renderings even when sources only carries the modern
  // pick). For each language, union the forms across the two dicts.
  let orderedSourceEntries = $derived.by(() => {
    const sources = morpheme.sources || {};
    const renderings = morpheme.renderings || {};
    const langs = new Set([
      ...Object.keys(sources),
      ...Object.keys(renderings),
    ]);
    const entries = [...langs].map((lang) => {
      const forms = new Set([
        ...(sources[lang] || []),
        ...Object.keys(renderings[lang] || {}),
      ]);
      return [lang, [...forms]];
    });
    entries.sort(([a], [b]) => {
      const ai = LANG_PRIORITY.indexOf(a);
      const bi = LANG_PRIORITY.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
    return entries;
  });

  // For each form within a source language, pull the matching
  // renderings slot (ipa / reader_pronunciation / original_script /
  // dialect) so the panel can render the four-column data with one
  // template iteration.
  function renderingFor(lang, form) {
    return morpheme.renderings?.[lang]?.[form] || {};
  }
</script>

<article class="morpheme" aria-labelledby="m-{morpheme.usage}-{morpheme._wordIndex}">
  <header>
    <!-- h5 for AT heading-nav: h3 (result name) → h4 (Morphemes
         section) → h5 (per-morpheme card). Visual styling stays
         the same; this just gives screen readers a real heading
         label for the card. -->
    <h5 class="usage" id="m-{morpheme.usage}-{morpheme._wordIndex}">
      {morpheme.usage}
    </h5>
    {#if morpheme.rendered && morpheme.rendered !== morpheme.usage}
      <span class="rendered" title="D18/D8 substituted variant">
        → {morpheme.rendered}
      </span>
    {/if}
  </header>

  {#if morpheme.meanings?.length}
    <p class="meanings">{morpheme.meanings.join(', ')}</p>
  {/if}

  {#if morpheme.tags?.length}
    <p class="tags">
      {#each morpheme.tags as tag}
        <span class="tag">{tag}</span>
      {/each}
    </p>
  {/if}

  {#each orderedSourceEntries as [lang, forms] (lang)}
    <section class="source-lang">
      <h4>{languageLabel(lang)}</h4>
      <table class="forms">
        <thead class="sr-only">
          <tr>
            <th>Form</th>
            <th>IPA</th>
            <th>Reader pronunciation</th>
            <th>Dialect</th>
          </tr>
        </thead>
        <tbody>
          {#each forms as form (form)}
            {@const r = renderingFor(lang, form)}
            <tr
              class="form-row"
              class:current={(morpheme.usage || '') === form ||
                (morpheme.usage || '').replace(/^-+|-+$/g, '') === form}
            >
              <td class="form">
                <!-- wyrd-hpjg: each form is a click-to-swap button.
                     Picks the form + appends a Swap step to the
                     pipeline targeting this (word, morpheme) cell.
                     The button approach is keyboard- + screen-reader-
                     accessible; the table semantics still apply. -->
                <button
                  type="button"
                  class="form-btn"
                  onclick={() => swapTo(form)}
                  title="Swap this morpheme's surface to: {form}"
                >
                  {#if r.original_script && r.original_script !== form}
                    <span class="original">{r.original_script}</span>
                    <span class="translit">({form})</span>
                  {:else}
                    {form}
                  {/if}
                </button>
              </td>
              <td class="ipa">{r.ipa || ''}</td>
              <td class="reader" title="reader-friendly pronunciation"
                >{r.reader_pronunciation || ''}</td>
              <td class="dialect">{r.dialect || ''}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/each}
</article>

<style>
  .morpheme {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
  }
  h5.usage {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  .rendered {
    font-size: 12px;
    color: var(--fg-muted);
  }
  .meanings {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--fg);
    line-height: 1.5;
  }
  .tags {
    margin: 0 0 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .tag {
    display: inline-block;
    font-size: 10px;
    padding: 1px 6px;
    background: var(--border);
    color: var(--fg-muted);
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .source-lang {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
  }
  .source-lang h4 {
    margin: 0 0 6px;
    font-size: 10px;
    font-weight: 600;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  table.forms {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  table.forms td {
    padding: 3px 8px 3px 0;
    vertical-align: top;
    line-height: 1.5;
  }
  td.form {
    font-weight: 600;
    color: var(--fg);
    min-width: 60px;
  }
  .form-btn {
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
    font: inherit;
    font-weight: 600;
    padding: 0;
    text-align: left;
  }
  .form-btn:hover {
    color: var(--accent);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  /* wyrd-hpjg round 2 (frontend MED): keyboard focus indicator.
     Without this, Tab users had no visible affordance on the
     swap targets (transparent bg + no border + no outline). */
  .form-btn:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 2px;
  }
  .form-row.current .form-btn {
    color: var(--accent);
  }
  .form-row.current .form-btn::before {
    content: '● ';
    font-size: 8px;
    vertical-align: middle;
  }
  .original {
    font-family: serif;
    font-size: 13px;
  }
  .translit {
    color: var(--fg-muted);
    font-size: 11px;
    margin-left: 4px;
  }
  td.ipa {
    color: var(--fg-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
  }
  td.reader {
    color: var(--accent);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 11px;
    letter-spacing: 0.04em;
  }
  td.dialect {
    color: var(--fg-muted);
    font-size: 10px;
    font-style: italic;
  }
</style>
